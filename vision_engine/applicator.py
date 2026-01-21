import cv2
import numpy as np
from pathlib import Path
from django.conf import settings
import os
import subprocess
import shutil


class ModelExecutor:
    """모델 유형별 실행 함수 매핑"""
    
    @staticmethod
    def get_executor(model):
        """모델의 task_type에 따라 적절한 executor 반환"""
        task_type = model.task_type
        
        executors = {
            'object_detection': ObjectDetectionExecutor,
            'super_resolution': SuperResolutionExecutor,
            'image_restoration': ImageRestorationExecutor,
            'image_classification': ImageClassificationExecutor,  # 추가
            'segmentation': SegmentationExecutor,  # 추가
        }
        
        executor_class = executors.get(task_type)
        if not executor_class:
            raise ValueError(f"지원하지 않는 작업 유형입니다: {task_type}")
        
        return executor_class(model)


class BaseExecutor:
    """모든 executor의 기본 클래스"""
    
    def __init__(self, model):
        self.model = model
        self.loaded_model = None
        self.load_model()
    
    def load_model(self):
        """모델 로드 - 하위 클래스에서 구현"""
        raise NotImplementedError
    
    def apply_frame(self, frame):
        """단일 프레임 처리 - 하위 클래스에서 구현"""
        raise NotImplementedError
    
    def process_video(self, input_path, output_path, progress_callback=None):
        """동영상/이미지 처리 - 공통 로직"""
        print(f"\n{'='*60}\n🔍 처리 시작\n{'='*60}")

        # 미디어 타입 판별
        is_image = input_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"파일을 열 수 없습니다: {input_path}")

        # 미디어 정보 추출
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if is_image:
            fps = 1
            total_frames = 1
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"🖼️ 해상도: {width}x{height} | FPS: {fps} | 총 프레임: {total_frames}")

        # 출력 설정
        out = None
        temp_output = output_path
        annotated_frame = None

        if not is_image:
            temp_output = str(
                Path(output_path).parent / f"temp_{Path(output_path).name}"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            if not out.isOpened():
                cap.release()
                raise ValueError("출력 VideoWriter 생성 실패")

        all_applications = []
        application_summary = {}
        total_applications_count = 0
        frame_count = 0

        try:
            print(f"🔄 처리 중...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 프레임 처리 (하위 클래스별로 다름)
                result = self.apply_frame(frame)
                
                # result 처리 방식이 executor마다 다름
                if isinstance(result, dict):  # detection의 경우
                    annotated_frame = result.get('frame', frame)
                    applications = result.get('applications', [])
                    
                    if applications:
                        all_applications.append(
                            {"frame": frame_count, "applications": applications}
                        )
                        total_applications_count += len(applications)
                        for det in applications:
                            label = det["label"]
                            application_summary[label] = application_summary.get(label, 0) + 1
                else:  # super_resolution, restoration의 경우
                    annotated_frame = result

                if not is_image and out:
                    out.write(annotated_frame)

                frame_count += 1
                if progress_callback and frame_count % 10 == 0:
                    progress = int((frame_count / total_frames) * 80)
                    progress_callback(frame_count, total_frames, progress)

        finally:
            cap.release()
            if out:
                out.release()

        # 최종 저장
        if is_image:
            if annotated_frame is not None:
                cv2.imwrite(output_path, annotated_frame)
                print(f"✅ 이미지 결과 저장: {output_path}")
            ffmpeg_success = True
        else:
            print(f"\n🎬 동영상 재인코딩 중...")
            if progress_callback:
                progress_callback(frame_count, total_frames, 85)
            ffmpeg_success = self.reencode_with_ffmpeg(temp_output, output_path)

            if ffmpeg_success and os.path.exists(temp_output):
                os.remove(temp_output)
            elif not ffmpeg_success:
                print(f"⚠️ ffmpeg 실패 - 원본 파일 사용")
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_output, output_path)

        if progress_callback:
            progress_callback(frame_count, total_frames, 100)

        return {
            "applications": all_applications,
            "total_applications": total_applications_count,
            "summary": application_summary,
        }
    
    def reencode_with_ffmpeg(self, input_path, output_path):
        """ffmpeg 재인코딩"""
        ffmpeg_path = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
        if not os.path.exists(ffmpeg_path):
            return False

        try:
            cmd = [
                ffmpeg_path,
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-y",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            return os.path.exists(output_path)
        except:
            return False


class ObjectDetectionExecutor(BaseExecutor):
    """객체 탐지 Executor (YOLOv5 + YOLOv8 지원)"""
    
    def __init__(self, model):
        self.yolo_version = None  # 'v5' 또는 'v8'
        super().__init__(model)
    
    def detect_yolo_version(self, model_path):
        """
        YOLO 버전 자동 감지
        
        Returns:
            'v5': YOLOv5
            'v8': YOLOv8/v9/v10/v11 (ultralytics)
            'onnx': ONNX 파일
        """
        model_path_str = str(model_path).lower()
        
        # 0. ONNX 파일 확인 (최우선)
        if model_path_str.endswith('.onnx'):
            print("🔍 ONNX 파일 확인")
            return 'onnx'
        
        # 1. 파일명 기반 감지
        if 'yolov5' in model_path_str or 'yolo5' in model_path_str:
            print("🔍 파일명으로 YOLOv5 감지")
            return 'v5'
        
        if any(v in model_path_str for v in ['yolov8', 'yolov9', 'yolov10', 'yolo11']):
            print("🔍 파일명으로 YOLOv8+ 감지")
            return 'v8'
        
        # 2. 모델 파일 구조 분석 (더 정확)
        if os.path.exists(model_path):
            try:
                # 파일 헤더로 ONNX 체크
                with open(model_path, 'rb') as f:
                    header = f.read(8)
                    # ONNX magic number: 0x08 0x03/0x07 ...
                    if header and header[0] == 0x08:
                        print("✅ ONNX 매직 넘버 확인")
                        return 'onnx'
                
                import torch
                print(f"🔍 모델 파일 구조 분석 중: {model_path}")
                
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if isinstance(checkpoint, dict):
                    # YOLOv5 특징: 'model' 키 + DetectionModel 구조
                    if 'model' in checkpoint:
                        model_obj = checkpoint.get('model')
                        # YOLOv5는 model이 객체이고 names 속성을 가짐
                        if hasattr(model_obj, 'names') or hasattr(model_obj, 'yaml'):
                            print("✅ 모델 구조로 YOLOv5 확인")
                            return 'v5'
                    
                    # YOLOv8 특징: 'train_args' 또는 다른 ultralytics 포맷
                    if 'train_args' in checkpoint or 'date' in checkpoint:
                        print("✅ 모델 구조로 YOLOv8 확인")
                        return 'v8'
            except Exception as e:
                print(f"⚠️ 모델 구조 분석 실패: {e}")
                error_str = str(e)
                
                # torch.load 실패 = ONNX일 가능성
                if 'invalid load key' in error_str or 'UnpicklingError' in error_str:
                    print("✅ PyTorch 로드 실패 → ONNX 파일로 판단")
                    return 'onnx'
                
                # models.yolo/common 오류 = YOLOv5
                if 'models.yolo' in error_str or 'models.common' in error_str:
                    print("✅ models.yolo 오류 감지 → YOLOv5로 판단")
                    return 'v5'
        
        # 3. 기본값: YOLOv8 (ultralytics가 더 최신)
        print("⚠️ 버전 감지 실패 - 기본값 YOLOv8 사용")
        return 'v8'
    
    def load_model(self):
        """YOLO 모델 로드"""
        print("\n" + "="*60)
        print("🔄 YOLO 모델 로딩 시작")
        print("="*60)
        
        # ⭐ 통합 Model의 source에 따라 분기
        if self.model.source == 'builtin':
            self._load_builtin_model()
        elif self.model.source == 'upload':
            self._load_upload_model()
        elif self.model.source == 'git':
            self._load_git_model()
        elif self.model.source == 'huggingface':
            self._load_huggingface_model()
        else:
            raise ValueError(f"지원하지 않는 모델 소스: {self.model.source}")
        
        print(f"✅ YOLO{self.yolo_version.upper()} 로드 완료")
        print("="*60 + "\n")
    
    def _load_builtin_model(self):
        """Built-in 모델 로드"""
        from ultralytics import YOLO
        
        # YOLOv8 builtin은 자동 다운로드
        self.yolo_version = 'v8'
        
        builtin_dir = settings.DEFAULT_MODELS_DIR
        os.makedirs(builtin_dir, exist_ok=True)
        
        # 프리셋 이름 (예: yolov8n.pt)
        preset = self.model.builtin_preset
        target_path = os.path.join(builtin_dir, preset)
        
        print(f"📦 Built-in 모델: {preset}")
        
        if os.path.exists(target_path):
            print(f"✅ 모델 파일 존재")
            print(f"   경로: {target_path}")
            self.loaded_model = YOLO(target_path)
        else:
            print(f"📥 모델 자동 다운로드 시작: {preset}")
            
            # ultralytics 환경변수 설정
            original_env = os.environ.get('YOLO_CONFIG_DIR')
            os.environ['YOLO_CONFIG_DIR'] = str(builtin_dir)
            
            try:
                # YOLO 모델 로드 (자동 다운로드)
                self.loaded_model = YOLO(preset)
                
                # 다운로드 후 경로 확인 및 복사
                possible_paths = [
                    os.path.join(builtin_dir, preset),
                    Path.home() / '.cache' / 'ultralytics' / preset,
                    os.path.join(os.getcwd(), preset),
                ]
                
                downloaded_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        downloaded_path = path
                        print(f"✅ 다운로드 완료: {downloaded_path}")
                        break
                
                # builtin 폴더로 복사
                if downloaded_path and str(downloaded_path) != target_path:
                    print(f"📋 모델을 builtin 폴더로 복사 중...")
                    shutil.copy2(str(downloaded_path), target_path)
                    print(f"✅ 복사 완료: {target_path}")
                    
                    # 원본이 현재 디렉토리에 있으면 삭제
                    if str(downloaded_path) == os.path.join(os.getcwd(), preset):
                        os.remove(downloaded_path)
                        print(f"🗑️ 임시 파일 삭제: {downloaded_path}")
                    
                    # DB에 파일 크기 저장
                    if os.path.exists(target_path):
                        file_size = os.path.getsize(target_path)
                        print(f"📊 파일 크기: {file_size / (1024*1024):.2f} MB")
                        
                        if self.model.file_size == 0:
                            self.model.file_size = file_size
                            self.model.save(update_fields=['file_size'])
                            print(f"💾 DB 업데이트 완료")
            finally:
                # 환경변수 복원
                if original_env:
                    os.environ['YOLO_CONFIG_DIR'] = original_env
                elif 'YOLO_CONFIG_DIR' in os.environ:
                    del os.environ['YOLO_CONFIG_DIR']
    
    def _load_upload_model(self):
        """업로드된 모델 로드"""
        model_path = self.model.model_file.path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
        
        print(f"📁 Upload 모델: {os.path.basename(model_path)}")
        print(f"   경로: {model_path}")
        
        # YOLO 버전 감지
        self.yolo_version = self.detect_yolo_version(model_path)
        
        # 버전별 로드
        if self.yolo_version == 'onnx':
            self._load_onnx(model_path)
        elif self.yolo_version == 'v5':
            self._load_yolov5(model_path)
        else:
            self._load_yolov8(model_path)
    
    def _load_git_model(self):
        """Git 모델 로드"""
        model_path = self.model.get_model_path()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Git 모델 디렉토리가 존재하지 않습니다: {model_path}")
        
        print(f"📁 Git 모델: {model_path}")
        
        # Git 디렉토리에서 모델 파일 찾기 (.pt, .onnx)
        pt_files = list(Path(model_path).rglob("*.pt"))
        onnx_files = list(Path(model_path).rglob("*.onnx"))
        
        # 우선순위: ONNX > PT
        if onnx_files:
            actual_model_path = str(onnx_files[0])
        elif pt_files:
            actual_model_path = str(pt_files[0])
        else:
            raise FileNotFoundError(f"Git 디렉토리에서 모델 파일(.pt, .onnx)을 찾을 수 없습니다")
        
        print(f"   모델 파일: {actual_model_path}")
        
        # YOLO 버전 감지 및 로드
        self.yolo_version = self.detect_yolo_version(actual_model_path)
        
        if self.yolo_version == 'onnx':
            self._load_onnx(actual_model_path)
        elif self.yolo_version == 'v5':
            self._load_yolov5(actual_model_path)
        else:
            self._load_yolov8(actual_model_path)
    
    def _load_huggingface_model(self):
        """HuggingFace 모델 로드"""
        model_id = self.model.hf_model_id
        
        print(f"🤗 HuggingFace 모델: {model_id}")
        
        # TODO: HuggingFace에서 모델 다운로드
        # 임시로 에러 발생
        raise NotImplementedError("HuggingFace 모델 로드는 아직 구현되지 않았습니다")
    
    def _load_yolov5(self, model_path):
        """YOLOv5 모델 로드"""
        import torch
        import sys
        from pathlib import Path
        
        print(f"🔄 YOLOv5 로딩 중...")
        
        # YOLOv5 저장소 경로
        yolov5_repo = Path(torch.hub.get_dir()) / 'ultralytics_yolov5_master'
        
        if not yolov5_repo.exists():
            print(f"⚠️ YOLOv5 저장소를 찾을 수 없습니다. 다운로드 중...")
            # 저장소 다운로드
            torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        
        # YOLOv5 경로를 sys.path에 추가
        yolov5_path = str(yolov5_repo)
        if yolov5_path not in sys.path:
            sys.path.insert(0, yolov5_path)
        
        try:
            # 직접 로컬 경로에서 로드
            self.loaded_model = torch.hub.load(
                str(yolov5_repo),
                'custom',
                path=model_path,
                source='local',  # ⭐ 중요: local로 지정
                force_reload=False
            )
            
            print(f"✅ YOLOv5 로드 완료")
            
        except Exception as e:
            print(f"⚠️ torch.hub.load 실패, 직접 로드 시도...")
            
            # Plan B: 직접 모델 로드
            try:
                from models.common import DetectMultiBackend
                from models.experimental import attempt_load
                
                # 직접 로드
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.loaded_model = attempt_load(model_path, device=device)
                
                # AutoShape 적용 (전처리 자동화)
                from models.common import AutoShape
                self.loaded_model = AutoShape(self.loaded_model)
                
                print(f"✅ YOLOv5 직접 로드 완료")
                
            except Exception as e2:
                print(f"❌ 직접 로드도 실패: {e2}")
                raise RuntimeError(
                    f"YOLOv5 모델 로드 실패.\n"
                    f"torch.hub 오류: {e}\n"
                    f"직접 로드 오류: {e2}\n"
                    f"YOLOv8 형식(.pt)을 사용하거나 ONNX 파일을 업로드하세요."
                )
    
    def _load_yolov8(self, model_path):
        """YOLOv8 모델 로드"""
        from ultralytics import YOLO
        
        print(f"🔄 YOLOv8 로딩 중...")
        
        self.loaded_model = YOLO(model_path)
        
        print(f"✅ YOLOv8 로드 완료")
    
    def _load_onnx(self, model_path):
        """ONNX 모델 로드"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime가 설치되지 않았습니다.\n"
                "pip install onnxruntime 또는 pip install onnxruntime-gpu 를 실행하세요."
            )
        
        print(f"🔄 ONNX 모델 로딩 중...")
        
        # ONNX Runtime 세션 생성
        self.loaded_model = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # GPU: CUDAExecutionProvider
        )
        
        # 입력/출력 정보 확인
        input_name = self.loaded_model.get_inputs()[0].name
        input_shape = self.loaded_model.get_inputs()[0].shape
        print(f"   입력: {input_name}, Shape: {input_shape}")
        
        output_names = [out.name for out in self.loaded_model.get_outputs()]
        print(f"   출력: {output_names}")
        
        print(f"✅ ONNX 모델 로드 완료")
    
    def apply_frame(self, frame):
        """YOLO 객체 탐지 (버전별 분기)"""
        if not self.loaded_model:
            return {'frame': frame, 'applications': []}

        try:
            # 4채널(RGBA) -> 3채널(BGR) 변환
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 버전별 추론
            if self.yolo_version == 'onnx':
                applications = self._apply_onnx(frame)
            elif self.yolo_version == 'v5':
                applications = self._apply_yolov5(frame)
            else:
                applications = self._apply_yolov8(frame)
            
            # 바운딩 박스 그리기
            annotated_frame = self.draw_applications(frame, applications)
            
            return {
                'frame': annotated_frame,
                'applications': applications
            }

        except Exception as e:
            print(f"⚠️ 탐지 오류: {e}")
            import traceback
            traceback.print_exc()
            return {'frame': frame, 'applications': []}
    
    def _apply_yolov5(self, frame):
        """YOLOv5 추론"""
        # BGR -> RGB 변환 (YOLOv5 필요)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 추론
        results = self.loaded_model(frame_rgb)
        
        # 결과 파싱
        applications = []
        
        # confidence threshold (기본값 0.25)
        conf_threshold = 0.25
        
        # results.xyxy[0]: [x1, y1, x2, y2, conf, cls]
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            confidence = float(conf)
            
            if confidence >= conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                label = results.names[class_id]
                
                applications.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                })
        
        return applications
    
    def _apply_yolov8(self, frame):
        """YOLOv8 추론"""
        results = self.loaded_model(frame, verbose=False)
        applications = []

        # confidence threshold (기본값 0.25)
        conf_threshold = 0.25

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                
                if confidence >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    label = self.loaded_model.names[class_id]

                    applications.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    })
        
        return applications
    
    def _apply_onnx(self, frame):
        """ONNX 모델 추론 (YOLOv5/v8 ONNX)"""
        import numpy as np
        
        # 입력 전처리
        input_name = self.loaded_model.get_inputs()[0].name
        input_shape = self.loaded_model.get_inputs()[0].shape
        
        # 입력 크기 (일반적으로 640x640)
        input_height = input_shape[2] if len(input_shape) > 2 else 640
        input_width = input_shape[3] if len(input_shape) > 3 else 640
        
        # 원본 프레임 크기
        orig_h, orig_w = frame.shape[:2]
        
        # BGR -> RGB & 리사이즈
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (input_width, input_height))
        
        # Normalize & Transpose (HWC -> CHW)
        input_tensor = frame_resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        # 추론
        outputs = self.loaded_model.run(None, {input_name: input_tensor})
        
        # 🔍 디버깅: 출력 형식 확인
        print(f"\n{'='*60}")
        print(f"🔍 ONNX 출력 디버깅")
        print(f"{'='*60}")
        print(f"출력 개수: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"outputs[{i}] shape: {output.shape}")
            if i == 0 and len(output.shape) >= 2:
                print(f"  → 첫 번째 detection shape: {output[0].shape if output.shape[0] > 0 else 'empty'}")
                if output.shape[0] > 0 and len(output[0]) > 0:
                    print(f"  → 첫 번째 값 샘플: {output[0][0][:10]}")  # 처음 10개 값
        
        # 결과 파싱
        applications = []
        conf_threshold = 0.25
        
        # YOLOv5 ONNX 출력 형식 확인
        # 가능한 형식:
        # 1. (1, 25200, 85) - YOLOv5 표준
        # 2. (1, 84, 8400) - YOLOv8 형식
        # 3. (1, N, 6) - [x1, y1, x2, y2, conf, class]
        
        output = outputs[0]
        
        # 형식 1: (1, 25200, 85) - YOLOv5 표준
        if len(output.shape) == 3 and output.shape[2] > output.shape[1]:
            print(f"✅ YOLOv5 표준 형식 감지: {output.shape}")
            detections = output[0]  # Remove batch dimension
            
            for detection in detections:
                # Confidence
                obj_conf = detection[4]
                
                if obj_conf >= conf_threshold:
                    # Class scores (index 5~)
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    confidence = obj_conf * class_conf
                    
                    if confidence >= conf_threshold:
                        # Bounding box (ONNX는 이미 픽셀 좌표일 수 있음)
                        x_center, y_center, width, height = detection[:4]
                        
                        # 스케일 조정 (640x640 -> 원본 크기)
                        scale_x = orig_w / input_width
                        scale_y = orig_h / input_height
                        
                        x_center *= scale_x
                        y_center *= scale_y
                        width *= scale_x
                        height *= scale_y
                        
                        # Convert to x1, y1, x2, y2
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        # Clamp to frame bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(orig_w, x2)
                        y2 = min(orig_h, y2)
                        
                        # Get label from model metadata or use class_id
                        if self.model.classes and len(self.model.classes) > class_id:
                            label = self.model.classes[class_id]
                        else:
                            label = f"class_{class_id}"
                        
                        applications.append({
                            'label': label,
                            'confidence': float(confidence),
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                        })
        
        # 형식 2: (1, 84, 8400) - YOLOv8 형식 (Transpose 필요)
        elif len(output.shape) == 3 and output.shape[1] < 100:
            print(f"✅ YOLOv8 형식 감지: {output.shape}")
            output = output[0]  # Remove batch (84, 8400)
            output = output.T  # Transpose to (8400, 84)
            
            # [x_center, y_center, width, height, class_probs...]
            for detection in output:
                # Class scores (index 4~)
                class_scores = detection[4:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence >= conf_threshold:
                    # Bounding box
                    x_center, y_center, width, height = detection[:4]
                    
                    # 스케일 조정
                    scale_x = orig_w / input_width
                    scale_y = orig_h / input_height
                    
                    x_center *= scale_x
                    y_center *= scale_y
                    width *= scale_x
                    height *= scale_y
                    
                    # Convert to x1, y1, x2, y2
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Clamp
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(orig_w, x2)
                    y2 = min(orig_h, y2)
                    
                    # Label
                    if self.model.classes and len(self.model.classes) > class_id:
                        label = self.model.classes[class_id]
                    else:
                        label = f"class_{class_id}"
                    
                    applications.append({
                        'label': label,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                    })
        
        else:
            print(f"⚠️ 알 수 없는 ONNX 출력 형식: {output.shape}")
            print(f"   수동으로 파싱 로직을 추가해야 합니다.")
        
        print(f"✅ 탐지된 객체: {len(applications)}개")
        print(f"{'='*60}\n")
        
        return applications
    
    def draw_applications(self, frame, applications):
        """탐지 결과를 프레임에 그리기"""
        result = frame.copy()
        for det in applications:
            x, y, w, h = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            color = self.get_color_for_label(label)

            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                result, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        return result

    def get_color_for_label(self, label):
        """라벨별 색상"""
        hash_val = hash(label)
        return (hash_val & 0xFF, (hash_val >> 8) & 0xFF, (hash_val >> 16) & 0xFF)


class SuperResolutionExecutor(BaseExecutor):
    """해상도 개선 Executor"""
    
    def load_model(self):
        """해상도 개선 모델 로드"""
        print("\n" + "="*60)
        print("🔄 해상도 개선 모델 로딩 시작")
        print("="*60)
        
        model_path = self.model.get_model_path()
        if not model_path:
            raise ValueError("모델 파일이 지정되지 않았습니다")
        
        # TODO: 실제 모델 로드 구현
        # 예시: Real-ESRGAN, SRGAN 등
        print(f"⚠️ 해상도 개선 모델 로드는 아직 구현되지 않았습니다")
        print(f"모델 경로: {model_path}")
        self.loaded_model = None
        
        print("="*60 + "\n")
    
    def apply_frame(self, frame):
        """이미지 해상도 개선"""
        if not self.loaded_model:
            # TODO: 실제 구현
            # 임시로 2배 업스케일링
            return cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # TODO: 실제 모델 추론
        return frame


class ImageRestorationExecutor(BaseExecutor):
    """이미지 복원 Executor"""
    
    def load_model(self):
        """이미지 복원 모델 로드"""
        print("\n" + "="*60)
        print("🔄 이미지 복원 모델 로딩 시작")
        print("="*60)
        
        model_path = self.model.get_model_path()
        if not model_path:
            raise ValueError("모델 파일이 지정되지 않았습니다")
        
        # TODO: 실제 모델 로드 구현
        # 예시: DeOldify, Bringing Old Photos Back to Life 등
        print(f"⚠️ 이미지 복원 모델 로드는 아직 구현되지 않았습니다")
        print(f"모델 경로: {model_path}")
        self.loaded_model = None
        
        print("="*60 + "\n")
    
    def apply_frame(self, frame):
        """이미지 복원"""
        if not self.loaded_model:
            # TODO: 실제 구현
            # 임시로 노이즈 제거만 수행
            return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # TODO: 실제 모델 추론
        return frame


class ImageClassificationExecutor(BaseExecutor):
    """이미지 분류 Executor (추후 구현)"""
    
    def load_model(self):
        """이미지 분류 모델 로드"""
        print("\n" + "="*60)
        print("🔄 이미지 분류 모델 로딩 시작")
        print("="*60)
        print(f"⚠️ 이미지 분류 모델은 아직 구현되지 않았습니다")
        print(f"모델: {self.model.name}")
        print("="*60 + "\n")
        raise NotImplementedError("이미지 분류 기능은 아직 구현되지 않았습니다. vision_engine은 현재 객체 탐지(object_detection)만 지원합니다.")
    
    def apply_frame(self, frame):
        """이미지 분류"""
        raise NotImplementedError("이미지 분류 기능은 아직 구현되지 않았습니다")


class SegmentationExecutor(BaseExecutor):
    """세그멘테이션 Executor (추후 구현)"""
    
    def load_model(self):
        """세그멘테이션 모델 로드"""
        print("\n" + "="*60)
        print("🔄 세그멘테이션 모델 로딩 시작")
        print("="*60)
        print(f"⚠️ 세그멘테이션 모델은 아직 구현되지 않았습니다")
        print(f"모델: {self.model.name}")
        print("="*60 + "\n")
        raise NotImplementedError("세그멘테이션 기능은 아직 구현되지 않았습니다. vision_engine은 현재 객체 탐지(object_detection)만 지원합니다.")
    
    def apply_frame(self, frame):
        """세그멘테이션"""
        raise NotImplementedError("세그멘테이션 기능은 아직 구현되지 않았습니다")


# 하위 호환성을 위한 기존 클래스 유지
# class ModelApplier:
#     """기존 코드 호환성을 위한 래퍼 클래스"""
    
#     def __init__(self, model):
#         self.executor = ModelExecutor.get_executor(model)
#         # 기존 속성 유지
#         self.model = model
#         self.yolo_model = getattr(self.executor, 'loaded_model', None)
#         self.model_type = model.task_type
    
#     def process_video(self, input_path, output_path, progress_callback=None):
#         return self.executor.process_video(input_path, output_path, progress_callback)
    
#     def apply_frame(self, frame):
#         result = self.executor.apply_frame(frame)
#         # 기존 인터페이스 맞추기
#         if isinstance(result, dict):
#             return result.get('applications', [])
#         return []
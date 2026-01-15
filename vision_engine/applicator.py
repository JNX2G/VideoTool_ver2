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
        task_type = getattr(model, 'task_type', 'detection')
        
        executors = {
            'detection': ObjectDetectionExecutor,
            'super_resolution': SuperResolutionExecutor,
            'restoration': ImageRestorationExecutor,
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
        """
        model_path_str = str(model_path).lower()
        
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
                import torch
                print(f"🔍 모델 파일 구조 분석 중: {model_path}")
                
                checkpoint = torch.load(model_path, map_location='cpu')
                
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
        
        # 3. 기본값: YOLOv8 (최신 버전)
        print("ℹ️ 버전 감지 실패, 기본값 YOLOv8 사용")
        return 'v8'
    
    def load_model(self):
        """YOLO 모델 로드 (v5 또는 v8 자동 감지)"""
        try:
            print("\n" + "="*60)
            print("🔄 객체 탐지 모델 로딩 시작")
            print("="*60)
            print(f"📂 MODELS_ROOT: {settings.MODELS_ROOT}")

            model_path = self.model.get_model_path()
            if not model_path:
                raise ValueError("모델 파일이 지정되지 않았습니다")

            print(f"📍 모델 경로: {model_path}")
            
            # ⭐ YOLO 버전 감지
            self.yolo_version = self.detect_yolo_version(model_path)
            print(f"🎯 감지된 YOLO 버전: {self.yolo_version}")
            
            # 버전별 로드
            if self.yolo_version == 'v5':
                self._load_yolov5(model_path)
            else:
                self._load_yolov8(model_path)
            
            print("="*60)
            print("✅ 객체 탐지 모델 로드 완료")
            print("="*60 + "\n")

        except Exception as e:
            print("="*60)
            print(f"❌ 모델 로드 실패: {e}")
            print("="*60 + "\n")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_yolov5(self, model_path):
        """YOLOv5 모델 로드"""
        print("📦 YOLOv5 모델 로딩...")
        
        try:
            import torch
            
            # YOLOv5는 torch.hub 사용
            print("🔧 torch.hub를 통한 YOLOv5 로드")
            
            # 모델 로드
            self.loaded_model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=model_path,
                force_reload=False,
                verbose=False
            )
            
            # 신뢰도 임계값 설정
            self.loaded_model.conf = 0.25
            self.loaded_model.iou = 0.45
            
            print(f"✅ YOLOv5 로드 완료")
            print(f"   클래스: {self.loaded_model.names}")
            
        except Exception as e:
            print(f"❌ YOLOv5 로드 실패: {e}")
            print("💡 torch가 설치되어 있는지 확인하세요: pip install torch torchvision")
            raise
    
    def _load_yolov8(self, model_path):
        """YOLOv8 모델 로드 (기존 코드)"""
        print("📦 YOLOv8 모델 로딩...")
        
        from ultralytics import YOLO
        
        # ⭐ YOLO 자동 다운로드 모델인 경우
        if hasattr(self.model, 'yolo_version') and self.model.yolo_version:
            builtin_dir = os.path.join(settings.MODELS_ROOT, "builtin")
            os.makedirs(builtin_dir, exist_ok=True)
            print(f"📁 Builtin 디렉토리: {builtin_dir}")
            
            target_path = os.path.join(builtin_dir, self.model.yolo_version)
            print(f"🎯 타겟 경로: {target_path}")
            
            # 이미 models/builtin/에 있으면 바로 사용
            if os.path.exists(target_path):
                print(f"✅ 기존 모델 발견!")
                print(f"   경로: {target_path}")
                print(f"   크기: {os.path.getsize(target_path) / (1024*1024):.2f} MB")
                self.loaded_model = YOLO(target_path)
            else:
                print(f"📥 모델 자동 다운로드 시작: {self.model.yolo_version}")
                
                # ⭐ ultralytics 환경변수 설정
                original_env = os.environ.get('YOLO_CONFIG_DIR')
                os.environ['YOLO_CONFIG_DIR'] = builtin_dir
                
                try:
                    # YOLO 모델 로드 (자동 다운로드)
                    self.loaded_model = YOLO(self.model.yolo_version)
                    
                    # 다운로드 후 경로 확인
                    possible_paths = [
                        os.path.join(builtin_dir, self.model.yolo_version),
                        Path.home() / '.cache' / 'ultralytics' / self.model.yolo_version,
                        os.path.join(os.getcwd(), self.model.yolo_version),
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
                        if str(downloaded_path) == os.path.join(os.getcwd(), self.model.yolo_version):
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
                
        else:
            # 직접 업로드된 파일 사용
            print(f"📁 직접 업로드된 모델 사용")
            self.loaded_model = YOLO(model_path)
        
        print(f"✅ YOLOv8 로드 완료")
    
    def apply_frame(self, frame):
        """YOLO 객체 탐지 (버전별 분기)"""
        if not self.loaded_model:
            return {'frame': frame, 'applications': []}

        try:
            # 4채널(RGBA) -> 3채널(BGR) 변환
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 버전별 추론
            if self.yolo_version == 'v5':
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
        
        # confidence threshold
        conf_threshold = 0.25
        if hasattr(self.model, "config") and isinstance(self.model.config, dict):
            conf_threshold = self.model.config.get("conf_threshold", 0.25)
        
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

        # confidence threshold
        conf_threshold = 0.25
        if hasattr(self.model, "config") and isinstance(self.model.config, dict):
            conf_threshold = self.model.config.get("conf_threshold", 0.25)

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


# 하위 호환성을 위한 기존 클래스 유지
class ModelApplier:
    """기존 코드 호환성을 위한 래퍼 클래스"""
    
    def __init__(self, model):
        self.executor = ModelExecutor.get_executor(model)
        # 기존 속성 유지
        self.model = model
        self.yolo_model = getattr(self.executor, 'loaded_model', None)
        self.model_type = getattr(model, 'task_type', 'detection')
    
    def process_video(self, input_path, output_path, progress_callback=None):
        return self.executor.process_video(input_path, output_path, progress_callback)
    
    def apply_frame(self, frame):
        result = self.executor.apply_frame(frame)
        # 기존 인터페이스 맞추기
        if isinstance(result, dict):
            return result.get('applications', [])
        return []
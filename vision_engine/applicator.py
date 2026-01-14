import cv2
import numpy as np
from pathlib import Path
from django.conf import settings
import os
import subprocess
import shutil


class ModelApplier:
    """동영상/이미지 객체 탐지 처리 (modelhub 통합)"""

    def __init__(self, model):
        """
        model: modelhub.BuiltinModel 또는 modelhub.CustomModel
        """
        self.model = model
        self.yolo_model = None
        self.model_type = getattr(model, "model_type", "yolo")

        # YOLO 모델 로드
        if self.model_type == "yolo":
            self.load_yolo_model()

    def load_yolo_model(self):
        """YOLO 모델 로드 (models/builtin/ 경로에 직접 다운로드)"""
        try:
            from ultralytics import YOLO
            
            print("\n" + "="*60)
            print("🔄 YOLO 모델 로딩 시작")
            print("="*60)
            print(f"📂 MODELS_ROOT: {settings.MODELS_ROOT}")

            model_path = self.model.get_model_path()
            if not model_path:
                raise ValueError("모델 파일이 지정되지 않았습니다")

            print(f"📍 모델 경로: {model_path}")
            
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
                    self.yolo_model = YOLO(target_path)
                else:
                    print(f"📥 모델 자동 다운로드 시작: {self.model.yolo_version}")
                    
                    # ⭐ ultralytics 환경변수 설정 - builtin 폴더에 직접 다운로드
                    # YOLO_CONFIG_DIR을 builtin으로 설정
                    original_env = os.environ.get('YOLO_CONFIG_DIR')
                    os.environ['YOLO_CONFIG_DIR'] = builtin_dir
                    
                    try:
                        # YOLO 모델 로드 (자동 다운로드)
                        self.yolo_model = YOLO(self.model.yolo_version)
                        
                        # 다운로드 후 경로 확인
                        # ultralytics는 여러 위치에 저장 가능
                        possible_paths = [
                            # 1. 설정한 YOLO_CONFIG_DIR
                            os.path.join(builtin_dir, self.model.yolo_version),
                            # 2. 기본 캐시 경로
                            Path.home() / '.cache' / 'ultralytics' / self.model.yolo_version,
                            # 3. 현재 디렉토리
                            os.path.join(os.getcwd(), self.model.yolo_version),
                        ]
                        
                        downloaded_path = None
                        for path in possible_paths:
                            if os.path.exists(path):
                                downloaded_path = path
                                print(f"✅ 다운로드 완료: {downloaded_path}")
                                break
                        
                        # builtin 폴더로 복사 (필요한 경우)
                        if downloaded_path and str(downloaded_path) != target_path:
                            print(f"📋 모델을 builtin 폴더로 복사 중...")
                            shutil.copy2(str(downloaded_path), target_path)
                            print(f"✅ 복사 완료: {target_path}")
                            
                            # 원본이 현재 디렉토리에 있으면 삭제
                            if str(downloaded_path) == os.path.join(os.getcwd(), self.model.yolo_version):
                                os.remove(downloaded_path)
                                print(f"🗑️  임시 파일 삭제: {downloaded_path}")
                        
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
                self.yolo_model = YOLO(model_path)
            
            print("="*60)
            print("✅ YOLO 모델 로드 완료")
            print("="*60 + "\n")

        except Exception as e:
            print("="*60)
            print(f"❌ YOLO 모델 로드 실패: {e}")
            print("="*60 + "\n")
            import traceback
            traceback.print_exc()
            raise

    def apply_frame(self, frame):
        """단일 프레임 감지"""
        if self.model_type == "yolo":
            return self.apply_yolo(frame)
        elif self.model_type == "custom":
            return self.apply_custom(frame)
        return []

    def apply_yolo(self, frame):
        """YOLO 객체 감지"""
        if not self.yolo_model:
            return []

        try:
            # 4채널(RGBA) -> 3채널(BGR) 변환
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = self.yolo_model(frame, verbose=False)
            applications = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = self.yolo_model.names[class_id]

                    # confidence threshold
                    conf_threshold = 0.25
                    if hasattr(self.model, "config") and isinstance(
                        self.model.config, dict
                    ):
                        conf_threshold = self.model.config.get("conf_threshold", 0.25)

                    if confidence >= conf_threshold:
                        applications.append(
                            {
                                "label": label,
                                "confidence": confidence,
                                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            }
                        )

            return applications

        except Exception as e:
            print(f"⚠️  YOLO 감지 오류: {e}")
            return []

    def apply_custom(self, frame):
        """커스텀 모델 감지 (확장 포인트)"""
        # TODO: 커스텀 모델 추론 로직
        print("⚠️  커스텀 모델 감지는 아직 구현되지 않았습니다")
        return []

    def process_video(self, input_path, output_path, progress_callback=None):
        """동영상/이미지 탐지 처리"""
        print(f"\n{'='*60}\n🔍 탐지 처리 시작\n{'='*60}")

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

        print(f"🖼️  해상도: {width}x{height} | FPS: {fps} | 총 프레임: {total_frames}")

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

                # 감지 수행
                applications = self.apply_frame(frame)
                annotated_frame = self.draw_applications(frame, applications)

                if not is_image and out:
                    out.write(annotated_frame)

                if applications:
                    all_applications.append(
                        {"frame": frame_count, "applications": applications}
                    )
                    total_applications_count += len(applications)
                    for det in applications:
                        label = det["label"]
                        application_summary[label] = application_summary.get(label, 0) + 1

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
                print(f"⚠️  ffmpeg 실패 - 원본 파일 사용")
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

    def draw_applications(self, frame, applications):
        """감지 결과를 프레임에 그리기"""
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
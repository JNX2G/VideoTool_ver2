from django.utils import timezone
from .models import Detection
from .detector import VideoDetector
import os
from pathlib import Path
from django.conf import settings


def process_detection(detection_id):
    """탐지 작업 실행 (백그라운드)"""
    detection = None

    try:
        print(f"\n{'='*60}")
        print(f"🔍 탐지 작업 시작: ID={detection_id}")
        print(f"{'='*60}\n")

        detection = Detection.objects.get(id=detection_id)
        task = detection.preprocessing_task
        model = detection.get_model()

        if not model:
            raise ValueError("모델이 선택되지 않았습니다")

        # 취소 확인
        detection.refresh_from_db()
        if detection.status == "cancelled":
            print(f"작업이 이미 취소됨: detection_id={detection_id}")
            return

        # 상태 업데이트
        detection.status = "processing"
        detection.started_at = timezone.now()
        detection.save()

        print(f"📹 전처리 작업 ID: {task.id}")
        print(f"🤖 모델: {detection.get_model_name()}")

        # ⭐ 헬퍼 메서드를 사용하여 실제 파일 경로 가져오기
        input_path = task.get_actual_file_path()

        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")

        print(f"📂 입력: {input_path}")

        # ⭐ 출력 경로 설정 - results/vision_engine/content_id/detection_id/
        content = task.get_content()
        
        output_dir = Path(settings.RESULTS_ROOT) / 'vision_engine' / str(content.id) / str(detection.id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 원본 파일명 가져오기
        if content and hasattr(content, "file") and content.file:
            original_filename = os.path.basename(content.file.name)
        else:
            original_filename = "detected_result.mp4"

        output_filename = f"detected_{original_filename}"
        output_path = output_dir / output_filename

        print(f"📤 출력: {output_path}")

        # 탐지 실행
        detector = VideoDetector(model)

        # 진행률 콜백 (취소 확인 포함)
        def progress_callback(current, total, progress):
            # DB에서 최신 상태 확인
            detection.refresh_from_db()
            
            # 취소되었으면 예외 발생
            if detection.status == "cancelled":
                print(f"작업 취소 감지: detection_id={detection_id}")
                raise InterruptedError("작업이 취소되었습니다.")
            
            # 진행률 업데이트
            detection.processed_frames = current
            detection.total_frames = total
            detection.progress = progress
            detection.save(update_fields=["processed_frames", "total_frames", "progress"])
            
            if progress % 10 == 0:
                print(f"⏳ 진행: {current}/{total} ({progress}%)")

        # 실행
        results = detector.process_video(
            str(input_path), str(output_path), progress_callback
        )

        # 완료 전 마지막 취소 확인
        detection.refresh_from_db()
        if detection.status == "cancelled":
            print(f"작업 완료 직전 취소 감지: detection_id={detection_id}")
            # 출력 파일 삭제
            if output_path.exists():
                output_path.unlink()
            return

        # 결과 저장
        detection.save_results(results["detections"])
        detection.total_detections = results["total_detections"]
        detection.detection_summary = results["summary"]

        # 출력 경로 저장 (RESULTS_ROOT 기준 상대 경로)
        relative_path = output_path.relative_to(settings.RESULTS_ROOT)
        detection.output_file_path = str(relative_path).replace("\\", "/")

        # 모델 사용 횟수 증가
        model.increment_usage()

        # 완료
        detection.status = "completed"
        detection.completed_at = timezone.now()
        detection.progress = 100
        detection.save()

        print(f"\n{'='*60}")
        print(f"✨ 탐지 완료!")
        print(f"   총 탐지: {detection.total_detections}")
        print(f"   클래스: {len(detection.detection_summary)}")
        print(f"{'='*60}\n")

        return True

    except InterruptedError as e:
        # 취소로 인한 중단
        print(f"🛑 작업 취소: detection_id={detection_id}, {e}")
        
        # 출력 파일 삭제
        if 'output_path' in locals() and Path(output_path).exists():
            try:
                Path(output_path).unlink()
                print(f"임시 출력 파일 삭제: {output_path}")
            except Exception as delete_error:
                print(f"임시 파일 삭제 실패: {delete_error}")

    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()

        if detection:
            # 취소가 아닌 진짜 오류인 경우만 failed로 설정
            detection.refresh_from_db()
            if detection.status != "cancelled":
                detection.status = "failed"
                detection.error_message = str(e)
                detection.save()

        return False


def start_detection_task(detection_id):
    """탐지 작업을 백그라운드 스레드로 시작"""
    import threading
    import logging
    
    logger = logging.getLogger(__name__)
    
    thread = threading.Thread(
        target=process_detection,
        args=(detection_id,),
        name=f"Detection-{detection_id}"
    )
    thread.daemon = True
    thread.start()
    
    logger.info(f"탐지 작업 스레드 시작: detection_id={detection_id}, thread={thread.name}")
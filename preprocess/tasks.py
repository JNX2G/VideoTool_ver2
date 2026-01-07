from django.utils import timezone
from .models import PreprocessingTask
import os
from pathlib import Path
import traceback


def process_preprocessing_task(task_id):
    """전처리 작업 실행"""
    task = None
    try:
        print(f"\n{'='*50}")
        print(f"🎬 전처리 작업 시작: ID={task_id}")

        task = PreprocessingTask.objects.get(id=task_id)

        # 컨텐츠 가져오기 (video 또는 image)
        content = task.get_content()
        content_type = task.get_content_type()

        if not content:
            raise ValueError("컨텐츠를 찾을 수 없습니다")

        print(f"📦 컨텐츠 타입: {content_type}")
        print(f"📄 파일명: {content.title}")

        # 상태 업데이트
        task.status = "processing"
        task.started_at = timezone.now()
        task.current_step = "전처리 시작"
        task.save()

        # 입력 파일 경로
        input_path = content.file.path

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")

        # 출력 경로 설정 (results/preprocessing/{task_id}/)
        from django.conf import settings
        output_dir = Path(settings.RESULTS_ROOT) / "preprocessing" / str(task.id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 파일 이름 정리 (특수문자 제거)
        original_name = content.file.name.split("/")[-1]
        clean_name = "".join(c for c in original_name if c.isalnum() or c in ".-_")

        # 컨텐츠 타입에 따라 확장자 결정
        if content_type == "image":
            output_filename = Path(clean_name).stem + "_processed.jpg"
        else:
            output_filename = Path(clean_name).stem + "_processed.mp4"

        output_path = output_dir / output_filename

        print(f"📤 출력 경로: {output_path}")

        # 전처리 엔진 생성
        from .preprocessing import PreprocessingEngine

        engine = PreprocessingEngine()

        # 진행률 콜백
        def progress_callback(current, total, progress):
            task.processed_frames = current
            task.total_frames = total
            task.progress = progress

            if content_type == "image":
                if progress < 90:
                    task.current_step = f"이미지 처리 중: {current}/{total}"
                else:
                    task.current_step = "완료 중..."
            else:
                if progress < 85:
                    task.current_step = f"프레임 처리 중: {current}/{total}"
                elif progress < 95:
                    task.current_step = "ffmpeg 재인코딩 중..."
                else:
                    task.current_step = "완료 중..."

            task.save()

            if current % 30 == 0 or content_type == "image":
                print(f"⏳ 진행률: {progress}%")

        # 파이프라인 실행
        pipeline = task.preprocessing_pipeline or []

        if not pipeline:
            # 파이프라인이 비어있으면 원본 복사
            import shutil

            shutil.copy(input_path, output_path)
            task.total_frames = 1
            task.processed_frames = 1
        else:
            # 컨텐츠 타입에 따라 다른 처리
            if content_type == "image":
                # 이미지 전처리
                engine.process_image(
                    input_path, pipeline, str(output_path), progress_callback
                )
            else:
                # 동영상 전처리
                engine.process_video(
                    input_path, pipeline, str(output_path), progress_callback
                )

        # 출력 파일 확인
        if not output_path.exists():
            raise FileNotFoundError(f"출력 파일이 생성되지 않았습니다: {output_path}")

        file_size = output_path.stat().st_size
        print(f"✅ 출력 파일: {file_size:,} bytes")

        # 상대 경로로 저장 (preprocessing/{task_id}/파일명)
        relative_path = output_path.relative_to(settings.RESULTS_ROOT)
        relative_path_str = str(relative_path).replace("\\", "/")

        print(f"💾 저장 경로: {relative_path_str}")

        # 완료 처리
        task.status = "completed"
        task.completed_at = timezone.now()
        task.progress = 100
        task.output_file_path = relative_path_str
        task.current_step = "완료"
        task.save()

        print(f"✨ 전처리 작업 완료!")

        return True

    except Exception as e:
        print(f"❌ 에러: {e}")
        traceback.print_exc()

        if task:
            task.status = "failed"
            task.error_message = str(e)
            task.current_step = "실패"
            task.save()

        return False


def start_preprocessing_task(task_id):
    """전처리 작업을 백그라운드에서 시작"""
    import threading

    thread = threading.Thread(target=process_preprocessing_task, args=(task_id,))
    thread.daemon = True
    thread.start()

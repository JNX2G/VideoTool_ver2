"""
전처리 작업 실행 (취소 확인 로직 포함)
"""
import logging
from pathlib import Path
from django.utils import timezone
from django.conf import settings

logger = logging.getLogger(__name__)


def process_preprocessing_task(task_id):
    """전처리 작업 실행 (백그라운드)"""
    from .models import PreprocessingTask
    from preprocess.preprocessing import PreprocessingEngine
    
    try:
        # 작업 조회
        task = PreprocessingTask.objects.get(id=task_id)
        
        # ⭐ 이미 취소되었는지 확인
        if task.status == 'cancelled':
            logger.info(f"작업이 이미 취소됨: task_id={task_id}")
            return
        
        # 상태 업데이트
        task.status = 'processing'
        task.started_at = timezone.now()
        task.save()
        
        logger.info(f"전처리 작업 시작: task_id={task_id}")
        
        # 컨텐츠 가져오기
        content = task.get_content()
        content_type = task.get_content_type()
        
        if not content:
            raise ValueError("컨텐츠를 찾을 수 없습니다.")
        
        # 입력 파일 경로
        input_path = content.file.path
        
        # ⭐ 출력 파일 경로 생성 - results/preprocess/content_id/
        output_dir = Path(settings.RESULTS_ROOT) / 'preprocess' / str(content.id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_filename = Path(input_path).stem
        output_filename = f"{input_filename}_preprocessed{Path(input_path).suffix}"
        output_path = output_dir / output_filename
        
        # 전처리 파이프라인
        pipeline = task.preprocessing_pipeline or []
        
        # ⭐ 진행률 콜백 (취소 확인 포함)
        def progress_callback(current, total, percent):
            # DB에서 최신 상태 확인
            task.refresh_from_db()
            
            # ⭐ 취소되었으면 예외 발생
            if task.status == 'cancelled':
                logger.info(f"작업 취소 감지: task_id={task_id}")
                raise InterruptedError("작업이 취소되었습니다.")
            
            # 진행률 업데이트
            task.progress = percent
            task.processed_frames = current
            task.total_frames = total
            task.save(update_fields=['progress', 'processed_frames', 'total_frames'])
        
        # 전처리 실행
        engine = PreprocessingEngine()
        
        if content_type == 'image':
            engine.process_image(
                input_path=str(input_path),
                pipeline=pipeline,
                output_path=str(output_path),
                progress_callback=progress_callback
            )
        else:
            engine.process_video(
                input_path=str(input_path),
                pipeline=pipeline,
                output_path=str(output_path),
                progress_callback=progress_callback,
                task_id=task.id
            )
        
        # ⭐ 완료 전 마지막 취소 확인
        task.refresh_from_db()
        if task.status == 'cancelled':
            logger.info(f"작업 완료 직전 취소 감지: task_id={task_id}")
            # 출력 파일 삭제
            if output_path.exists():
                output_path.unlink()
            return
        
        # 출력 경로 저장 (RESULTS_ROOT 기준 상대 경로)
        relative_path = output_path.relative_to(settings.RESULTS_ROOT)
        task.output_file_path = str(relative_path).replace("\\", "/")
        
        # 완료
        task.status = 'completed'
        task.completed_at = timezone.now()
        task.progress = 100
        task.save()
        
        logger.info(f"✅ 전처리 완료: task_id={task_id}, output={task.output_file_path}")
    
    except InterruptedError as e:
        # 취소로 인한 중단
        logger.info(f"🛑 작업 취소: task_id={task_id}, {e}")
        
        # 출력 파일 삭제
        if 'output_path' in locals() and output_path.exists():
            try:
                output_path.unlink()
                logger.info(f"임시 출력 파일 삭제: {output_path}")
            except Exception as delete_error:
                logger.warning(f"임시 파일 삭제 실패: {delete_error}")
        
        # 작업 상태는 이미 'cancelled'로 설정되어 있음
    
    except Exception as e:
        logger.exception(f"❌ 전처리 작업 실패: task_id={task_id}, {e}")
        
        try:
            task = PreprocessingTask.objects.get(id=task_id)
            
            # 취소가 아닌 진짜 오류인 경우만 failed로 설정
            if task.status != 'cancelled':
                task.status = 'failed'
                task.error_message = str(e)
                task.save()
        except Exception as save_error:
            logger.error(f"작업 상태 저장 실패: {save_error}")


def start_preprocessing_task(task_id):
    """전처리 작업을 백그라운드 스레드로 시작"""
    import threading
    
    thread = threading.Thread(
        target=process_preprocessing_task,
        args=(task_id,),
        name=f"Preprocessing-{task_id}"
    )
    thread.daemon = True
    thread.start()
    
    logger.info(f"전처리 작업 스레드 시작: task_id={task_id}, thread={thread.name}")
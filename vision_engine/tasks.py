"""
Vision Engine 백그라운드 작업
- Application 실행
"""

from django.utils import timezone
from pathlib import Path
import os

from .models import Application
from .applicator import ModelExecutor  # 통합 Executor 사용


def process_application(application_id):
    """
    모델 적용 백그라운드 작업
    
    Args:
        application_id: Application 인스턴스 ID
    """
    try:
        application = Application.objects.get(id=application_id)
    except Application.DoesNotExist:
        print(f"Application을 찾을 수 없습니다: {application_id}")
        return
    
    try:
        # 상태 업데이트
        application.status = "processing"
        application.started_at = timezone.now()
        application.save(update_fields=['status', 'started_at'])
        
        print(f"\n{'='*60}")
        print(f"🚀 Application 시작: {application.title}")
        print(f"{'='*60}")
        
        # 모델 가져오기
        model = application.get_model()
        if not model:
            raise ValueError("모델이 지정되지 않았습니다")
        
        print(f"📦 사용 모델: {model.name} ({model.get_source_display()})")
        print(f"🎯 작업 유형: {model.get_task_type_display()}")
        
        # Executor 생성
        executor = ModelExecutor.get_executor(model)
        
        # 입력/출력 경로 설정
        task = application.preprocessing_task
        content = task.get_content()
        content_type = task.get_content_type()
        
        # 입력 파일 (전처리 결과 또는 원본)
        if task.output_file_path:
            from django.conf import settings
            input_path = os.path.join(
                settings.RESULTS_ROOT,
                task.output_file_path
            )
        else:
            input_path = content.file.path
        
        # 출력 파일 경로 설정
        from django.conf import settings
        
        # results/application/video/1/application_2/output.mp4
        output_dir = Path(settings.RESULTS_ROOT) / 'application' / content_type / str(content.id) / str(application.id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_ext = Path(input_path).suffix
        output_filename = f"output{file_ext}"
        output_path = output_dir / output_filename
        
        # 상대 경로 저장 (results 기준)
        relative_output_path = f"application/{content_type}/{content.id}/{application.id}/{output_filename}"
        
        # 진행률 콜백
        def progress_callback(current, total, progress):
            application.processed_frames = current
            application.total_frames = total
            application.progress = progress
            application.save(update_fields=['processed_frames', 'total_frames', 'progress'])
        
        # 모델 실행
        print(f"🎬 처리 시작...")
        print(f"   입력: {input_path}")
        print(f"   출력: {output_path}")
        
        result = executor.process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            progress_callback=progress_callback
        )
        
        # 결과 저장
        application.output_file_path = relative_output_path
        application.application_data = result.get('applications', [])
        application.total_applications = result.get('total_applications', 0)
        application.application_summary = result.get('summary', {})
        application.status = "completed"
        application.completed_at = timezone.now()
        application.progress = 100
        application.save()
        
        # 모델 사용 횟수 증가
        model.increment_usage()
        
        print(f"✅ Application 완료!")
        print(f"   총 탐지: {application.total_applications}개")
        print(f"   클래스: {len(application.application_summary)}개")
        print(f"{'='*60}\n")
        
    except Exception as e:
        # 에러 처리
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        
        application.status = "failed"
        application.error_message = error_msg
        application.completed_at = timezone.now()
        application.save()
        
        print(f"❌ Application 실패: {e}")
        print(traceback.format_exc())
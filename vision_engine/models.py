from django.db import models
from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver
from pathlib import Path
import shutil
import os


# ============================================
# Application 모델
# ============================================

class Application(models.Model):
    """모델 적용 작업"""

    STATUS_CHOICES = [
        ("ready", "대기"),
        ("processing", "처리 중"),
        ("completed", "완료"),
        ("failed", "실패"),
        ("cancelled", "취소"),
    ]

    # 연결 - PreprocessingTask와 연결
    preprocessing_task = models.ForeignKey(
        'preprocess.PreprocessingTask',
        on_delete=models.CASCADE,
        related_name="applications",
        verbose_name="전처리 작업",
    )

    # ⭐ 단일 모델 필드 (통합)
    model = models.ForeignKey(
        "modelhub.Model",
        on_delete=models.CASCADE,
        verbose_name="모델",
        related_name="applications",
    )

    # 기본 정보
    title = models.CharField(max_length=200, verbose_name="제목")
    description = models.TextField(blank=True, verbose_name="설명")

    # 실행 상태
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="ready", verbose_name="상태"
    )

    # 진행률
    total_frames = models.IntegerField(default=0, verbose_name="총 프레임")
    processed_frames = models.IntegerField(default=0, verbose_name="처리된 프레임")
    progress = models.IntegerField(default=0, verbose_name="진행률 (%)")

    # 결과 - results 디렉토리 사용
    output_file_path = models.CharField(
        max_length=500, blank=True, verbose_name="결과 파일 경로"
    )
    application_data = models.JSONField(
        default=list, blank=True, verbose_name="모델 적용 결과 데이터"
    )
    total_applications = models.IntegerField(default=0, verbose_name="총 탐지 수")
    application_summary = models.JSONField(
        default=dict, blank=True, verbose_name="모델 적용 결과 요약"
    )

    # 에러
    error_message = models.TextField(blank=True, verbose_name="에러 메시지")

    # 시간
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    started_at = models.DateTimeField(null=True, blank=True, verbose_name="시작 시간")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="완료 시간")

    class Meta:
        verbose_name = "모델 적용"
        verbose_name_plural = "모델 적용들"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} - {self.get_status_display()}"

    def get_model(self):
        """사용된 모델 반환"""
        return self.model

    def get_model_name(self):
        """모델 이름 반환"""
        return self.model.name if self.model else "모델 없음"

    def get_content(self):
        """연결된 콘텐츠 반환"""
        return self.preprocessing_task.get_content()

    def get_content_type(self):
        """콘텐츠 타입 반환"""
        return self.preprocessing_task.get_content_type()

    def get_output_url(self):
        """결과 파일 URL 반환"""
        if self.output_file_path:
            return f"/results-media/{self.output_file_path}"
        return None

    def save_results(self, applications):
        """모델 결과 저장"""
        self.application_data = applications
        self.save(update_fields=["application_data"])

    def get_summary_stats(self):
        """요약 통계 반환"""
        if not self.application_summary:
            return []
        
        return [
            {"label": label, "count": count}
            for label, count in sorted(
                self.application_summary.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        ]
    
    def get_duration_display(self):
        """실행 시간 표시"""
        if self.started_at and self.completed_at:
            duration = self.completed_at - self.started_at
            total_seconds = int(duration.total_seconds())
            
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}시간 {minutes}분 {seconds}초"
            elif minutes > 0:
                return f"{minutes}분 {seconds}초"
            else:
                return f"{seconds}초"
        return "-"

    def delete(self, *args, **kwargs):
        """삭제 시 결과 파일도 함께 삭제"""
        if self.output_file_path:
            try:
                output_full_path = Path(settings.RESULTS_ROOT) / self.output_file_path
                application_dir = output_full_path.parent
                
                if application_dir.exists():
                    shutil.rmtree(application_dir)
                    print(f"✅ 모델 적용 결과 디렉토리 삭제: {application_dir}")
            except Exception as e:
                print(f"⚠️ 모델 적용 결과 삭제 실패: {e}")
        
        super().delete(*args, **kwargs)

    def get_status_display_badge(self):
        """상태에 따른 Bootstrap 색상 클래스 반환"""
        status_colors = {
            'ready': 'secondary',
            'processing': 'primary',
            'completed': 'success',
            'failed': 'danger',
            'cancelled': 'warning',
        }
        return status_colors.get(self.status, 'secondary')


@receiver(post_delete, sender=Application)
def application_delete_files(sender, instance, **kwargs):
    """Application 삭제 시 results 폴더의 파일도 삭제"""
    if instance.output_file_path:
        try:
            file_path = Path(settings.RESULTS_ROOT) / instance.output_file_path
            
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                print(f"✅ 모델 적용 결과 파일 삭제: {file_path}")
                
                # 빈 폴더 정리
                remove_empty_directories(file_path, Path(settings.RESULTS_ROOT))
                
        except Exception as e:
            print(f"❌ 모델 적용 결과 파일 삭제 실패: {e}")


def remove_empty_directories(file_path, base_path):
    """파일 삭제 후 빈 폴더를 재귀적으로 삭제"""
    try:
        file_path = Path(file_path)
        base_path = Path(base_path)
        
        current_dir = file_path.parent
        
        while current_dir != base_path and current_dir.is_relative_to(base_path):
            if current_dir.exists() and current_dir.is_dir():
                try:
                    current_dir.rmdir()
                    print(f"✅ 빈 폴더 삭제: {current_dir}")
                except OSError:
                    break
            current_dir = current_dir.parent
            
    except Exception as e:
        print(f"⚠️ 빈 폴더 삭제 중 오류: {e}")

    # 기본 정보
    title = models.CharField(max_length=200, verbose_name="제목")
    description = models.TextField(blank=True, verbose_name="설명")

    # 실행 상태
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="ready", verbose_name="상태"
    )

    # 진행률
    total_frames = models.IntegerField(default=0, verbose_name="총 프레임")
    processed_frames = models.IntegerField(default=0, verbose_name="처리된 프레임")
    progress = models.IntegerField(default=0, verbose_name="진행률 (%)")

    # 결과 - results 디렉토리 사용
    output_file_path = models.CharField(
        max_length=500, blank=True, verbose_name="결과 파일 경로"
    )
    application_data = models.JSONField(
        default=list, blank=True, verbose_name="모델 적용 결과 데이터"
    )
    total_applications = models.IntegerField(default=0, verbose_name="총 탐지 수")
    application_summary = models.JSONField(
        default=dict, blank=True, verbose_name="모델 적용 결과 요약"
    )

    # 에러
    error_message = models.TextField(blank=True, verbose_name="에러 메시지")

    # 시간
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    started_at = models.DateTimeField(null=True, blank=True, verbose_name="시작 시간")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="완료 시간")

    class Meta:
        verbose_name = "모델 적용"
        verbose_name_plural = "모델 적용들"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} - {self.get_status_display()}"

    def get_model(self):
        """
        사용된 모델 반환
        builtin_model과 custom_model은 사실상 같은 Model 테이블의 Proxy
        """
        if self.builtin_model:
            return self.builtin_model
        return self.custom_model

    def get_model_name(self):
        """모델 이름 반환"""
        model = self.get_model()
        if model:
            # BuiltinModel은 display_name 프로퍼티가 있음
            if hasattr(model, 'display_name'):
                return model.display_name or model.name
            else:
                return model.name
        return "모델 없음"

    def get_content(self):
        """연결된 콘텐츠 반환"""
        return self.preprocessing_task.get_content()

    def get_content_type(self):
        """콘텐츠 타입 반환"""
        return self.preprocessing_task.get_content_type()

    def get_output_url(self):
        """결과 파일 URL 반환"""
        if self.output_file_path:
            # results 디렉토리의 파일을 /results-media/ URL로 서빙
            return f"/results-media/{self.output_file_path}"
        return None

    def save_results(self, applications):
        """모델 결과 저장"""
        self.application_data = applications
        self.save(update_fields=["application_data"])

    def get_summary_stats(self):
        """요약 통계 반환"""
        if not self.application_summary:
            return []
        
        return [
            {"label": label, "count": count}
            for label, count in sorted(
                self.application_summary.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        ]
    
    def get_duration_display(self):
        """실행 시간 표시"""
        if self.started_at and self.completed_at:
            duration = self.completed_at - self.started_at
            total_seconds = int(duration.total_seconds())
            
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}시간 {minutes}분 {seconds}초"
            elif minutes > 0:
                return f"{minutes}분 {seconds}초"
            else:
                return f"{seconds}초"
        return "-"

    def delete(self, *args, **kwargs):
        """삭제 시 결과 파일도 함께 삭제"""
        # 결과 파일 삭제
        if self.output_file_path:
            try:
                # 전체 모델 적용 디렉토리 삭제 (application/콘텐츠타입/콘텐츠ID/application_ID/)
                output_full_path = Path(settings.RESULTS_ROOT) / self.output_file_path
                application_dir = output_full_path.parent
                
                if application_dir.exists():
                    shutil.rmtree(application_dir)
                    print(f"✅ 모델 적용 결과 디렉토리 삭제: {application_dir}")
            except Exception as e:
                print(f"⚠️ 모델 적용 결과 삭제 실패: {e}")
        
        super().delete(*args, **kwargs)

    def get_status_display_badge(self):
        """상태에 따른 Bootstrap 색상 클래스 반환"""
        status_colors = {
            'ready': 'secondary',
            'processing': 'primary',
            'completed': 'success',
            'failed': 'danger',
            'cancelled': 'warning',
        }
        return status_colors.get(self.status, 'secondary')


@receiver(post_delete, sender=Application)
def application_delete_files(sender, instance, **kwargs):
    """Application 삭제 시 results 폴더의 파일도 삭제"""
    if instance.output_file_path:
        try:
            file_path = Path(settings.RESULTS_ROOT) / instance.output_file_path
            
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                print(f"✅ 모델 적용 결과 파일 삭제: {file_path}")
                
                # 빈 폴더 정리
                remove_empty_directories(file_path, Path(settings.RESULTS_ROOT))
                
        except Exception as e:
            print(f"❌ 모델 적용 결과 파일 삭제 실패: {e}")


def remove_empty_directories(file_path, base_path):
    """
    파일 삭제 후 빈 폴더를 재귀적으로 삭제
    """
    try:
        file_path = Path(file_path)
        base_path = Path(base_path)
        
        current_dir = file_path.parent
        
        while current_dir != base_path and current_dir.is_relative_to(base_path):
            if current_dir.exists() and current_dir.is_dir():
                try:
                    current_dir.rmdir()
                    print(f"✅ 빈 폴더 삭제: {current_dir}")
                except OSError:
                    break
            current_dir = current_dir.parent
            
    except Exception as e:
        print(f"⚠️ 빈 폴더 삭제 중 오류: {e}")
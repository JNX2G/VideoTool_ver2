import os
import shutil
from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver
from pathlib import Path

from django.db import models
from django.utils import timezone
from contents.models import Video, Image



class PreprocessingTask(models.Model):
    """전처리 작업"""
    
    STATUS_CHOICES = [
        ("ready", "준비"),
        ("pending", "대기 중"),
        ("processing", "처리 중"),
        ("completed", "완료"),
        ("failed", "실패"),
        ("cancelled", "취소됨"), 
    ]

    # video 또는 image 중 하나만 필수
    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name="preprocessing_tasks",
        null=True,
        blank=True,
        verbose_name="동영상",
    )

    image = models.ForeignKey(
        Image,
        on_delete=models.CASCADE,
        related_name="preprocessing_tasks",
        null=True,
        blank=True,
        verbose_name="이미지",
    )

    # ⭐ prephub의 PreprocessingMethod를 참조하는 파이프라인
    # [{"method_id": 1, "params": {"ksize": 5}}, {"method_id": 3, "params": {...}}]
    preprocessing_pipeline = models.JSONField(
        default=list, 
        blank=True, 
        verbose_name="전처리 파이프라인",
        help_text="적용할 전처리 기법들의 순서와 파라미터"
    )

    status = models.CharField(
        max_length=20, 
        choices=STATUS_CHOICES, 
        default="pending", 
        verbose_name="상태"
    )

    progress = models.IntegerField(
        default=0, 
        verbose_name="진행률 (%)"
    )

    current_step = models.CharField(
        max_length=200, 
        blank=True, 
        null=True,
        default="", 
        verbose_name="현재 단계"
    )

    # 처리 정보
    started_at = models.DateTimeField(
        null=True, 
        blank=True, 
        verbose_name="시작 시간"
    )
    
    completed_at = models.DateTimeField(
        null=True, 
        blank=True, 
        verbose_name="완료 시간"
    )
    
    processed_frames = models.IntegerField(
        default=0, 
        verbose_name="처리된 프레임"
    )
    
    total_frames = models.IntegerField(
        default=0, 
        verbose_name="총 프레임"
    )

    # 결과 저장 (results/preprocessing/{task_id}/ 경로)
    output_file_path = models.CharField(
        max_length=500, 
        blank=True, 
        verbose_name="출력 파일 경로"
    )

    # 에러
    error_message = models.TextField(
        blank=True, 
        null=True,
        default="",
        verbose_name="에러 메시지"
    )

    created_at = models.DateTimeField(
        auto_now_add=True, 
        verbose_name="생성일"
    )
    
    updated_at = models.DateTimeField(
        auto_now=True, 
        verbose_name="수정일"
    )

    class Meta:
        verbose_name = "전처리 작업"
        verbose_name_plural = "전처리 작업들"
        ordering = ["-created_at"]

    def __str__(self):
        content = self.get_content()
        if content:
            return f"전처리 #{self.id} - {content.title}"
        return f"전처리 #{self.id}"

    def clean(self):
        """video와 image 중 하나만 있어야 함"""
        from django.core.exceptions import ValidationError

        if not self.video and not self.image:
            raise ValidationError("동영상 또는 이미지 중 하나는 필수입니다.")
        if self.video and self.image:
            raise ValidationError("동영상과 이미지를 동시에 지정할 수 없습니다.")

    def get_content(self):
        """컨텐츠 객체 반환 (video 또는 image)"""
        return self.video if self.video else self.image

    def get_content_type(self):
        """컨텐츠 타입 반환"""
        return "video" if self.video else "image"

    def is_using_original_file(self):
        """원본 파일을 사용하는지 확인"""
        return self.output_file_path and self.output_file_path.startswith("__original__:")
    
    def get_actual_file_path(self):
        """
        실제 파일 시스템 경로 반환
        
        Returns:
            str: 파일의 절대 경로
        """
        from django.conf import settings
        import os
        
        if not self.output_file_path:
            return None
        
        if self.is_using_original_file():
            # 원본 파일 경로 추출
            original_path = self.output_file_path.replace("__original__:", "")
            return os.path.join(settings.MEDIA_ROOT, original_path)
        else:
            # 전처리 결과 파일 경로
            return os.path.join(settings.RESULTS_ROOT, self.output_file_path)

    # ⭐ 전처리 파이프라인 관리 메서드들
    def add_preprocessing_step(self, method_id, params=None):
        """전처리 단계 추가"""
        if not isinstance(self.preprocessing_pipeline, list):
            self.preprocessing_pipeline = []

        step = {
            "method_id": method_id,
            "params": params or {}
        }
        self.preprocessing_pipeline.append(step)
        self.save()

    def remove_preprocessing_step(self, index):
        """특정 인덱스의 전처리 단계 제거"""
        if isinstance(self.preprocessing_pipeline, list) and 0 <= index < len(self.preprocessing_pipeline):
            self.preprocessing_pipeline.pop(index)
            self.save()
            return True
        return False

    def reorder_preprocessing_step(self, from_index, to_index):
        """전처리 단계 순서 변경"""
        if not isinstance(self.preprocessing_pipeline, list):
            return False
        
        if not (0 <= from_index < len(self.preprocessing_pipeline) and 
                0 <= to_index < len(self.preprocessing_pipeline)):
            return False
        
        step = self.preprocessing_pipeline.pop(from_index)
        self.preprocessing_pipeline.insert(to_index, step)
        self.save()
        return True

    def clear_preprocessing_pipeline(self):
        """전처리 파이프라인 초기화"""
        self.preprocessing_pipeline = []
        self.save()

    def get_pipeline_display(self):
        """파이프라인을 읽기 쉬운 형식으로 반환"""
        if not self.preprocessing_pipeline:
            return []

        from prephub.models import PreprocessingMethod
        
        result = []
        for step in self.preprocessing_pipeline:
            method_id = step.get("method_id")
            params = step.get("params", {})
            
            try:
                method = PreprocessingMethod.objects.get(id=method_id)
                display_name = method.name
                
                if params:
                    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                    display_name = f"{display_name} ({param_str})"
                
                result.append(display_name)
            except PreprocessingMethod.DoesNotExist:
                result.append(f"알 수 없는 기법 (ID: {method_id})")
        
        return result

    def get_status_display_badge(self):
        """상태 배지 색상 반환"""
        status_colors = {
            "ready": "info",
            "pending": "secondary",
            "processing": "primary",
            "completed": "success",
            "failed": "danger",
            "cancelled": "warning",  # ⭐ 추가
        }
        return status_colors.get(self.status, "secondary")

    def get_duration(self):
        """작업 소요 시간 반환 (초)"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds()
        return 0

    def delete_files(self):
        """전처리 결과 파일 삭제"""
        import os
        import shutil
        from django.conf import settings

        deleted_files = []

        # 출력 파일 삭제
        if self.output_file_path:
            try:
                # ⭐ 원본 파일 사용 플래그 확인 - 원본 파일은 삭제하지 않음
                if self.output_file_path.startswith("__original__:"):
                    # 원본 파일은 건드리지 않음
                    pass
                else:
                    # 전처리 결과 파일 삭제
                    path = os.path.join(settings.RESULTS_ROOT, self.output_file_path)
                    if os.path.exists(path):
                        os.remove(path)
                        deleted_files.append(path)
            except Exception as e:
                print(f"파일 삭제 실패: {e}")

        # 전처리 결과 폴더 삭제
        result_dir = os.path.join(
            settings.RESULTS_ROOT, "preprocessing", str(self.id)
        )
        if os.path.exists(result_dir):
            try:
                shutil.rmtree(result_dir)
                deleted_files.append(result_dir)
            except Exception as e:
                print(f"폴더 삭제 실패: {e}")

        return deleted_files
    

@receiver(post_delete, sender=PreprocessingTask)
def preprocessing_task_delete_files(sender, instance, **kwargs):
    """PreprocessingTask 삭제 시 results 폴더의 파일도 삭제"""
    if instance.output_file_path:
        # ⭐ 원본 파일 사용 플래그 확인 - 원본 파일은 삭제하지 않음
        if instance.output_file_path.startswith("__original__:"):
            # 원본 파일은 건드리지 않음
            return
        
        # 전처리 결과 파일 삭제
        try:
            # results/preprocess/content_id/ 경로의 파일 삭제
            file_path = os.path.join(settings.RESULTS_ROOT, instance.output_file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ 전처리 결과 파일 삭제: {file_path}")
        except Exception as e:
            print(f"❌ 전처리 결과 파일 삭제 실패: {e}")


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


@receiver(post_delete, sender=PreprocessingTask)
def preprocessing_task_delete_files_with_cleanup(sender, instance, **kwargs):
    """PreprocessingTask 삭제 시 results 폴더의 파일도 삭제 (빈 폴더 정리 포함)"""
    if instance.output_file_path:
        # ⭐ 원본 파일 사용 플래그 확인 - 원본 파일은 삭제하지 않음
        if instance.output_file_path.startswith("__original__:"):
            # 원본 파일은 건드리지 않음
            return
        
        try:
            file_path = Path(settings.RESULTS_ROOT) / instance.output_file_path
            
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                print(f"✅ 전처리 결과 파일 삭제: {file_path}")
                
                remove_empty_directories(file_path, Path(settings.RESULTS_ROOT))
                
        except Exception as e:
            print(f"❌ 전처리 결과 파일 삭제 실패: {e}")
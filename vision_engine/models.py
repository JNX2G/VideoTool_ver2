import re, os

from django.db import models
from django.utils import timezone
from django.conf import settings
from pathlib import Path
import shutil

def sanitize_filename(filename):
    """파일명 정리 - 특수문자, 공백 제거"""
    # 확장자 분리
    name, ext = os.path.splitext(filename)  
    # 공백을 언더스코어로 변경
    name = name.replace(" ", "_")           
    # 특수문자 제거 (알파벳, 숫자, 언더스코어, 하이픈 허용)
    name = re.sub(r"[^\w\-]", "", name)    
    # 연속 언더스코어 제거 
    name = re.sub(r"_+", "_", name)         

    # 파일명 없으면, 기본값 적용
    if not name:                            
        name = "file"
    
    return name, ext.lower()

def video_upload_path(instance, filename):
    """동영상 업로드 경로: videos/YYYY/원본파일명_MMDDhhmmss.확장자"""
    now = timezone.now()
    name, ext = sanitize_filename(filename)
    new_filename = f"{name}_{now.strftime('%m%d%H%M%S')}{ext}"
    return os.path.join("videos", str(now.year), new_filename)

def image_upload_path(instance, filename):
    """이미지 업로드 경로: images/YYYY/원본파일명_MMDDhhmmss.확장자"""
    now = timezone.now()
    name, ext = sanitize_filename(filename)
    new_filename = f"{name}_{now.strftime('%m%d%H%M%S')}{ext}"
    return os.path.join("images", str(now.year), new_filename)

def thumbnail_upload_path(instance, filename):
    """썸네일 업로드 경로"""
    now = timezone.now()
    name, ext = sanitize_filename(filename)
    new_filename = f"{name}_{now.strftime('%m%d%H%M%S')}_thumb.jpg"
    return os.path.join("thumbnails", str(now.year), new_filename)


class Video(models.Model):
    """동영상 모델"""
    title = models.CharField(max_length=200, verbose_name="제목")
    description = models.TextField(blank=True, verbose_name="설명")
    file = models.FileField(upload_to=video_upload_path, verbose_name="동영상 파일")
    thumbnail = models.ImageField(
        upload_to=thumbnail_upload_path, blank=True, null=True, verbose_name="썸네일"
    )
    file_size = models.BigIntegerField(default=0, verbose_name="파일 크기")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="업로드 시간")

    class Meta:
        verbose_name = "동영상"
        verbose_name_plural = "동영상들"
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.title
    
    def get_file_size_display(self):
        """파일 크기를 읽기 쉬운 형식으로 반환"""
        size = self.file_size
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"
    

class Image(models.Model):
    """이미지 모델"""
    title = models.CharField(max_length=200, verbose_name="제목")
    description = models.TextField(blank=True, verbose_name="설명")
    file = models.ImageField(upload_to=image_upload_path, verbose_name="이미지 파일")
    file_size = models.BigIntegerField(default=0, verbose_name="파일 크기")
    width = models.IntegerField(default=0, verbose_name="너비")
    height = models.IntegerField(default=0, verbose_name="높이")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="업로드 시간")

    class Meta:
        verbose_name = "이미지"
        verbose_name_plural = "이미지들"
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.title
    
    def get_file_size_display(self):
        """파일 크기를 읽기 쉬운 형식으로 반환"""
        size = self.file_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    def get_resolution_display(self):
        """해상도 표시"""
        if self.width and self.height:
            return f"{self.width} × {self.height}"
        return "-"


class Detection(models.Model):
    """객체 탐지 작업"""

    STATUS_CHOICES = [
        ("ready", "대기"),
        ("processing", "처리 중"),
        ("completed", "완료"),
        ("failed", "실패"),
    ]

    # 연결 - PreprocessingTask와 연결
    preprocessing_task = models.ForeignKey(
        'preprocess.PreprocessingTask',
        on_delete=models.CASCADE,
        related_name="detections",
        verbose_name="전처리 작업",
    )

    # 모델 선택 - CASCADE 설정으로 모델 삭제 시 탐지 결과도 삭제
    base_model = models.ForeignKey(
        "modelhub.BaseModel",
        on_delete=models.CASCADE,  # ⭐ SET_NULL에서 CASCADE로 변경
        null=True,
        blank=True,
        verbose_name="기본 모델",
        related_name="detections",
    )
    custom_model = models.ForeignKey(
        "modelhub.CustomModel",
        on_delete=models.CASCADE,  # ⭐ SET_NULL에서 CASCADE로 변경
        null=True,
        blank=True,
        verbose_name="커스텀 모델",
        related_name="detections",
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
    detection_data = models.JSONField(
        default=list, blank=True, verbose_name="탐지 데이터"
    )
    total_detections = models.IntegerField(default=0, verbose_name="총 탐지 수")
    detection_summary = models.JSONField(
        default=dict, blank=True, verbose_name="탐지 요약"
    )

    # 에러
    error_message = models.TextField(blank=True, verbose_name="에러 메시지")

    # 시간
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    started_at = models.DateTimeField(null=True, blank=True, verbose_name="시작 시간")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="완료 시간")

    class Meta:
        verbose_name = "객체 탐지"
        verbose_name_plural = "객체 탐지들"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} - {self.get_status_display()}"

    def get_model(self):
        """사용된 모델 반환"""
        if self.base_model:
            return self.base_model
        return self.custom_model

    def get_model_name(self):
        """모델 이름 반환"""
        model = self.get_model()
        if model:
            return model.display_name or model.name
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

    def save_results(self, detections):
        """탐지 결과 저장"""
        import json
        self.detection_data = detections
        self.save(update_fields=["detection_data"])

    def get_summary_stats(self):
        """요약 통계 반환"""
        if not self.detection_summary:
            return []
        
        return [
            {"label": label, "count": count}
            for label, count in sorted(
                self.detection_summary.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        ]

    def delete(self, *args, **kwargs):
        """삭제 시 결과 파일도 함께 삭제"""
        # 결과 파일 삭제
        if self.output_file_path:
            try:
                # 전체 탐지 디렉토리 삭제 (detection/콘텐츠타입/콘텐츠ID/detection_ID/)
                output_full_path = Path(settings.RESULTS_ROOT) / self.output_file_path
                detection_dir = output_full_path.parent
                
                if detection_dir.exists():
                    shutil.rmtree(detection_dir)
                    print(f"✅ 탐지 결과 디렉토리 삭제: {detection_dir}")
            except Exception as e:
                print(f"⚠️ 탐지 결과 삭제 실패: {e}")
        
        super().delete(*args, **kwargs)
import re, os

from django.db import models
from django.utils import timezone

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
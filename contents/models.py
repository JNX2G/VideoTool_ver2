import re, os

from django.db import models
from django.utils import timezone
from django.db.models.signals import post_delete, pre_save
from django.dispatch import receiver
from django.conf import settings
from pathlib import Path



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
    """썸네일 업로드 경로: videos/YYYY/원본파일명_MMDDhhmmss_thumb.jpg"""
    now = timezone.now()
    name, ext = sanitize_filename(filename)
    new_filename = f"{name}_{now.strftime('%m%d%H%M%S')}_thumb.jpg"
    return os.path.join("videos", str(now.year), new_filename)


def remove_empty_directories(file_path, base_path):
    """
    파일 삭제 후 빈 폴더를 재귀적으로 삭제
    
    Args:
        file_path (str|Path): 삭제된 파일의 경로
        base_path (str|Path): 삭제를 멈출 기준 경로
    """
    try:
        file_path = Path(file_path)
        base_path = Path(base_path)
        
        # 파일의 부모 디렉토리부터 시작
        current_dir = file_path.parent
        
        # base_path에 도달할 때까지 반복
        while current_dir != base_path and current_dir.is_relative_to(base_path):
            if current_dir.exists() and current_dir.is_dir():
                try:
                    # 비어있으면 삭제
                    current_dir.rmdir()
                    print(f"✅ 빈 폴더 삭제: {current_dir}")
                except OSError:
                    # 폴더가 비어있지 않으면 중단
                    break
            
            # 상위 폴더로 이동
            current_dir = current_dir.parent
            
    except Exception as e:
        print(f"⚠️ 빈 폴더 삭제 중 오류: {e}")


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


# ============================================
# Signals - 파일 자동 삭제 + 빈 폴더 삭제
# ============================================

@receiver(post_delete, sender=Video)
def video_delete_files(sender, instance, **kwargs):
    """Video 삭제 시 관련 파일들도 삭제"""
    base_path = Path(settings.MEDIA_ROOT)
    
    # 동영상 파일 삭제
    if instance.file:
        try:
            file_path = Path(instance.file.path)
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                print(f"✅ 동영상 파일 삭제: {file_path}")
                
                # ⭐ 빈 폴더 재귀 삭제
                remove_empty_directories(file_path, base_path)
        except Exception as e:
            print(f"❌ 동영상 파일 삭제 실패: {e}")
    
    # 썸네일 파일 삭제
    if instance.thumbnail:
        try:
            thumb_path = Path(instance.thumbnail.path)
            if thumb_path.exists() and thumb_path.is_file():
                thumb_path.unlink()
                print(f"✅ 썸네일 파일 삭제: {thumb_path}")
                
                # ⭐ 빈 폴더 재귀 삭제
                remove_empty_directories(thumb_path, base_path)
        except Exception as e:
            print(f"❌ 썸네일 파일 삭제 실패: {e}")


@receiver(pre_save, sender=Video)
def video_update_files(sender, instance, **kwargs):
    """Video 업데이트 시 이전 파일 삭제 (파일이 변경된 경우)"""
    if not instance.pk:
        return  # 새로운 객체는 건너뜀
    
    try:
        old_instance = Video.objects.get(pk=instance.pk)
    except Video.DoesNotExist:
        return
    
    base_path = Path(settings.MEDIA_ROOT)
    
    # 동영상 파일이 변경된 경우 이전 파일 삭제
    if old_instance.file and old_instance.file != instance.file:
        try:
            old_path = Path(old_instance.file.path)
            if old_path.exists() and old_path.is_file():
                old_path.unlink()
                print(f"✅ 이전 동영상 파일 삭제: {old_path}")
                remove_empty_directories(old_path, base_path)
        except Exception as e:
            print(f"❌ 이전 동영상 파일 삭제 실패: {e}")
    
    # 썸네일이 변경된 경우 이전 파일 삭제
    if old_instance.thumbnail and old_instance.thumbnail != instance.thumbnail:
        try:
            old_thumb = Path(old_instance.thumbnail.path)
            if old_thumb.exists() and old_thumb.is_file():
                old_thumb.unlink()
                print(f"✅ 이전 썸네일 파일 삭제: {old_thumb}")
                remove_empty_directories(old_thumb, base_path)
        except Exception as e:
            print(f"❌ 이전 썸네일 파일 삭제 실패: {e}")


@receiver(post_delete, sender=Image)
def image_delete_files(sender, instance, **kwargs):
    """Image 삭제 시 파일도 삭제"""
    if instance.file:
        try:
            file_path = Path(instance.file.path)
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                print(f"✅ 이미지 파일 삭제: {file_path}")
                
                # ⭐ 빈 폴더 재귀 삭제
                remove_empty_directories(file_path, Path(settings.MEDIA_ROOT))
        except Exception as e:
            print(f"❌ 이미지 파일 삭제 실패: {e}")


@receiver(pre_save, sender=Image)
def image_update_files(sender, instance, **kwargs):
    """Image 업데이트 시 이전 파일 삭제 (파일이 변경된 경우)"""
    if not instance.pk:
        return  # 새로운 객체는 건너뜀
    
    try:
        old_instance = Image.objects.get(pk=instance.pk)
    except Image.DoesNotExist:
        return
    
    # 이미지 파일이 변경된 경우 이전 파일 삭제
    if old_instance.file and old_instance.file != instance.file:
        try:
            old_path = Path(old_instance.file.path)
            if old_path.exists() and old_path.is_file():
                old_path.unlink()
                print(f"✅ 이전 이미지 파일 삭제: {old_path}")
                remove_empty_directories(old_path, Path(settings.MEDIA_ROOT))
        except Exception as e:
            print(f"❌ 이전 이미지 파일 삭제 실패: {e}")
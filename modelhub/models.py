from django.db import models
from django.utils import timezone
from django.conf import settings
import os
import re


# Create your models here.
def sanitize_model_filename(filename):
    """모델 파일명을 안전하게 정리"""
    name, ext = os.path.splitext(filename)
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w\-]", "", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if not name:
        name = "model"

    return name, ext.lower()

def default_model_upload_path(instance, filename):
    """기본 모델 업로드 경로: models/default/파일명.확장자"""
    name, ext = sanitize_model_filename(filename)
    new_filename = f"{name}{ext}"

    return os.path.join("default", new_filename)


def custom_model_upload_path(instance, filename):
    """커스텀 모델 업로드 경로: models/custom/YYYY/파일명_MMDDhhmmss.확장자"""
    now = timezone.now()
    name, ext = sanitize_model_filename(filename)
    new_filename = f"{name}_{now.strftime('%m%d%H%M%S')}{ext}"

    return os.path.join("custom", str(now.year), new_filename)


class BaseModel(models.Model):
    """기본 모델 (사전 학습된 YOLO 등)"""
    
    # 기본 정보
    name = models.CharField(max_length=200, verbose_name="모델 이름")
    display_name = models.CharField(max_length=200, verbose_name="표시 이름")
    description = models.TextField(blank=True, verbose_name="설명")

    # 모델 정보
    model_type = models.CharField(
        max_length=50, default="yolo", verbose_name="모델 타입"
    )
    version = models.CharField(max_length=50, blank=True, verbose_name="버전")

    # 파일 정보
    model_file = models.FileField(
        upload_to=default_model_upload_path,
        blank=True,
        null=True,
        verbose_name="모델 파일",
    )
    file_size = models.BigIntegerField(default=0, verbose_name="파일 크기 (bytes)")

    # YOLO 특화 정보
    yolo_version = models.CharField(
        max_length=50,
        blank=True,
        verbose_name="YOLO 버전",
        help_text="예: yolov8n.pt, yolov8s.pt",
    )
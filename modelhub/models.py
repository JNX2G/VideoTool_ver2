from django.db import models
from django.utils import timezone
from django.conf import settings
import os
import re


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

    # 파일 정보 (선택적 - 자동 다운로드 모델은 파일 없음)
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

    # 성능 정보
    classes = models.JSONField(default=list, blank=True, verbose_name="탐지 클래스")
    input_size = models.CharField(max_length=50, blank=True, verbose_name="입력 크기")

    # 추가 설정
    config = models.JSONField(default=dict, blank=True, verbose_name="설정")

    # 상태
    is_active = models.BooleanField(default=True, verbose_name="활성화")
    is_default = models.BooleanField(default=False, verbose_name="기본 모델")

    # 통계
    usage_count = models.IntegerField(default=0, verbose_name="사용 횟수")

    # 메타정보
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")

    class Meta:
        verbose_name = "기본 모델"
        verbose_name_plural = "기본 모델들"
        ordering = ["-is_default", "-created_at"]

    def __str__(self):
        return self.display_name or self.name

    def get_model_path(self):
        """실제 모델 경로 반환"""
        # 파일이 업로드된 경우
        if self.model_file:
            return self.model_file.path

        # YOLO 자동 다운로드 모델인 경우
        if self.yolo_version:
            # ultralytics가 자동으로 관리하는 경로
            return self.yolo_version

        return None

    def get_file_size_display(self):
        """파일 크기를 읽기 쉬운 형식으로 변환"""
        if self.file_size == 0:
            return "자동 다운로드" if self.yolo_version else "-"

        size = self.file_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def increment_usage(self):
        """사용 횟수 증가"""
        self.usage_count += 1
        self.save(update_fields=["usage_count", "updated_at"])


class CustomModel(models.Model):
    """커스텀 모델 (사용자가 학습시킨 모델)"""

    # 기본 정보
    name = models.CharField(max_length=200, verbose_name="모델 이름")
    description = models.TextField(blank=True, verbose_name="설명")

    # 파일 정보 (필수)
    model_file = models.FileField(
        upload_to=custom_model_upload_path, verbose_name="모델 파일"
    )
    file_size = models.BigIntegerField(default=0, verbose_name="파일 크기 (bytes)")
    file_format = models.CharField(max_length=10, blank=True, verbose_name="파일 형식")

    # 모델 정보
    model_type = models.CharField(
        max_length=50,
        default="custom",
        verbose_name="모델 타입",
        help_text="예: yolo, resnet, custom",
    )
    framework = models.CharField(
        max_length=50,
        blank=True,
        verbose_name="프레임워크",
        help_text="예: PyTorch, TensorFlow, ONNX",
    )
    version = models.CharField(max_length=50, blank=True, verbose_name="버전")

    # 학습 정보
    dataset_name = models.CharField(
        max_length=200, blank=True, verbose_name="데이터셋 이름"
    )
    classes = models.JSONField(default=list, blank=True, verbose_name="탐지 클래스")
    num_classes = models.IntegerField(null=True, blank=True, verbose_name="클래스 수")

    # 성능 정보
    accuracy = models.FloatField(null=True, blank=True, verbose_name="정확도 (%)")
    precision = models.FloatField(null=True, blank=True, verbose_name="정밀도 (%)")
    recall = models.FloatField(null=True, blank=True, verbose_name="재현율 (%)")
    map_score = models.FloatField(null=True, blank=True, verbose_name="mAP 점수")
    inference_time = models.FloatField(
        null=True, blank=True, verbose_name="추론 시간 (ms)"
    )

    # 메타데이터
    author = models.CharField(max_length=100, blank=True, verbose_name="작성자")
    tags = models.JSONField(default=list, blank=True, verbose_name="태그")
    config = models.JSONField(default=dict, blank=True, verbose_name="설정")

    # 상태
    is_active = models.BooleanField(default=True, verbose_name="활성화")
    is_validated = models.BooleanField(default=False, verbose_name="검증됨")

    # 통계
    usage_count = models.IntegerField(default=0, verbose_name="사용 횟수")

    # 메타정보
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")

    class Meta:
        verbose_name = "커스텀 모델"
        verbose_name_plural = "커스텀 모델들"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def get_model_path(self):
        """실제 모델 파일 경로 반환"""
        if self.model_file:
            return self.model_file.path
        return None

    def get_file_size_display(self):
        """파일 크기를 읽기 쉬운 형식으로 변환"""
        size = self.file_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def increment_usage(self):
        """사용 횟수 증가"""
        self.usage_count += 1
        self.save(update_fields=["usage_count", "updated_at"])

    def validate_model(self):
        """모델 검증"""
        try:
            # 파일 존재 확인
            if not os.path.exists(self.get_model_path()):
                return False, "모델 파일을 찾을 수 없습니다"

            # 파일 크기 확인
            if self.file_size == 0:
                return False, "모델 파일이 비어있습니다"

            # 추가 검증 로직...

            self.is_validated = True
            self.save(update_fields=["is_validated", "updated_at"])
            return True, "모델 검증 완료"

        except Exception as e:
            return False, f"검증 실패: {str(e)}"


class ModelCategory(models.Model):
    """모델 카테고리 (선택적)"""

    name = models.CharField(max_length=100, unique=True, verbose_name="카테고리 이름")
    description = models.TextField(blank=True, verbose_name="설명")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")

    class Meta:
        verbose_name = "모델 카테고리"
        verbose_name_plural = "모델 카테고리들"

    def __str__(self):
        return self.name
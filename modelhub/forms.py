from django import forms
from .models import BaseModel, CustomModel
from django.core.exceptions import ValidationError
import os


class BaseModelForm(forms.ModelForm):
    """기본 모델 폼"""

    class Meta:
        model = BaseModel
        fields = [
            "name",
            "display_name",
            "description",
            "model_type",
            "version",
            "yolo_version",
            "model_file",
            "config",
            "is_active",
            "is_default",
        ]
        widgets = {
            "name": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "예: yolov8s"}
            ),
            "display_name": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "예: YOLOv8 Small (기본 모델)",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 3,
                    "placeholder": "모델 설명을 입력하세요",
                }
            ),
            "model_type": forms.TextInput(
                attrs={"class": "form-control", "value": "yolo"}
            ),
            "version": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "예: v8"}
            ),
            "yolo_version": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "예: yolov8s.pt (자동 다운로드)",
                }
            ),
            "model_file": forms.FileInput(attrs={"class": "form-control"}),
            "is_active": forms.CheckboxInput(attrs={"class": "form-check-input"}),
            "is_default": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }

    def clean(self):
        cleaned_data = super().clean()
        model_file = cleaned_data.get("model_file")
        yolo_version = cleaned_data.get("yolo_version")

        # 모델 파일 또는 YOLO 버전 중 하나는 있어야 함
        if not model_file and not yolo_version:
            raise ValidationError("모델 파일을 업로드하거나 YOLO 버전을 입력해주세요.")

        return cleaned_data


class CustomModelForm(forms.ModelForm):
    """커스텀 모델 폼"""

    class Meta:
        model = CustomModel
        fields = [
            "name",
            "description",
            "model_file",
            "model_type",
            "framework",
            "version",
            "dataset_name",
            "classes",
            "num_classes",
            "accuracy",
            "precision",
            "recall",
            "map_score",
            "author",
            "tags",
            "config",
        ]
        widgets = {
            "name": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "커스텀 모델 이름"}
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 3,
                    "placeholder": "모델에 대한 설명을 입력하세요",
                }
            ),
            "model_file": forms.FileInput(
                attrs={"class": "form-control", "accept": ".pt,.pth,.onnx,.h5,.pb"}
            ),
            "model_type": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "예: yolo, custom"}
            ),
            "framework": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "예: PyTorch, TensorFlow",
                }
            ),
            "version": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "모델 버전"}
            ),
            "dataset_name": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "학습에 사용한 데이터셋 이름",
                }
            ),
            "num_classes": forms.NumberInput(
                attrs={"class": "form-control", "placeholder": "탐지 가능한 클래스 수"}
            ),
            "accuracy": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "step": "0.01",
                    "placeholder": "정확도 (%)",
                }
            ),
            "precision": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "step": "0.01",
                    "placeholder": "정밀도 (%)",
                }
            ),
            "recall": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "step": "0.01",
                    "placeholder": "재현율 (%)",
                }
            ),
            "map_score": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "step": "0.01",
                    "placeholder": "mAP 점수",
                }
            ),
            "author": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "모델 작성자"}
            ),
        }

    def clean_model_file(self):
        """모델 파일 검증"""
        file = self.cleaned_data.get("model_file")

        if file:
            # 파일 크기 체크 (500MB)
            max_size = 500 * 1024 * 1024
            if file.size > max_size:
                raise ValidationError(
                    f"모델 파일 크기는 500MB를 초과할 수 없습니다. "
                    f"(현재: {file.size / 1024 / 1024:.2f}MB)"
                )

            # 파일 확장자 검증
            ext = os.path.splitext(file.name)[1].lower()
            allowed_extensions = [".pt", ".pth", ".onnx", ".h5", ".pb"]

            if ext not in allowed_extensions:
                raise ValidationError(
                    f"허용되지 않는 파일 형식입니다. "
                    f'허용 형식: {", ".join(allowed_extensions)}'
                )

        return file

    def clean(self):
        cleaned_data = super().clean()
        model_file = cleaned_data.get("model_file")

        # 커스텀 모델은 파일 필수
        if not model_file:
            raise ValidationError("커스텀 모델은 파일 업로드가 필수입니다.")

        return cleaned_data
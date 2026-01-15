from django import forms
from .models import BuiltinModel, CustomModel
from django.core.exceptions import ValidationError
import os


class BuiltinModelForm(forms.ModelForm):
    """기본 모델 폼"""

    class Meta:
        model = BuiltinModel
        fields = [
            "name",
            "display_name",
            "description",
            "task_type",  # 추가
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
            "task_type": forms.Select(
                attrs={"class": "form-control"}
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
            "task_type",  # 추가
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
            "inference_time",
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
            "task_type": forms.Select(
                attrs={"class": "form-control"}
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
            "inference_time": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "step": "0.01",
                    "placeholder": "추론 시간 (ms)",
                }
            ),
            "author": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "작성자"}
            ),
        }

    def clean_model_file(self):
        model_file = self.cleaned_data.get("model_file")
        if model_file:
            # 파일 확장자 검증
            valid_extensions = ['.pt', '.pth', '.onnx', '.h5', '.pb']
            file_ext = os.path.splitext(model_file.name)[1].lower()
            
            if file_ext not in valid_extensions:
                raise ValidationError(
                    f"지원하지 않는 파일 형식입니다. "
                    f"허용된 형식: {', '.join(valid_extensions)}"
                )
        
        return model_file
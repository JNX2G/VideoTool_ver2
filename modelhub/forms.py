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
    """커스텀 모델 폼 - CustomModel의 실제 필드에 맞춰 수정"""

    class Meta:
        model = CustomModel
        fields = [
            "name",
            "description",
            "model_file",
            "model_type",
            "training_dataset",
            "training_epochs",
            "performance_metrics",
            "config",
            "is_active",
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
            "training_dataset": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "학습에 사용한 데이터셋 이름",
                }
            ),
            "training_epochs": forms.NumberInput(
                attrs={"class": "form-control", "placeholder": "학습 에포크 수"}
            ),
            "is_active": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }

    # JSONField를 CharField로 오버라이드하여 UI에서 입력 편의성 제공
    performance_metrics = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control font-monospace',
            'rows': 5,
            'placeholder': '''예시:
{
  "accuracy": 0.95,
  "precision": 0.92,
  "recall": 0.90,
  "map": 0.88
}'''
        }),
        help_text="JSON 형식으로 성능 지표를 입력하세요"
    )
    
    config = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control font-monospace',
            'rows': 5,
            'placeholder': '''예시:
{
  "conf_threshold": 0.25,
  "iou_threshold": 0.45
}'''
        }),
        help_text="JSON 형식으로 추가 설정을 입력하세요"
    )

    def clean_performance_metrics(self):
        """JSON 형식 검증"""
        import json
        data = self.cleaned_data.get('performance_metrics', '').strip()
        
        if not data:
            return {}
        
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"올바른 JSON 형식이 아닙니다: {e}")

    def clean_config(self):
        """JSON 형식 검증"""
        import json
        data = self.cleaned_data.get('config', '').strip()
        
        if not data:
            return {}
        
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"올바른 JSON 형식이 아닙니다: {e}")

    def clean_model_file(self):
        """파일 크기 및 형식 검증"""
        file = self.cleaned_data.get("model_file")
        
        if file:
            # 최대 파일 크기: 500MB
            max_size = 500 * 1024 * 1024
            if file.size > max_size:
                raise ValidationError("모델 파일 크기는 500MB를 초과할 수 없습니다.")
            
            # 허용된 확장자
            ext = os.path.splitext(file.name)[1].lower()
            allowed_extensions = [".pt", ".pth", ".onnx", ".h5", ".pb"]
            
            if ext not in allowed_extensions:
                raise ValidationError(
                    f"허용되지 않는 파일 형식입니다. "
                    f'허용 형식: {", ".join(allowed_extensions)}'
                )
        
        return file
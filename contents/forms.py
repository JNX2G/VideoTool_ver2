import os

from django import forms
from django.core.exceptions import ValidationError

from .models import Video, Image

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ["title", "description", "file"]
        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "동영상 제목을 입력하세요",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 4,
                    "placeholder": "동영상 설명을 입력하세요 (선택사항)"
                }
            ),
            "file": forms.FileInput(
                attrs={
                    "class": "form-control",
                    "accept": "video/*"
                }
            ),
        }
    
    def clean_file(self):
        file = self.cleaned_data.get("file")
        if file:
            if file.size > 1024 * 1024 * 1024:
                raise forms.ValidationError('파일 크기는 1GB를 초과할 수 없습니다.')
        return file
    

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ["title", "description", "file"]
        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "이미지 제목을 입력하세요",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 4,
                    "placeholder": "이미지 설명을 입력하세요 (선택사항)",
                }
            ),
            "file": forms.FileInput(
                attrs={
                    "class": "form-control",
                    "accept": "image/*"
                }
            ),
        }
    
    def clean_file(self):
        file = self.cleaned_data.get("file")

        if file:
            max_size = 10 * 1024 * 1024
            if file.size > max_size:
                raise ValidationError("이미지 크기는 10MB를 초과할 수 없습니다.")
            
            ext = os.path.splitext(file.name)[1].lower()
            allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]

            if ext not in allowed_extensions:
                raise ValidationError(
                    f"허용되지 않는 파일 형식입니다. "
                    f'허용 형식: {", ".join(allowed_extensions)}'
                )
        return file
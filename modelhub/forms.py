from django import forms
from .models import Model


# ========================================
# 1. Built-in 모델 폼
# ========================================
class BuiltinModelForm(forms.ModelForm):
    """Built-in 모델 추가 폼 (프리셋 선택)"""
    
    class Meta:
        model = Model
        fields = ['builtin_preset', 'description']
        widgets = {
            'builtin_preset': forms.Select(attrs={
                'class': 'form-select',
                'required': True,
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': '이 모델의 용도나 특징을 설명하세요 (선택사항)',
            }),
        }
        labels = {
            'builtin_preset': '모델 프리셋',
            'description': '설명',
        }
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.source = 'builtin'
        if commit:
            instance.save()
        return instance


# ========================================
# 2. 파일 업로드 폼
# ========================================
class FileUploadModelForm(forms.ModelForm):
    """파일 업로드 모델 폼"""
    
    class Meta:
        model = Model
        fields = ['model_file', 'name', 'task_type', 'description']
        widgets = {
            'model_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pt,.pth,.onnx,.h5,.pb,.tflite',
                'required': True,
            }),
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '비워두면 파일명에서 자동 생성',
                'required': False,
            }),
            'task_type': forms.Select(attrs={
                'class': 'form-select',
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': '학습 데이터셋, 성능, 특징 등을 설명하세요 (선택사항)',
            }),
        }
        labels = {
            'model_file': '모델 파일',
            'name': '모델 이름',
            'task_type': '작업 유형',
            'description': '설명',
        }
        help_texts = {
            'task_type': '파일 분석 후 자동으로 추천되지만, 직접 선택할 수 있습니다.',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # name 필드를 필수가 아니도록 설정
        self.fields['name'].required = False
    
    def clean_model_file(self):
        """파일 크기 검증 (최대 1GB)"""
        model_file = self.cleaned_data.get('model_file')
        if model_file:
            if model_file.size > 1024 * 1024 * 1024:  # 1GB
                raise forms.ValidationError('파일 크기는 1GB를 초과할 수 없습니다.')
        return model_file
    
    def clean(self):
        """전체 폼 검증 - 이름 자동 생성"""
        cleaned_data = super().clean()
        name = cleaned_data.get('name')
        model_file = cleaned_data.get('model_file')
        
        # 이름이 비어있으면 파일명에서 생성
        if not name and model_file:
            import os
            filename = os.path.basename(model_file.name)
            cleaned_data['name'] = os.path.splitext(filename)[0]
        
        return cleaned_data
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.source = 'upload'
        if instance.model_file:
            instance.file_size = instance.model_file.size
        if commit:
            instance.save()
        return instance


# ========================================
# 3. Git Repository 폼
# ========================================
class GitModelForm(forms.ModelForm):
    """Git Repository 모델 폼"""
    
    class Meta:
        model = Model
        fields = ['git_url', 'git_branch', 'git_subfolder', 'name', 'task_type', 'description']
        widgets = {
            'git_url': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://github.com/username/repo.git',
                'required': True,
            }),
            'git_branch': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'main',
                'value': 'main',
            }),
            'git_subfolder': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'models/weights (선택사항)',
            }),
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '비워두면 Repository 이름에서 자동 생성',
                'required': False,
            }),
            'task_type': forms.Select(attrs={
                'class': 'form-select',
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Repository 설명 (선택사항)',
            }),
        }
        labels = {
            'git_url': 'Git Repository URL',
            'git_branch': 'Branch',
            'git_subfolder': 'Subfolder',
            'name': '모델 이름',
            'task_type': '작업 유형',
            'description': '설명',
        }
        help_texts = {
            'task_type': '모델이 수행할 작업 유형을 선택하세요.',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # name 필드를 필수가 아니도록 설정
        self.fields['name'].required = False
    
    def clean_git_url(self):
        """Git URL 형식 검증"""
        git_url = self.cleaned_data.get('git_url')
        if git_url:
            # 기본적인 Git URL 검증
            if not (git_url.endswith('.git') or 'github.com' in git_url or 'gitlab.com' in git_url):
                raise forms.ValidationError('유효한 Git Repository URL을 입력하세요.')
        return git_url
    
    def clean(self):
        """전체 폼 검증 - 이름 자동 생성"""
        cleaned_data = super().clean()
        name = cleaned_data.get('name')
        git_url = cleaned_data.get('git_url')
        
        # 이름이 비어있으면 Git URL에서 생성
        if not name and git_url:
            # URL에서 repository 이름 추출
            # https://github.com/username/repo.git -> repo
            repo_name = git_url.rstrip('/').split('/')[-1]
            cleaned_data['name'] = repo_name.replace('.git', '')
        
        return cleaned_data
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.source = 'git'
        if commit:
            instance.save()
        return instance


# ========================================
# 4. HuggingFace 폼
# ========================================
class HuggingFaceModelForm(forms.ModelForm):
    """HuggingFace Hub 모델 폼"""
    
    class Meta:
        model = Model
        fields = ['hf_model_id', 'hf_revision', 'name', 'task_type', 'description']
        widgets = {
            'hf_model_id': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'facebook/detr-resnet-50',
                'required': True,
            }),
            'hf_revision': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'main',
                'value': 'main',
            }),
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '비워두면 Model ID에서 자동 생성',
                'required': False,
            }),
            'task_type': forms.Select(attrs={
                'class': 'form-select',
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': '모델 설명 (선택사항)',
            }),
        }
        labels = {
            'hf_model_id': 'Model ID',
            'hf_revision': 'Revision',
            'name': '모델 이름',
            'task_type': '작업 유형',
            'description': '설명',
        }
        help_texts = {
            'task_type': 'HuggingFace에서 자동 추론되지만, 직접 선택할 수 있습니다.',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # name 필드를 필수가 아니도록 설정
        self.fields['name'].required = False
    
    def clean_hf_model_id(self):
        """HuggingFace Model ID 형식 검증"""
        hf_model_id = self.cleaned_data.get('hf_model_id')
        if hf_model_id:
            # 기본 형식: username/model-name
            if '/' not in hf_model_id:
                raise forms.ValidationError('Model ID는 "username/model-name" 형식이어야 합니다.')
        return hf_model_id
    
    def clean(self):
        """전체 폼 검증 - 이름 자동 생성"""
        cleaned_data = super().clean()
        name = cleaned_data.get('name')
        hf_model_id = cleaned_data.get('hf_model_id')
        
        # 이름이 비어있으면 Model ID에서 생성
        if not name and hf_model_id:
            # facebook/detr-resnet-50 -> detr-resnet-50
            cleaned_data['name'] = hf_model_id.split('/')[-1]
        
        return cleaned_data
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.source = 'huggingface'
        if commit:
            instance.save()
        return instance


# ========================================
# 5. 모델 수정 폼
# ========================================
class ModelUpdateForm(forms.ModelForm):
    """모델 수정 폼 (이름, 설명, 작업 유형 등)"""
    
    class Meta:
        model = Model
        fields = ['name', 'task_type', 'description', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'required': True,
            }),
            'task_type': forms.Select(attrs={
                'class': 'form-select',
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
            }),
            'is_active': forms.CheckboxInput(attrs={
                'class': 'form-check-input',
            }),
        }
        labels = {
            'name': '모델 이름',
            'task_type': '작업 유형',
            'description': '설명',
            'is_active': '활성화',
        }
        help_texts = {
            'task_type': '모델이 수행할 작업 유형입니다. 잘못 추론된 경우 수정할 수 있습니다.',
        }
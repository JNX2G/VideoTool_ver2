from django import forms
from .models import PreprocessingMethod
import json


class MethodForm(forms.ModelForm):
    """전처리 기법 추가/수정 폼"""
    
    # JSONField를 CharField로 오버라이드
    parameter_schema = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control font-monospace',
            'rows': 10,
            'placeholder': '''{
  "ksize": {
    "type": "int",
    "default": 5,
    "min": 1,
    "max": 31,
    "description": "커널 크기"
  }
}'''
        })
    )
    
    default_parameters = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control font-monospace',
            'rows': 5,
            'placeholder': '''{
  "ksize": 5
}'''
        })
    )
    
    class Meta:
        model = PreprocessingMethod
        fields = [
            'name', 
            'code', 
            'description', 
            'category',
            'python_file',
            'function_name',
            'parameter_schema',
            'default_parameters',
            'is_active'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '예: 가우시안 블러'
            }),
            'code': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '예: gaussian_blur'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': '이 기법에 대한 설명을 입력하세요'
            }),
            'category': forms.Select(attrs={
                'class': 'form-select'
            }),
            'python_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.py'
            }),
            'function_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'process'
            }),
            'is_active': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # ← 오타 수정!
        
        # 수정 모드일 때만 JSON을 문자열로 변환
        if self.instance.pk:
            # parameter_schema 처리
            if self.instance.parameter_schema and self.instance.parameter_schema != {}:
                self.initial['parameter_schema'] = json.dumps(
                    self.instance.parameter_schema, 
                    indent=2, 
                    ensure_ascii=False
                )
            else:
                # 빈 dict면 빈 문자열로
                self.initial['parameter_schema'] = ''
            
            # default_parameters 처리
            if self.instance.default_parameters and self.instance.default_parameters != {}:
                self.initial['default_parameters'] = json.dumps(
                    self.instance.default_parameters, 
                    indent=2, 
                    ensure_ascii=False
                )
            else:
                # 빈 dict면 빈 문자열로
                self.initial['default_parameters'] = ''
        else:
            # 새로 생성할 때는 빈 문자열
            self.initial['parameter_schema'] = ''
            self.initial['default_parameters'] = ''
    
    def clean_parameter_schema(self):
        """문자열을 JSON으로 변환"""
        data = self.cleaned_data.get('parameter_schema', '').strip()
        
        # 빈 문자열이면 빈 dict 반환
        if not data:
            return {}
        
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise forms.ValidationError(f'올바른 JSON 형식이 아닙니다: {e}')
    
    def clean_default_parameters(self):
        """문자열을 JSON으로 변환"""
        data = self.cleaned_data.get('default_parameters', '').strip()
        
        # 빈 문자열이면 빈 dict 반환
        if not data:
            return {}
        
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise forms.ValidationError(f'올바른 JSON 형식이 아닙니다: {e}')
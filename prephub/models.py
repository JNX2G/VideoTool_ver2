from django.db import models
import os


def method_upload_path(instance, filename):
    """전처리 기법 Python 파일 업로드 경로"""
    # prephub_methods/기법코드/파일명.py
    return os.path.join('prephub_methods', instance.code, filename)


class PreprocessingMethod(models.Model):
    """전처리 기법 정의"""
    
    # 기본 정보
    name = models.CharField(max_length=100, verbose_name="기법 이름")
    code = models.CharField(
        max_length=50, 
        unique=True, 
        verbose_name="기법 코드",
        help_text="프로그램에서 사용되는 고유 식별자 (예: my_blur)"
    )
    description = models.TextField(blank=True, verbose_name="설명")
    
    # 카테고리
    CATEGORY_CHOICES = [
        ('edge', '경계 검출'),
        ('blur', '블러/노이즈 제거'),
        ('threshold', '임계값 처리'),
        ('morphology', '형태학적 변환'),
        ('color', '색상 변환'),
        ('feature', '특징 검출'),
        ('enhancement', '이미지 향상'),
        ('custom', '사용자 정의'),
    ]
    category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
        default='custom',
        verbose_name="카테고리"
    )
    
    # Python 파일 업로드
    python_file = models.FileField(
        upload_to=method_upload_path,
        blank=True,
        null=True,
        verbose_name="Python 파일",
        help_text=".py 파일을 업로드하세요"
    )
    
    # 함수 이름 (파일 내 실행할 함수명)
    function_name = models.CharField(
        max_length=100,
        default='process',
        verbose_name="함수 이름",
        help_text="Python 파일 내에서 실행할 함수 이름 (기본: process)"
    )
    
    # 파라미터 스키마 (JSON)
    parameter_schema = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="파라미터 스키마",
        help_text="각 파라미터의 타입, 기본값, 범위 등을 정의"
    )
    
    # 기본 파라미터 값
    default_parameters = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="기본 파라미터"
    )
    
    # 상태
    is_active = models.BooleanField(default=True, verbose_name="활성화")
    is_builtin = models.BooleanField(
        default=False, 
        verbose_name="내장 기법",
        help_text="시스템 기본 제공 기법 (삭제 불가)"
    )
    
    # 메타데이터
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")
    
    class Meta:
        verbose_name = "전처리 기법"
        verbose_name_plural = "전처리 기법들"
        ordering = ['category', 'name']
    
    def __str__(self):
        return f"{self.name} ({self.code})"
    
    def get_parameter_display(self):
        """파라미터를 읽기 쉬운 형식으로 반환"""
        if not self.default_parameters:
            return "파라미터 없음"
        
        items = [f"{k}={v}" for k, v in self.default_parameters.items()]
        return ", ".join(items)
    
    def execute(self, frame, params=None):
        """
        전처리 기법 실행
        
        Args:
            frame: numpy array (입력 이미지) - BGR 형식
            params: dict (파라미터)
            
        Returns:
            numpy array (처리된 이미지)
        """
        import cv2
        import numpy as np
        
        # 파라미터 병합
        merged_params = {**self.default_parameters}
        if params:
            merged_params.update(params)
        
        # 내장 기법
        if self.is_builtin:
            try:
                from . import builtin_methods
                func = getattr(builtin_methods, self.code)
                return func(frame, **merged_params)
            except Exception as e:
                print(f"❌ 내장 기법 실행 오류 ({self.code}): {e}")
                import traceback
                traceback.print_exc()
                return frame
        
        # 커스텀 기법
        else:
            if not self.python_file:
                print(f"⚠️ 커스텀 기법 '{self.code}'에 Python 파일이 없습니다.")
                return frame
                
            try:
                import importlib.util
                import sys
                
                file_path = self.python_file.path
                spec = importlib.util.spec_from_file_location(
                    f"prephub_method_{self.code}", 
                    file_path
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                
                if hasattr(module, self.function_name):
                    func = getattr(module, self.function_name)
                    return func(frame, **merged_params)
                else:
                    print(f"❌ 함수 '{self.function_name}'를 찾을 수 없습니다.")
                    return frame
                    
            except Exception as e:
                print(f"❌ 커스텀 기법 실행 오류: {e}")
                import traceback
                traceback.print_exc()
                return frame



    
def execute(self, frame, params=None):
    """
    전처리 기법 실행
    
    Args:
        frame: numpy array (입력 이미지)
        params: dict (파라미터)
        
    Returns:
        numpy array (처리된 이미지)
    """
    import cv2
    import numpy as np
    
    # 파라미터 병합
    merged_params = {**self.default_parameters}
    if params:
        merged_params.update(params)
    
    # ⭐ 내장 기법
    if self.is_builtin:
        try:
            from . import builtin_methods
            # 함수명은 code와 동일 (예: gaussian_blur)
            func = getattr(builtin_methods, self.code)
            return func(frame, **merged_params)
        except AttributeError:
            print(f"내장 기법 '{self.code}' 함수를 찾을 수 없습니다.")
            return frame
        except Exception as e:
            print(f"내장 기법 실행 오류 ({self.code}): {e}")
            import traceback
            traceback.print_exc()
            return frame
    
    # ⭐ 커스텀 기법 (업로드된 파일)
    else:
        if self.python_file:
            try:
                import importlib.util
                import sys
                
                file_path = self.python_file.path
                
                # 동적으로 모듈 로드
                spec = importlib.util.spec_from_file_location(
                    f"prephub_method_{self.code}", 
                    file_path
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                
                # 함수 가져오기
                if hasattr(module, self.function_name):
                    func = getattr(module, self.function_name)
                    return func(frame, **merged_params)
                else:
                    print(f"함수 '{self.function_name}'를 찾을 수 없습니다.")
                    return frame
                    
            except Exception as e:
                print(f"커스텀 기법 실행 오류: {e}")
                import traceback
                traceback.print_exc()
                return frame
        
        # 파일이 없으면 원본 반환
        return frame

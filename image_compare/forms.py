from django import forms
from .models import ComparisonMethod, ImageComparison


class ComparisonConfigForm(forms.Form):
    """비교 설정 폼 - 단순화 버전"""
    
    # 비교 방법 (Integer로 받아서 나중에 ComparisonMethod 객체로 변환)
    comparison_method = forms.IntegerField(
        required=True,
        error_messages={'required': '비교 방법을 선택해주세요.'}
    )
    
    # 특징점 기반 파라미터
    n_features = forms.IntegerField(
        initial=1000,
        min_value=100,
        max_value=10000,
        required=False
    )
    
    match_threshold = forms.FloatField(
        initial=0.75,
        min_value=0.5,
        max_value=0.95,
        required=False
    )
    
    # SSIM 파라미터
    window_size = forms.IntegerField(
        initial=11,
        min_value=3,
        max_value=21,
        required=False
    )
    
    # 히스토그램 파라미터
    histogram_method = forms.ChoiceField(
        choices=[
            ('correlation', 'Correlation'),
            ('chi_square', 'Chi-Square'),
            ('intersection', 'Intersection'),
            ('bhattacharyya', 'Bhattacharyya'),
        ],
        initial='correlation',
        required=False
    )
    
    histogram_bins = forms.IntegerField(
        initial=256,
        min_value=16,
        max_value=256,
        required=False
    )
    
    # 픽셀 차이 파라미터
    pixel_diff_method = forms.ChoiceField(
        choices=[
            ('absolute', 'Absolute Difference'),
            ('squared', 'Squared Difference'),
        ],
        initial='absolute',
        required=False
    )
    
    pixel_threshold = forms.IntegerField(
        initial=30,
        min_value=1,
        max_value=255,
        required=False
    )
    
    # 공통 파라미터
    color_space = forms.ChoiceField(
        choices=[
            ('RGB', 'RGB'),
            ('HSV', 'HSV'),
            ('Lab', 'Lab'),
        ],
        initial='HSV',
        required=False
    )
    
    # 시각화 옵션 (체크박스는 on/off만 체크)
    viz_matches = forms.BooleanField(required=False)
    viz_heatmap = forms.BooleanField(required=False)
    viz_ssim_map = forms.BooleanField(required=False)
    viz_side_by_side = forms.BooleanField(required=False)
    
    def clean_comparison_method(self):
        """comparison_method를 ComparisonMethod 객체로 변환"""
        method_id = self.cleaned_data.get('comparison_method')
        try:
            method = ComparisonMethod.objects.get(id=method_id, is_active=True)
            return method
        except ComparisonMethod.DoesNotExist:
            raise forms.ValidationError("유효하지 않은 비교 방법입니다.")
    
    def clean_window_size(self):
        """윈도우 크기가 홀수인지 확인"""
        size = self.cleaned_data.get('window_size')
        if size and size % 2 == 0:
            raise forms.ValidationError("윈도우 크기는 홀수여야 합니다.")
        return size
    
    def get_comparison_method_object(self):
        """ComparisonMethod 객체 반환"""
        return self.cleaned_data.get('comparison_method')
    
    def get_parameters(self):
        """폼 데이터를 파라미터 딕셔너리로 변환"""
        method = self.cleaned_data.get('comparison_method')
        
        if not method:
            return {}
        
        params = {}
        
        # 방법별 파라미터 설정
        if method.category == 'feature':
            params['method'] = method.name  # ORB, SIFT 등
            params['n_features'] = self.cleaned_data.get('n_features', 1000)
            params['match_threshold'] = self.cleaned_data.get('match_threshold', 0.75)
        
        elif method.category == 'structural':
            params['window_size'] = self.cleaned_data.get('window_size', 11)
        
        elif method.category == 'histogram':
            params['method'] = self.cleaned_data.get('histogram_method', 'correlation')
            params['bins'] = self.cleaned_data.get('histogram_bins', 256)
            params['color_space'] = self.cleaned_data.get('color_space', 'HSV')
        
        elif method.category == 'pixel':
            params['method'] = self.cleaned_data.get('pixel_diff_method', 'absolute')
            params['threshold'] = self.cleaned_data.get('pixel_threshold', 30)
            params['color_space'] = self.cleaned_data.get('color_space', 'RGB')
        
        # 시각화 타입
        viz_types = []
        if self.cleaned_data.get('viz_matches'):
            viz_types.append('matches')
        if self.cleaned_data.get('viz_heatmap'):
            viz_types.append('diff_heatmap')
        if self.cleaned_data.get('viz_ssim_map'):
            viz_types.append('ssim_map')
        if self.cleaned_data.get('viz_side_by_side'):
            viz_types.append('side_by_side')
        
        params['visualization_types'] = viz_types or ['side_by_side']  # 기본값
        
        return params
import os
import json
from django.db import models
from django.conf import settings
from django.utils import timezone
from pathlib import Path

from contents.models import Image


def comparison_result_path(instance, filename):
    """비교 결과 이미지 저장 경로"""
    now = timezone.now()
    ext = os.path.splitext(filename)[1].lower()
    new_filename = f"comparison_{instance.id}_{now.strftime('%m%d%H%M%S')}{ext}"
    return os.path.join("comparisons", str(now.year), str(now.month).zfill(2), new_filename)


class ComparisonMethod(models.Model):
    """비교 방법 모델"""
    
    CATEGORY_CHOICES = [
        ('feature', '특징점 기반'),
        ('structural', '구조적 유사도'),
        ('pixel', '픽셀 차이'),
        ('histogram', '히스토그램'),
        ('deep', '딥러닝'),
        ('optical', '광학 흐름'),
        ('phase', '위상 상관'),
        ('color', '색상 기반'),
    ]
    
    name = models.CharField(max_length=50, unique=True, verbose_name="방법 이름")
    display_name = models.CharField(max_length=100, verbose_name="표시 이름")
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, verbose_name="카테고리")
    description = models.TextField(blank=True, verbose_name="설명")
    
    # 기본 파라미터 (JSON)
    default_params = models.JSONField(
        default=dict,
        verbose_name="기본 파라미터",
        help_text="예: {'n_features': 1000, 'threshold': 0.75}"
    )
    
    # 파라미터 스키마 (UI 생성용)
    param_schema = models.JSONField(
        default=dict,
        verbose_name="파라미터 스키마",
        help_text="UI에서 파라미터 입력 폼을 생성하기 위한 스키마"
    )
    
    is_active = models.BooleanField(default=True, verbose_name="활성화")
    order = models.IntegerField(default=0, verbose_name="정렬 순서")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "비교 방법"
        verbose_name_plural = "비교 방법들"
        ordering = ['order', 'name']
    
    def __str__(self):
        return f"{self.display_name} ({self.name})"


class ImageComparison(models.Model):
    """이미지 비교 모델 - 고급 버전"""
    
    # 비교할 두 이미지
    image_1 = models.ForeignKey(
        Image, 
        on_delete=models.CASCADE, 
        related_name='comparisons_as_first',
        verbose_name="첫 번째 이미지"
    )
    image_2 = models.ForeignKey(
        Image, 
        on_delete=models.CASCADE, 
        related_name='comparisons_as_second',
        verbose_name="두 번째 이미지"
    )
    
    # 비교 방법
    comparison_method = models.ForeignKey(
        ComparisonMethod,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name="비교 방법"
    )
    
    # 비교 메타 정보
    title = models.CharField(max_length=200, blank=True, verbose_name="비교 제목")
    description = models.TextField(blank=True, verbose_name="비교 설명")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="비교 생성 시간")
    
    # 비교 파라미터 (사용자 지정)
    parameters = models.JSONField(
        default=dict,
        verbose_name="비교 파라미터",
        help_text="예: {'n_features': 1000, 'threshold': 0.75, 'visualization_types': ['matches', 'heatmap']}"
    )
    
    # 비교 결과 이미지들 (여러 시각화)
    result_images = models.JSONField(
        default=list,
        verbose_name="결과 이미지 목록",
        help_text="예: [{'type': 'matches', 'path': '...'}, {'type': 'heatmap', 'path': '...'}]"
    )
    
    # 유사도 점수들
    similarity_scores = models.JSONField(
        default=dict,
        verbose_name="유사도 점수",
        help_text="예: {'overall': 0.85, 'feature_match': 0.90, 'color_similarity': 0.80}"
    )
    
    # 피처 비교 결과 (상세 데이터)
    feature_comparison_data = models.JSONField(
        blank=True,
        null=True,
        verbose_name="피처 비교 데이터"
    )
    
    # 처리 시간
    processing_time = models.FloatField(default=0.0, verbose_name="처리 시간(초)")
    
    # 비교 상태
    STATUS_CHOICES = [
        ('pending', '대기 중'),
        ('processing', '처리 중'),
        ('completed', '완료'),
        ('failed', '실패'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name="비교 상태"
    )
    
    # 에러 메시지
    error_message = models.TextField(blank=True, verbose_name="에러 메시지")
    
    class Meta:
        verbose_name = "이미지 비교"
        verbose_name_plural = "이미지 비교들"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['comparison_method']),
        ]
    
    def __str__(self):
        if self.title:
            return self.title
        method_name = self.comparison_method.display_name if self.comparison_method else "기본"
        return f"{self.image_1.title} vs {self.image_2.title} ({method_name})"
    
    def save(self, *args, **kwargs):
        if not self.title:
            method_name = self.comparison_method.display_name if self.comparison_method else "비교"
            self.title = f"{self.image_1.title} vs {self.image_2.title} - {method_name}"
        super().save(*args, **kwargs)
    
    def get_overall_similarity(self):
        """전체 유사도 점수 반환"""
        return self.similarity_scores.get('overall', 0.0)
    
    def get_similarity_percentage(self):
        """유사도를 퍼센트로 반환"""
        return round(self.get_overall_similarity() * 100, 2)
    
    def get_difference_percentage(self):
        """차이도를 퍼센트로 반환"""
        return round((1.0 - self.get_overall_similarity()) * 100, 2)
    
    def get_result_image_by_type(self, viz_type):
        """특정 타입의 결과 이미지 경로 반환"""
        for img in self.result_images:
            if img.get('type') == viz_type:
                return img.get('path')
        return None
    
    def get_all_visualization_types(self):
        """모든 시각화 타입 리스트 반환"""
        return [img.get('type') for img in self.result_images]


class ComparisonFeatureExtraction(models.Model):
    """이미지별 피처 추출 결과 저장 (캐싱용)"""
    
    image = models.ForeignKey(
        Image,
        on_delete=models.CASCADE,
        related_name='feature_extractions',
        verbose_name="이미지"
    )
    
    # 사용된 방법
    method = models.ForeignKey(
        ComparisonMethod,
        on_delete=models.CASCADE,
        verbose_name="추출 방법"
    )
    
    # 추출 파라미터
    extraction_params = models.JSONField(
        default=dict,
        verbose_name="추출 파라미터"
    )
    
    # 추출된 피처 (JSON 형태)
    features_data = models.JSONField(verbose_name="피처 데이터")
    
    # 메타 정보
    extracted_at = models.DateTimeField(auto_now_add=True, verbose_name="추출 시간")
    processing_time = models.FloatField(default=0.0, verbose_name="처리 시간(초)")
    
    class Meta:
        verbose_name = "피처 추출 결과"
        verbose_name_plural = "피처 추출 결과들"
        ordering = ['-extracted_at']
        indexes = [
            models.Index(fields=['image', 'method']),
        ]
    
    def __str__(self):
        return f"{self.image.title} - {self.method.display_name}"
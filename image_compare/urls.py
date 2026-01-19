from django.urls import path
from . import views

app_name = 'image_compare'

urlpatterns = [
    # 비교 이력 목록
    path('', views.ComparisonListView.as_view(), name='comparison_list'),
    
    # 두 번째 이미지 선택
    path('select/<int:first_image_id>/', 
         views.SelectSecondImageView.as_view(), 
         name='select_second_image'),
    
    # 비교 설정 (새로 추가)
    path('config/<int:first_image_id>/<int:second_image_id>/', 
         views.ComparisonConfigView.as_view(), 
         name='comparison_config'),
    
    # 비교 실행 (빠른 실행 - 기본 설정으로 리다이렉트)
    path('compare/<int:first_image_id>/<int:second_image_id>/', 
         views.CompareImagesView.as_view(), 
         name='compare_images'),
    
    # 비교 결과 상세
    path('result/<int:pk>/', 
         views.ComparisonResultView.as_view(), 
         name='comparison_result'),
    
    # 비교 삭제
    path('delete/<int:pk>/', 
         views.ComparisonDeleteView.as_view(), 
         name='comparison_delete'),
    
    # 피처 추출 (AJAX)
    path('extract-features/<int:image_id>/', 
         views.FeatureExtractionView.as_view(), 
         name='extract_features'),

     # bulk 삭제
     path('bulk-delete/', 
          views.BulkDeleteView.as_view(), 
          name='bulk_delete'),
]
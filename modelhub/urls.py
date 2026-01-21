from django.urls import path
from . import views

app_name = 'modelhub'

urlpatterns = [
    # ========================================
    # 목록
    # ========================================
    path('', views.model_list, name='model_list'),
    
    # ========================================
    # 추가 (통합 페이지)
    # ========================================
    path('add/', views.model_add, name='model_add'),
    
    # ========================================
    # 추가 처리 (소스별 POST)
    # ========================================
    path('add/builtin/', views.model_add_builtin, name='model_add_builtin'),
    path('add/upload/', views.model_add_upload, name='model_add_upload'),
    path('add/git/', views.model_add_git, name='model_add_git'),
    path('add/huggingface/', views.model_add_huggingface, name='model_add_huggingface'),
    
    # ========================================
    # 상세/수정/삭제
    # ========================================
    path('<int:model_id>/', views.model_detail, name='model_detail'),
    path('<int:model_id>/update/', views.model_update, name='model_update'),
    path('<int:model_id>/delete/', views.model_delete, name='model_delete'),
    
    # ========================================
    # 활성화 토글 (AJAX)
    # ========================================
    path('<int:model_id>/toggle/', views.model_toggle, name='model_toggle'),
    
    # ========================================
    # 모델 검증
    # ========================================
    path('<int:model_id>/validate/', views.model_validate, name='model_validate'),
    
]

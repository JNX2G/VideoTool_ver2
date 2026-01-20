from django.urls import path
from . import views

app_name = "modelhub"

urlpatterns = [
    # 모델 목록
    path("", views.model_list, name="model_list"),
    
    # 모델 추가
    path("add/", views.model_add, name="model_add"),
    path("builtin/add/", views.builtin_model_add, name="builtin_model_add"),
    path("builtin/add/preset/", views.builtin_model_add_preset, name="builtin_model_add_preset"),
    path("custom/add/", views.custom_model_add, name="custom_model_add"),
    
    # 통합 상세/삭제/토글 (model_type으로 구분)
    path("<str:model_type>/<int:model_id>/", views.model_detail, name="model_detail"),
    path("<str:model_type>/<int:model_id>/delete/", views.model_delete, name="model_delete"),
    path("<str:model_type>/<int:model_id>/toggle/", views.model_toggle, name="model_toggle"), 
    
    # 커스텀 모델 전용
    path("custom/<int:model_id>/validate/", views.custom_model_validate, name="custom_model_validate"),
    
    # API
    path("api/models/", views.api_all_models, name="api_all_models"),
]
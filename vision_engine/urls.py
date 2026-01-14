from django.urls import path
from . import views

app_name = "vision_engine"

urlpatterns = [
    # 탐지 목록
    path("", views.application_list, name="application_list"),
    
    # 모델 선택 - task_id 사용
    path("select/<int:task_id>/", views.select_model, name="select_model"),
    
    # 탐지 실행
    path("<int:application_id>/execute/", views.execute_application, name="execute_application"),
    
    # 진행 상황
    path("<int:application_id>/progress/", views.application_progress, name="application_progress"),
    
    # 상태 API
    path("<int:application_id>/status/", views.application_status, name="application_status"),
    
    # 결과
    path("<int:application_id>/result/", views.application_result, name="application_result"),
    
    # 삭제
    path("<int:application_id>/delete/", views.application_delete, name="application_delete"),
    
    # 취소 
    path("<int:application_id>/cancel/", views.cancel_application, name="cancel_application"),
    
    # 결과 파일 제공
    path("<int:application_id>/stream/", views.serve_applied_video, name="serve_applied_video"),
    path("<int:application_id>/image/", views.serve_applied_image, name="serve_applied_image"),
]
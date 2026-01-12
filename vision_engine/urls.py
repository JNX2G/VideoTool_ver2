from django.urls import path
from . import views

app_name = "vision_engine"

urlpatterns = [
    # 탐지 목록
    path("", views.detection_list, name="detection_list"),
    
    # 모델 선택 - task_id 사용
    path("select/<int:task_id>/", views.select_model, name="select_model"),
    
    # 탐지 실행
    path("<int:detection_id>/execute/", views.execute_detection, name="execute_detection"),
    
    # 진행 상황
    path("<int:detection_id>/progress/", views.detection_progress, name="detection_progress"),
    
    # 상태 API
    path("<int:detection_id>/status/", views.detection_status, name="detection_status"),
    
    # 결과
    path("<int:detection_id>/result/", views.detection_result, name="detection_result"),
    
    # 삭제
    path("<int:detection_id>/delete/", views.detection_delete, name="detection_delete"),
    
    # 취소 
    path("<int:detection_id>/cancel/", views.cancel_detection, name="cancel_detection"),
    
    # 결과 파일 제공
    path("<int:detection_id>/stream/", views.serve_detected_video, name="serve_detected_video"),
    path("<int:detection_id>/image/", views.serve_detected_image, name="serve_detected_image"),
]
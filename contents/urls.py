from django.urls import path

from . import views

urlpatterns = [
    # 통합 컨텐츠 목록
    path("", views.ContentListView.as_view(), name="content_list"),
    # 통합 업로드
    path("upload/", views.ContentUploadView.as_view(), name="upload_content"),
    
    # 동영상 관련
    path("video/<int:pk>/", views.VideoDetailView.as_view(), name="video_detail"),
    path("video/<int:pk>/delete/", views.VideoDeleteView.as_view(), name="video_delete"),
    path("video/<int:pk>/stream/", views.VideoStreamView.as_view(), name="serve_video"),

    # 이미지 관련
    path("image/<int:pk>/", views.ImageDetailView.as_view(), name="image_detail"),
    path("image/<int:pk>/delete/", views.ImageDeleteView.as_view(), name="image_delete"),
]

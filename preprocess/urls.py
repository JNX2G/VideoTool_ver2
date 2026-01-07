from django.urls import path
from . import views

app_name = 'preprocess'

urlpatterns = [
    # 전처리 시작 (파이프라인 구성)
    path(
        "start/<int:content_id>/",
        views.StartPreprocessingView.as_view(),
        name="start_preprocessing",
    ),
    
    # 전처리 파이프라인 관리
    path(
        "<int:task_id>/add-step/",
        views.AddPreprocessingStepView.as_view(),
        name="add_step",
    ),
    path(
        "<int:task_id>/remove-step/",
        views.RemovePreprocessingStepView.as_view(),
        name="remove_step",
    ),
    path(
        "<int:task_id>/reorder-step/",
        views.ReorderPreprocessingStepView.as_view(),
        name="reorder_step",
    ),
    path(
        "<int:task_id>/clear-pipeline/",
        views.ClearPipelineView.as_view(),
        name="clear_pipeline",
    ),
    
    # 전처리 실행
    path(
        "<int:task_id>/execute/",
        views.ExecutePreprocessingView.as_view(),
        name="execute_preprocessing",
    ),
    
    # 진행 상황
    path(
        "<int:task_id>/progress/",
        views.PreprocessingProgressView.as_view(),
        name="preprocessing_progress",
    ),
    path(
        "<int:task_id>/status/",
        views.PreprocessingStatusView.as_view(),
        name="preprocessing_status",
    ),
    
    # 결과
    path(
        "<int:task_id>/result/",
        views.PreprocessingResultView.as_view(),
        name="preprocessing_result",
    ),
    
    # 결과 파일 제공
    path(
        "<int:task_id>/stream/",
        views.ServePreprocessedVideoView.as_view(),
        name="serve_preprocessed_video",
    ),
    path(
        "<int:task_id>/image/",
        views.ServePreprocessedImageView.as_view(),
        name="serve_preprocessed_image",
    ),
    
    # 전처리 작업 삭제
    path(
        "<int:task_id>/delete/",
        views.PreprocessingDeleteView.as_view(),
        name="preprocessing_delete",
    ),

    # 전처리 작업 편집
    path(
        "<int:task_id>/update-step/",
        views.UpdatePreprocessingStepView.as_view(),
        name="update_step",
    ),

]


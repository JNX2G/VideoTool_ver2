from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, Http404, HttpResponse, StreamingHttpResponse, FileResponse
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
from preprocess.models import PreprocessingTask
from modelhub.models import BuiltinModel, CustomModel
from .models import Application
from wsgiref.util import FileWrapper
import threading
import os
import re
import mimetypes


def select_model(request, task_id):
    """모델 선택 페이지"""
    task = get_object_or_404(PreprocessingTask, id=task_id)

    # 전처리가 완료되지 않았으면 리다이렉트
    if task.status != "completed":
        messages.error(request, "전처리가 완료되지 않았습니다.")
        return redirect("preprocess:preprocessing_result", task_id=task_id)

    # 활성화된 모델 목록
    builtin_models = BuiltinModel.objects.filter(is_active=True)
    custom_models = CustomModel.objects.filter(is_active=True)

    if request.method == "POST":
        model_type = request.POST.get("model_type")  # 'base' or 'custom'
        model_id = request.POST.get("model_id")
        title = request.POST.get("title", "")
        description = request.POST.get("description", "")

        # Application 생성
        application = Application.objects.create(
            preprocessing_task=task,
            title=title or f"모델 적용 - {timezone.now().strftime('%Y%m%d_%H%M%S')}",
            description=description,
            status="ready",
        )

        # 모델 할당
        if model_type == "base":
            model = get_object_or_404(BuiltinModel, id=model_id)
            application.builtin_model = model
        else:
            model = get_object_or_404(CustomModel, id=model_id)
            application.custom_model = model

        application.save()

        return redirect("vision_engine:execute_application", application_id=application.id)

    context = {
        "task": task,
        "content": task.get_content(),
        "content_type": task.get_content_type(),
        "builtin_models": builtin_models,
        "custom_models": custom_models,
    }
    return render(request, "vision_engine/select_model.html", context)


def execute_application(request, application_id):
    """모델 실행 페이지"""
    application = get_object_or_404(Application, id=application_id)

    if request.method == "POST":
        # 상태를 즉시 processing으로 변경
        application.status = "processing"
        application.save(update_fields=['status'])
        
        # 백그라운드로 탐지 실행
        from .tasks import process_application

        thread = threading.Thread(target=process_application, args=(application_id,))
        thread.daemon = True
        thread.start()

        return redirect("vision_engine:application_progress", application_id=application.id)

    context = {
        "application": application,
        "task": application.preprocessing_task,
        "content": application.get_content(),
        "content_type": application.get_content_type(),
    }
    return render(request, "vision_engine/execute_application.html", context)


def application_progress(request, application_id):
    """모델 적용 진행 상황 페이지"""
    application = get_object_or_404(Application, id=application_id)

    context = {
        "application": application,
        "task": application.preprocessing_task,
        "content": application.get_content(),
        "content_type": application.get_content_type(),
    }
    return render(request, "vision_engine/application_progress.html", context)


def application_status(request, application_id):
    """탐지 상태 API (AJAX)"""
    application = get_object_or_404(Application, id=application_id)

    return JsonResponse(
        {
            "status": application.status,
            "progress": application.progress,
            "processed_frames": application.processed_frames,
            "total_frames": application.total_frames,
            "error_message": application.error_message,
        }
    )


def application_result(request, application_id):
    """탐지 결과 페이지"""
    application = get_object_or_404(Application, id=application_id)
    task = application.preprocessing_task
    
    # summary_stats 계산
    summary_stats = []
    if application.application_summary:
        for label, count in application.application_summary.items():
            summary_stats.append({
                'label': label,
                'count': count
            })
        # 카운트 기준 내림차순 정렬
        summary_stats.sort(key=lambda x: x['count'], reverse=True)

    context = {
        "application": application,
        "task": task,
        "content": application.get_content(),
        "content_type": application.get_content_type(),
        "summary_stats": summary_stats,
        "output_url": f"/vision_engine/{application_id}/stream/",  # ⭐ output_url 추가
    }
    return render(request, "vision_engine/application_result.html", context)


def application_list(request):
    """전체 탐지 목록"""
    applications = Application.objects.all().order_by("-created_at")

    # 상태별 필터
    status = request.GET.get("status")
    if status:
        applications = applications.filter(status=status)

    context = {
        "applications": applications,
    }
    return render(request, "vision_engine/application_list.html", context)


def application_delete(request, application_id):
    """탐지 삭제"""
    application = get_object_or_404(Application, id=application_id)

    if request.method == "POST":
        task_id = application.preprocessing_task.id
        content = application.get_content()
        content_type = application.get_content_type()
        
        application.delete()

        messages.success(request, "탐지 작업이 삭제되었습니다.")

        # redirect 파라미터에 따라 분기
        redirect_to = request.POST.get("redirect", "application_list")
        
        if redirect_to == "preprocessing_result":
            return redirect("preprocess:preprocessing_result", task_id=task_id)
        
        elif redirect_to == "video_detail" and content_type == "video" and content:
            return redirect("video_detail", pk=content.id)  # 동영상 상세로
        
        elif redirect_to == "image_detail" and content_type == "image" and content:
            return redirect("image_detail", pk=content.id)  # 이미지 상세로
        
        # 기본값: 탐지 목록으로
        return redirect("vision_engine:application_list")

    context = {
        "application": application,
        "task": application.preprocessing_task,
        "content": application.get_content(),
        "content_type": application.get_content_type(),
    }
    return render(request, "vision_engine/application_delete.html", context)

def cancel_application(request, application_id):
    """탐지 작업 취소"""
    application = get_object_or_404(Application, id=application_id)
    task_id = application.preprocessing_task.id

    if request.method == "POST":
        if application.status == "processing":
            # 상태를 cancelled로 변경
            application.status = "cancelled"
            application.save(update_fields=['status'])
            
            messages.success(request, "탐지 작업이 취소되었습니다.")
            
            # ⭐ 전처리 결과 페이지로 리다이렉트
            return redirect("preprocess:preprocessing_result", task_id=task_id)
        else:
            messages.warning(request, "처리 중인 작업만 취소할 수 있습니다.")
            return redirect("vision_engine:application_progress", application_id=application_id)

    # GET 요청 시에도 전처리 결과로 리다이렉트
    return redirect("preprocess:preprocessing_result", task_id=task_id)


# ============================================
# 탐지 결과 파일 제공
# ============================================

def serve_applied_video(request, application_id):
    """탐지 결과 동영상 스트리밍"""
    application = get_object_or_404(Application, id=application_id)

    if not application.output_file_path:
        raise Http404("처리된 동영상 파일이 없습니다.")

    # results 디렉토리에서 파일 찾기
    video_path = os.path.join(settings.RESULTS_ROOT, application.output_file_path)

    if not os.path.exists(video_path):
        raise Http404("동영상 파일을 찾을 수 없습니다.")

    file_size = os.path.getsize(video_path)
    content_type, _ = mimetypes.guess_type(video_path)
    content_type = content_type or "video/mp4"

    # Range 요청 처리
    range_header = request.META.get("HTTP_RANGE", "").strip()
    range_match = re.match(r"bytes\s*=\s*(\d+)\s*-\s*(\d*)", range_header, re.I)

    if range_match:
        first_byte, last_byte = range_match.groups()
        first_byte = int(first_byte) if first_byte else 0
        last_byte = int(last_byte) if last_byte else file_size - 1

        if last_byte >= file_size:
            last_byte = file_size - 1

        length = last_byte - first_byte + 1

        with open(video_path, "rb") as file:
            file.seek(first_byte)
            data = file.read(length)

        response = HttpResponse(data, status=206, content_type=content_type)
        response["Content-Length"] = str(length)
        response["Content-Range"] = f"bytes {first_byte}-{last_byte}/{file_size}"
        response["Accept-Ranges"] = "bytes"
        return response

    # 전체 파일 스트리밍
    response = StreamingHttpResponse(
        FileWrapper(open(video_path, "rb"), 8192), content_type=content_type
    )
    response["Content-Length"] = str(file_size)
    response["Accept-Ranges"] = "bytes"
    return response


def serve_applied_image(request, application_id):
    """탐지 결과 이미지 제공"""
    application = get_object_or_404(Application, id=application_id)

    if not application.output_file_path:
        raise Http404("처리된 이미지 파일이 없습니다.")

    # results 디렉토리에서 파일 찾기
    image_path = os.path.join(settings.RESULTS_ROOT, application.output_file_path)

    if not os.path.exists(image_path):
        raise Http404("이미지 파일을 찾을 수 없습니다.")

    content_type, _ = mimetypes.guess_type(image_path)
    content_type = content_type or "image/jpeg"

    return FileResponse(open(image_path, "rb"), content_type=content_type)


# vision_engine/views.py에 추가할 코드

def application_list(request):
    """전체 탐지 목록 (정렬 기능 포함)"""
    
    # 기본 쿼리셋
    applications = Application.objects.all()

    # 상태별 필터
    status = request.GET.get("status")
    if status:
        applications = applications.filter(status=status)

    # 정렬 처리
    sort = request.GET.get("sort", "-created_at")  # 기본값: 최신순
    
    # 허용된 정렬 필드만 사용 (보안)
    allowed_sort_fields = [
        'id', '-id',
        'status', '-status',
        'total_applications', '-total_applications',
        'progress', '-progress',
        'created_at', '-created_at',
    ]
    
    if sort in allowed_sort_fields:
        applications = applications.order_by(sort)
    else:
        # 유효하지 않은 정렬 필드면 기본값 사용
        applications = applications.order_by("-created_at")
        sort = "-created_at"

    context = {
        "applications": applications,
        "current_sort": sort,  # 현재 정렬 상태 전달
    }
    return render(request, "vision_engine/application_list.html", context)
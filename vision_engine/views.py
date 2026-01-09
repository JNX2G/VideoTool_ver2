from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, Http404, HttpResponse, StreamingHttpResponse, FileResponse
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
from preprocess.models import PreprocessingTask
from modelhub.models import BaseModel, CustomModel
from .models import Detection
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
    base_models = BaseModel.objects.filter(is_active=True)
    custom_models = CustomModel.objects.filter(is_active=True)

    if request.method == "POST":
        model_type = request.POST.get("model_type")  # 'base' or 'custom'
        model_id = request.POST.get("model_id")
        title = request.POST.get("title", "")
        description = request.POST.get("description", "")

        # Detection 생성
        detection = Detection.objects.create(
            preprocessing_task=task,
            title=title or f"객체 탐지 - {timezone.now().strftime('%Y%m%d_%H%M%S')}",
            description=description,
            status="ready",
        )

        # 모델 할당
        if model_type == "base":
            model = get_object_or_404(BaseModel, id=model_id)
            detection.base_model = model
        else:
            model = get_object_or_404(CustomModel, id=model_id)
            detection.custom_model = model

        detection.save()

        return redirect("vision_engine:execute_detection", detection_id=detection.id)

    context = {
        "task": task,
        "content": task.get_content(),
        "content_type": task.get_content_type(),
        "base_models": base_models,
        "custom_models": custom_models,
    }
    return render(request, "vision_engine/select_model.html", context)


def execute_detection(request, detection_id):
    """탐지 실행 페이지"""
    detection = get_object_or_404(Detection, id=detection_id)

    if request.method == "POST":
        # 상태를 즉시 processing으로 변경
        detection.status = "processing"
        detection.save(update_fields=['status'])
        
        # 백그라운드로 탐지 실행
        from .tasks import process_detection

        thread = threading.Thread(target=process_detection, args=(detection_id,))
        thread.daemon = True
        thread.start()

        return redirect("vision_engine:detection_progress", detection_id=detection.id)

    context = {
        "detection": detection,
        "task": detection.preprocessing_task,
        "content": detection.get_content(),
        "content_type": detection.get_content_type(),
    }
    return render(request, "vision_engine/execute_detection.html", context)


def detection_progress(request, detection_id):
    """탐지 진행 상황 페이지"""
    detection = get_object_or_404(Detection, id=detection_id)

    context = {
        "detection": detection,
        "task": detection.preprocessing_task,
        "content": detection.get_content(),
        "content_type": detection.get_content_type(),
    }
    return render(request, "vision_engine/detection_progress.html", context)


def detection_status(request, detection_id):
    """탐지 상태 API (AJAX)"""
    detection = get_object_or_404(Detection, id=detection_id)

    return JsonResponse(
        {
            "status": detection.status,
            "progress": detection.progress,
            "processed_frames": detection.processed_frames,
            "total_frames": detection.total_frames,
            "error_message": detection.error_message,
        }
    )


def detection_result(request, detection_id):
    """탐지 결과 페이지"""
    detection = get_object_or_404(Detection, id=detection_id)
    task = detection.preprocessing_task
    
    # summary_stats 계산
    summary_stats = []
    if detection.detection_summary:
        for label, count in detection.detection_summary.items():
            summary_stats.append({
                'label': label,
                'count': count
            })
        # 카운트 기준 내림차순 정렬
        summary_stats.sort(key=lambda x: x['count'], reverse=True)

    context = {
        "detection": detection,
        "task": task,
        "content": detection.get_content(),
        "content_type": detection.get_content_type(),
        "summary_stats": summary_stats,
        "output_url": f"/vision_engine/{detection_id}/stream/",  # ⭐ output_url 추가
    }
    return render(request, "vision_engine/detection_result.html", context)


def detection_list(request):
    """전체 탐지 목록"""
    detections = Detection.objects.all().order_by("-created_at")

    # 상태별 필터
    status = request.GET.get("status")
    if status:
        detections = detections.filter(status=status)

    context = {
        "detections": detections,
    }
    return render(request, "vision_engine/detection_list.html", context)


def detection_delete(request, detection_id):
    """탐지 삭제"""
    detection = get_object_or_404(Detection, id=detection_id)

    if request.method == "POST":
        task_id = detection.preprocessing_task.id
        detection.delete()

        messages.success(request, "탐지 작업이 삭제되었습니다.")

        redirect_to = request.POST.get("redirect", "detection_list")
        if redirect_to == "preprocessing_result":
            return redirect("preprocess:preprocessing_result", task_id=task_id)

        return redirect("vision_engine:detection_list")

    context = {
        "detection": detection,
        "task": detection.preprocessing_task,
        "content": detection.get_content(),
        "content_type": detection.get_content_type(),
    }
    return render(request, "vision_engine/detection_delete.html", context)


# 탐지 결과 파일 제공 뷰 추가
def serve_detected_video(request, detection_id):
    """탐지 결과 동영상 스트리밍"""
    detection = get_object_or_404(Detection, id=detection_id)

    if not detection.output_file_path:
        raise Http404("처리된 동영상 파일이 없습니다.")

    # results 디렉토리에서 파일 찾기
    video_path = os.path.join(settings.RESULTS_ROOT, detection.output_file_path)

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


def serve_detected_image(request, detection_id):
    """탐지 결과 이미지 제공"""
    detection = get_object_or_404(Detection, id=detection_id)

    if not detection.output_file_path:
        raise Http404("처리된 이미지 파일이 없습니다.")

    # results 디렉토리에서 파일 찾기
    image_path = os.path.join(settings.RESULTS_ROOT, detection.output_file_path)

    if not os.path.exists(image_path):
        raise Http404("이미지 파일을 찾을 수 없습니다.")

    content_type, _ = mimetypes.guess_type(image_path)
    content_type = content_type or "image/jpeg"

    return FileResponse(open(image_path, "rb"), content_type=content_type)
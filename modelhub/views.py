from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from .models import BaseModel, CustomModel
from .forms import BaseModelForm, CustomModelForm
import os


# ============================================
# 통합 모델 목록
# ============================================


def model_list(request):
    """전체 모델 목록 (통합 - 탭으로 구분)"""
    base_models = BaseModel.objects.filter(is_active=True).order_by(
        "-is_default", "-created_at"
    )
    custom_models = CustomModel.objects.filter(is_active=True).order_by("-created_at")

    total_base = base_models.count()
    total_custom = custom_models.count()
    total_models = total_base + total_custom

    context = {
        "base_models": base_models,
        "custom_models": custom_models,
        "total_models": total_models,
        "total_base": total_base,
        "total_custom": total_custom,
    }
    return render(request, "modelhub/model_list.html", context)


def model_add(request):
    """모델 추가 선택 페이지"""
    return render(request, "modelhub/model_add.html")


# ============================================
# 기본 모델 추가
# ============================================


def base_model_add(request):
    """기본 모델 추가 (YOLO 등)"""
    if request.method == "POST":
        form = BaseModelForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save(commit=False)
            if model.model_file:
                model.file_size = model.model_file.size
            model.save()
            messages.success(
                request, f'기본 모델 "{model.display_name}"이 추가되었습니다.'
            )
            return redirect("modelhub:model_list")
    else:
        form = BaseModelForm()

    # YOLO 사전 정의 모델 목록
    yolo_presets = [
        {
            "name": "yolov8n",
            "display_name": "YOLOv8 Nano",
            "version": "yolov8n.pt",
            "description": "가장 빠른 모델, 실시간 처리에 적합 (정확도: 낮음)",
        },
        {
            "name": "yolov8s",
            "display_name": "YOLOv8 Small",
            "version": "yolov8s.pt",
            "description": "속도와 정확도의 균형 (권장)",
        },
        {
            "name": "yolov8m",
            "display_name": "YOLOv8 Medium",
            "version": "yolov8m.pt",
            "description": "높은 정확도, 중간 속도",
        },
        {
            "name": "yolov8l",
            "display_name": "YOLOv8 Large",
            "version": "yolov8l.pt",
            "description": "매우 높은 정확도 (속도: 느림)",
        },
        {
            "name": "yolov8x",
            "display_name": "YOLOv8 XLarge",
            "version": "yolov8x.pt",
            "description": "최고 정확도, 가장 느림",
        },
    ]

    context = {
        "form": form,
        "yolo_presets": yolo_presets,
    }
    return render(request, "modelhub/base_model_add.html", context)


def base_model_add_preset(request):
    """YOLO 사전 정의 모델 빠른 추가"""
    if request.method == "POST":
        yolo_version = request.POST.get("yolo_version")
        display_name = request.POST.get("display_name")
        description = request.POST.get("description", "")

        model = BaseModel.objects.create(
            name=yolo_version.replace(".pt", ""),
            display_name=display_name,
            description=description,
            model_type="yolo",
            yolo_version=yolo_version,
            is_active=True,
        )

        messages.success(request, f'YOLO 모델 "{model.display_name}"이 추가되었습니다.')
        return redirect("modelhub:model_list")

    return redirect("modelhub:base_model_add")


# ============================================
# 커스텀 모델 추가
# ============================================


def custom_model_add(request):
    """커스텀 모델 업로드"""
    if request.method == "POST":
        form = CustomModelForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save(commit=False)

            if model.model_file:
                model.file_size = model.model_file.size
                model.file_format = os.path.splitext(model.model_file.name)[1]

            model.save()
            messages.success(request, f'커스텀 모델 "{model.name}"이 업로드되었습니다.')
            return redirect("modelhub:model_list")
    else:
        form = CustomModelForm()

    context = {
        "form": form,
    }
    return render(request, "modelhub/custom_model_add.html", context)


# ============================================
# 통합 상세/수정/삭제 (model_type으로 구분)
# ============================================


def model_detail(request, model_type, model_id):
    """통합 모델 상세 (모델 정보 + 탐지 결과 목록)"""
    if model_type == "base":
        model = get_object_or_404(BaseModel, id=model_id)
        # 이 모델을 사용한 탐지 목록
        from vision_engine.models import Detection

        detections = Detection.objects.filter(base_model=model).order_by("-created_at")
    else:
        model = get_object_or_404(CustomModel, id=model_id)
        # 이 모델을 사용한 탐지 목록
        from vision_engine.models import Detection

        detections = Detection.objects.filter(custom_model=model).order_by(
            "-created_at"
        )

    # 통계 계산
    completed_count = detections.filter(status="completed").count()

    context = {
        "model": model,
        "model_type": model_type,
        "detections": detections,
        "completed_count": completed_count,  # 완료된 탐지 수
    }
    return render(request, "modelhub/model_detail.html", context)


def model_edit(request, model_type, model_id):
    """통합 모델 수정"""
    if model_type == "base":
        model = get_object_or_404(BaseModel, id=model_id)
        if request.method == "POST":
            form = BaseModelForm(request.POST, request.FILES, instance=model)
            if form.is_valid():
                form.save()
                messages.success(request, "모델 정보가 수정되었습니다.")
                return redirect("modelhub:model_detail", model_type, model.id)
        else:
            form = BaseModelForm(instance=model)
    else:
        model = get_object_or_404(CustomModel, id=model_id)
        if request.method == "POST":
            model.name = request.POST.get("name", model.name)
            model.description = request.POST.get("description", model.description)
            model.save()
            messages.success(request, "모델 정보가 수정되었습니다.")
            return redirect("modelhub:model_detail", model_type, model.id)
        form = None

    context = {
        "model": model,
        "model_type": model_type,
        "form": form,
    }
    return render(request, "modelhub/model_edit.html", context)


def model_delete(request, model_type, model_id):
    """통합 모델 삭제"""
    if model_type == "base":
        model = get_object_or_404(BaseModel, id=model_id)
    else:
        model = get_object_or_404(CustomModel, id=model_id)

    if request.method == "POST":
        model_name = model.display_name if model_type == "base" else model.name

        # 파일 삭제
        try:
            if model_type == "base" and model.model_file:
                if os.path.exists(model.model_file.path):
                    os.remove(model.model_file.path)
            elif model_type == "custom" and model.model_file:
                if os.path.exists(model.model_file.path):
                    os.remove(model.model_file.path)
        except Exception as e:
            print(f"파일 삭제 실패: {e}")

        model.delete()
        messages.success(request, f'모델 "{model_name}"이 삭제되었습니다.')
        return redirect("modelhub:model_list")

    context = {
        "model": model,
        "model_type": model_type,
    }
    return render(request, "modelhub/model_delete.html", context)


def model_toggle(request, model_type, model_id):
    """모델 활성화/비활성화 토글"""
    if model_type == "base":
        model = get_object_or_404(BaseModel, id=model_id)
    else:
        model = get_object_or_404(CustomModel, id=model_id)

    model.is_active = not model.is_active
    model.save()

    status = "활성화" if model.is_active else "비활성화"
    messages.success(request, f"모델이 {status}되었습니다.")
    return redirect("modelhub:model_detail", model_type, model_id)


# ============================================
# 커스텀 모델 전용 기능
# ============================================


def custom_model_validate(request, model_id):
    """커스텀 모델 검증"""
    model = get_object_or_404(CustomModel, id=model_id)

    is_valid, message = model.validate_model()

    if is_valid:
        messages.success(request, f"모델 검증 성공: {message}")
    else:
        messages.error(request, f"모델 검증 실패: {message}")

    return redirect("modelhub:model_detail", "custom", model_id)


# ============================================
# API 엔드포인트
# ============================================


def api_all_models(request):
    """전체 모델 목록 API"""
    base_models = BaseModel.objects.filter(is_active=True).values(
        "id", "name", "display_name", "yolo_version"
    )
    custom_models = CustomModel.objects.filter(is_active=True).values(
        "id", "name", "model_type"
    )

    return JsonResponse(
        {
            "success": True,
            "base_models": list(base_models),
            "custom_models": list(custom_models),
        }
    )
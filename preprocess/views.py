import json
import mimetypes
import os
import re

from django.conf import settings
from django.contrib import messages
from django.http import (
    FileResponse,
    Http404,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import DeleteView
from wsgiref.util import FileWrapper

from contents.models import Image, Video
from prephub.models import PreprocessingMethod
from .models import PreprocessingTask


class StartPreprocessingView(View):
    """전처리 시작 (파이프라인 구성 화면)"""
    template_name = "preprocess/preprocessing.html"

    def _get_content(self, content_type, content_id):
        """컨텐츠 가져오기"""
        if content_type == "image":
            return get_object_or_404(Image, pk=content_id)
        return get_object_or_404(Video, pk=content_id)

    def _get_ready_task(self, content, content_type):
        """준비 상태의 전처리 작업 가져오기 또는 생성"""
        if content_type == "image":
            task = PreprocessingTask.objects.filter(
                image=content, status="ready"
            ).first()
            if not task:
                task = PreprocessingTask.objects.create(
                    image=content, status="ready"
                )
            return task

        task = PreprocessingTask.objects.filter(
            video=content, status="ready"
        ).first()
        if not task:
            task = PreprocessingTask.objects.create(
                video=content, status="ready"
            )
        return task

    def _skip_preprocessing(self, request, content, content_type):
        """전처리 건너뛰기 (원본 파일 그대로 사용)"""
        if content_type == "image":
            task = PreprocessingTask.objects.create(
                image=content,
                preprocessing_pipeline=[],
                status="completed",
            )
            task.output_file_path = content.file.name
        else:
            task = PreprocessingTask.objects.create(
                video=content,
                preprocessing_pipeline=[],
                status="completed",
            )
            task.output_file_path = content.file.name

        task.total_frames = 0
        task.processed_frames = 0
        task.completed_at = timezone.now()
        task.save()

        messages.success(
            request,
            "전처리를 건너뛰었습니다. 이후 모델을 적용할 수 있습니다.",
        )

        if content_type == "image":
            return redirect("image_detail", pk=content.pk)
        return redirect("preprocessing_result", task_id=task.id)

    def get(self, request, content_id):
        content_type = request.GET.get("type", "video")
        content = self._get_content(content_type, content_id)
        task = self._get_ready_task(content, content_type)

        # ⭐ prephub에서 활성화된 전처리 기법 가져오기
        active_methods = PreprocessingMethod.objects.filter(
            is_active=True
        ).order_by("category", "name")

        # 카테고리별로 그룹화
        methods_by_category = {}
        for method in active_methods:
            category = method.get_category_display()
            if category not in methods_by_category:
                methods_by_category[category] = []
            methods_by_category[category].append(method)

        context = {
            "content": content,
            "content_type": content_type,
            "task": task,
            "methods_by_category": methods_by_category,
            "current_pipeline": task.get_pipeline_display(),
        }
        return render(request, self.template_name, context)

    def post(self, request, content_id):
        content_type = request.GET.get("type", "video")
        content = self._get_content(content_type, content_id)
        skip_preprocessing = request.POST.get("skip_preprocessing") == "true"

        if skip_preprocessing:
            return self._skip_preprocessing(request, content, content_type)

        # 전처리를 건너뛰지 않는 경우 기존 화면 그대로 렌더
        return self.get(request, content_id)


class AddPreprocessingStepView(View):
    """전처리 단계 추가"""
    
    def post(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, id=task_id)
        data = json.loads(request.body or "{}")
        method_id = data.get("method_id")
        params = data.get("params", {})

        # method_id 유효성 검사
        try:
            method = PreprocessingMethod.objects.get(id=method_id, is_active=True)
        except PreprocessingMethod.DoesNotExist:
            return JsonResponse({
                "success": False,
                "error": "전처리 기법을 찾을 수 없거나 비활성화되었습니다."
            }, status=400)

        task.add_preprocessing_step(method_id, params)

        return JsonResponse({
            "success": True,
            "pipeline": task.get_pipeline_display(),
            "pipeline_full": task.preprocessing_pipeline,
        })

    def get(self, request, task_id):
        return JsonResponse({
            "success": False, 
            "error": "Invalid request"
        }, status=405)


class RemovePreprocessingStepView(View):
    """전처리 단계 제거 (특정 인덱스 지원)"""

    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def post(self, request, task_id):
        try:
            task = PreprocessingTask.objects.get(id=task_id)
            data = json.loads(request.body)
            index = data.get("index")

            if index is not None:
                success = task.remove_preprocessing_step(index)
                if not success:
                    return JsonResponse({
                        "success": False, 
                        "error": "잘못된 인덱스입니다."
                    })
            else:
                if task.preprocessing_pipeline:
                    task.remove_preprocessing_step(len(task.preprocessing_pipeline) - 1)

            return JsonResponse({
                "success": True, 
                "pipeline": task.get_pipeline_display()
            })

        except PreprocessingTask.DoesNotExist:
            return JsonResponse({
                "success": False, 
                "error": "전처리 작업을 찾을 수 없습니다."
            })
        except Exception as e:
            return JsonResponse({
                "success": False, 
                "error": str(e)
            })


class ReorderPreprocessingStepView(View):
    """전처리 단계 순서 변경"""

    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def post(self, request, task_id):
        try:
            task = PreprocessingTask.objects.get(id=task_id)
            data = json.loads(request.body)
            from_index = data.get("from_index")
            to_index = data.get("to_index")

            if from_index is None or to_index is None:
                return JsonResponse({
                    "success": False, 
                    "error": "from_index와 to_index가 필요합니다."
                })

            success = task.reorder_preprocessing_step(from_index, to_index)
            if not success:
                return JsonResponse({
                    "success": False, 
                    "error": "잘못된 인덱스입니다."
                })

            return JsonResponse({
                "success": True, 
                "pipeline": task.get_pipeline_display()
            })

        except PreprocessingTask.DoesNotExist:
            return JsonResponse({
                "success": False, 
                "error": "전처리 작업을 찾을 수 없습니다."
            })
        except Exception as e:
            return JsonResponse({
                "success": False, 
                "error": str(e)
            })


class ClearPipelineView(View):
    """전처리 파이프라인 전체 초기화"""

    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def post(self, request, task_id):
        try:
            task = PreprocessingTask.objects.get(id=task_id)
            task.clear_preprocessing_pipeline()

            return JsonResponse({
                "success": True, 
                "pipeline": []
            })

        except PreprocessingTask.DoesNotExist:
            return JsonResponse({
                "success": False, 
                "error": "전처리 작업을 찾을 수 없습니다."
            })
        except Exception as e:
            return JsonResponse({
                "success": False, 
                "error": str(e)
            })


class ExecutePreprocessingView(View):
    """전처리 실행"""
    
    def post(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, id=task_id)
        content = task.get_content()

        if not content:
            messages.error(request, "컨텐츠를 찾을 수 없습니다.")
            return redirect("content_list")

        if task.status == "processing":
            messages.warning(request, "이미 처리 중입니다.")
            return redirect("preprocessing_progress", task_id=task_id)

        from .tasks import start_preprocessing_task

        start_preprocessing_task(task_id)
        messages.success(request, "전처리를 시작했습니다.")
        return redirect("preprocessing_progress", task_id=task_id)

    def get(self, request, task_id):
        return redirect("preprocessing_progress", task_id=task_id)


class PreprocessingProgressView(View):
    """전처리 진행 상황"""
    template_name = "preprocess/progress.html"

    def get(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, id=task_id)
        content = task.get_content()

        if not content:
            messages.error(request, "컨텐츠를 찾을 수 없습니다.")
            return redirect("content_list")

        context = {
            "task": task,
            "content": content,
            "content_type": task.get_content_type(),
        }
        return render(request, self.template_name, context)


class PreprocessingStatusView(View):
    """전처리 상태 조회 (AJAX)"""
    
    def get(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, id=task_id)
        return JsonResponse({
            "status": task.status,
            "status_display": task.get_status_display(),
            "status_badge": task.get_status_display_badge(),
            "progress": task.progress,
            "current_step": task.current_step,
            "processed_frames": task.processed_frames,
            "total_frames": task.total_frames,
            "error_message": task.error_message,
        })


class PreprocessingResultView(View):
    """전처리 결과"""
    template_name = "preprocess/result.html"

    def get(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, id=task_id)
        content = task.get_content()
        content_type = task.get_content_type()

        if not content:
            messages.error(request, "컨텐츠를 찾을 수 없습니다.")
            return redirect("content_list")

        context = {
            "task": task,
            "content": content,
            "content_type": content_type,
        }
        return render(request, self.template_name, context)


class PreprocessingDeleteView(DeleteView):
    """전처리 작업 삭제"""
    model = PreprocessingTask
    template_name = "preprocess/task_delete.html"
    context_object_name = "task"
    pk_url_kwarg = "task_id"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['content'] = self.object.get_content()
        context['content_type'] = self.object.get_content_type()
        return context

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        content = self.object.get_content()
        content_type = self.object.get_content_type()

        # 결과 파일 삭제
        self.object.delete_files()
        self.object.delete()

        messages.success(request, "전처리 작업이 삭제되었습니다.")

        # 리다이렉트
        redirect_to = request.POST.get("redirect", "content_detail")
        if redirect_to == "video_detail" and content_type == "video" and content:
            return redirect("video_detail", pk=content.id)
        if redirect_to == "image_detail" and content_type == "image" and content:
            return redirect("image_detail", pk=content.id)

        return redirect("content_list")


class ServePreprocessedVideoView(View):
    """전처리 결과 동영상 스트리밍"""

    def get(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, id=task_id)

        if not task.output_file_path:
            raise Http404("처리된 동영상 파일이 없습니다.")

        # results 디렉토리에서 파일 찾기
        video_path = os.path.join(
            settings.RESULTS_ROOT, task.output_file_path
        )

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
            FileWrapper(open(video_path, "rb"), 8192),
            content_type=content_type,
        )
        response["Content-Length"] = str(file_size)
        response["Accept-Ranges"] = "bytes"
        return response


class ServePreprocessedImageView(View):
    """전처리 결과 이미지 제공"""

    def get(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, id=task_id)

        if not task.output_file_path:
            raise Http404("처리된 이미지 파일이 없습니다.")

        # results 디렉토리에서 파일 찾기
        image_path = os.path.join(
            settings.RESULTS_ROOT, task.output_file_path
        )

        if not os.path.exists(image_path):
            raise Http404("이미지 파일을 찾을 수 없습니다.")

        content_type, _ = mimetypes.guess_type(image_path)
        content_type = content_type or "image/jpeg"

        return FileResponse(open(image_path, "rb"), content_type=content_type)

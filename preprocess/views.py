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

    def _get_or_create_task(self, content, content_type, task_id=None):
        """
        task_id가 제공되면 해당 task 반환 (편집 모드)
        task_id가 없으면 ready 상태의 task를 가져오거나 새로 생성
        """
        # task_id가 제공된 경우 - 특정 task 편집
        if task_id:
            if content_type == "image":
                task = get_object_or_404(
                    PreprocessingTask, 
                    id=task_id, 
                    image=content
                )
            else:
                task = get_object_or_404(
                    PreprocessingTask, 
                    id=task_id, 
                    video=content
                )
            
            # 편집 가능한 상태로 변경 (completed, failed, cancelled, pending -> ready)
            if task.status in ['completed', 'failed', 'cancelled', 'pending']:
                task.status = 'ready'
                task.progress = 0
                task.processed_frames = 0
                task.current_step = None
                task.error_message = ''
                task.started_at = None
                task.completed_at = None
                task.save()
            
            return task
        
        # task_id가 없는 경우 - ready 상태 task 찾기 또는 생성
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
        else:
            task = PreprocessingTask.objects.create(
                video=content,
                preprocessing_pipeline=[],
                status="completed",
            )
        
        # 원본 파일 사용을 나타내는 특별한 플래그 추가
        task.output_file_path = f"__original__:{content.file.name}"
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
        return redirect("video_detail", pk=content.pk)

    def get(self, request, content_id, task_id=None):
        """
        task_id 파라미터 추가
        - task_id가 있으면: 해당 task 편집
        - task_id가 없으면: ready 상태 task 찾기 또는 생성
        """
        content_type = request.GET.get("type", "video")
        content = self._get_content(content_type, content_id)
        
        # new=true 파라미터가 있으면 기존 ready 작업 정리 후 리다이렉트
        force_new = request.GET.get("new") == "true"
        
        if force_new and not task_id:
            # 기존 ready 상태의 작업들을 cancelled로 변경
            if content_type == "image":
                PreprocessingTask.objects.filter(
                    image=content, status="ready"
                ).update(status="cancelled")
            else:
                PreprocessingTask.objects.filter(
                    video=content, status="ready"
                ).update(status="cancelled")
            
            # new 파라미터 제거하고 리다이렉트
            return redirect(f'/preprocess/start/{content_id}/?type={content_type}')
        
        # task 가져오기 또는 생성 (task_id 파라미터 전달)
        task = self._get_or_create_task(content, content_type, task_id)

        # prephub에서 활성화된 전처리 기법 가져오기
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
            "current_pipeline": task.get_pipeline_display() if task else [],
        }
        return render(request, self.template_name, context)

    def post(self, request, content_id, task_id=None):
        """
        task_id 파라미터 추가
        """
        content_type = request.GET.get("type", "video")
        content = self._get_content(content_type, content_id)
        skip_preprocessing = request.POST.get("skip_preprocessing") == "true"

        if skip_preprocessing:
            return self._skip_preprocessing(request, content, content_type)

        # 전처리를 건너뛰지 않는 경우 기존 화면 그대로 렌더
        return self.get(request, content_id, task_id)
    
    
class CreateTaskAndAddStepView(View):
    """새 전처리 작업 생성 및 첫 단계 추가"""
    
    def post(self, request):
        data = json.loads(request.body or "{}")
        content_id = data.get("content_id")
        content_type = data.get("content_type", "video")
        method_id = data.get("method_id")
        params = data.get("params", {})
        
        if not content_id:
            return JsonResponse({
                "success": False,
                "error": "content_id가 필요합니다."
            }, status=400)
        
        # 컨텐츠 가져오기
        if content_type == "image":
            content = get_object_or_404(Image, pk=content_id)
            task = PreprocessingTask.objects.create(
                image=content, status="ready"
            )
        else:
            content = get_object_or_404(Video, pk=content_id)
            task = PreprocessingTask.objects.create(
                video=content, status="ready"
            )
        
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
            "task_id": task.id,
            "pipeline": task.get_pipeline_display(),
            "pipeline_full": task.preprocessing_pipeline,
        })


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
            "task_id": task.id,
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
                return redirect("preprocess:preprocessing_progress", task_id=task_id)

            # 실제 실행 버튼을 눌렀을 때 기존 데이터 청소
            # 1. 연결된 탐지 결과(Application) 삭제
            app_count = task.applications.count()
            if app_count > 0:
                task.applications.all().delete()
                # 사용자에게 알림 (선택 사항)
                # messages.info(request, f"기존 탐지 결과 {app_count}개가 삭제되었습니다.")

            # 2. 기존 전처리 결과 파일 삭제 (재실행 시 덮어쓰기 방지)
            if task.output_file_path and not task.output_file_path.startswith("__original__"):
                task.delete_files()
                task.output_file_path = ""
                task.save()

            # 3. 전처리 상태 초기화 및 시작
            from .tasks import start_preprocessing_task
            start_preprocessing_task(task_id)
            
            messages.success(request, "전처리를 시작했습니다.")
            return redirect("preprocess:preprocessing_progress", task_id=task_id)

    def get(self, request, task_id):
        return redirect("preprocess:preprocessing_progress", task_id=task_id)


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
    template_name = "preprocess/preprocess_result.html"

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

        # 원본 파일 사용 플래그 확인
        if task.output_file_path.startswith("__original__:"):
            # 원본 파일 경로 추출
            original_file_path = task.output_file_path.replace("__original__:", "")
            video_path = os.path.join(settings.MEDIA_ROOT, original_file_path)
        else:
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

        # 원본 파일 사용 플래그 확인
        if task.output_file_path.startswith("__original__:"):
            # 원본 파일 경로 추출
            original_file_path = task.output_file_path.replace("__original__:", "")
            image_path = os.path.join(settings.MEDIA_ROOT, original_file_path)
        else:
            # results 디렉토리에서 파일 찾기
            image_path = os.path.join(
                settings.RESULTS_ROOT, task.output_file_path
            )

        if not os.path.exists(image_path):
            raise Http404("이미지 파일을 찾을 수 없습니다.")

        content_type, _ = mimetypes.guess_type(image_path)
        content_type = content_type or "image/jpeg"

        return FileResponse(open(image_path, "rb"), content_type=content_type)


class UpdatePreprocessingStepView(View):
    """전처리 단계의 파라미터 수정"""
    def post(self, request, task_id):
        try:
            task = get_object_or_404(PreprocessingTask, id=task_id)
            data = json.loads(request.body)
            index = data.get("index")
            params = data.get("params", {})

            if index is None or not (0 <= index < len(task.preprocessing_pipeline)):
                return JsonResponse({"success": False, "error": "잘못된 인덱스입니다."}, status=400)

            # 해당 인덱스의 파라미터만 교체
            task.preprocessing_pipeline[index]["params"] = params
            task.save()

            return JsonResponse({
                "success": True,
                "pipeline": task.get_pipeline_display(),
                "pipeline_full": task.preprocessing_pipeline # JS 갱신용 데이터
            })
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)
        
class RestartTaskView(View):
    """전처리 작업 재시작"""
    
    def post(self, request, task_id):
        task = get_object_or_404(PreprocessingTask, pk=task_id)
        
        # 완료된 작업은 재시작 불가
        if task.status == 'completed':
            messages.error(request, '완료된 작업은 재시작할 수 없습니다.')
            return redirect('preprocess:preprocessing_result', task_id=task.id)
        
        # 상태 초기화
        task.status = 'pending'
        task.progress = 0
        task.processed_frames = 0
        task.current_step = None
        task.error_message = None
        task.started_at = None
        task.completed_at = None
        task.save()
        
        # 백그라운드 작업 실행
        from .tasks import process_preprocessing_task
        import threading
        thread = threading.Thread(target=process_preprocessing_task, args=(task.id,))
        thread.daemon = True
        thread.start()
        
        messages.success(request, f'전처리 작업 #{task.id}를 재시작했습니다.')
        
        # AJAX 요청이면 JSON 응답
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': '작업이 재시작되었습니다.',
                'redirect_url': f'/preprocess/{task.id}/progress/'
            })
        
        # 일반 요청이면 진행 상황 페이지로 리다이렉트
        return redirect('preprocess:preprocessing_progress', task_id=task.id)



class CancelTaskView(View):
    """전처리 작업 취소"""
    
    def post(self, request, task_id):
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        task = get_object_or_404(PreprocessingTask, pk=task_id)
        
        # 완료된 작업은 취소 불가
        if task.status == 'completed':
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'message': '완료된 작업은 취소할 수 없습니다.'
                }, status=400)
            messages.error(request, '완료된 작업은 취소할 수 없습니다.')
            return redirect('preprocess:preprocessing_result', task_id=task.id)
        
        content = task.get_content()
        content_type = task.get_content_type()
        content_id = content.id if content else None
        
        # 1단계: 'cancelled'로 변경 (백그라운드 스레드 중단 신호)
        task.status = 'cancelled'
        task.save(update_fields=['status'])
        
        logger.info(f"✋ 작업 취소 신호 전송: task_id={task_id}")
        
        # 2단계: 백그라운드 스레드가 중단될 때까지 대기 (최대 3초)
        max_wait = 3.0
        wait_interval = 0.1
        waited = 0.0
        
        while waited < max_wait:
            time.sleep(wait_interval)
            waited += wait_interval
            task.refresh_from_db()
            
            # 스레드가 cancelled를 확인하고 중단했는지 확인
            # (진행률이 변경되지 않으면 중단된 것으로 간주)
            if task.status == 'cancelled':
                logger.info(f"⏳ 대기 중... ({waited:.1f}초)")
                # 한 번 더 확인
                time.sleep(0.3)
                task.refresh_from_db()
                
                # 여전히 cancelled 상태면 중단 성공
                if task.status == 'cancelled':
                    break
            else:
                # 상태가 변경되었으면 (예: completed) 취소 실패
                logger.warning(f"⚠️ 작업이 이미 완료됨: task_id={task_id}, status={task.status}")
                break
        
        # 3단계: 최종 상태 확인
        task.refresh_from_db()
        
        if task.status == 'completed':
            # 취소 실패 - 이미 완료됨
            logger.warning(f"❌ 취소 실패 (작업 완료): task_id={task_id}")
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'message': '작업이 이미 완료되어 취소할 수 없습니다.',
                    'completed': True,
                    'redirect_url': f'/preprocess/{task_id}/result/'
                })
            
            messages.info(request, '작업이 이미 완료되었습니다.')
            return redirect('preprocess:preprocessing_result', task_id=task.id)
        
        # 4단계: 'ready' 상태로 변경 (편집 가능)
        task.status = 'ready'
        task.progress = 0
        task.processed_frames = 0
        task.total_frames = 0
        task.current_step = None
        task.error_message = ''
        task.started_at = None
        task.completed_at = None
        # 중요: output_file_path는 유지하지 않음 (삭제 예정)
        task.save()
        
        logger.info(f"✅ 작업 취소 완료: task_id={task_id}, 상태='ready'로 변경")
        
        # 5단계: 임시 출력 파일 삭제
        try:
            if task.output_file_path:
                import os
                from django.conf import settings
                
                # 원본 파일 사용 플래그 확인 - 원본 파일은 삭제하지 않음
                if task.output_file_path.startswith("__original__:"):
                    # 원본 파일은 건드리지 않음
                    pass
                else:
                    # 전처리 결과 파일만 삭제
                    path = os.path.join(settings.RESULTS_ROOT, task.output_file_path)
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"🗑️ 임시 출력 파일 삭제: {path}")
                
                task.output_file_path = ""
                task.save(update_fields=['output_file_path'])
        except Exception as e:
            logger.warning(f"파일 삭제 실패: {e}")
        
        # AJAX 응답
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # preprocessing.html로 리다이렉트 (편집 가능) - task_id 포함
            return JsonResponse({
                'success': True,
                'message': '작업이 취소되었습니다. 파이프라인을 수정하고 다시 시작할 수 있습니다.',
                'cancelled': True,
                'redirect_url': f'/preprocess/start/{content_id}/?type={content_type}&task_id={task_id}'
            })
        
        messages.warning(request, '작업이 취소되었습니다. 파이프라인을 수정하고 다시 시작할 수 있습니다.')
        
        # preprocessing.html로 리다이렉트 (편집 가능) - task_id 포함
        return redirect(f'/preprocess/start/{content_id}/?type={content_type}&task_id={task_id}')
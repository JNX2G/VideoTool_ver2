from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.db.models import Q
from .models import Model
from .forms import (
    BuiltinModelForm,
    FileUploadModelForm,
    GitModelForm,
    HuggingFaceModelForm,
    ModelUpdateForm,
)
import os


# ========================================
# 1. 목록
# ========================================
def model_list(request):
    """모델 목록 (탭별 필터링)"""
    tab = request.GET.get('tab', 'all')
    search = request.GET.get('search', '')
    
    # 기본 쿼리셋
    models = Model.objects.all()
    
    # 탭별 필터
    if tab != 'all':
        models = models.filter(source=tab)
    
    # 검색
    if search:
        models = models.filter(
            Q(name__icontains=search) |
            Q(description__icontains=search) |
            Q(architecture__icontains=search)
        )
    
    # 소스별 개수
    total_models = Model.objects.count()
    total_builtin = Model.objects.filter(source='builtin').count()
    total_upload = Model.objects.filter(source='upload').count()
    total_git = Model.objects.filter(source='git').count()
    total_huggingface = Model.objects.filter(source='huggingface').count()
    active_count = Model.objects.filter(is_active=True).count()
    
    context = {
        'models': models,
        'tab': tab,
        'search': search,
        'total_models': total_models,
        'total_builtin': total_builtin,
        'total_upload': total_upload,
        'total_git': total_git,
        'total_huggingface': total_huggingface,
        'active_count': active_count,
    }
    return render(request, 'modelhub/model_list.html', context)


# ========================================
# 2. 추가 (통합 페이지)
# ========================================
def model_add(request):
    """모델 추가 - 통합 탭 페이지 (GET만)"""
    return render(request, 'modelhub/model_add.html')


# ========================================
# 3. Built-in 모델 추가 (POST)
# ========================================
def model_add_builtin(request):
    """Built-in 모델 추가 처리"""
    if request.method == 'POST':
        form = BuiltinModelForm(request.POST)
        if form.is_valid():
            model = form.save()
            messages.success(request, f'✅ Built-in 모델 "{model.name}"이 추가되었습니다.')
            return redirect('modelhub:model_list')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    
    return redirect('modelhub:model_add')


# ========================================
# 4. 파일 업로드 (POST)
# ========================================
def model_add_upload(request):
    """파일 업로드 모델 추가 처리"""
    if request.method == 'POST':
        if 'model_file' not in request.FILES:
            messages.error(request, '❌ 파일이 선택되지 않았습니다.')
            return redirect('modelhub:model_add')
        
        form = FileUploadModelForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save()
            messages.success(
                request,
                f'✅ 모델 "{model.name}"이 업로드되었습니다. 메타데이터를 추출 중입니다...'
            )
            
            # 메타데이터 추출 트리거
            from .tasks import extract_and_update_metadata
            try:
                extract_and_update_metadata(model.id)
            except Exception as e:
                messages.warning(request, f'메타데이터 추출 실패: {e}')
            
            return redirect('modelhub:model_detail', model_id=model.id)
        else:
            messages.error(request, '❌ 폼 검증 실패:')
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    
    return redirect('modelhub:model_add')


# ========================================
# 5. Git Repository (POST)
# ========================================
def model_add_git(request):
    """Git Repository 모델 추가 처리"""
    if request.method == 'POST':
        form = GitModelForm(request.POST)
        if form.is_valid():
            model = form.save()
            messages.success(
                request,
                f'✅ Git 모델 "{model.name}"을 클론 중입니다. 잠시 후 확인하세요.'
            )
            
            # Git 클론 트리거
            from .tasks import download_git_model
            try:
                download_git_model(model.id)
            except Exception as e:
                messages.error(request, f'Git 클론 실패: {e}')
            
            return redirect('modelhub:model_detail', model_id=model.id)
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    
    return redirect('modelhub:model_add')


# ========================================
# 6. HuggingFace (POST)
# ========================================
def model_add_huggingface(request):
    """HuggingFace 모델 추가 처리"""
    if request.method == 'POST':
        form = HuggingFaceModelForm(request.POST)
        if form.is_valid():
            model = form.save()
            messages.success(
                request,
                f'✅ HuggingFace 모델 "{model.name}"을 가져오는 중입니다.'
            )
            
            # HuggingFace 메타데이터 추출 트리거
            from .tasks import download_huggingface_model
            try:
                download_huggingface_model(model.id)
            except Exception as e:
                messages.warning(request, f'메타데이터 가져오기 실패: {e}')
            
            return redirect('modelhub:model_detail', model_id=model.id)
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    
    return redirect('modelhub:model_add')


# ========================================
# 7. 상세
# ========================================
def model_detail(request, model_id):
    """모델 상세 + 사용 기록"""
    model = get_object_or_404(Model, id=model_id)
    
    # 이 모델을 사용한 Application 목록
    applications = []
    try:
        from vision_engine.models import Application
        applications = Application.objects.filter(model=model).order_by('-created_at')[:10]
    except:
        pass  # vision_engine 앱이 없을 수도 있음
    
    context = {
        'model': model,
        'applications': applications,
    }
    return render(request, 'modelhub/model_detail.html', context)


# ========================================
# 8. 수정
# ========================================
def model_update(request, model_id):
    """모델 수정 (이름, 설명, 활성화 상태)"""
    model = get_object_or_404(Model, id=model_id)
    
    if request.method == 'POST':
        form = ModelUpdateForm(request.POST, instance=model)
        if form.is_valid():
            form.save()
            messages.success(request, f'✅ 모델 "{model.name}"이 수정되었습니다.')
            return redirect('modelhub:model_detail', model_id=model.id)
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = ModelUpdateForm(instance=model)
    
    context = {
        'model': model,
        'form': form,
    }
    return render(request, 'modelhub/model_update.html', context)


# ========================================
# 9. 삭제
# ========================================
def model_delete(request, model_id):
    """모델 삭제"""
    model = get_object_or_404(Model, id=model_id)
    
    if request.method == 'POST':
        model_name = model.name
        
        # 연관된 Application이 있는지 확인
        try:
            from vision_engine.models import Application
            app_count = Application.objects.filter(model=model).count()
            if app_count > 0:
                messages.warning(
                    request,
                    f'⚠️ 이 모델을 사용하는 {app_count}개의 작업이 있습니다. '
                    f'삭제하면 해당 작업들이 영향을 받을 수 있습니다.'
                )
        except:
            pass
        
        # 삭제
        model.delete()
        messages.success(request, f'✅ 모델 "{model_name}"이 삭제되었습니다.')
        return redirect('modelhub:model_list')
    
    # 사용 통계
    usage_info = {
        'applications': 0,
    }
    try:
        from vision_engine.models import Application
        usage_info['applications'] = Application.objects.filter(model=model).count()
    except:
        pass
    
    context = {
        'model': model,
        'usage_info': usage_info,
    }
    return render(request, 'modelhub/model_delete.html', context)


# ========================================
# 10. 활성화 토글 (AJAX)
# ========================================
def model_toggle(request, model_id):
    """모델 활성화/비활성화 토글"""
    if request.method == 'POST':
        model = get_object_or_404(Model, id=model_id)
        model.is_active = not model.is_active
        model.save(update_fields=['is_active'])
        
        active_count = Model.objects.filter(is_active=True).count()
        
        return JsonResponse({
            'success': True,
            'is_active': model.is_active,
            'active_count': active_count,
        })
    
    return JsonResponse({'success': False, 'error': 'POST 요청만 허용됩니다.'}, status=400)


# ========================================
# 11. 검증
# ========================================
def model_validate(request, model_id):
    """모델 파일 검증 (로딩 테스트)"""
    model = get_object_or_404(Model, id=model_id)
    
    validation_passed = False
    error_message = None
    
    try:
        # Built-in 모델 검증
        if model.source == 'builtin':
            try:
                from ultralytics import YOLO
                # 실제 모델 로딩 테스트
                m = YOLO(model.builtin_preset)
                validation_passed = True
                messages.success(
                    request, 
                    f'✅ Built-in 모델 "{model.name}"이 정상적으로 로딩됩니다.'
                )
            except ImportError:
                error_message = 'Ultralytics 패키지가 설치되지 않았습니다. (pip install ultralytics)'
            except Exception as e:
                error_message = f'모델 로딩 실패: {str(e)}'
        
        # Upload 모델 검증
        elif model.source == 'upload' and model.model_file:
            file_path = model.model_file.path
            
            if not os.path.exists(file_path):
                error_message = '모델 파일이 존재하지 않습니다.'
            else:
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # PyTorch 파일
                if file_ext in ['.pt', '.pth']:
                    try:
                        import torch
                        checkpoint = torch.load(file_path, map_location='cpu')
                        validation_passed = True
                        messages.success(
                            request,
                            f'✅ PyTorch 모델 "{model.name}"이 정상적으로 로딩됩니다.'
                        )
                    except ImportError:
                        error_message = 'PyTorch가 설치되지 않았습니다. (pip install torch)'
                    except Exception as e:
                        error_message = f'PyTorch 모델 로딩 실패: {str(e)}'
                
                # ONNX 파일
                elif file_ext == '.onnx':
                    try:
                        import onnx
                        onnx_model = onnx.load(file_path)
                        onnx.checker.check_model(onnx_model)
                        validation_passed = True
                        messages.success(
                            request,
                            f'✅ ONNX 모델 "{model.name}"이 유효합니다.'
                        )
                    except ImportError:
                        error_message = 'ONNX가 설치되지 않았습니다. (pip install onnx)'
                    except Exception as e:
                        error_message = f'ONNX 검증 실패: {str(e)}'
                
                # TensorFlow 파일
                elif file_ext in ['.h5', '.pb']:
                    try:
                        import tensorflow as tf
                        if file_ext == '.h5':
                            tf.keras.models.load_model(file_path, compile=False)
                        validation_passed = True
                        messages.success(
                            request,
                            f'✅ TensorFlow 모델 "{model.name}"이 정상적으로 로딩됩니다.'
                        )
                    except ImportError:
                        error_message = 'TensorFlow가 설치되지 않았습니다. (pip install tensorflow)'
                    except Exception as e:
                        error_message = f'TensorFlow 모델 로딩 실패: {str(e)}'
                
                else:
                    error_message = f'지원하지 않는 파일 형식입니다: {file_ext}'
        
        # Git 모델 검증
        elif model.source == 'git':
            git_path = model.get_model_path()
            if git_path and os.path.exists(git_path):
                validation_passed = True
                messages.success(
                    request,
                    f'✅ Git 모델 "{model.name}" 디렉토리가 존재합니다.'
                )
            else:
                error_message = 'Git 클론 디렉토리를 찾을 수 없습니다.'
        
        # HuggingFace 모델 검증
        elif model.source == 'huggingface':
            try:
                from huggingface_hub import model_info
                info = model_info(model.hf_model_id)
                validation_passed = True
                messages.success(
                    request,
                    f'✅ HuggingFace 모델 "{model.name}"을 찾았습니다.'
                )
            except ImportError:
                error_message = 'huggingface-hub가 설치되지 않았습니다. (pip install huggingface-hub)'
            except Exception as e:
                error_message = f'HuggingFace 모델 확인 실패: {str(e)}'
        
        else:
            error_message = '검증할 모델 파일이 없습니다.'
        
        # 에러 메시지 표시
        if error_message:
            messages.error(request, f'❌ {error_message}')
    
    except Exception as e:
        messages.error(request, f'❌ 검증 중 오류 발생: {str(e)}')
    
    return redirect('modelhub:model_detail', model_id=model.id)
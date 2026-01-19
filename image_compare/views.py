from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, View
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse

from contents.models import Image
from .models import ImageComparison, ComparisonFeatureExtraction, ComparisonMethod
from .forms import ComparisonConfigForm
from .utils import compare_images_comprehensive


class SelectSecondImageView(ListView):
    """두 번째 이미지 선택 페이지"""
    model = Image
    template_name = 'image_compare/select_second_image.html'
    context_object_name = 'images'
    paginate_by = 12
    
    def get_queryset(self):
        """첫 번째 이미지를 제외한 이미지 목록"""
        first_image_id = self.kwargs.get('first_image_id')
        queryset = Image.objects.exclude(id=first_image_id).order_by('-uploaded_at')
        
        # 검색 기능
        search = self.request.GET.get('search', '')
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) | Q(description__icontains=search)
            )
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        first_image_id = self.kwargs.get('first_image_id')
        context['first_image'] = get_object_or_404(Image, id=first_image_id)
        context['search'] = self.request.GET.get('search', '')
        return context


class ComparisonConfigView(View):
    """비교 설정 페이지"""
    
    def get(self, request, first_image_id, second_image_id):
        """비교 설정 폼 표시"""
        first_image = get_object_or_404(Image, id=first_image_id)
        second_image = get_object_or_404(Image, id=second_image_id)
        
        # 같은 이미지 비교 방지
        if first_image.id == second_image.id:
            messages.error(request, "같은 이미지는 비교할 수 없습니다.")
            return redirect('image_detail', pk=first_image_id)
        
        form = ComparisonConfigForm()
        
        context = {
            'first_image': first_image,
            'second_image': second_image,
            'form': form,
            'methods': ComparisonMethod.objects.filter(is_active=True)
        }
        
        # 디버그 모드 (URL에 ?debug=1 추가 시)
        if request.GET.get('debug') == '1':
            return render(request, 'image_compare/comparison_config_test.html', context)
        
        return render(request, 'image_compare/comparison_config.html', context)
    
    def post(self, request, first_image_id, second_image_id):
        """비교 실행"""
        first_image = get_object_or_404(Image, id=first_image_id)
        second_image = get_object_or_404(Image, id=second_image_id)
        
        # 디버깅: POST 데이터 확인
        print("=" * 50)
        print("🚀 비교 POST 요청 받음!")
        print(f"Image 1 ID: {first_image_id}")
        print(f"Image 2 ID: {second_image_id}")
        print(f"POST 데이터: {dict(request.POST)}")
        print("=" * 50)
        
        form = ComparisonConfigForm(request.POST)
        
        if not form.is_valid():
            # 디버깅: 폼 에러 출력
            print("❌ 폼 검증 실패!")
            print("에러:", form.errors)
            for field, errors in form.errors.items():
                messages.error(request, f"{field}: {', '.join(errors)}")
            
            # 에러가 있어도 설정 페이지로 돌아가서 다시 시도
            context = {
                'first_image': first_image,
                'second_image': second_image,
                'form': form,
                'methods': ComparisonMethod.objects.filter(is_active=True)
            }
            return render(request, 'image_compare/comparison_config.html', context)
        
        # ComparisonMethod 객체 가져오기
        comparison_method = form.get_comparison_method_object()
        
        print(f"✅ 폼 검증 성공! 선택된 방법: {comparison_method.display_name}")
        
        # 비교 객체 생성
        comparison = ImageComparison.objects.create(
            image_1=first_image,
            image_2=second_image,
            comparison_method=comparison_method,
            parameters=form.get_parameters(),
            status='processing'
        )
        
        try:
            # 비교 실행
            result = compare_images_comprehensive(first_image, second_image, comparison)
            
            # 결과 저장
            comparison.similarity_scores = result.get('similarity_scores', {})
            comparison.feature_comparison_data = result.get('feature_comparison_data', {})
            comparison.result_images = result.get('result_images', [])
            comparison.processing_time = result.get('processing_time', 0.0)
            comparison.status = result.get('status', 'completed')
            comparison.save()
            
            messages.success(request, "이미지 비교가 완료되었습니다!")
            return redirect('image_compare:comparison_result', pk=comparison.id)
            
        except Exception as e:
            comparison.status = 'failed'
            comparison.error_message = str(e)
            comparison.save()
            
            print(f"❌ 비교 실행 중 에러: {e}")
            import traceback
            traceback.print_exc()
            
            messages.error(request, f"비교 중 오류가 발생했습니다: {str(e)}")
            return redirect('image_compare:comparison_config', 
                          first_image_id=first_image_id, 
                          second_image_id=second_image_id)


class CompareImagesView(View):
    """빠른 비교 (기본 설정)"""
    
    def post(self, request, first_image_id, second_image_id):
        """기본 설정으로 비교 실행"""
        # 설정 페이지로 리다이렉트
        return redirect('image_compare:comparison_config',
                       first_image_id=first_image_id,
                       second_image_id=second_image_id)


class ComparisonResultView(DetailView):
    """비교 결과 상세 페이지"""
    model = ImageComparison
    template_name = 'image_compare/comparison_result.html'
    context_object_name = 'comparison'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        comparison = self.object
        
        # breadcrumb용 이미지 정보 추가
        context['first_image'] = comparison.image_1
        context['second_image'] = comparison.image_2
        
        # 방법별 세부 정보 추가
        if comparison.comparison_method:
            context['method_category'] = comparison.comparison_method.category
            context['method_name'] = comparison.comparison_method.display_name
        
        # 시각화 이미지들
        context['result_images_list'] = comparison.result_images
        
        return context


class ComparisonListView(ListView):
    """비교 이력 목록"""
    model = ImageComparison
    template_name = 'image_compare/comparison_list.html'
    context_object_name = 'comparisons'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = ImageComparison.objects.select_related(
            'image_1', 'image_2', 'comparison_method'
        ).all()
        
        # 필터링
        status = self.request.GET.get('status', '')
        if status:
            queryset = queryset.filter(status=status)
        
        # 방법 필터
        method = self.request.GET.get('method', '')
        if method:
            queryset = queryset.filter(comparison_method__name=method)
        
        # 검색
        search = self.request.GET.get('search', '')
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) | 
                Q(description__icontains=search) |
                Q(image_1__title__icontains=search) |
                Q(image_2__title__icontains=search)
            )
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_filter'] = self.request.GET.get('status', '')
        context['method_filter'] = self.request.GET.get('method', '')
        context['search'] = self.request.GET.get('search', '')
        context['methods'] = ComparisonMethod.objects.filter(is_active=True)
        return context


class ComparisonDeleteView(View):
    """비교 삭제"""
    
    def post(self, request, pk):
        comparison = get_object_or_404(ImageComparison, pk=pk)
        
        # 결과 이미지 파일들 삭제
        for result_img in comparison.result_images:
            try:
                import os
                from django.conf import settings
                img_path = os.path.join(settings.MEDIA_ROOT, result_img.get('path', ''))
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                print(f"결과 이미지 삭제 실패: {e}")
        
        comparison.delete()
        messages.success(request, "비교 결과가 삭제되었습니다.")
        return redirect('image_compare:comparison_list')


class FeatureExtractionView(View):
    """이미지 피처 추출 (AJAX)"""
    
    def post(self, request, image_id):
        """피처 추출 실행"""
        from .utils import extract_features_from_image
        
        image = get_object_or_404(Image, id=image_id)
        method = request.POST.get('method', 'ORB')
        
        try:
            # 피처 추출
            result = extract_features_from_image(image, method=method)
            
            return JsonResponse({
                'success': True,
                'message': '피처 추출이 완료되었습니다.',
                'data': {
                    'processing_time': result.get('processing_time', 0.0)
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'피처 추출 중 오류가 발생했습니다: {str(e)}'
            }, status=500)
        
class BulkDeleteView(View):
    def post(self, request):
        comparison_ids = request.POST.get('comparison_ids', '')
        ids = [int(id.strip()) for id in comparison_ids.split(',')]
        deleted_count = ImageComparison.objects.filter(id__in=ids).delete()[0]
        messages.success(request, f'{deleted_count}개 삭제됨')
        return redirect('image_compare:comparison_list')
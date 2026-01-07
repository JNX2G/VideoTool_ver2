from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView
from django.contrib import messages
from django.db import models
from django.http import JsonResponse 
from urllib.parse import urlencode

from django.views import View
from .models import PreprocessingMethod
from .forms import MethodForm


class MethodListView(ListView):
    """전처리 기법 목록"""
    model = PreprocessingMethod
    template_name = 'prephub/method_list.html'
    context_object_name = 'methods'
    paginate_by = 20
    
    def get_queryset(self):
            queryset = super().get_queryset()
            
            # 탭 필터 (내장 vs 커스텀)
            tab = self.request.GET.get('tab', 'all')
            if tab == 'builtin':
                queryset = queryset.filter(is_builtin=True)
            elif tab == 'custom':
                queryset = queryset.filter(is_builtin=False)
            
            # 카테고리 필터
            category = self.request.GET.get('category')
            if category:
                queryset = queryset.filter(category=category)
            
            # 검색
            search = self.request.GET.get('search')
            if search:
                queryset = queryset.filter(
                    models.Q(name__icontains=search) |
                    models.Q(description__icontains=search) |
                    models.Q(code__icontains=search)
                )
            
            # 활성화 상태 필터
            status = self.request.GET.get('status')
            if status == 'active':
                queryset = queryset.filter(is_active=True)
            elif status == 'inactive':
                queryset = queryset.filter(is_active=False)
            
            return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = PreprocessingMethod.CATEGORY_CHOICES
        context['current_tab'] = self.request.GET.get('tab', 'all')
        context['current_category'] = self.request.GET.get('category', '')
        context['current_status'] = self.request.GET.get('status', '')
        context['search_query'] = self.request.GET.get('search', '')
        
        # 통계
        context['total_count'] = PreprocessingMethod.objects.count()
        context['builtin_count'] = PreprocessingMethod.objects.filter(is_builtin=True).count()
        context['custom_count'] = PreprocessingMethod.objects.filter(is_builtin=False).count()
        context['active_count'] = PreprocessingMethod.objects.filter(is_active=True).count()
        
        return context

class MethodCreateView(CreateView):
    """전처리 기법 추가"""
    model = PreprocessingMethod
    form_class = MethodForm
    template_name = 'prephub/method_form.html'
    success_url = reverse_lazy('prephub:method_list')
    
    def form_valid(self, form):
        # 커스텀 기법은 is_builtin=False
        form.instance.is_builtin = False
        messages.success(self.request, f'전처리 기법 "{form.instance.name}"이(가) 추가되었습니다.')
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = '전처리 기법 추가'
        context['button_text'] = '추가'
        return context
    

class MethodDetailView(DetailView):
    """전처리 기법 상세"""
    model = PreprocessingMethod
    template_name = 'prephub/method_detail.html'
    context_object_name = 'method'


class MethodUpdateView(UpdateView):
    """전처리 기법 수정"""
    model = PreprocessingMethod
    form_class = MethodForm
    template_name = 'prephub/method_form.html'
    success_url = reverse_lazy('prephub:method_list')
    
    def get_queryset(self):
        # 내장 기법은 수정 불가
        return super().get_queryset().filter(is_builtin=False)

    def form_valid(self, form):
        messages.success(self.request, f'전처리 기법 "{form.instance.name}"이(가) 수정되었습니다.')
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = '전처리 기법 수정'
        context['button_text'] = '수정'
        return context


class MethodDeleteView(DeleteView):
    """전처리 기법 삭제 (커스텀만 가능)"""
    model = PreprocessingMethod
    template_name = 'prephub/method_confirm_delete.html'
    success_url = reverse_lazy('prephub:method_list')
    context_object_name = 'method'
    
    def get_queryset(self):
        # 내장 기법은 삭제 불가
        return super().get_queryset().filter(is_builtin=False)
    
    def delete(self, request, *args, **kwargs):
        method = self.get_object()
        messages.success(request, f'전처리 기법 "{method.name}"이(가) 삭제되었습니다.')
        return super().delete(request, *args, **kwargs)
    
class BulkToggleActiveView(View):
    """일괄 활성화/비활성화 (필터 조건 반영)"""
    def post(self, request):
        # 1. 파라미터 수집
        action = request.POST.get('action')
        tab = request.POST.get('tab', 'all')
        category = request.POST.get('category')
        status = request.POST.get('status')
        search = request.POST.get('search')
        
        # 2. QuerySet 필터링 로직 (기존과 동일)
        queryset = PreprocessingMethod.objects.all()
        if tab == 'builtin':
            queryset = queryset.filter(is_builtin=True)
        elif tab == 'custom':
            queryset = queryset.filter(is_builtin=False)
        
        if category:
            queryset = queryset.filter(category=category)
        if status:
            queryset = queryset.filter(is_active=(status == 'active'))
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(code__icontains=search)
            )
        
        # 3. 일괄 처리 수행
        new_status = (action == 'activate')
        queryset.update(is_active=new_status)
        
        # 4. 현재 필터 상태를 유지하며 리다이렉트
        params = {
            'tab': tab,
            'category': category,
            'status': status,
            'search': search
        }
        params = {k: v for k, v in params.items() if v}
        
        redirect_url = reverse('prephub:method_list')
        if params:
            redirect_url += '?' + urlencode(params)
            
        return redirect(redirect_url)
    

# 개별 토글 (AJAX)
class ToggleActiveView(View):
    """개별 활성화/비활성화 토글"""
    
    def post(self, request, pk):
        try:
            method = get_object_or_404(PreprocessingMethod, pk=pk)
            method.is_active = not method.is_active
            method.save()
            
            # 활성화된 기법 개수 계산
            active_count = PreprocessingMethod.objects.filter(is_active=True).count()
            
            return JsonResponse({
                'success': True,
                'is_active': method.is_active,
                'active_count': active_count,  # ← 추가
                'message': f'"{method.name}" {"활성화" if method.is_active else "비활성화"}됨'
            }, content_type='application/json')
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400, content_type='application/json')
from django.contrib import admin
from .models import Model


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    """Model Admin 설정"""
    
    list_display = [
        'id',
        'name',
        'source',
        'task_type',
        'framework',
        'is_active',
        'usage_count',
        'created_at',
    ]
    
    list_filter = [
        'source',
        'task_type',
        'framework',
        'is_active',
        'created_at',
    ]
    
    search_fields = [
        'name',
        'description',
        'architecture',
    ]
    
    readonly_fields = [
        'id',
        'file_size',
        'num_classes',
        'usage_count',
        'created_at',
        'updated_at',
    ]
    
    fieldsets = (
        ('기본 정보', {
            'fields': ('id', 'source', 'name', 'description', 'is_active')
        }),
        ('소스별 필드', {
            'fields': (
                'builtin_preset',
                'model_file', 'file_size',
                'git_url', 'git_branch', 'git_subfolder', 'git_commit_hash',
                'hf_model_id', 'hf_revision',
            ),
            'classes': ('collapse',),
        }),
        ('메타데이터', {
            'fields': (
                'task_type',
                'framework',
                'architecture',
                'num_classes',
                'input_size',
                'classes',
                'metadata',
            ),
            'classes': ('collapse',),
        }),
        ('통계', {
            'fields': ('usage_count', 'created_at', 'updated_at'),
        }),
    )
    
    actions = [
        'activate_models',
        'deactivate_models',
        'extract_metadata',
    ]
    
    def activate_models(self, request, queryset):
        """선택한 모델 활성화"""
        count = queryset.update(is_active=True)
        self.message_user(request, f'{count}개 모델이 활성화되었습니다.')
    activate_models.short_description = '선택한 모델 활성화'
    
    def deactivate_models(self, request, queryset):
        """선택한 모델 비활성화"""
        count = queryset.update(is_active=False)
        self.message_user(request, f'{count}개 모델이 비활성화되었습니다.')
    deactivate_models.short_description = '선택한 모델 비활성화'
    
    def extract_metadata(self, request, queryset):
        """메타데이터 재추출"""
        from .tasks import extract_and_update_metadata
        
        count = 0
        for model in queryset.filter(source='upload'):
            try:
                extract_and_update_metadata(model.id)
                count += 1
            except Exception as e:
                self.message_user(request, f'{model.name} 추출 실패: {e}', level='error')
        
        self.message_user(request, f'{count}개 모델의 메타데이터가 추출되었습니다.')
    extract_metadata.short_description = '메타데이터 재추출 (Upload만)'
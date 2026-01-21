from django.db import models
from django.core.validators import FileExtensionValidator
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from pathlib import Path


# 모델 전용 스토리지 (MODELS_ROOT 사용)
models_storage = FileSystemStorage(
    location=settings.MODELS_ROOT,
    base_url='/models/'
)


def model_upload_path(instance, filename):
    """모델 파일 업로드 경로 생성"""
    # instance.id가 None이면 임시로 'temp' 사용
    model_id = instance.id if instance.id else 'temp'
    # MODELS_ROOT 기준이므로 'models/' 제거
    return f'custom/{model_id}/{filename}'


class Model(models.Model):
    """통합 모델 관리 클래스"""
    
    # ========================================
    # 소스 타입
    # ========================================
    SOURCE_CHOICES = [
        ('builtin', 'Built-in'),
        ('upload', 'File Upload'),
        ('git', 'Git Repository'),
        ('huggingface', 'HuggingFace'),
    ]
    source = models.CharField(
        max_length=20,
        choices=SOURCE_CHOICES,
        verbose_name='소스'
    )
    
    # ========================================
    # Built-in 모델 전용 필드
    # ========================================
    BUILTIN_PRESETS = [
        ('yolov8n', 'YOLOv8 Nano (빠름)'),
        ('yolov8s', 'YOLOv8 Small (균형) ⭐'),
        ('yolov8m', 'YOLOv8 Medium (정확)'),
        ('yolov8l', 'YOLOv8 Large (매우 정확)'),
        ('yolov8x', 'YOLOv8 XLarge (최고)'),
    ]
    builtin_preset = models.CharField(
        max_length=50,
        choices=BUILTIN_PRESETS,
        blank=True,
        null=True,
        verbose_name='Built-in 프리셋'
    )
    
    # ========================================
    # Upload 모델 전용 필드
    # ========================================
    model_file = models.FileField(
        upload_to=model_upload_path,
        storage=models_storage,  # ← MODELS_ROOT 사용
        blank=True,
        null=True,
        validators=[FileExtensionValidator(['pt', 'pth', 'onnx', 'h5', 'pb', 'tflite'])],
        verbose_name='모델 파일'
    )
    file_size = models.BigIntegerField(default=0, verbose_name='파일 크기(bytes)')
    
    # ========================================
    # Git Repository 전용 필드
    # ========================================
    git_url = models.URLField(blank=True, null=True, verbose_name='Git URL')
    git_branch = models.CharField(max_length=100, blank=True, default='main', verbose_name='Branch')
    git_subfolder = models.CharField(max_length=255, blank=True, verbose_name='Subfolder')
    git_commit_hash = models.CharField(max_length=40, blank=True, verbose_name='Commit Hash')
    
    # ========================================
    # HuggingFace 전용 필드
    # ========================================
    hf_model_id = models.CharField(max_length=255, blank=True, null=True, verbose_name='HF Model ID')
    hf_revision = models.CharField(max_length=100, blank=True, default='main', verbose_name='Revision')
    
    # ========================================
    # 공통 필드
    # ========================================
    name = models.CharField(max_length=200, verbose_name='모델 이름')
    description = models.TextField(blank=True, verbose_name='설명')
    is_active = models.BooleanField(default=True, verbose_name='활성화')
    
    # ========================================
    # 자동 추출 메타데이터
    # ========================================
    TASK_TYPE_CHOICES = [
        ('object_detection', 'Object Detection'),
        ('image_classification', 'Image Classification'),
        ('segmentation', 'Segmentation'),
        ('super_resolution', 'Super Resolution'),
        ('image_restoration', 'Image Restoration'),
    ]
    task_type = models.CharField(
        max_length=50,
        choices=TASK_TYPE_CHOICES,
        blank=True,
        verbose_name='작업 유형'
    )
    
    framework = models.CharField(max_length=50, blank=True, verbose_name='프레임워크')
    architecture = models.CharField(max_length=100, blank=True, verbose_name='아키텍처')
    
    # JSON 필드
    classes = models.JSONField(default=list, blank=True, verbose_name='클래스 목록')
    num_classes = models.IntegerField(default=0, verbose_name='클래스 개수')
    input_size = models.JSONField(default=list, blank=True, verbose_name='입력 크기')
    metadata = models.JSONField(default=dict, blank=True, verbose_name='추가 메타데이터')
    
    # ========================================
    # 통계
    # ========================================
    usage_count = models.IntegerField(default=0, verbose_name='사용 횟수')
    
    # ========================================
    # 타임스탬프
    # ========================================
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일')
    
    class Meta:
        verbose_name = '모델'
        verbose_name_plural = '모델들'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['source', 'is_active']),
            models.Index(fields=['task_type']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.get_source_display()})"
    
    def save(self, *args, **kwargs):
        """저장 시 자동 처리"""
        is_new = self.pk is None
        
        # Built-in 모델: 자동 메타데이터 설정
        if self.source == 'builtin' and self.builtin_preset:
            self._set_builtin_metadata()
        
        # Upload: 파일명에서 이름 생성
        elif self.source == 'upload' and self.model_file:
            if not self.name:
                filename = os.path.basename(self.model_file.name)
                self.name = os.path.splitext(filename)[0]
            if self.model_file and hasattr(self.model_file, 'size'):
                self.file_size = self.model_file.size
        
        # Git: URL에서 이름 생성
        elif self.source == 'git' and self.git_url and not self.name:
            repo_name = self.git_url.rstrip('/').split('/')[-1]
            self.name = repo_name.replace('.git', '')
        
        # HuggingFace: Model ID에서 이름 생성
        elif self.source == 'huggingface' and self.hf_model_id and not self.name:
            self.name = self.hf_model_id.split('/')[-1]
        
        super().save(*args, **kwargs)
        
        # Upload 파일 경로 재조정 (id 생성 후)
        if is_new and self.source == 'upload' and self.model_file:
            old_path = self.model_file.path
            if 'temp' in old_path:
                import shutil
                # 올바른 경로 생성 (MODELS_ROOT 기준)
                new_name = f'custom/{self.id}/{os.path.basename(old_path)}'
                new_path = os.path.join(settings.MODELS_ROOT, new_name)
                
                # 디렉토리 생성
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                
                # 파일 이동
                if os.path.exists(old_path):
                    shutil.move(old_path, new_path)
                    
                    # model_file 필드 업데이트
                    self.model_file.name = new_name
                    super().save(update_fields=['model_file'])
    
    def _set_builtin_metadata(self):
        """Built-in 모델 메타데이터 자동 설정"""
        preset_names = {
            'yolov8n': 'YOLOv8 Nano',
            'yolov8s': 'YOLOv8 Small',
            'yolov8m': 'YOLOv8 Medium',
            'yolov8l': 'YOLOv8 Large',
            'yolov8x': 'YOLOv8 XLarge',
        }
        
        if not self.name:
            self.name = preset_names.get(self.builtin_preset, self.builtin_preset)
        
        # COCO 80 클래스
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.task_type = 'object_detection'
        self.framework = 'PyTorch'
        self.architecture = 'YOLOv8'
        self.classes = coco_classes
        self.num_classes = 80
        self.input_size = [640, 640]
        
        # 프리셋별 파일 크기 (대략)
        size_map = {
            'yolov8n': 6_200_000,      # ~6 MB
            'yolov8s': 22_500_000,     # ~22 MB
            'yolov8m': 52_000_000,     # ~52 MB
            'yolov8l': 87_700_000,     # ~88 MB
            'yolov8x': 136_700_000,    # ~137 MB
        }
        self.file_size = size_map.get(self.builtin_preset, 0)
    
    # ========================================
    # 헬퍼 메서드
    # ========================================
    
    def get_file_size_display(self):
        """파일 크기를 읽기 쉬운 형식으로"""
        if not self.file_size:
            return '-'
        
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def get_source_badge_class(self):
        """소스별 Bootstrap 배지 클래스"""
        return {
            'builtin': 'success',
            'upload': 'primary',
            'git': 'warning',
            'huggingface': 'info',
        }.get(self.source, 'secondary')
    
    def get_task_icon(self):
        """작업 유형별 Bootstrap 아이콘"""
        return {
            'object_detection': 'bounding-box',
            'image_classification': 'tags',
            'segmentation': 'segmented-nav',
            'super_resolution': 'zoom-in',
            'image_restoration': 'magic',
        }.get(self.task_type, 'cpu')
    
    def increment_usage(self):
        """사용 횟수 증가"""
        self.usage_count += 1
        self.save(update_fields=['usage_count'])
    
    def get_model_path(self):
        """실제 모델 파일 경로 반환"""
        if self.source == 'builtin':
            # Built-in 모델은 자동 다운로드되므로 프리셋만 반환
            return self.builtin_preset
        
        elif self.source == 'upload' and self.model_file:
            return self.model_file.path
        
        elif self.source == 'git':
            # Git 클론 디렉토리
            git_dir = Path(settings.MODELS_ROOT) / 'git' / f'model_{self.id}'
            if self.git_subfolder:
                return git_dir / self.git_subfolder
            return git_dir
        
        elif self.source == 'huggingface':
            # HuggingFace 캐시 경로 (자동)
            return self.hf_model_id
        
        return None
    
    def delete(self, *args, **kwargs):
        """삭제 시 파일도 함께 삭제"""
        # Upload 파일 삭제
        if self.model_file:
            try:
                if os.path.exists(self.model_file.path):
                    os.remove(self.model_file.path)
                    # 빈 디렉토리도 삭제
                    dir_path = os.path.dirname(self.model_file.path)
                    if os.path.exists(dir_path) and not os.listdir(dir_path):
                        os.rmdir(dir_path)
            except Exception as e:
                print(f"파일 삭제 실패: {e}")
        
        # Git 디렉토리 삭제
        if self.source == 'git':
            import shutil
            git_dir = Path(settings.MODELS_ROOT) / 'git' / f'model_{self.id}'
            if git_dir.exists():
                try:
                    shutil.rmtree(git_dir)
                except Exception as e:
                    print(f"Git 디렉토리 삭제 실패: {e}")
        
        super().delete(*args, **kwargs)
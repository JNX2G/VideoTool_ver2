# image_compare/management/commands/init_comparison_methods.py

from django.core.management.base import BaseCommand
from image_compare.models import ComparisonMethod


class Command(BaseCommand):
    help = '초기 비교 방법 데이터 생성'

    def handle(self, *args, **kwargs):
        methods = [
            # ORB
            {
                'name': 'ORB',
                'display_name': 'ORB 특징점 매칭',
                'category': 'feature',
                'description': '빠르고 효율적인 특징점 기반 매칭. 회전 불변성 제공.',
                'default_params': {
                    'n_features': 1000,
                    'match_threshold': 0.75
                },
                'param_schema': {
                    'n_features': {
                        'type': 'integer',
                        'label': '특징점 개수',
                        'min': 100,
                        'max': 10000,
                        'step': 100,
                        'default': 1000
                    },
                    'match_threshold': {
                        'type': 'float',
                        'label': '매칭 임계값',
                        'min': 0.5,
                        'max': 0.95,
                        'step': 0.05,
                        'default': 0.75
                    }
                },
                'order': 1
            },
            # SIFT
            {
                'name': 'SIFT',
                'display_name': 'SIFT 특징점 매칭',
                'category': 'feature',
                'description': '고품질 특징점 매칭. 스케일 및 회전 불변성이 뛰어남.',
                'default_params': {
                    'n_features': 1000,
                    'match_threshold': 0.75
                },
                'param_schema': {
                    'n_features': {
                        'type': 'integer',
                        'label': '특징점 개수',
                        'min': 100,
                        'max': 10000,
                        'step': 100,
                        'default': 1000
                    },
                    'match_threshold': {
                        'type': 'float',
                        'label': '매칭 임계값',
                        'min': 0.5,
                        'max': 0.95,
                        'step': 0.05,
                        'default': 0.75
                    }
                },
                'order': 2
            },
            # AKAZE
            {
                'name': 'AKAZE',
                'display_name': 'AKAZE 특징점 매칭',
                'category': 'feature',
                'description': '세밀한 디테일 매칭에 적합. 비선형 스케일 공간 사용.',
                'default_params': {
                    'n_features': 1000,
                    'match_threshold': 0.75
                },
                'param_schema': {
                    'n_features': {
                        'type': 'integer',
                        'label': '특징점 개수',
                        'min': 100,
                        'max': 10000,
                        'step': 100,
                        'default': 1000
                    },
                    'match_threshold': {
                        'type': 'float',
                        'label': '매칭 임계값',
                        'min': 0.5,
                        'max': 0.95,
                        'step': 0.05,
                        'default': 0.75
                    }
                },
                'order': 3
            },
            # SSIM
            {
                'name': 'SSIM',
                'display_name': 'SSIM 구조적 유사도',
                'category': 'structural',
                'description': '이미지의 구조적 유사도를 측정. 인간의 시각 인지와 유사.',
                'default_params': {
                    'window_size': 11
                },
                'param_schema': {
                    'window_size': {
                        'type': 'integer',
                        'label': '윈도우 크기',
                        'min': 3,
                        'max': 21,
                        'step': 2,
                        'default': 11,
                        'help_text': '홀수만 가능'
                    }
                },
                'order': 4
            },
            # 히스토그램
            {
                'name': 'Histogram',
                'display_name': '히스토그램 비교',
                'category': 'histogram',
                'description': '색상 분포를 비교하여 전반적인 색감 유사도 측정.',
                'default_params': {
                    'method': 'correlation',
                    'bins': 256,
                    'color_space': 'HSV'
                },
                'param_schema': {
                    'method': {
                        'type': 'select',
                        'label': '비교 방법',
                        'choices': [
                            ('correlation', 'Correlation'),
                            ('chi_square', 'Chi-Square'),
                            ('intersection', 'Intersection'),
                            ('bhattacharyya', 'Bhattacharyya')
                        ],
                        'default': 'correlation'
                    },
                    'bins': {
                        'type': 'integer',
                        'label': '빈 개수',
                        'min': 16,
                        'max': 256,
                        'step': 16,
                        'default': 256
                    },
                    'color_space': {
                        'type': 'select',
                        'label': '색공간',
                        'choices': [
                            ('RGB', 'RGB'),
                            ('HSV', 'HSV'),
                            ('Lab', 'Lab')
                        ],
                        'default': 'HSV'
                    }
                },
                'order': 5
            },
            # 픽셀 차이
            {
                'name': 'PixelDiff',
                'display_name': '픽셀 단위 차이',
                'category': 'pixel',
                'description': '픽셀별 직접 비교. 정렬된 이미지에 적합.',
                'default_params': {
                    'method': 'absolute',
                    'threshold': 30,
                    'color_space': 'RGB'
                },
                'param_schema': {
                    'method': {
                        'type': 'select',
                        'label': '차이 방법',
                        'choices': [
                            ('absolute', 'Absolute Difference'),
                            ('squared', 'Squared Difference')
                        ],
                        'default': 'absolute'
                    },
                    'threshold': {
                        'type': 'integer',
                        'label': '임계값',
                        'min': 1,
                        'max': 255,
                        'default': 30
                    },
                    'color_space': {
                        'type': 'select',
                        'label': '색공간',
                        'choices': [
                            ('RGB', 'RGB'),
                            ('HSV', 'HSV'),
                            ('Lab', 'Lab')
                        ],
                        'default': 'RGB'
                    }
                },
                'order': 6
            },
        ]
        
        created_count = 0
        updated_count = 0
        
        for method_data in methods:
            method, created = ComparisonMethod.objects.update_or_create(
                name=method_data['name'],
                defaults={
                    'display_name': method_data['display_name'],
                    'category': method_data['category'],
                    'description': method_data['description'],
                    'default_params': method_data['default_params'],
                    'param_schema': method_data['param_schema'],
                    'order': method_data['order'],
                    'is_active': True
                }
            )
            
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'✅ 생성: {method.display_name}')
                )
            else:
                updated_count += 1
                self.stdout.write(
                    self.style.WARNING(f'🔄 업데이트: {method.display_name}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'\n완료! 생성: {created_count}개, 업데이트: {updated_count}개'
            )
        )
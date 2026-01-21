"""
백그라운드 작업 (메타데이터 추출, Git 클론, HuggingFace 다운로드)
"""
import os
import subprocess
from pathlib import Path
from django.conf import settings


def extract_and_update_metadata(model_id):
    """
    백그라운드에서 메타데이터 추출
    
    Args:
        model_id: Model 인스턴스 ID
    """
    from .models import Model
    from .extractors import extract_metadata
    
    try:
        print(f'\n{"="*60}')
        print(f'📊 메타데이터 추출 시작: Model ID={model_id}')
        print(f'{"="*60}\n')
        
        model = Model.objects.get(id=model_id)
        
        # 메타데이터 추출
        extract_metadata(model)
        
        print(f'\n✅ 메타데이터 추출 완료: {model.name}')
        print(f'{"="*60}\n')
        
    except Model.DoesNotExist:
        print(f'❌ Model ID={model_id}를 찾을 수 없습니다.')
    except Exception as e:
        print(f'❌ 메타데이터 추출 실패: {e}')
        import traceback
        traceback.print_exc()


def download_git_model(model_id):
    """
    Git Repository 클론
    
    Args:
        model_id: Model 인스턴스 ID
    """
    from .models import Model
    
    try:
        print(f'\n{"="*60}')
        print(f'📦 Git 클론 시작: Model ID={model_id}')
        print(f'{"="*60}\n')
        
        model = Model.objects.get(id=model_id)
        
        if model.source != 'git':
            print(f'⚠️ Git 모델이 아닙니다: {model.source}')
            return
        
        # 클론 디렉토리
        git_dir = Path(settings.MODELS_ROOT) / 'git' / f'model_{model.id}'
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        
        print(f'📂 클론 경로: {git_dir}')
        print(f'🔗 Git URL: {model.git_url}')
        print(f'🌿 Branch: {model.git_branch or "main"}')
        
        # Git 클론
        clone_cmd = [
            'git', 'clone',
            '--branch', model.git_branch or 'main',
            '--depth', '1',  # Shallow clone (빠름)
            model.git_url,
            str(git_dir)
        ]
        
        result = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode != 0:
            raise Exception(f'Git 클론 실패: {result.stderr}')
        
        print(f'✅ Git 클론 완료')
        
        # Commit hash 저장
        commit_cmd = ['git', 'rev-parse', 'HEAD']
        commit_result = subprocess.run(
            commit_cmd,
            cwd=git_dir,
            capture_output=True,
            text=True
        )
        
        if commit_result.returncode == 0:
            model.git_commit_hash = commit_result.stdout.strip()
            print(f'📌 Commit: {model.git_commit_hash[:8]}')
        
        # README 파일 찾기 및 파싱 (선택적)
        readme_files = ['README.md', 'readme.md', 'README.txt', 'README']
        for readme in readme_files:
            readme_path = git_dir / readme
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 처음 500자를 description으로
                        if not model.description and len(content) > 0:
                            model.description = content[:500]
                            if len(content) > 500:
                                model.description += '...'
                    print(f'📄 README 파싱 완료')
                    break
                except Exception as e:
                    print(f'⚠️ README 파싱 실패: {e}')
        
        model.save()
        
        print(f'\n✅ Git 모델 다운로드 완료: {model.name}')
        print(f'{"="*60}\n')
        
    except Model.DoesNotExist:
        print(f'❌ Model ID={model_id}를 찾을 수 없습니다.')
    except subprocess.TimeoutExpired:
        print(f'❌ Git 클론 타임아웃 (5분 초과)')
    except Exception as e:
        print(f'❌ Git 클론 실패: {e}')
        import traceback
        traceback.print_exc()


def download_huggingface_model(model_id):
    """
    HuggingFace Hub에서 메타데이터 가져오기
    
    Args:
        model_id: Model 인스턴스 ID
    """
    from .models import Model
    
    try:
        print(f'\n{"="*60}')
        print(f'🤗 HuggingFace 메타데이터 가져오기: Model ID={model_id}')
        print(f'{"="*60}\n')
        
        model = Model.objects.get(id=model_id)
        
        if model.source != 'huggingface':
            print(f'⚠️ HuggingFace 모델이 아닙니다: {model.source}')
            return
        
        print(f'🔍 Model ID: {model.hf_model_id}')
        
        # HuggingFace Hub API 사용
        try:
            from huggingface_hub import model_info
            
            info = model_info(model.hf_model_id)
            
            print(f'✅ 모델 정보 가져오기 성공')
            
            # 메타데이터 저장
            model.metadata = {
                'downloads': getattr(info, 'downloads', 0),
                'likes': getattr(info, 'likes', 0),
                'tags': getattr(info, 'tags', []),
                'pipeline_tag': getattr(info, 'pipeline_tag', None),
                'library_name': getattr(info, 'library_name', None),
            }
            
            # Pipeline tag로 task type 매핑
            pipeline_tag = getattr(info, 'pipeline_tag', None)
            if pipeline_tag:
                task_mapping = {
                    'object-detection': 'object_detection',
                    'image-classification': 'image_classification',
                    'image-segmentation': 'segmentation',
                }
                if pipeline_tag in task_mapping:
                    model.task_type = task_mapping[pipeline_tag]
                    print(f'📋 Task Type: {model.task_type}')
            
            # Library name으로 framework 설정
            library_name = getattr(info, 'library_name', None)
            if library_name:
                if library_name in ['transformers', 'pytorch']:
                    model.framework = 'PyTorch'
                elif library_name in ['tensorflow', 'keras']:
                    model.framework = 'TensorFlow'
                print(f'🔧 Framework: {model.framework}')
            
            # Description 설정 (없는 경우)
            if not model.description and hasattr(info, 'cardData'):
                card_data = info.cardData
                if card_data and isinstance(card_data, dict):
                    desc = card_data.get('description', '')
                    if desc:
                        model.description = desc[:500]
            
            # 통계 출력
            print(f'📊 다운로드: {model.metadata.get("downloads", 0):,}회')
            print(f'❤️  좋아요: {model.metadata.get("likes", 0)}개')
            print(f'🏷️  태그: {", ".join(model.metadata.get("tags", [])[:5])}')
            
            model.save()
            
            print(f'\n✅ HuggingFace 메타데이터 가져오기 완료: {model.name}')
            print(f'{"="*60}\n')
            
        except ImportError:
            print(f'❌ huggingface-hub 패키지가 설치되지 않았습니다.')
            print(f'   설치: pip install huggingface-hub')
        
    except Model.DoesNotExist:
        print(f'❌ Model ID={model_id}를 찾을 수 없습니다.')
    except Exception as e:
        print(f'❌ HuggingFace 메타데이터 가져오기 실패: {e}')
        import traceback
        traceback.print_exc()


# 편의 함수들
def bulk_extract_metadata(model_ids):
    """여러 모델의 메타데이터를 한 번에 추출"""
    for model_id in model_ids:
        extract_and_update_metadata(model_id)


def cleanup_unused_files():
    """사용하지 않는 모델 파일 정리"""
    from .models import Model
    
    # Upload 파일 정리
    upload_dir = Path(settings.MEDIA_ROOT) / 'models' / 'custom'
    if upload_dir.exists():
        db_files = set(
            Model.objects.filter(source='upload')
            .exclude(model_file='')
            .values_list('model_file', flat=True)
        )
        
        for file_path in upload_dir.rglob('*'):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(settings.MEDIA_ROOT))
                if rel_path not in db_files:
                    print(f'🗑️  삭제: {rel_path}')
                    file_path.unlink()
    
    # Git 디렉토리 정리
    git_dir = Path(settings.MODELS_ROOT) / 'git'
    if git_dir.exists():
        db_git_ids = set(
            Model.objects.filter(source='git')
            .values_list('id', flat=True)
        )
        
        for model_dir in git_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith('model_'):
                model_id = int(model_dir.name.split('_')[1])
                if model_id not in db_git_ids:
                    print(f'🗑️  삭제: {model_dir}')
                    import shutil
                    shutil.rmtree(model_dir)

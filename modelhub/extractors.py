"""
모델 파일에서 메타데이터 자동 추출
"""
import os
from pathlib import Path


class ModelExtractor:
    """모델 메타데이터 추출기"""
    
    @staticmethod
    def extract_from_pytorch(file_path):
        """
        PyTorch 모델 (.pt, .pth) 메타데이터 추출
        
        Args:
            file_path: 모델 파일 경로
            
        Returns:
            dict: 메타데이터 딕셔너리
        """
        try:
            import torch
            
            # CPU로 로드 (메모리 절약)
            # 먼저 weights_only=True로 시도 (안전)
            try:
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            except Exception:
                # weights_only=True 실패 시 False로 재시도
                # (YOLOv5 등 구버전 모델 호환)
                try:
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                except Exception:
                    # 둘 다 실패하면 기본값으로 시도 (PyTorch 2.5 이하)
                    checkpoint = torch.load(file_path, map_location='cpu')
            
            metadata = {
                'framework': 'PyTorch',
                'architecture': None,
                'classes': [],
                'num_classes': 0,
                'input_size': [],
                'model_type': None,
            }
            
            # 체크포인트가 딕셔너리인 경우
            if isinstance(checkpoint, dict):
                # YOLOv8 Ultralytics 형식
                if 'model' in checkpoint:
                    model = checkpoint.get('model')
                    
                    # 클래스 이름
                    if hasattr(model, 'names'):
                        metadata['classes'] = list(model.names.values())
                        metadata['num_classes'] = len(model.names)
                    
                    # 아키텍처
                    metadata['architecture'] = 'YOLOv8'
                    metadata['model_type'] = 'detection'
                    
                    # 입력 크기
                    if hasattr(model, 'yaml'):
                        yaml_dict = model.yaml
                        if isinstance(yaml_dict, dict) and 'imgsz' in yaml_dict:
                            size = yaml_dict['imgsz']
                            if isinstance(size, int):
                                metadata['input_size'] = [size, size]
                            elif isinstance(size, (list, tuple)):
                                metadata['input_size'] = list(size)
                
                # YOLOv5 형식 (딕셔너리에 직접 정보가 있음)
                elif 'names' in checkpoint or 'nc' in checkpoint:
                    # 클래스 정보
                    if 'names' in checkpoint:
                        names = checkpoint['names']
                        if isinstance(names, dict):
                            metadata['classes'] = list(names.values())
                        elif isinstance(names, list):
                            metadata['classes'] = names
                        metadata['num_classes'] = len(metadata['classes'])
                    elif 'nc' in checkpoint:
                        metadata['num_classes'] = checkpoint['nc']
                    
                    # 아키텍처
                    metadata['architecture'] = 'YOLOv5'
                    metadata['model_type'] = 'detection'
                    
                    # 입력 크기 (일반적으로 640x640)
                    if 'imgsz' in checkpoint:
                        size = checkpoint['imgsz']
                        if isinstance(size, int):
                            metadata['input_size'] = [size, size]
                        elif isinstance(size, (list, tuple)):
                            metadata['input_size'] = list(size)
                    else:
                        metadata['input_size'] = [640, 640]  # 기본값
                
                # 일반 PyTorch 체크포인트
                else:
                    # 클래스 정보
                    if 'classes' in checkpoint:
                        metadata['classes'] = checkpoint['classes']
                        metadata['num_classes'] = len(checkpoint['classes'])
                    elif 'class_names' in checkpoint:
                        metadata['classes'] = checkpoint['class_names']
                        metadata['num_classes'] = len(checkpoint['class_names'])
                    
                    # 아키텍처 정보
                    if 'arch' in checkpoint:
                        metadata['architecture'] = checkpoint['arch']
                    elif 'model_name' in checkpoint:
                        metadata['architecture'] = checkpoint['model_name']
            
            return metadata
            
        except ImportError as e:
            print(f'⚠️ PyTorch가 설치되지 않았습니다: {e}')
            return None
        except Exception as e:
            print(f'❌ PyTorch 메타데이터 추출 실패: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def extract_from_onnx(file_path):
        """
        ONNX 모델 메타데이터 추출
        
        Args:
            file_path: 모델 파일 경로
            
        Returns:
            dict: 메타데이터 딕셔너리
        """
        try:
            import onnx
            
            model = onnx.load(file_path)
            
            metadata = {
                'framework': 'ONNX',
                'architecture': None,
                'classes': [],
                'num_classes': 0,
                'input_size': [],
            }
            
            # 입력 크기 추출
            if model.graph.input:
                input_shape = model.graph.input[0].type.tensor_type.shape.dim
                metadata['input_size'] = [
                    d.dim_value for d in input_shape if d.dim_value > 0
                ]
            
            # 메타데이터 프로퍼티에서 정보 추출
            for prop in model.metadata_props:
                if prop.key == 'classes' or prop.key == 'names':  # YOLOv5 ONNX는 'names' 사용
                    # 문자열 형식 확인: "['fire', 'smoke']" 또는 "fire,smoke"
                    value = prop.value
                    if value.startswith('[') and value.endswith(']'):
                        # ['fire', 'smoke'] 형식
                        import ast
                        try:
                            metadata['classes'] = ast.literal_eval(value)
                        except:
                            metadata['classes'] = value.strip('[]').replace("'", "").split(', ')
                    else:
                        # fire,smoke 형식
                        metadata['classes'] = value.split(',')
                    metadata['num_classes'] = len(metadata['classes'])
                elif prop.key == 'architecture':
                    metadata['architecture'] = prop.value
                elif prop.key == 'model_type':
                    metadata['model_type'] = prop.value
            
            # 출력 레이어에서 클래스 개수 추정
            if not metadata['num_classes'] and model.graph.output:
                for output in model.graph.output:
                    output_shape = output.type.tensor_type.shape.dim
                    if len(output_shape) > 0:
                        # 마지막 차원이 클래스 개수일 가능성
                        last_dim = output_shape[-1].dim_value
                        if last_dim > 0 and last_dim < 10000:
                            metadata['num_classes'] = last_dim
                            break
            
            return metadata
            
        except ImportError:
            print('⚠️ ONNX가 설치되지 않았습니다.')
            return None
        except Exception as e:
            print(f'❌ ONNX 메타데이터 추출 실패: {e}')
            return None
    
    @staticmethod
    def extract_from_tensorflow(file_path):
        """
        TensorFlow/Keras 모델 메타데이터 추출
        
        Args:
            file_path: 모델 파일 경로
            
        Returns:
            dict: 메타데이터 딕셔너리
        """
        try:
            import tensorflow as tf
            
            metadata = {
                'framework': 'TensorFlow',
                'architecture': None,
                'classes': [],
                'num_classes': 0,
                'input_size': [],
            }
            
            # .h5 파일 (Keras)
            if file_path.endswith('.h5'):
                model = tf.keras.models.load_model(file_path, compile=False)
                
                # 입력 크기
                if model.input_shape:
                    metadata['input_size'] = [
                        d for d in model.input_shape[1:] if d is not None
                    ]
                
                # 출력 크기 (클래스 개수)
                if model.output_shape:
                    output_dim = model.output_shape[-1]
                    if output_dim:
                        metadata['num_classes'] = output_dim
            
            # SavedModel 형식
            elif os.path.isdir(file_path):
                model = tf.saved_model.load(file_path)
                # TODO: SavedModel 메타데이터 추출
            
            return metadata
            
        except ImportError:
            print('⚠️ TensorFlow가 설치되지 않았습니다.')
            return None
        except Exception as e:
            print(f'❌ TensorFlow 메타데이터 추출 실패: {e}')
            return None


def extract_metadata(model_instance):
    """
    Model 인스턴스에서 메타데이터 추출 및 업데이트
    
    Args:
        model_instance: modelhub.models.Model 인스턴스
    """
    if model_instance.source != 'upload' or not model_instance.model_file:
        print('⚠️ Upload 모델이 아니거나 파일이 없습니다.')
        return
    
    file_path = model_instance.model_file.path
    if not os.path.exists(file_path):
        print(f'❌ 파일이 존재하지 않습니다: {file_path}')
        return
    
    file_ext = Path(file_path).suffix.lower()
    
    extractor = ModelExtractor()
    metadata = None
    
    # 파일 형식별 추출
    if file_ext in ['.pt', '.pth']:
        print(f'🔍 PyTorch 모델 분석 중: {file_path}')
        metadata = extractor.extract_from_pytorch(file_path)
    
    elif file_ext == '.onnx':
        print(f'🔍 ONNX 모델 분석 중: {file_path}')
        metadata = extractor.extract_from_onnx(file_path)
    
    elif file_ext in ['.h5', '.pb']:
        print(f'🔍 TensorFlow 모델 분석 중: {file_path}')
        metadata = extractor.extract_from_tensorflow(file_path)
    
    else:
        print(f'⚠️ 지원하지 않는 파일 형식: {file_ext}')
        return
    
    # 메타데이터 업데이트
    if metadata:
        print(f'✅ 메타데이터 추출 성공')
        
        if metadata.get('framework'):
            model_instance.framework = metadata['framework']
        
        if metadata.get('architecture'):
            model_instance.architecture = metadata['architecture']
        
        if metadata.get('classes'):
            model_instance.classes = metadata['classes']
            model_instance.num_classes = len(metadata['classes'])
        elif metadata.get('num_classes'):
            model_instance.num_classes = metadata['num_classes']
        
        if metadata.get('input_size'):
            model_instance.input_size = metadata['input_size']
        
        # Task type 추론 (사용자가 선택하지 않은 경우에만 추천)
        if not model_instance.task_type:
            recommended_task = None
            
            # 1. 명시적 model_type이 있으면 사용
            if metadata.get('model_type') == 'detection':
                recommended_task = 'object_detection'
            # 2. YOLO 계열은 무조건 객체 탐지
            elif 'yolo' in str(metadata.get('architecture', '')).lower():
                recommended_task = 'object_detection'
            # 3. ONNX 파일에서 'names' 메타데이터가 있으면 객체 탐지 (YOLOv5 ONNX)
            elif metadata.get('framework') == 'ONNX' and metadata.get('classes'):
                recommended_task = 'object_detection'
            # 4. 클래스가 1000개 이상이면 분류
            elif metadata.get('num_classes') and metadata['num_classes'] >= 1000:
                recommended_task = 'image_classification'
            # 5. 기본 추천값은 객체 탐지
            else:
                recommended_task = 'object_detection'
            
            # 추천된 task_type 설정
            model_instance.task_type = recommended_task
            print(f'💡 추천 Task Type: {recommended_task} (사용자가 수정할 수 있습니다)')
        
        model_instance.save()
        
        print(f'📊 추출된 정보:')
        print(f'  - Framework: {model_instance.framework}')
        print(f'  - Architecture: {model_instance.architecture}')
        print(f'  - Classes: {model_instance.num_classes}개')
        print(f'  - Input Size: {model_instance.input_size}')
        print(f'  - Task Type: {model_instance.task_type}')
    else:
        print(f'❌ 메타데이터 추출 실패')


# 편의 함수
def extract_pytorch_classes(file_path):
    """PyTorch 모델에서 클래스 목록만 추출"""
    metadata = ModelExtractor.extract_from_pytorch(file_path)
    return metadata.get('classes', []) if metadata else []


def extract_onnx_input_size(file_path):
    """ONNX 모델에서 입력 크기만 추출"""
    metadata = ModelExtractor.extract_from_onnx(file_path)
    return metadata.get('input_size', []) if metadata else []
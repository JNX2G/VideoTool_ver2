"""
전처리 실행 엔진
prephub의 PreprocessingMethod를 사용하여 이미지/동영상 전처리 수행
"""
import cv2
import numpy as np
import logging
import subprocess
from pathlib import Path
from prephub.models import PreprocessingMethod


# 로거 설정
logger = logging.getLogger(__name__)


class PreprocessingEngine:
    """전처리 실행 엔진"""
    
    def __init__(self, log_level=logging.INFO):
        """
        Args:
            log_level: 로그 레벨 (logging.DEBUG, INFO, WARNING, ERROR)
        """
        self.current_frame = 0
        self.total_frames = 0
        
        # 로거 설정
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
    
    def _validate_frame_size(self, frame, expected_width, expected_height, step_name=""):
        """
        프레임 크기 검증 및 자동 복원
        
        Args:
            frame: 검증할 프레임
            expected_width: 예상 너비
            expected_height: 예상 높이
            step_name: 단계 이름 (로그용)
        
        Returns:
            tuple: (validated_frame, size_changed)
        """
        current_height, current_width = frame.shape[:2]
        
        if current_height == expected_height and current_width == expected_width:
            return frame, False
        
        # 크기 불일치 감지
        logger.warning(
            f"크기 불일치 - {step_name} | "
            f"예상: {expected_width}x{expected_height}, "
            f"실제: {current_width}x{current_height}"
        )
        
        # 원본 크기로 복원
        logger.info("원본 크기로 복원 중...")
        restored_frame = cv2.resize(
            frame,
            (expected_width, expected_height),
            interpolation=cv2.INTER_LINEAR
        )
        logger.info("복원 완료")
        
        return restored_frame, True
    
    def process_image(self, input_path, pipeline, output_path, progress_callback=None):
        """
        이미지 전처리 실행
        
        Args:
            input_path (str|Path): 입력 이미지 경로
            pipeline (list): [{"method_id": 1, "params": {...}}, ...]
            output_path (str|Path): 출력 이미지 경로
            progress_callback (callable): 진행률 콜백 함수(current, total, progress)
        
        Raises:
            ValueError: 이미지 로드/저장 실패
            FileNotFoundError: 입력 파일 없음
        """
        logger.info("=" * 80)
        logger.info("이미지 전처리 작업 시작")
        logger.info(f"입력 파일: {input_path}")
        logger.info(f"출력 파일: {output_path}")
        logger.info(f"파이프라인 단계: {len(pipeline)}개")
        
        # ========================================
        # 1. 이미지 로드
        # ========================================
        logger.info("-" * 80)
        logger.info("이미지 로드 중...")
        
        frame = cv2.imread(str(input_path))
        if frame is None:
            logger.error(f"이미지 로드 실패: {input_path}")
            raise ValueError(f"이미지를 읽을 수 없습니다: {input_path}")
        
        # 원본 정보 저장
        original_height, original_width = frame.shape[:2]
        original_channels = frame.shape[2] if len(frame.shape) == 3 else 1
        
        logger.info(
            f"이미지 로드 완료 - "
            f"{original_width}x{original_height}, "
            f"{original_channels}채널"
        )
        
        # ========================================
        # 2. 전처리 파이프라인 적용
        # ========================================
        self.total_frames = len(pipeline) + 1
        self.current_frame = 0
        
        if progress_callback:
            progress_callback(self.current_frame, self.total_frames, 0)
        
        if not pipeline:
            logger.warning("전처리 파이프라인이 비어있음 - 원본 이미지 사용")
        else:
            logger.info("-" * 80)
            logger.info(f"전처리 파이프라인 실행 시작 ({len(pipeline)}단계)")
            
            for i, step in enumerate(pipeline, 1):
                method_id = step.get("method_id")
                params = step.get("params", {})
                
                try:
                    # 전처리 기법 가져오기
                    method = PreprocessingMethod.objects.get(id=method_id)
                    
                    logger.info(f"[{i}/{len(pipeline)}] {method.name} 적용 중...")
                    
                    if params:
                        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        logger.debug(f"  파라미터: {params_str}")
                    
                    # 전처리 실행
                    frame = method.execute(frame, params)
                    
                    # 크기 검증
                    frame, size_changed = self._validate_frame_size(
                        frame, 
                        original_width, 
                        original_height,
                        step_name=f"단계 {i} ({method.name})"
                    )
                    
                    # 진행률 업데이트
                    self.current_frame = i
                    progress = int((self.current_frame / self.total_frames) * 100)
                    
                    if progress_callback:
                        progress_callback(self.current_frame, self.total_frames, progress)
                    
                    logger.info(f"[{i}/{len(pipeline)}] {method.name} 완료")
                    
                except PreprocessingMethod.DoesNotExist:
                    logger.error(f"전처리 기법을 찾을 수 없음 (ID: {method_id})")
                    continue
                    
                except Exception as e:
                    logger.exception(f"전처리 실행 중 오류 발생: {e}")
                    raise
            
            logger.info("전처리 파이프라인 완료")
        
        # ========================================
        # 3. 최종 검증
        # ========================================
        logger.info("-" * 80)
        logger.info("최종 검증 중...")
        
        final_height, final_width = frame.shape[:2]
        logger.info(
            f"원본 크기: {original_width}x{original_height}, "
            f"최종 크기: {final_width}x{final_height}"
        )
        
        # 최종 크기 강제 검증
        if final_height != original_height or final_width != original_width:
            logger.warning("최종 크기 불일치 - 강제 복원 수행")
            frame = cv2.resize(
                frame,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR
            )
            logger.info("복원 완료")
        
        # ========================================
        # 4. 이미지 저장
        # ========================================
        logger.info("-" * 80)
        logger.info("이미지 저장 중...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 확장자에 따라 저장 옵션 설정
        ext = output_path.suffix.lower()
        
        if ext == '.png':
            # PNG - 무손실
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
            logger.debug("PNG 형식으로 저장 (압축 없음)")
        elif ext in ['.jpg', '.jpeg']:
            # JPEG - 최고 품질
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
            logger.debug("JPEG 형식으로 저장 (품질 100)")
        else:
            encode_params = []
            logger.debug(f"{ext} 형식으로 저장 (기본 설정)")
        
        success = cv2.imwrite(str(output_path), frame, encode_params)
        
        if not success:
            logger.error(f"이미지 저장 실패: {output_path}")
            raise ValueError(f"이미지 저장 실패: {output_path}")
        
        # 저장된 파일 검증
        file_size = output_path.stat().st_size
        logger.info(f"이미지 저장 완료 - 크기: {file_size:,} bytes")
        
        # 저장된 이미지 다시 읽어서 검증
        saved_img = cv2.imread(str(output_path))
        if saved_img is not None:
            saved_height, saved_width = saved_img.shape[:2]
            
            if saved_height == original_height and saved_width == original_width:
                logger.info(f"저장 검증 완료 - {saved_width}x{saved_height}")
            else:
                logger.warning(
                    f"저장 후 크기 변경 감지: "
                    f"{saved_width}x{saved_height} (원본: {original_width}x{original_height})"
                )
        
        # 진행률 100%
        self.current_frame = self.total_frames
        if progress_callback:
            progress_callback(self.current_frame, self.total_frames, 100)
        
        logger.info("=" * 80)
        logger.info("이미지 전처리 작업 완료")
        logger.info("=" * 80)
    
    def process_video(self, input_path, pipeline, output_path, progress_callback=None, task_id=None):
        """
        동영상 전처리 실행
        
        Args:
            input_path (str|Path): 입력 동영상 경로
            pipeline (list): [{"method_id": 1, "params": {...}}, ...]
            output_path (str|Path): 출력 동영상 경로
            progress_callback (callable): 진행률 콜백 함수(current, total, progress)
            task_id (int): 작업 ID (임시 파일명에 사용)
        
        Raises:
            ValueError: 동영상 열기/쓰기 실패
            FileNotFoundError: 입력 파일 없음
        """
        logger.info("=" * 80)
        logger.info("동영상 전처리 작업 시작")
        logger.info(f"입력 파일: {input_path}")
        logger.info(f"출력 파일: {output_path}")
        logger.info(f"파이프라인 단계: {len(pipeline)}개")
        
        # ========================================
        # 1. 동영상 열기
        # ========================================
        logger.info("-" * 80)
        logger.info("동영상 정보 로드 중...")
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"동영상 열기 실패: {input_path}")
            raise ValueError(f"동영상을 열 수 없습니다: {input_path}")
        
        # 동영상 정보 추출
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"FPS: {fps}")
        logger.info(f"해상도: {original_width}x{original_height}")
        logger.info(f"총 프레임: {self.total_frames}")
        
        # ========================================
        # 2. VideoWriter 설정
        # ========================================
        logger.info("-" * 80)
        logger.info("VideoWriter 초기화 중...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ⭐ 임시 파일명을 고유하게 생성 (task_id + timestamp)
        import time
        timestamp = int(time.time() * 1000)  # 밀리초 단위
        temp_filename = f"temp_{task_id}_{timestamp}.avi" if task_id else f"temp_{timestamp}.avi"
        temp_output = output_path.parent / temp_filename
        logger.debug(f"임시 파일: {temp_output}")
        
        # XVID 코덱 사용
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            str(temp_output),
            fourcc,
            fps,
            (original_width, original_height)
        )
        
        if not out.isOpened():
            cap.release()
            logger.error("VideoWriter 초기화 실패")
            raise ValueError("VideoWriter 초기화 실패")
        
        logger.info("VideoWriter 초기화 완료")
        
        # ========================================
        # 3. 프레임별 전처리
        # ========================================
        logger.info("-" * 80)
        logger.info(f"프레임 처리 시작 (총 {self.total_frames}프레임)")
        
        self.current_frame = 0
        size_mismatch_count = 0  # 크기 불일치 카운트
        error_count = 0  # 오류 카운트
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 전처리 파이프라인 적용
                for step_idx, step in enumerate(pipeline, 1):
                    method_id = step.get("method_id")
                    params = step.get("params", {})
                    
                    try:
                        method = PreprocessingMethod.objects.get(id=method_id)
                        frame = method.execute(frame, params)
                        
                        # 크기 검증 (매 프레임마다)
                        frame, size_changed = self._validate_frame_size(
                            frame,
                            original_width,
                            original_height,
                            step_name=f"프레임 {self.current_frame}, 단계 {step_idx}"
                        )
                        
                        if size_changed:
                            size_mismatch_count += 1
                        
                    except PreprocessingMethod.DoesNotExist:
                        # 첫 프레임에만 경고
                        if self.current_frame == 0:
                            logger.error(f"전처리 기법을 찾을 수 없음 (ID: {method_id})")
                        error_count += 1
                        continue
                        
                    except Exception as e:
                        # 100프레임마다 로그 출력
                        if self.current_frame % 100 == 0:
                            logger.error(
                                f"프레임 {self.current_frame} 처리 오류: {e}"
                            )
                        error_count += 1
                        continue
                
                # 프레임 저장
                out.write(frame)
                
                self.current_frame += 1
                
                # 진행률 업데이트 (30프레임마다)
                if self.current_frame % 30 == 0 or self.current_frame == self.total_frames:
                    progress = int((self.current_frame / self.total_frames) * 85)
                    
                    if progress_callback:
                        progress_callback(
                            self.current_frame,
                            self.total_frames,
                            progress
                        )
                    
                    # 100프레임마다 진행 상황 로그
                    if self.current_frame % 100 == 0:
                        logger.info(
                            f"진행: {self.current_frame}/{self.total_frames} "
                            f"({progress}%)"
                        )
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"프레임 처리 완료 - 총 {self.current_frame}프레임")
        
        if size_mismatch_count > 0:
            logger.warning(f"크기 불일치 발생 횟수: {size_mismatch_count}")
        
        if error_count > 0:
            logger.warning(f"오류 발생 횟수: {error_count}")
        
        # ========================================
        # 4. ffmpeg 재인코딩 (MP4 H.264)
        # ========================================
        logger.info("-" * 80)
        logger.info("ffmpeg 재인코딩 시작...")
        
        if progress_callback:
            progress_callback(self.current_frame, self.total_frames, 90)
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(temp_output),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',  # 고품질 (18-23 권장)
            '-pix_fmt', 'yuv420p',
            '-s', f'{original_width}x{original_height}',  # 명시적 크기 지정
            str(output_path)
        ]
        
        logger.debug(f"ffmpeg 명령: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=300  # 5분 타임아웃
            )
            logger.info("ffmpeg 재인코딩 완료")
            
            # ⭐ Windows에서 파일 핸들이 완전히 닫히도록 짧은 대기
            import time
            time.sleep(0.3)
            
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg 재인코딩 타임아웃 (5분 초과)")
            raise ValueError("ffmpeg 재인코딩 타임아웃")
            
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg 재인코딩 실패")
            logger.error(f"stderr: {e.stderr.decode('utf-8', errors='ignore')}")
            raise ValueError("ffmpeg 재인코딩 실패")
            
        finally:
            # ⭐ 임시 파일 삭제 (Windows 호환 - 재시도 로직)
            if temp_output.exists():
                import time
                max_retries = 5
                retry_delay = 0.5  # 0.5초
                
                for attempt in range(max_retries):
                    try:
                        temp_output.unlink()
                        logger.debug(f"임시 파일 삭제 성공: {temp_output}")
                        break
                    except PermissionError as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"임시 파일 삭제 실패 (시도 {attempt + 1}/{max_retries}), "
                                f"{retry_delay}초 후 재시도: {e}"
                            )
                            time.sleep(retry_delay)
                        else:
                            logger.error(
                                f"임시 파일 삭제 최종 실패: {temp_output}\n"
                                f"파일이 다른 프로세스에서 사용 중일 수 있습니다. "
                                f"수동으로 삭제해주세요."
                            )
                            # 삭제 실패해도 전처리는 완료된 것으로 처리
                    except Exception as e:
                        logger.error(f"임시 파일 삭제 중 예상치 못한 오류: {e}")
                        break
        
        # 최종 파일 검증
        if output_path.exists():
            file_size = output_path.stat().st_size
            logger.info(f"최종 파일 크기: {file_size:,} bytes")
        else:
            logger.error(f"최종 파일이 생성되지 않음: {output_path}")
            raise ValueError("최종 파일 생성 실패")
        
        # 진행률 100%
        if progress_callback:
            progress_callback(self.total_frames, self.total_frames, 100)
        
        logger.info("=" * 80)
        logger.info("동영상 전처리 작업 완료")
        logger.info("=" * 80)
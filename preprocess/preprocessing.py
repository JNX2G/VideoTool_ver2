"""
전처리 실행 엔진
prephub의 PreprocessingMethod를 사용하여 이미지/동영상 전처리 수행
"""
import cv2
import numpy as np
from pathlib import Path
from prephub.models import PreprocessingMethod


class PreprocessingEngine:
    """전처리 실행 엔진"""
    
    def __init__(self):
        self.current_frame = 0
        self.total_frames = 0
    
    def process_image(self, input_path, pipeline, output_path, progress_callback=None):
        """
        이미지 전처리 실행
        
        Args:
            input_path: 입력 이미지 경로
            pipeline: [{"method_id": 1, "params": {...}}, ...]
            output_path: 출력 이미지 경로
            progress_callback: 진행률 콜백 함수(current, total, progress)
        """
        print(f"\n{'='*50}")
        print(f"🖼️ 이미지 전처리 시작")
        print(f"📥 입력: {input_path}")
        print(f"📤 출력: {output_path}")
        print(f"🔧 파이프라인: {len(pipeline)}단계")
        
        # 이미지 읽기
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {input_path}")
        
        print(f"✅ 이미지 로드 완료: {frame.shape}")
        
        # 전처리 파이프라인 적용
        self.total_frames = len(pipeline) + 1  # 파이프라인 단계 + 저장
        self.current_frame = 0
        
        if progress_callback:
            progress_callback(self.current_frame, self.total_frames, 0)
        
        for i, step in enumerate(pipeline):
            method_id = step.get("method_id")
            params = step.get("params", {})
            
            try:
                # PreprocessingMethod 가져오기
                method = PreprocessingMethod.objects.get(id=method_id)
                
                print(f"\n🔧 [{i+1}/{len(pipeline)}] {method.name} 적용 중...")
                print(f"   파라미터: {params}")
                
                # 전처리 실행
                frame = method.execute(frame, params)
                
                self.current_frame = i + 1
                progress = int((self.current_frame / self.total_frames) * 100)
                
                if progress_callback:
                    progress_callback(self.current_frame, self.total_frames, progress)
                
                print(f"   ✅ 완료")
                
            except PreprocessingMethod.DoesNotExist:
                print(f"   ⚠️ 전처리 기법을 찾을 수 없습니다 (ID: {method_id})")
                continue
            except Exception as e:
                print(f"   ❌ 오류: {e}")
                raise
        
        # 결과 저장
        print(f"\n💾 결과 저장 중...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(output_path), frame)
        if not success:
            raise ValueError(f"이미지 저장 실패: {output_path}")
        
        self.current_frame = self.total_frames
        if progress_callback:
            progress_callback(self.current_frame, self.total_frames, 100)
        
        print(f"✅ 저장 완료: {output_path}")
        print(f"{'='*50}\n")
    
    def process_video(self, input_path, pipeline, output_path, progress_callback=None):
        """
        동영상 전처리 실행
        
        Args:
            input_path: 입력 동영상 경로
            pipeline: [{"method_id": 1, "params": {...}}, ...]
            output_path: 출력 동영상 경로
            progress_callback: 진행률 콜백 함수(current, total, progress)
        """
        print(f"\n{'='*50}")
        print(f"🎬 동영상 전처리 시작")
        print(f"📥 입력: {input_path}")
        print(f"📤 출력: {output_path}")
        print(f"🔧 파이프라인: {len(pipeline)}단계")
        
        # 동영상 열기
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"동영상을 열 수 없습니다: {input_path}")
        
        # 동영상 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✅ 동영상 정보: {width}x{height} @ {fps}fps, {self.total_frames}프레임")
        
        # 임시 출력 파일 (코덱 없이)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
        
        # VideoWriter 생성 (무손실 코덱)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError("VideoWriter 초기화 실패")
        
        self.current_frame = 0
        
        try:
            # 프레임별 처리
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 전처리 파이프라인 적용
                for step in pipeline:
                    method_id = step.get("method_id")
                    params = step.get("params", {})
                    
                    try:
                        method = PreprocessingMethod.objects.get(id=method_id)
                        frame = method.execute(frame, params)
                    except PreprocessingMethod.DoesNotExist:
                        print(f"⚠️ 전처리 기법을 찾을 수 없습니다 (ID: {method_id})")
                        continue
                    except Exception as e:
                        print(f"❌ 프레임 {self.current_frame} 처리 오류: {e}")
                        # 오류 발생 시 원본 프레임 사용
                        continue
                
                # 프레임 저장
                out.write(frame)
                
                self.current_frame += 1
                
                # 진행률 업데이트 (매 30프레임마다)
                if self.current_frame % 30 == 0 or self.current_frame == self.total_frames:
                    progress = int((self.current_frame / self.total_frames) * 85)  # 85%까지
                    if progress_callback:
                        progress_callback(self.current_frame, self.total_frames, progress)
                    
                    if self.current_frame % 100 == 0:
                        print(f"⏳ 진행: {self.current_frame}/{self.total_frames} ({progress}%)")
        
        finally:
            cap.release()
            out.release()
        
        print(f"\n✅ 프레임 처리 완료")
        
        # ffmpeg로 재인코딩 (MP4 H.264)
        print(f"🎞️ ffmpeg 재인코딩 중...")
        
        if progress_callback:
            progress_callback(self.current_frame, self.total_frames, 90)
        
        import subprocess
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(temp_output),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            print(f"✅ 재인코딩 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg 오류:")
            print(e.stderr.decode('utf-8'))
            raise ValueError("ffmpeg 재인코딩 실패")
        finally:
            # 임시 파일 삭제
            if temp_output.exists():
                temp_output.unlink()
        
        if progress_callback:
            progress_callback(self.total_frames, self.total_frames, 100)
        
        print(f"✅ 저장 완료: {output_path}")
        print(f"{'='*50}\n")

"""
내장 전처리 기법들
각 함수는 (frame, **params) 형태로 정의
"""
import cv2
import numpy as np


def gaussian_blur(frame, **params):
    """가우시안 블러"""
    ksize = params.get('ksize', 5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def median_blur(frame, **params):
    """미디언 블러"""
    ksize = params.get('ksize', 5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(frame, ksize)


def canny_edge(frame, **params):
    """Canny 엣지 검출"""
    threshold1 = params.get('threshold1', 100)
    threshold2 = params.get('threshold2', 200)
    
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    edges = cv2.Canny(gray, threshold1, threshold2)
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return result


def sobel_edge(frame, **params):
    """Sobel 엣지 검출"""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    
    result = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    return result


def grayscale(frame, **params):
    """그레이스케일 변환"""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        result = frame
    return result


def binary_threshold(frame, **params):
    """이진화"""
    threshold = params.get('threshold', 127)
    
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return result


def adaptive_threshold(frame, **params):
    """적응형 이진화"""
    block_size = params.get('block_size', 11)
    c = params.get('c', 2)
    
    if block_size % 2 == 0:
        block_size += 1
    
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )
    
    result = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    return result


def morphology_open(frame, **params):
    """모폴로지 열기"""
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)


def morphology_close(frame, **params):
    """모폴로지 닫기"""
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)


def histogram_equalization(frame, **params):
    """히스토그램 평활화"""
    if len(frame.shape) == 3:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = cv2.equalizeHist(frame)
    return result

def clahe(frame, **params):
    """CLAHE 대비 개선"""
    clip_limit = params.get('clip_limit', 2.0)
    tile_grid_size = params.get('tile_grid_size', 8)
    
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    
    if len(frame.shape) == 3:
        # 컬러 이미지의 경우 밝기 채널(Y)만 처리하여 색 왜곡 방지
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe_obj.apply(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = clahe_obj.apply(frame)
    return result


def bilateral_filter(frame, **params):
    """양방향 필터 (엣지 보존 노이즈 제거)"""
    d = params.get('d', 9)
    sigma_color = params.get('sigma_color', 75)
    sigma_space = params.get('sigma_space', 75)
    
    # Bilateral 필터는 8비트 3채널(BGR) 또는 1채널 이미지를 지원함
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)


def harris_corner(frame, **params):
    """해리스 코너 검출"""
    block_size = params.get('block_size', 2)
    ksize = params.get('ksize', 3)
    k = params.get('k', 0.04)
    
    if ksize % 2 == 0: ksize += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # 결과 시각화: 임계값을 넘는 지점을 빨간색 점으로 표시
    result = frame.copy()
    result[dst > 0.01 * dst.max()] = [0, 0, 255]
    return result


def shi_tomasi(frame, **params):
    """Shi-Tomasi 코너 검출 (Good Features to Track)"""
    max_corners = params.get('max_corners', 100)
    quality_level = params.get('quality_level', 0.01)
    min_distance = params.get('min_distance', 10)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    
    result = frame.copy()
    if corners is not None:
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1) # 초록색 점 표시
    return result


def orb_feature(frame, **params):
    """ORB 특징점 검출 및 시각화"""
    n_features = params.get('n_features', 500)
    
    orb = cv2.ORB_create(nfeatures=n_features)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # 특징점 검출
    keypoints = orb.detect(gray, None)
    
    # 특징점의 위치와 방향을 이미지 위에 그림
    result = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 255), flags=0)
    return result
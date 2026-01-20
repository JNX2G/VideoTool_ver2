"""
고급 이미지 비교 유틸리티

지원 방법:
1. ORB/SIFT/AKAZE 특징점 기반
2. SSIM 구조적 유사도
3. 히스토그램 비교
4. 픽셀 차이
"""

import os
import cv2
import numpy as np
from pathlib import Path
from django.conf import settings
import time
from skimage.metrics import structural_similarity as ssim

from .models import ComparisonFeatureExtraction, ComparisonMethod


# =============================================================================
# 1. 특징점 기반 비교 (Feature-based)
# =============================================================================

def compare_with_features(img1, img2, params):
    """
    특징점 기반 비교
    
    Args:
        img1, img2: OpenCV 이미지
        params: {
            'method': 'ORB'|'SIFT'|'AKAZE'|'BRISK',
            'n_features': int,
            'match_threshold': float
        }
    
    Returns:
        dict: 비교 결과
    """
    method = params.get('method', 'ORB')
    n_features = params.get('n_features', 1000)
    match_threshold = params.get('match_threshold', 0.75)
    
    # 그레이스케일 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 디텍터 생성
    if method == 'ORB':
        detector = cv2.ORB_create(nfeatures=n_features)
    elif method == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=n_features)
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
    elif method == 'BRISK':
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")
    
    # 특징점 및 디스크립터 추출
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
        return {
            'similarity': 0.0,
            'details': {
                'keypoints1': 0,
                'keypoints2': 0,
                'matches': 0,
                'good_matches': 0
            },
            'visualization_data': None
        }
    
    # 매칭
    if method in ['ORB', 'AKAZE', 'BRISK']:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < match_threshold * n.distance:
                good_matches.append(m)
    
    # 유사도 계산
    max_possible = min(len(kp1), len(kp2))
    similarity = len(good_matches) / max_possible if max_possible > 0 else 0.0
    similarity = min(similarity * 2.0, 1.0)  # 스케일링
    
    return {
        'similarity': similarity,
        'details': {
            'keypoints1': len(kp1),
            'keypoints2': len(kp2),
            'matches': len(matches),
            'good_matches': len(good_matches),
            'match_ratio': len(good_matches) / len(matches) if matches else 0
        },
        'visualization_data': {
            'kp1': kp1,
            'kp2': kp2,
            'good_matches': good_matches[:50]  # 상위 50개만
        }
    }


# =============================================================================
# 2. SSIM 구조적 유사도
# =============================================================================

def compare_with_ssim(img1, img2, params):
    """
    SSIM 구조적 유사도 비교
    
    Args:
        img1, img2: OpenCV 이미지
        params: {
            'window_size': int,
            'channel_axis': int|None
        }
    
    Returns:
        dict: 비교 결과
    """
    window_size = params.get('window_size', 11)
    
    # 같은 크기로 리사이즈
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    img1_resized = cv2.resize(img1, (target_w, target_h))
    img2_resized = cv2.resize(img2, (target_w, target_h))
    
    # SSIM 계산
    score, diff = ssim(img1_resized, img2_resized, 
                       channel_axis=2,
                       full=True,
                       win_size=window_size)
    
    # 차이 맵 정규화
    diff = (diff * 255).astype("uint8")
    
    return {
        'similarity': score,
        'details': {
            'window_size': window_size,
            'mean_diff': float(np.mean(diff)),
            'max_diff': float(np.max(diff))
        },
        'visualization_data': {
            'diff_map': diff
        }
    }


# =============================================================================
# 3. 히스토그램 비교
# =============================================================================

def compare_with_histogram(img1, img2, params):
    """
    히스토그램 비교
    
    Args:
        img1, img2: OpenCV 이미지
        params: {
            'method': 'correlation'|'chi_square'|'intersection'|'bhattacharyya',
            'bins': int,
            'color_space': 'RGB'|'HSV'|'Lab'
        }
    
    Returns:
        dict: 비교 결과
    """
    method = params.get('method', 'correlation')
    bins = params.get('bins', 256)
    color_space = params.get('color_space', 'HSV')
    
    # 색공간 변환
    if color_space == 'HSV':
        img1_converted = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_converted = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    elif color_space == 'Lab':
        img1_converted = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
        img2_converted = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    else:  # RGB
        img1_converted = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_converted = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 히스토그램 계산
    hist1 = cv2.calcHist([img1_converted], [0, 1, 2], None, 
                         [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_converted], [0, 1, 2], None, 
                         [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    
    # 정규화
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # 비교 방법 선택
    method_map = {
        'correlation': cv2.HISTCMP_CORREL,
        'chi_square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
    }
    
    cv_method = method_map.get(method, cv2.HISTCMP_CORREL)
    score = cv2.compareHist(hist1, hist2, cv_method)
    
    # 점수 정규화 (0~1 범위)
    if method == 'correlation':
        similarity = (score + 1) / 2  # -1~1 → 0~1
    elif method == 'chi_square':
        similarity = 1.0 / (1.0 + score)  # 낮을수록 유사
    elif method == 'intersection':
        similarity = score  # 이미 0~1
    elif method == 'bhattacharyya':
        similarity = 1.0 - score  # 낮을수록 유사
    else:
        similarity = score
    
    # 각 채널별 히스토그램도 계산
    channel_scores = []
    for i in range(3):
        h1 = cv2.calcHist([img1_converted], [i], None, [bins], [0, 256])
        h2 = cv2.calcHist([img2_converted], [i], None, [bins], [0, 256])
        cv2.normalize(h1, h1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(h2, h2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        channel_score = cv2.compareHist(h1, h2, cv_method)
        channel_scores.append(float(channel_score))
    
    return {
        'similarity': similarity,
        'details': {
            'method': method,
            'bins': bins,
            'color_space': color_space,
            'raw_score': float(score),
            'channel_scores': channel_scores
        },
        'visualization_data': {
            'hist1': hist1,
            'hist2': hist2
        }
    }


# =============================================================================
# 4. 픽셀 차이
# =============================================================================

def compare_with_pixel_diff(img1, img2, params):
    """
    픽셀 단위 차이 비교
    
    Args:
        img1, img2: OpenCV 이미지
        params: {
            'method': 'absolute'|'squared',
            'threshold': int,
            'color_space': 'RGB'|'HSV'|'Lab'
        }
    
    Returns:
        dict: 비교 결과
    """
    method = params.get('method', 'absolute')
    threshold = params.get('threshold', 30)
    color_space = params.get('color_space', 'RGB')
    
    # 같은 크기로 리사이즈
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    img1_resized = cv2.resize(img1, (target_w, target_h))
    img2_resized = cv2.resize(img2, (target_w, target_h))
    
    # 색공간 변환
    if color_space == 'HSV':
        img1_converted = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2HSV)
        img2_converted = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2HSV)
    elif color_space == 'Lab':
        img1_converted = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2Lab)
        img2_converted = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2Lab)
    else:  # RGB
        img1_converted = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
        img2_converted = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB)
    
    # 차이 계산
    if method == 'absolute':
        diff = cv2.absdiff(img1_converted, img2_converted)
    elif method == 'squared':
        diff = (img1_converted.astype(float) - img2_converted.astype(float)) ** 2
        diff = np.sqrt(diff).astype(np.uint8)
    else:
        diff = cv2.absdiff(img1_converted, img2_converted)
    
    # 그레이스케일 변환 (시각화용)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if color_space != 'HSV' else diff[:,:,2]
    
    # 임계값 적용
    _, thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 차이 픽셀 비율
    diff_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    diff_ratio = diff_pixels / total_pixels
    
    # 유사도 = 1 - 차이 비율
    similarity = 1.0 - diff_ratio
    
    # 평균 차이
    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))
    
    return {
        'similarity': similarity,
        'details': {
            'method': method,
            'threshold': threshold,
            'color_space': color_space,
            'diff_pixels': int(diff_pixels),
            'total_pixels': int(total_pixels),
            'diff_ratio': float(diff_ratio),
            'mean_diff': mean_diff,
            'max_diff': max_diff
        },
        'visualization_data': {
            'diff_map': diff,
            'diff_gray': diff_gray,
            'thresh_map': thresh
        }
    }


# =============================================================================
# 통합 비교 함수
# =============================================================================

def compare_images_comprehensive(image1_obj, image2_obj, comparison_obj):
    """
    이미지 종합 비교 (파라미터 기반)
    
    Args:
        image1_obj: Image 인스턴스
        image2_obj: Image 인스턴스
        comparison_obj: ImageComparison 인스턴스
    
    Returns:
        dict: 비교 결과
    """
    start_time = time.time()
    
    # 이미지 로드
    img1 = cv2.imread(image1_obj.file.path)
    img2 = cv2.imread(image2_obj.file.path)
    
    if img1 is None or img2 is None:
        raise ValueError("이미지를 로드할 수 없습니다")
    
    # 비교 방법 및 파라미터
    method = comparison_obj.comparison_method
    params = comparison_obj.parameters
    
    # 방법별 비교 실행
    if method.category == 'feature':
        result = compare_with_features(img1, img2, params)
    elif method.category == 'structural':
        result = compare_with_ssim(img1, img2, params)
    elif method.category == 'histogram':
        result = compare_with_histogram(img1, img2, params)
    elif method.category == 'pixel':
        result = compare_with_pixel_diff(img1, img2, params)
    else:
        raise ValueError(f"지원하지 않는 카테고리: {method.category}")
    
    # 시각화 생성
    viz_types = params.get('visualization_types', ['main'])
    result_images = create_visualizations(
        img1, img2, 
        result, 
        viz_types, 
        method.category,
        comparison_obj
    )
    
    processing_time = time.time() - start_time
    
    return {
        'similarity_scores': {
            'overall': result['similarity'],
            'method_specific': result.get('details', {})
        },
        'feature_comparison_data': result.get('details', {}),
        'result_images': result_images,
        'processing_time': processing_time,
        'status': 'completed'
    }


# =============================================================================
# 시각화 생성
# =============================================================================

def create_visualizations(img1, img2, comparison_result, viz_types, category, comparison_obj):
    """
    비교 결과 시각화 생성
    
    Returns:
        list: [{'type': 'matches', 'path': '...'}, ...]
    """
    result_images = []
    viz_data = comparison_result.get('visualization_data', {})
    
    for viz_type in viz_types:
        if category == 'feature' and viz_type == 'matches':
            path = create_feature_match_viz(img1, img2, viz_data, comparison_obj)
            if path:
                result_images.append({'type': 'matches', 'path': path})
        
        elif category == 'structural' and viz_type == 'ssim_map':
            path = create_ssim_map_viz(img1, img2, viz_data, comparison_obj)
            if path:
                result_images.append({'type': 'ssim_map', 'path': path})
        
        # ✅ 수정: pixel 조건 제거 - 모든 카테고리에서 히트맵 생성 가능
        elif viz_type == 'diff_heatmap':
            path = create_diff_heatmap_viz(img1, img2, viz_data, comparison_obj)
            if path:
                result_images.append({'type': 'diff_heatmap', 'path': path})
        
        # ✅ 수정: 특징점 기반에서는 side_by_side skip
        elif viz_type == 'side_by_side':
            if category != 'feature':
                path = create_side_by_side_viz(img1, img2, comparison_obj)
                if path:
                    result_images.append({'type': 'side_by_side', 'path': path})
        
        # ✅ 추가: 차이 오버레이
        elif viz_type == 'overlay':
            path = create_overlay_viz(img1, img2, viz_data, comparison_obj)
            if path:
                result_images.append({'type': 'overlay', 'path': path})
        
        # ✅ 추가: 3분할 비교
        elif viz_type == 'triple':
            path = create_triple_viz(img1, img2, viz_data, comparison_obj)
            if path:
                result_images.append({'type': 'triple', 'path': path})
        
        # ✅ 추가: 차이 영역 박스 강조
        elif viz_type == 'highlighted':
            path = create_highlighted_diff_viz(img1, img2, viz_data, comparison_obj)
            if path:
                result_images.append({'type': 'highlighted', 'path': path})
    
    return result_images


def create_feature_match_viz(img1, img2, viz_data, comparison_obj):
    """특징점 매칭 시각화"""
    try:
        kp1 = viz_data.get('kp1', [])
        kp2 = viz_data.get('kp2', [])
        matches = viz_data.get('good_matches', [])
        
        if not matches:
            return None
        
        result = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # 텍스트 추가
        text = f"Matches: {len(matches)} | KP1: {len(kp1)}, KP2: {len(kp2)}"
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return save_result_image(result, comparison_obj, 'matches')
    except Exception as e:
        print(f"특징점 시각화 실패: {e}")
        return None


def create_ssim_map_viz(img1, img2, viz_data, comparison_obj):
    """SSIM 차이 맵 시각화"""
    try:
        diff_map = viz_data.get('diff_map')
        if diff_map is None:
            return None
        
        # 히트맵 적용
        heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
        
        return save_result_image(heatmap, comparison_obj, 'ssim_map')
    except Exception as e:
        print(f"SSIM 맵 시각화 실패: {e}")
        return None


def create_diff_heatmap_viz(img1, img2, viz_data, comparison_obj):
    """픽셀 차이 히트맵 시각화 (수정)"""
    try:
        import numpy as np
        
        # viz_data에 있으면 사용, 없으면 직접 계산
        diff_gray = viz_data.get('diff_gray')
        
        if diff_gray is None:
            # img1, img2는 이미 numpy 배열
            # 같은 크기로
            h = max(img1.shape[0], img2.shape[0])
            w = max(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (w, h))
            img2_resized = cv2.resize(img2, (w, h))
            
            # 차이 계산
            diff = cv2.absdiff(img1_resized, img2_resized)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 히트맵 적용
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)
        
        # 범례 추가
        legend_height = 50
        legend = np.zeros((legend_height, heatmap.shape[1], 3), dtype=np.uint8)
        
        # 색상 바
        for i in range(heatmap.shape[1]):
            color_val = int(255 * i / heatmap.shape[1])
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_HOT)[0][0]
            cv2.line(legend, (i, 10), (i, 30), color.tolist(), 1)
        
        # 텍스트
        cv2.putText(legend, "Low Difference", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, "High Difference", (heatmap.shape[1]-150, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 합치기
        result = np.vstack([heatmap, legend])
        
        return save_result_image(result, comparison_obj, 'diff_heatmap')
    except Exception as e:
        print(f"차이 히트맵 시각화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_side_by_side_viz(img1, img2, comparison_obj):
    """나란히 비교 시각화"""
    try:
        # 같은 높이로 리사이즈
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_h = max(h1, h2)
        
        img1_resized = cv2.resize(img1, (int(w1 * max_h / h1), max_h))
        img2_resized = cv2.resize(img2, (int(w2 * max_h / h2), max_h))
        
        # 나란히 배치
        result = np.hstack([img1_resized, img2_resized])
        
        # 구분선
        h, w = result.shape[:2]
        cv2.line(result, (w//2, 0), (w//2, h), (255, 255, 255), 3)
        
        return save_result_image(result, comparison_obj, 'side_by_side')
    except Exception as e:
        print(f"나란히 비교 시각화 실패: {e}")
        return None


def save_result_image(img, comparison_obj, viz_type):
    """결과 이미지 저장"""
    try:
        from django.utils import timezone
        now = timezone.now()
        
        filename = f"comparison_{comparison_obj.id}_{viz_type}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(
            settings.MEDIA_ROOT,
            'comparisons',
            str(now.year),
            str(now.month).zfill(2),
            filename
        )
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, img)
        
        return os.path.relpath(filepath, settings.MEDIA_ROOT)
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        return None
    

def create_overlay_viz(img1, img2, viz_data, comparison_obj):
    """차이 영역을 원본 위에 오버레이 (수정)"""
    try:
        import numpy as np
        
        # img1, img2는 이미 numpy 배열
        # 같은 크기로
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # 차이 계산
        diff = cv2.absdiff(img1_resized, img2_resized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 임계값 적용 (차이가 큰 부분만)
        _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # 빨간색 오버레이 생성
        overlay = img1_resized.copy()
        overlay[mask > 0] = [0, 0, 255]  # BGR: 빨간색
        
        # 블렌딩 (원본 70% + 오버레이 30%)
        result = cv2.addWeighted(img1_resized, 0.7, overlay, 0.3, 0)
        
        # 범례 추가
        cv2.rectangle(result, (10, 10), (250, 60), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (250, 60), (255, 255, 255), 2)
        cv2.putText(result, "Red Area = Different", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return save_result_image(result, comparison_obj, 'overlay')
    except Exception as e:
        print(f"오버레이 시각화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_triple_viz(img1, img2, viz_data, comparison_obj):
    """원본1 | 차이맵 | 원본2 3분할 표시 (수정)"""
    try:
        import numpy as np
        
        # img1, img2는 이미 numpy 배열
        # 크기 통일
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # 차이 계산
        diff = cv2.absdiff(img1_resized, img2_resized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)
        
        # 라벨 추가 함수
        def add_label(img, text, color=(255, 255, 255)):
            labeled = img.copy()
            # 배경
            cv2.rectangle(labeled, (0, 0), (w, 50), (0, 0, 0), -1)
            # 테두리
            cv2.rectangle(labeled, (0, 0), (w, 50), color, 3)
            # 텍스트
            cv2.putText(labeled, text, (15, 33), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            return labeled
        
        img1_labeled = add_label(img1_resized, "Image 1", (100, 200, 255))
        img2_labeled = add_label(img2_resized, "Image 2", (100, 255, 100))
        heatmap_labeled = add_label(heatmap, "Difference Map", (255, 100, 100))
        
        # 3개 이미지 가로로 연결
        result = np.hstack([img1_labeled, heatmap_labeled, img2_labeled])
        
        return save_result_image(result, comparison_obj, 'triple')
    except Exception as e:
        print(f"3분할 시각화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_highlighted_diff_viz(img1, img2, viz_data, comparison_obj):
    """차이가 있는 부분을 박스로 강조 (수정)"""
    try:
        import numpy as np
        
        # img1, img2는 이미 numpy 배열
        # 크기 통일
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # 차이 계산
        diff = cv2.absdiff(img1_resized, img2_resized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 차이 영역 찾기
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거 (모폴로지 연산)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # 이미지 2개 복사
        result1 = img1_resized.copy()
        result2 = img2_resized.copy()
        
        # 차이 영역에 박스 그리기
        diff_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 작은 노이즈 제거
                x, y, w_box, h_box = cv2.boundingRect(contour)
                # 박스 그리기
                cv2.rectangle(result1, (x, y), (x+w_box, y+h_box), 
                            (0, 0, 255), 3)
                cv2.rectangle(result2, (x, y), (x+w_box, y+h_box), 
                            (0, 0, 255), 3)
                # 번호 표시
                diff_count += 1
                cv2.putText(result1, str(diff_count), (x+5, y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(result2, str(diff_count), (x+5, y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 상단 라벨 배경
        cv2.rectangle(result1, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.rectangle(result2, (0, 0), (w, 50), (0, 0, 0), -1)
        
        # 라벨 추가
        cv2.putText(result1, f"Image 1 ({diff_count} differences)", 
                   (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2)
        cv2.putText(result2, f"Image 2 ({diff_count} differences)", 
                   (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
        
        # 2개 이미지 가로로 연결
        result = np.hstack([result1, result2])
        
        # 중앙 구분선
        cv2.line(result, (w, 0), (w, h), (255, 255, 255), 3)
        
        return save_result_image(result, comparison_obj, 'highlighted')
    except Exception as e:
        print(f"강조 시각화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
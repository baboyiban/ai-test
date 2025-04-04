import cv2
import numpy as np
import math

def merge_similar_lines(lines, angle_thresh=5.0, dist_thresh=20):
    """유사한 선분 통합 (병합)"""
    if lines is None or len(lines) < 2:
        return lines

    segments = np.array([line[0] for line in lines])
    angles = []
    centers = []

    for (x1, y1, x2, y2) in segments:
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
        centers.append(center)

    angles = np.array(angles)
    centers = np.array(centers)
    merged_lines = []
    used = [False] * len(segments)

    for i in range(len(segments)):
        if used[i]:
            continue

        similar_indices = [i]
        for j in range(i+1, len(segments)):
            if used[j]:
                continue

            angle_diff = abs(angles[i] - angles[j])
            angle_diff = min(angle_diff, 180 - angle_diff)
            center_dist = np.linalg.norm(centers[i] - centers[j])

            if angle_diff < angle_thresh and center_dist < dist_thresh:
                similar_indices.append(j)
                used[j] = True

        similar_lines = segments[similar_indices]
        x = np.concatenate([similar_lines[:, 0], similar_lines[:, 2]])
        y = np.concatenate([similar_lines[:, 1], similar_lines[:, 3]])

        if len(x) > 1:
            coefficients = np.polyfit(x, y, 1)
            x_min, x_max = min(x), max(x)
            y_min = np.polyval(coefficients, x_min)
            y_max = np.polyval(coefficients, x_max)
            merged_lines.append([x_min, y_min, x_max, y_max])

    return np.array(merged_lines).reshape(-1, 1, 4) if merged_lines else None

def filter_by_lane_properties(lines, img_shape, min_angle=15, max_angle=75):
    """차선 특성에 맞는 선분 필터링"""
    if lines is None:
        return None

    height, width = img_shape[:2]
    filtered_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if abs(x1 - x2) < 5:
            continue

        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = abs(angle)

        if min_angle <= angle <= max_angle:
            if max(y1, y2) > height * 0.7:
                filtered_lines.append(line)

    return np.array(filtered_lines) if filtered_lines else None

def advanced_lane_detection(image_path):
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    if img is None:
        print("이미지 로드 실패")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI 설정 (하단 50%만 처리)
    height, width = edges.shape
    mask = np.zeros_like(edges)

    vertices = np.array([
        [[0, height], [width//2, height//2], [width, height]]
    ], dtype=np.int32)

    # drawContours로 변경 (안정적인 구현)
    cv2.drawContours(mask, [vertices], -1, 255, thickness=cv2.FILLED)

    masked_edges = cv2.bitwise_and(edges, mask)

    # 허프 변환
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=30,
        maxLineGap=20
    )

    # 선분 후처리 및 시각화
    if lines is not None:
        merged_lines = merge_similar_lines(lines)
        filtered_lines = filter_by_lane_properties(merged_lines, img.shape)

        line_img = np.zeros_like(img)
        if filtered_lines is not None:
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
        debug = np.hstack([img, result])
        cv2.imshow('Debug', cv2.resize(debug, (width//2, height//2)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("차선을 찾을 수 없습니다.")

if __name__ == "__main__":
    advanced_lane_detection("../image/image_2.webp")

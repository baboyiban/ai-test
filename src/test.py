"""
차선 인식 시스템 (Lane Detection System)
- 이미지/실시간 영상에서 차선을 검출하는 파이프라인
- 주요 처리 단계: 전처리 → 엣지 검출 → 선분 검출 → 차선 필터링 → 시각화
"""

import math
import cv2 as cv

def preprocess_image(img):
    """
    이미지 전처리 함수
    - 입력: BGR 컬러 이미지
    - 출력: 그레이스케일 + 가우시안 블러 적용 이미지
    """
    if img is None:
        return None

    # 고정 해상도 설정 (주석 해제시 사용)
    # img = cv.resize(img, (640, 480))

    # 컬러 → 그레이스케일 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 가우시안 블러로 노이즈 제거 (커널 크기 7x7)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    return blurred

def detect_edges(img):
    """
    Canny 엣지 검출
    - 입력: 전처리된 그레이스케일 이미지
    - 출력: 엣지 마스크 (이진 이미지)
    """
    edges = cv.Canny(img, 40, 100)  # 낮은/높은 임계값 설정
    cv.imshow('Edges', edges)  # 엣지 결과 실시간 표시
    return edges

def find_lines(edges):
    """
    Hough 변환을 이용한 선분 검출
    - 입력: 엣지 이미지
    - 출력: 검출된 선분 리스트 (Nx1x4 배열)
    """
    lines = cv.HoughLinesP(
        edges,
        rho=1,                  # 거리 해상도 (픽셀 단위)
        theta=math.pi/180,      # 각도 해상도 (라디안)
        threshold=80,           # 투표(vote) 임계값
        minLineLength=20,       # 최소 선분 길이
        maxLineGap=20           # 최대 허용 간격
    )
    return lines

def visualize_lines(edges, lines, color=(255, 0, 255)):
    """
    검출된 선분 시각화
    - 입력: 엣지 이미지, 선분 리스트
    - 출력: 선분이 그려진 컬러 이미지
    """
    if lines is None:
        return edges

    # 그레이스케일 → BGR 변환 (컬러 표시용)
    edges_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # 모든 선분 그리기
    for line in lines:
        x1, y1, x2, y2 = line[0]  # 선분의 시작점/끝점 좌표
        cv.line(edges_color, (x1, y1), (x2, y2), color, 3, cv.LINE_AA)

    cv.imshow('Detected Lines', edges_color)
    return edges_color

def filter_lanes(lines, min_angle=10):
    """
    차선 후보 필터링
    - 입력: 모든 선분 리스트
    - 출력: 차선으로 판단된 선분 리스트
    """
    if lines is None:
        return None

    lanes = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 수직선 처리 (분모 0 방지)
        if x1 == x2:
            lanes.append([x1, y1, x2, y2])
            continue

        # 각도 계산 (라디안 → 도)
        angle = math.atan2(y2-y1, x2-x1) * 180 / math.pi

        # 경사각이 큰 선분만 선택 (차선은 일반적으로 10도 이상)
        if abs(angle) > min_angle:
            lanes.append([x1, y1, x2, y2])

    return lanes

def visualize_lanes(img, lanes, color=(0, 255, 0)):
    """
    최종 차선 시각화
    - 입력: 원본 이미지, 차선 리스트
    - 출력: 차선이 표시된 원본 이미지
    """
    if lanes is not None:
        for lane in lanes:
            x1, y1, x2, y2 = lane
            cv.line(img, (x1, y1), (x2, y2), color, 3, cv.LINE_AA)

    cv.imshow('Detected Lanes', img)

def lane_detection_pipeline(image_path):
    """
    정적 이미지 차선 인식 메인 파이프라인
    - 처리 단계: 전처리 → 엣지 → 선분 → 필터링 → 결과
    """
    # 이미지 로드
    img = cv.imread(image_path)
    if img is None:
        print("이미지를 로드할 수 없습니다.")
        return

    cv.imshow('Original Image', img)

    # 파이프라인 실행
    blurred = preprocess_image(img)          # 1. 전처리
    edges = detect_edges(blurred)            # 2. 엣지 검출
    lines = find_lines(edges)                # 3. 선분 검출
    line_visualization = visualize_lines(edges, lines)  # 4. 시각화
    lanes = filter_lanes(lines)              # 5. 차선 필터링
    visualize_lanes(img.copy(), lanes)       # 6. 최종 결과

    cv.waitKey(0)
    cv.destroyAllWindows()

def realtime_lane_detection():
    """
    실시간 차선 인식 (웹캠 사용)
    - ESC 키로 종료
    """
    cap = cv.VideoCapture(0)  # 0 = 기본 웹캠

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 파이프라인 실행 (정적 이미지와 동일)
        blurred = preprocess_image(frame)
        edges = detect_edges(blurred)
        lines = find_lines(edges)
        lanes = filter_lanes(lines)
        visualize_lanes(frame, lanes)

        # 'q' 키로 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # 정적 이미지 테스트
    image_url = '../image/image_2.webp'
    lane_detection_pipeline(image_url)

    # 실시간 테스트 (주석 해제시 사용)
    # realtime_lane_detection()

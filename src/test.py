import math
import cv2 as cv

def preprocess_image(img):
    """이미지를 그레이스케일로 변환하고 가우시안 블러 적용"""
    if img is None:
        return None

    img = cv.resize(img, (640, 480))  # 고정 해상도
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    return blurred

def detect_edges(img):
    """Canny 엣지 검출 수행"""
    edges = cv.Canny(img, 40, 100)
    cv.imshow('Edges', edges)
    return edges

def find_lines(edges):
    """Hough 변환을 사용하여 선분 검출"""
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi/180,
        threshold=80,
        minLineLength=20,
        maxLineGap=20
    )
    return lines

def visualize_lines(edges, lines, color=(255, 0, 255)):
    """검출된 선분을 시각화"""
    if lines is None:
        return edges

    edges_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(edges_color, (x1, y1), (x2, y2), color, 3, cv.LINE_AA)

    cv.imshow('Detected Lines', edges_color)
    return edges_color

def filter_lanes(lines, min_angle=10):
    """검출된 선분 중 차선으로 판단될만한 것만 필터링"""
    if lines is None:
        return None

    lanes = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 수직선 처리
        if x1 == x2:
            lanes.append([x1, y1, x2, y2])
            continue

        # 각도 계산 및 필터링
        angle = math.atan2(y2-y1, x2-x1) * 180 / math.pi
        if abs(angle) > min_angle:
            lanes.append([x1, y1, x2, y2])

    return lanes

def visualize_lanes(img, lanes, color=(0, 255, 0)):
    """최종 차선을 원본 이미지에 표시"""
    if lanes is not None:
        for lane in lanes:
            x1, y1, x2, y2 = lane
            cv.line(img, (x1, y1), (x2, y2), color, 3, cv.LINE_AA)

    cv.imshow('Detected Lanes', img)

def lane_detection_pipeline(image_path):
    """전체 차선 인식 파이프라인 실행"""
    # 이미지 로드
    img = cv.imread(image_path)
    if img is None:
        print("이미지를 로드할 수 없습니다.")
        return

    cv.imshow('Original Image', img)

    # 1. 이미지 전처리
    blurred = preprocess_image(img)

    # 2. 엣지 검출
    edges = detect_edges(blurred)

    # 3. 선분 검출
    lines = find_lines(edges)

    # 4. 검출된 선분 시각화
    line_visualization = visualize_lines(edges, lines)

    # 5. 차선 필터링
    lanes = filter_lanes(lines)

    # 6. 최종 차선 시각화
    visualize_lanes(img.copy(), lanes)

    cv.waitKey(0)
    cv.destroyAllWindows()

# 메인 실행
if __name__ == "__main__":
    image_url = './../image/image_1.webp'
    lane_detection_pipeline(image_url)
    # realtime_lane_detection()  # ← 이 함수로 변경

# 추가 코드
def realtime_lane_detection():
    cap = cv.VideoCapture(0)  # 웹캠 사용시

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 이미지 전처리
        blurred = preprocess_image(frame)

        # 2~5. 차선 검출 과정
        edges = detect_edges(blurred)
        lines = find_lines(edges)
        lanes = filter_lanes(lines)

        # 6. 결과 시각화
        visualize_lanes(frame, lanes)

        if cv.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            break

    cap.release()
    cv.destroyAllWindows()

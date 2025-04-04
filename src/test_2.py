import cv2
import numpy as np
import os

def process_frame(frame):
    """차선 검출을 위한 프레임 처리"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ROI 설정 (하단 50%)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([
        [0, height],
        [width, height],
        [width, height//2],
        [0, height//2]
    ], dtype=np.int32)

    # fillPoly 호환성 개선
    cv2.fillPoly(mask, [polygon], color=(255,))  # color를 튜플로 지정

    # 선분 검출 및 차선 표시
    lines = cv2.HoughLinesP(
        image=cv2.bitwise_and(edges, mask),
        rho=2,
        theta=np.pi/180,
        threshold=50,
        minLineLength=40.0,  # float으로 전달
        maxLineGap=100.0     # float으로 전달
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return frame

def save_processed_video(input_path, output_path):
    """동영상 처리 및 저장"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("동영상 열기 실패")

    # 출력 설정
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc 코드 직접 사용 (MP4V)
    fourcc_code = 0x7634706D
    out = cv2.VideoWriter(output_path, fourcc_code, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise IOError("출력 파일 생성 실패")

    print(f"처리 시작: {input_path} -> {output_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(process_frame(frame))
    finally:
        cap.release()
        out.release()
        print("처리 완료")

if __name__ == "__main__":
    # 절대 경로 사용 권장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(current_dir, "../video/video_1.mp4")  # 원본 동영상
    output_video = os.path.join(current_dir, "output.mp4")  # 결과물

    try:
        save_processed_video(input_video, output_video)
    except Exception as e:
        print(f"오류 발생: {str(e)}")

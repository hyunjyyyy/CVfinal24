import cv2
import dlib
import numpy as np

# EAR 계산 함수
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

# 시선 방향 계산 함수
def calculate_gaze_direction(eye_points, landmarks):
    eye_center = np.mean(eye_points, axis=0).astype(int)
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])  # 코 끝 (랜드마크 30번)
    dx = eye_center[0] - nose_tip[0]
    dy = eye_center[1] - nose_tip[1]

    if abs(dx) > abs(dy):
        return "Left" if dx < 0 else "Right"
    else:
        return "Up" if dy < 0 else "Down"

# 입 벌림 비율(MAR) 계산 함수
def calculate_mar(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # 상하 좌표
    B = np.linalg.norm(mouth_points[4] - mouth_points[8])   # 좌우 좌표
    C = np.linalg.norm(mouth_points[0] - mouth_points[6])   # 입 너비
    return (A + B) / (2.0 * C)

# EAR 및 MAR 임계값 및 프레임 제한값
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CONSECUTIVE_FRAMES = 3
EYE_CLOSED_WARNING_FRAMES = 90  # 3초 (30 FPS 기준)

# Dlib 얼굴 탐지 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 및 입 랜드마크 인덱스
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 68))

# 깜빡임 및 하품 횟수 초기화
left_blink_count = 0
right_blink_count = 0
left_frame_counter = 0
right_frame_counter = 0
yawn_count = 0
yawn_frame_counter = 0
eye_closed_frame_counter = 0

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    eye_closed = False

    for face in faces:
        landmarks = predictor(gray, face)
        
        # 왼쪽 눈과 오른쪽 눈 좌표 가져오기
        left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in LEFT_EYE_POINTS])
        right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in RIGHT_EYE_POINTS])

        # EAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # 눈 영역 표시
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # 왼쪽 눈 깜빡임 감지
        if left_ear < EAR_THRESHOLD:
            left_frame_counter += 1
        else:
            if left_frame_counter >= CONSECUTIVE_FRAMES:
                left_blink_count += 1
                print(f"Left Eye Blink Count: {left_blink_count}")
            left_frame_counter = 0

        # 오른쪽 눈 깜빡임 감지
        if right_ear < EAR_THRESHOLD:
            right_frame_counter += 1
        else:
            if right_frame_counter >= CONSECUTIVE_FRAMES:
                right_blink_count += 1
                print(f"Right Eye Blink Count: {right_blink_count}")
            right_frame_counter = 0

        # 눈 감음 상태 확인
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            eye_closed = True
            eye_closed_frame_counter += 1
        else:
            eye_closed_frame_counter = 0

        # 경고: 3초 이상 눈을 감고 있는 경우
        if eye_closed_frame_counter >= EYE_CLOSED_WARNING_FRAMES:
            cv2.putText(frame, "WARNING: Eyes closed for too long!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 입 좌표 가져오기
        mouth = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in MOUTH_POINTS])

        # MAR 계산
        mar = calculate_mar(mouth)

        cv2.polylines(frame, [mouth], True, (0, 0, 255), 1)

        # 하품 감지
        if mar > MAR_THRESHOLD:
            yawn_frame_counter += 1
        else:
            if yawn_frame_counter >= CONSECUTIVE_FRAMES:
                yawn_count += 1
                print(f"Yawn Count: {yawn_count}")
            yawn_frame_counter = 0

        # 입 영역 표시
        cv2.polylines(frame, [mouth], True, (255, 0, 0), 1)

        # 시선 방향 계산
        left_gaze = calculate_gaze_direction(left_eye, landmarks)
        right_gaze = calculate_gaze_direction(right_eye, landmarks)
        gaze_text = f"Left Eye: {left_gaze}, Right Eye: {right_gaze}"

        # 시선 방향 텍스트 표시
        cv2.putText(frame, gaze_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 결과 화면 출력
    cv2.putText(frame, f"Left Blinks: {left_blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Right Blinks: {right_blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Yawns: {yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Driver Safety System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

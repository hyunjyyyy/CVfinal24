import cv2
import dlib
import numpy as np
from sleep_detect import calculate_ear, calculate_mar, detect_blink_and_yawn
import time

# Dlib 얼굴 탐지 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 및 입 랜드마크 인덱스
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 68))

# 초기 thresholds 설정
thresholds = {
    'EAR_THRESHOLD': 0.2, 
    'MAR_THRESHOLD': 0.5,  
    'CONSECUTIVE_FRAMES': 3,
    'EYE_CLOSED_WARNING_FRAMES': 60,
    'BLINK_COUNT_WARNING_THRESHOLD': 5
}

# 상태 변수 초기화
state = {
    "left_blink_count": 0,
    "right_blink_count": 0,
    "yawn_count": 0,
    "left_eye_frames": 0,
    "right_eye_frames": 0,
    "eye_closed_frames": 0,
    "yawn_frames": 0,
    "eye_closed_warning_duration": 0,
    "blink_warning_duration": 0,
    "yawn_warning_duration": 0
}

# 웹캠 열기
cap = cv2.VideoCapture(0)

# **초기 5초 동안 사용자 EAR 평균값 계산**
print("Calibrating... Please keep your eyes open naturally for 5 seconds.")
start_time = time.time()
ear_values = []

while time.time() - start_time < 5:  # 5초 동안 캘리브레이션
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 눈 좌표 가져오기
        left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in LEFT_EYE_POINTS])
        right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in RIGHT_EYE_POINTS])

        # EAR 계산 및 저장
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        ear_values.append(avg_ear)

    cv2.putText(frame, "Calibrating... Keep eyes open.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

# 사용자 평균 EAR 값을 기준으로 새로운 임계값 설정
if ear_values:
    thresholds['EAR_THRESHOLD'] = np.mean(ear_values) * 0.75  # 평균 EAR의 75%를 임계값으로 설정
    print(f"Calibration complete. EAR_THRESHOLD set to: {thresholds['EAR_THRESHOLD']:.2f}")

cv2.destroyAllWindows()

# 메인 루프 시작
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 눈과 입 좌표 가져오기
        left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in LEFT_EYE_POINTS])
        right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in RIGHT_EYE_POINTS])
        mouth = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in MOUTH_POINTS])

        # EAR 및 MAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        mar = calculate_mar(mouth)

        # 눈 및 입 영역 강조 표시
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)  # 왼쪽 눈
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)  # 오른쪽 눈
        cv2.polylines(frame, [mouth], True, (0, 0, 255), 2)  # 입

        # EAR 및 MAR 값 표시
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 깜빡임 및 하품 감지
        eye_closed_warning, blink_warning, yawn_warning = detect_blink_and_yawn(
            left_ear, right_ear, mar, thresholds, state
        )

        # 경고 메시지 표시
        if eye_closed_warning:
            cv2.putText(frame, eye_closed_warning, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if blink_warning:
            cv2.putText(frame, blink_warning, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if yawn_warning:
            cv2.putText(frame, yawn_warning, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # 깜빡임 및 하품 카운트 출력
        cv2.putText(frame, f"Left Blinks: {state['left_blink_count']}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Right Blinks: {state['right_blink_count']}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Yawns: {state['yawn_count']}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 화면 출력
    cv2.imshow("Driver Safety System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

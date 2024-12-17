import time
import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from sleep_detect import SleepDetector

# 초기화
thresholds = {
    'EAR_THRESHOLD': 0.2,  # 기본값 (캘리브레이션으로 수정)
    'MAR_THRESHOLD': 0.5,  # 기본값
    'CONSECUTIVE_FRAMES': 3
}

LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 68))

# SleepDetector 클래스 초기화
sleep_detector = SleepDetector(thresholds)

# YOLO 및 Dlib 초기화
model = YOLO('yolov8_best_4.pt')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def calibrate_threshold(cap, detector, predictor, sleep_detector, duration=5):
    """EAR 및 MAR 임계값을 동시에 캘리브레이션하는 함수"""
    ear_values = []
    mar_values = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # 왼쪽 및 오른쪽 눈 EAR 계산
            left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in LEFT_EYE_POINTS])
            right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in RIGHT_EYE_POINTS])

            left_ear = sleep_detector.calculate_ear(left_eye)
            right_ear = sleep_detector.calculate_ear(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)

            # 입 (MOUTH) MAR 계산
            mouth = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in MOUTH_POINTS])
            mar = sleep_detector.calculate_mar(mouth)
            mar_values.append(mar)

        # 캘리브레이션 메시지 출력
        cv2.putText(frame, "Calibrating... Keep eyes open and mouth closed.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

    # EAR 임계값 설정
    if ear_values:
        new_ear_threshold = np.mean(ear_values) * 0.7  # 비율을 0.7로 설정하여 EAR 임계값 조정
        sleep_detector.set_ear_threshold(new_ear_threshold)
        print(f"Calibration complete. EAR_THRESHOLD set to: {new_ear_threshold:.2f}")

    # MAR 임계값 설정
    if mar_values:
        new_mar_threshold = np.mean(mar_values) * 1.5  # 비율을 1.5배로 설정하여 MAR 임계값 조정
        sleep_detector.set_mar_threshold(new_mar_threshold)
        print(f"MAR Calibration complete. MAR_THRESHOLD set to: {new_mar_threshold:.2f}")

    # 캘리브레이션 창 닫기
    cv2.destroyWindow("Calibration")


def process_frame(frame, detector, predictor, sleep_detector, model, show_values=True):
    """프레임 처리 함수"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 눈 및 입 좌표 계산
        left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in LEFT_EYE_POINTS])
        right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in RIGHT_EYE_POINTS])
        mouth = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in MOUTH_POINTS])

        # EAR 및 MAR 계산
        left_ear = sleep_detector.calculate_ear(left_eye)
        right_ear = sleep_detector.calculate_ear(right_eye)
        mar = sleep_detector.calculate_mar(mouth)

        # EAR과 MAR의 평균 값 계산
        avg_ear = (left_ear + right_ear) / 2.0

        # 눈 및 입 표시
        cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)
        cv2.polylines(frame, [mouth], True, (255, 0, 255), 1)

        # 경고 감지
        warnings = sleep_detector.detect(left_ear, right_ear, mar)

        # 경고 메시지 출력
        for i, (key, warning) in enumerate(warnings.items()):
            if warning:
                cv2.putText(frame, warning, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # EAR 및 MAR 값 화면에 표시
        if show_values:
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # YOLO 객체 탐지
    results = model(frame, stream=True)
    for result in results:
        for box, cls_id in zip(result.boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(cls_id)

            if conf > 0.5:
                class_name = model.names[cls_id] if cls_id in model.names else "Unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    calibrate_threshold(cap, detector, predictor, sleep_detector, duration=5)

    show_values = True  # EAR과 MAR 값을 표시할지 여부
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        frame = process_frame(frame, detector, predictor, sleep_detector, model, show_values)
        cv2.imshow("Driver Safety System", frame)

        # 'i' 키로 EAR, MAR 값 표시 토글
        if cv2.waitKey(1) & 0xFF == ord('i'):
            show_values = not show_values

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

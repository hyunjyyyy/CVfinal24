import numpy as np

# EAR 계산 함수
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

# 입 벌림 비율(MAR) 계산 함수
def calculate_mar(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # 상하 좌표
    B = np.linalg.norm(mouth_points[4] - mouth_points[8])   # 좌우 좌표
    C = np.linalg.norm(mouth_points[0] - mouth_points[6])   # 입 너비
    return (A + B) / (2.0 * C)

# 눈 깜빡임 감지 및 경고 처리 함수
def detect_blink_and_yawn(left_ear, right_ear, mar, thresholds, state):
    eye_closed_warning = ""
    blink_warning = ""
    yawn_warning = ""

    # 눈 감은 상태 확인
    if left_ear < thresholds['EAR_THRESHOLD'] and right_ear < thresholds['EAR_THRESHOLD']:
        state["eye_closed_frames"] += 1
    else:
        if state["eye_closed_frames"] >= thresholds['EYE_CLOSED_WARNING_FRAMES']:
            state["eye_closed_warning_duration"] = 60  # 경고 유지 프레임 수
        state["eye_closed_frames"] = 0

    # 경고 지속
    if state.get("eye_closed_warning_duration", 0) > 0:
        eye_closed_warning = "WARNING: Eyes closed for too long!"
        state["eye_closed_warning_duration"] -= 1

    # 왼쪽 눈 깜빡임 감지
    if left_ear < thresholds['EAR_THRESHOLD']:
        state["left_eye_frames"] += 1
    else:
        if state["left_eye_frames"] >= thresholds['CONSECUTIVE_FRAMES']:
            state["left_blink_count"] += 1
            state["blink_warning_duration"] = 60
        state["left_eye_frames"] = 0

    # 오른쪽 눈 깜빡임 감지
    if right_ear < thresholds['EAR_THRESHOLD']:
        state["right_eye_frames"] += 1
    else:
        if state["right_eye_frames"] >= thresholds['CONSECUTIVE_FRAMES']:
            state["right_blink_count"] += 1
            state["blink_warning_duration"] = 60
        state["right_eye_frames"] = 0

    # 깜빡임 경고 지속
    if state.get("blink_warning_duration", 0) > 0:
        blink_warning = "WARNING: Unusual blink rate detected!"
        state["blink_warning_duration"] -= 1

    # 하품 감지
    if mar > thresholds['MAR_THRESHOLD']:
        state["yawn_frames"] += 1
    else:
        if state["yawn_frames"] >= thresholds['CONSECUTIVE_FRAMES']:
            state["yawn_count"] += 1
            state["yawn_warning_duration"] = 60
        state["yawn_frames"] = 0

    # 하품 경고 지속
    if state.get("yawn_warning_duration", 0) > 0:
        yawn_warning = "WARNING: Yawning detected!"
        state["yawn_warning_duration"] -= 1

    return eye_closed_warning, blink_warning, yawn_warning

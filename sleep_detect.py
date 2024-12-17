import numpy as np

class SleepDetector:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.state = {
            "eye_closed_frames": 0,
            "left_eye_frames": 0,
            "right_eye_frames": 0,
            "yawn_frames": 0,
            "left_blink_count": 0,
            "right_blink_count": 0,
            "yawn_count": 0,
            "eye_closed_warning_duration": 0,
            "blink_warning_duration": 0,
            "yawn_warning_duration": 0
        }

    def set_ear_threshold(self, new_threshold):
        """EAR 임계값을 설정하는 메소드"""
        self.thresholds['EAR_THRESHOLD'] = new_threshold
        print(f"Updated EAR_THRESHOLD: {new_threshold:.2f}")

    def set_mar_threshold(self, new_threshold):
        """MAR 임계값을 설정하는 메소드"""
        self.thresholds['MAR_THRESHOLD'] = new_threshold
        print(f"Updated MAR_THRESHOLD: {new_threshold:.2f}")
    
    def calculate_ear(self, eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth_points):
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # 상하
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])   # 좌우
        C = np.linalg.norm(mouth_points[0] - mouth_points[6])   # 입 너비
        return (A + B) / (2.0 * C)

    def detect(self, left_ear, right_ear, mar):
        warnings = {"eye_closed": "", "blink": "", "yawn": ""}
        
        # 눈 감김 상태
        if left_ear < self.thresholds['EAR_THRESHOLD'] and right_ear < self.thresholds['EAR_THRESHOLD']:
            self.state["eye_closed_frames"] += 1
        else:
            if self.state["eye_closed_frames"] >= self.thresholds['CONSECUTIVE_FRAMES']:
                self.state["eye_closed_warning_duration"] = 60
            self.state["eye_closed_frames"] = 0

        if self.state["eye_closed_warning_duration"] > 0:
            warnings["eye_closed"] = "WARNING: Eyes closed for too long!"
            self.state["eye_closed_warning_duration"] -= 1

        # 눈 깜빡임
        self._detect_blink(left_ear, right_ear, warnings)

        # 하품 감지
        if mar > self.thresholds['MAR_THRESHOLD']:
            self.state["yawn_frames"] += 1
        else:
            if self.state["yawn_frames"] >= self.thresholds['CONSECUTIVE_FRAMES']:
                self.state["yawn_count"] += 1
                self.state["yawn_warning_duration"] = 60
            self.state["yawn_frames"] = 0

        if self.state["yawn_warning_duration"] > 0:
            warnings["yawn"] = "WARNING: Yawning detected!"
            self.state["yawn_warning_duration"] -= 1

        return warnings

    def _detect_blink(self, left_ear, right_ear, warnings):
        if left_ear < self.thresholds['EAR_THRESHOLD']:
            self.state["left_eye_frames"] += 1
        else:
            if self.state["left_eye_frames"] >= self.thresholds['CONSECUTIVE_FRAMES']:
                self.state["left_blink_count"] += 1
                self.state["blink_warning_duration"] = 60
            self.state["left_eye_frames"] = 0

        if right_ear < self.thresholds['EAR_THRESHOLD']:
            self.state["right_eye_frames"] += 1
        else:
            if self.state["right_eye_frames"] >= self.thresholds['CONSECUTIVE_FRAMES']:
                self.state["right_blink_count"] += 1
                self.state["blink_warning_duration"] = 60
            self.state["right_eye_frames"] = 0

        if self.state["blink_warning_duration"] > 0:
            warnings["blink"] = "WARNING: Unusual blink rate detected!"
            self.state["blink_warning_duration"] -= 1

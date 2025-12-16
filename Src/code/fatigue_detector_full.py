import os
import cv2
import time
import csv
import math
import numpy as np
from collections import deque

# ---------- OPTIONAL GPIO (Jetson) ----------
# On Windows this will be ignored safely.
GPIO_AVAILABLE = False
try:
    import Jetson.GPIO as GPIO  # only works on Jetson
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False

# ---------- PATHS ----------
LOG_DIR = r"C:\Users\asus\Desktop\fatigue\logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "fatigue_log.csv")

# ---------- THRESHOLDS / SETTINGS ----------
EAR_THRESH = 0.22          # lower = more strict
MAR_THRESH = 0.60          # mouth open threshold (tune if needed)
YAWN_SECONDS = 2.0         # mouth open for >= this = yawn
PERCLOS_WINDOW_SEC = 30.0  # use 30s window (common); you can set 60
PERCLOS_THRESH = 0.40      # fatigue if >= 40% eyes closed in window
PITCH_DOWN_THRESH = 18.0   # degrees (head looking down) for fatigue hint
FATIGUE_HOLD_SEC = 2.0     # how long condition must persist to trigger alert

# ---------- Jetson GPIO SETTINGS ----------
# Change pins later on Jetson side only.
GPIO_PIN = 18  # example physical pin mapping depends on mode
GPIO_MODE = None  # will set if available
GPIO_SET = False

def setup_gpio():
    global GPIO_SET
    if not GPIO_AVAILABLE:
        return
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO_SET = True
    except Exception:
        GPIO_SET = False

def gpio_alert(on: bool):
    if not GPIO_AVAILABLE or not GPIO_SET:
        return
    GPIO.output(GPIO_PIN, GPIO.HIGH if on else GPIO.LOW)

# ---------- MEDIAPIPE FACE MESH ----------
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# FaceMesh landmark indices (commonly used)
# EAR needs 6 points per eye
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# MAR needs mouth vertical + horizontal
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Head pose points (2D from FaceMesh)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# 3D model points (generic face model)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

def dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def ear_from_points(pts):
    # pts = [p1,p2,p3,p4,p5,p6]
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = pts
    A = dist(p2, p6)
    B = dist(p3, p5)
    C = dist(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mar_from_points(m_left, m_right, m_top, m_bottom):
    # MAR = vertical / horizontal
    horiz = dist(m_left, m_right)
    vert = dist(m_top, m_bottom)
    if horiz == 0:
        return 0.0
    return vert / horiz

def get_angles_from_pnp(image_points_2d, frame_w, frame_h):
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion
    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch = float(euler[0])
    yaw = float(euler[1])
    roll = float(euler[2])
    return pitch, yaw, roll

# ---------- METRICS BUFFERS ----------
# Store (timestamp, bool) for eyes closed / mouth open
eye_state = deque()
mouth_state = deque()

blink_count_times = deque()  # timestamps when blink detected
yawn_active = False
yawn_start_time = 0.0

fatigue_start = None  # for "hold" logic
alert_on = False

def prune_deque(dq, now, window_sec):
    while dq and (now - dq[0][0]) > window_sec:
        dq.popleft()

def prune_times(dq, now, window_sec):
    while dq and (now - dq[0]) > window_sec:
        dq.popleft()

def perclos_from_states(dq):
    if not dq:
        return 0.0
    closed = sum(1 for _, v in dq if v)
    return closed / len(dq)

def rate_from_states(dq):
    if not dq:
        return 0.0
    opened = sum(1 for _, v in dq if v)
    return opened / len(dq)

def write_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "ear", "mar",
                "eyes_closed", "mouth_open",
                "perclos", "mor",
                "blink_per_min",
                "yawn_active", "yawn_seconds",
                "pitch", "yaw", "roll",
                "fatigue_flag"
            ])

def append_log(row):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def main():
    global yawn_active, yawn_start_time, fatigue_start, alert_on

    setup_gpio()
    write_log_header()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not opening. Close any app using camera (Zoom/Teams), then try again.")
        return

    print("Webcam opened. Starting FULL detector (EAR + PERCLOS + MAR + HeadPose + Fusion + Logs)...")

    prev_eye_closed = False  # for blink detection
    last_frame_time = time.time()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            ear = 0.0
            mar = 0.0
            pitch = yaw = roll = 0.0
            eyes_closed = False
            mouth_open = False
            fatigue_flag = False
            yawn_seconds = 0.0

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                def lm_xy(i):
                    return (int(lm[i].x * w), int(lm[i].y * h))

                # ----- EAR -----
                left_eye_pts = [lm_xy(i) for i in LEFT_EYE_IDX]
                right_eye_pts = [lm_xy(i) for i in RIGHT_EYE_IDX]
                ear_left = ear_from_points(left_eye_pts)
                ear_right = ear_from_points(right_eye_pts)
                ear = (ear_left + ear_right) / 2.0
                eyes_closed = (ear < EAR_THRESH)

                # ----- MAR -----
                m_left = lm_xy(MOUTH_LEFT)
                m_right = lm_xy(MOUTH_RIGHT)
                m_top = lm_xy(MOUTH_TOP)
                m_bottom = lm_xy(MOUTH_BOTTOM)
                mar = mar_from_points(m_left, m_right, m_top, m_bottom)
                mouth_open = (mar > MAR_THRESH)

                # ----- Head Pose (PnP) -----
                image_points = np.array([
                    lm_xy(NOSE_TIP),
                    lm_xy(CHIN),
                    lm_xy(LEFT_EYE_OUTER),
                    lm_xy(RIGHT_EYE_OUTER),
                    lm_xy(LEFT_MOUTH),
                    lm_xy(RIGHT_MOUTH)
                ], dtype=np.float64)
                pitch, yaw, roll = get_angles_from_pnp(image_points, w, h)

                # ----- Update rolling windows -----
                eye_state.append((now, eyes_closed))
                mouth_state.append((now, mouth_open))
                prune_deque(eye_state, now, PERCLOS_WINDOW_SEC)
                prune_deque(mouth_state, now, PERCLOS_WINDOW_SEC)

                perclos = perclos_from_states(eye_state)   # % eyes closed
                mor = rate_from_states(mouth_state)        # % mouth open (rate)

                # ----- Blink detection (simple) -----
                # Blink = transition open->closed->open (use ear threshold)
                if (not prev_eye_closed) and eyes_closed:
                    # started closing
                    prev_eye_closed = True
                elif prev_eye_closed and (not eyes_closed):
                    # reopened => blink
                    prev_eye_closed = False
                    blink_count_times.append(now)

                prune_times(blink_count_times, now, 60.0)
                blink_per_min = len(blink_count_times)

                # ----- Yawn duration -----
                if mouth_open and (not yawn_active):
                    yawn_active = True
                    yawn_start_time = now
                elif (not mouth_open) and yawn_active:
                    yawn_active = False
                    yawn_start_time = 0.0

                if yawn_active:
                    yawn_seconds = now - yawn_start_time

                yawn_detected = (yawn_seconds >= YAWN_SECONDS)

                # ----- Fusion rule (edit as you like) -----
                # Baseline fatigue: high perclos OR long eye closure hint OR head pitch down
                head_down = (pitch > PITCH_DOWN_THRESH)

                fatigue_flag = (perclos >= PERCLOS_THRESH) or yawn_detected or head_down

                # Hold logic to prevent false triggers
                if fatigue_flag:
                    if fatigue_start is None:
                        fatigue_start = now
                    if (now - fatigue_start) >= FATIGUE_HOLD_SEC:
                        alert_on = True
                else:
                    fatigue_start = None
                    alert_on = False

                gpio_alert(alert_on)

                # ----- Draw overlays -----
                cv2.putText(frame, f"EAR: {ear:.3f} ({'CLOSED' if eyes_closed else 'OPEN'})",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(frame, f"PERCLOS({int(PERCLOS_WINDOW_SEC)}s): {perclos:.2f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(frame, f"MAR: {mar:.3f} ({'OPEN' if mouth_open else 'CLOSED'})",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                cv2.putText(frame, f"YawnSec: {yawn_seconds:.1f}  Blink/min: {blink_per_min}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                cv2.putText(frame, f"Pitch:{pitch:.1f} Yaw:{yaw:.1f} Roll:{roll:.1f}",
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                if yawn_detected:
                    cv2.putText(frame, "YAWN DETECTED", (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

                if alert_on:
                    cv2.putText(frame, "FATIGUE ALERT!", (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)

                # ----- Log -----
                append_log([
                    now,
                    ear, mar,
                    int(eyes_closed), int(mouth_open),
                    perclos, mor,
                    blink_per_min,
                    int(yawn_active), yawn_seconds,
                    pitch, yaw, roll,
                    int(alert_on)
                ])

            else:
                # No face detected
                cv2.putText(frame, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                gpio_alert(False)
                alert_on = False
                fatigue_start = None

            cv2.imshow("Fatigue Detector FULL", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if GPIO_AVAILABLE and GPIO_SET:
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup()

    print("Stopped. Log saved at:", LOG_PATH)

if __name__ == "__main__":
    main()

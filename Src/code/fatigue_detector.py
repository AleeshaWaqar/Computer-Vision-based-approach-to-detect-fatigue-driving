import os
import csv
import time
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ...\fatigue
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

YAWN_MODEL_PATH = MODELS_DIR / "yawn_cnn.h5"

# Must match train_yawn_model.py
IMG_SIZE = 64

# =========================
# THRESHOLDS / SETTINGS
# =========================
EAR_THRESH = 0.20             # tune (0.18-0.25 common)
EAR_CONSEC_FRAMES = 3         # blink if closed for >= this
PERCLOS_WINDOW_SEC = 30       # sliding window
PERCLOS_THRESH = 0.40         # drowsy if > 0.4
MAR_THRESH = 0.60             # tune based on your camera
YAWN_PROB_THRESH = 0.60       # CNN sigmoid threshold
YAWN_MIN_SEC = 2.0            # mouth open for >= 2 sec -> yawn
HEAD_PITCH_DOWN_THRESH = 15.0 # degrees (down)

# =========================
# MediaPipe FaceMesh setup
# =========================
mp_face = mp.solutions.face_mesh

# FaceMesh landmark indices (MediaPipe)
# Eye landmarks for EAR (6 points per eye)
LEFT_EYE = [33, 160, 158, 133, 153, 144]    # p1,p2,p3,p4,p5,p6
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR (outer lips)
# horizontal: 61-291, vertical: 13-14 (inner) works well
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# PnP head pose 2D points from MediaPipe landmarks
PNP_POINTS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291
}

# A simple generic 3D face model (mm) for solvePnP (approx)
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye outer
    (43.3, 32.7, -26.0),    # right eye outer
    (-28.9, -28.9, -24.1),  # left mouth
    (28.9, -28.9, -24.1),   # right mouth
], dtype=np.float64)

def euclid(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def landmark_to_xy(lm, w, h):
    return (lm.x * w, lm.y * h)

def ear_from_landmarks(pts):
    # pts order: [p1,p2,p3,p4,p5,p6]
    p1, p2, p3, p4, p5, p6 = pts
    A = euclid(p2, p6)
    B = euclid(p3, p5)
    C = euclid(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mar_from_landmarks(m_left, m_right, m_top, m_bottom):
    # simple MAR = vertical / horizontal
    horiz = euclid(m_left, m_right)
    vert = euclid(m_top, m_bottom)
    if horiz == 0:
        return 0.0
    return vert / horiz

def crop_mouth_gray(frame, m_left, m_right, m_top, m_bottom, pad=25):
    h, w = frame.shape[:2]
    xs = [m_left[0], m_right[0], m_top[0], m_bottom[0]]
    ys = [m_left[1], m_right[1], m_top[1], m_bottom[1]]
    x1 = max(0, int(min(xs) - pad))
    y1 = max(0, int(min(ys) - pad))
    x2 = min(w, int(max(xs) + pad))
    y2 = min(h, int(max(ys) + pad))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    gray = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return gray

def rotation_vector_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return (math.degrees(x), math.degrees(y), math.degrees(z))  # pitch, yaw, roll approx

def head_pose_pnp(face_landmarks, w, h, camera_matrix, dist_coeffs):
    image_points = []
    for k in ["nose_tip","chin","left_eye_outer","right_eye_outer","left_mouth","right_mouth"]:
        idx = PNP_POINTS[k]
        lm = face_landmarks.landmark[idx]
        x, y = landmark_to_xy(lm, w, h)
        image_points.append((x, y))
    image_points = np.array(image_points, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_3D, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    pitch, yaw, roll = rotation_vector_to_euler(rvec)
    return pitch, yaw, roll

def main():
    if not YAWN_MODEL_PATH.exists():
        print(f"ERROR: yawn model not found: {YAWN_MODEL_PATH}")
        print("Train it first: python code/train_yawn_model.py")
        return

    yawn_model = load_model(str(YAWN_MODEL_PATH))

    # Windows camera: CAP_DSHOW reduces “opens then closes” issues
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Webcam not opened. Try camera index 1 or 2.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps > 120:
        fps = 30.0
    win_len = int(PERCLOS_WINDOW_SEC * fps)

    # Camera intrinsics approx
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # PERCLOS queue: 1 if eyes closed else 0
    closed_queue = deque(maxlen=win_len)

    eye_closed_streak = 0
    yawn_open_streak = 0

    # CSV log
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"session_{ts}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "EAR", "MAR", "PERCLOS", "yawn_prob", "yawn_event", "pitch", "yaw", "roll", "drowsy"])

    print("Running fatigue detector... press 'q' to quit.")
    print("Logging:", log_path)

    with mp_face.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)

            ear = 0.0
            mar = 0.0
            perclos = 0.0
            yawn_prob = 0.0
            yawn_event = 0
            pitch = yaw = roll = 0.0
            drowsy = 0

            if result.multi_face_landmarks:
                fl = result.multi_face_landmarks[0]

                # --- EAR
                left_pts = [landmark_to_xy(fl.landmark[i], w, h) for i in LEFT_EYE]
                right_pts = [landmark_to_xy(fl.landmark[i], w, h) for i in RIGHT_EYE]
                ear = (ear_from_landmarks(left_pts) + ear_from_landmarks(right_pts)) / 2.0

                eyes_closed = ear < EAR_THRESH
                if eyes_closed:
                    eye_closed_streak += 1
                    closed_queue.append(1)
                else:
                    eye_closed_streak = 0
                    closed_queue.append(0)

                perclos = float(sum(closed_queue)) / float(len(closed_queue)) if len(closed_queue) > 0 else 0.0

                # --- MAR
                m_left = landmark_to_xy(fl.landmark[MOUTH_LEFT], w, h)
                m_right = landmark_to_xy(fl.landmark[MOUTH_RIGHT], w, h)
                m_top = landmark_to_xy(fl.landmark[MOUTH_TOP], w, h)
                m_bottom = landmark_to_xy(fl.landmark[MOUTH_BOTTOM], w, h)
                mar = mar_from_landmarks(m_left, m_right, m_top, m_bottom)

                # --- CNN Yawn
                mouth_in = crop_mouth_gray(frame, m_left, m_right, m_top, m_bottom, pad=25)
                if mouth_in is not None:
                    try:
                        yawn_prob = float(yawn_model.predict(mouth_in, verbose=0)[0][0])
                    except Exception:
                        yawn_prob = 0.0

                mouth_open = (yawn_prob >= YAWN_PROB_THRESH) or (mar >= MAR_THRESH)

                if mouth_open:
                    yawn_open_streak += 1
                else:
                    yawn_open_streak = 0

                # yawn event if mouth open for >= YAWN_MIN_SEC
                if yawn_open_streak >= int(YAWN_MIN_SEC * fps):
                    yawn_event = 1

                # --- Head pose (PnP)
                pose = head_pose_pnp(fl, w, h, camera_matrix, dist_coeffs)
                if pose is not None:
                    pitch, yaw, roll = pose

                head_down = pitch > HEAD_PITCH_DOWN_THRESH

                # --- Fusion rule (rule-based)
                # Drowsy if high perclos AND (yawn OR head_down)
                if (perclos >= PERCLOS_THRESH) and (yawn_event == 1 or head_down):
                    drowsy = 1

                # --- Draw
                cv2.putText(frame, f"EAR: {ear:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"YawnProb: {yawn_prob:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                if eyes_closed:
                    cv2.putText(frame, "EYES CLOSED", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    cv2.putText(frame, "EYES OPEN", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                if mouth_open:
                    cv2.putText(frame, "MOUTH OPEN", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                else:
                    cv2.putText(frame, "MOUTH CLOSED", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                if yawn_event:
                    cv2.putText(frame, "YAWN!", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 3)

                if drowsy:
                    cv2.putText(frame, "DROWSY!", (20, 295), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 4)

            # log
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), ear, mar, perclos, yawn_prob, yawn_event, pitch, yaw, roll, drowsy])

            cv2.imshow("Fatigue Detector (EAR+MAR+CNN+PnP)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

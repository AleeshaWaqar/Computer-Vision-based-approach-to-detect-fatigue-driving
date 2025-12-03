import os
print("WORKING DIR:", os.getcwd())

import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

print("SCRIPT STARTED")

# NOTE: blink_detector.py is inside the 'code' folder
# and eye_model.h5 is inside the 'models' folder at the SAME LEVEL as 'code'
# so we go one level up: ../models/eye_model.h5
model = load_model("models/eye_model.h5")
print("MODEL LOADED")

detector = MTCNN()
print("MTCNN LOADED")

def crop_eye(frame, point):
    x, y = point
    h, w = frame.shape[:2]
    size = 30

    x1, y1 = max(0, x - size), max(0, y - size)
    x2, y2 = min(w, x + size), min(h, y + size)

    eye = frame[y1:y2, x1:x2]
    if eye is None or eye.size == 0:
        return None

    eye = cv2.resize(eye, (24, 24))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye.reshape(1, 24, 24, 1) / 255.0
    return eye

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("WEBCAM OPENED")
closed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if faces:
        kp = faces[0]['keypoints']
        left_eye_pt = kp['left_eye']
        right_eye_pt = kp['right_eye']

        left_eye = crop_eye(frame, left_eye_pt)
        right_eye = crop_eye(frame, right_eye_pt)

        if left_eye is not None and right_eye is not None:
            left_pred = np.argmax(model.predict(left_eye)[0])
            right_pred = np.argmax(model.predict(right_eye)[0])

            if left_pred == 0 and right_pred == 0:
                closed_frames += 1
                cv2.putText(frame, "EYES CLOSED", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                closed_frames = 0
                cv2.putText(frame, "EYES OPEN", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if closed_frames > 15:
                cv2.putText(frame, "DROWSY!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("SCRIPT ENDED")

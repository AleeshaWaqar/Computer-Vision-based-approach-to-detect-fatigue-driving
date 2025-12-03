import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Load models
eye_model = load_model(r"C:\Users\asus\Desktop\fatigue\models\eye_model.h5")
yawn_model = load_model(r"C:\Users\asus\Desktop\fatigue\models\mouth_model.h5")

detector = MTCNN()

# ----------- EYE REGION (24x24 grayscale) -----------
def crop_eye(frame, point, size=30):
    x, y = point
    h, w = frame.shape[:2]

    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w, x + size)
    y2 = min(h, y + size)

    region = frame[y1:y2, x1:x2]

    if region.size == 0:
        return None

    region = cv2.resize(region, (24, 24))
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = region.reshape(1, 24, 24, 1) / 255.0
    return region

# ----------- MOUTH REGION (24x24 grayscale for yawn model) -----------
def crop_mouth(frame, mouth_left, mouth_right):
    x1, y1 = mouth_left
    x2, y2 = mouth_right

    # Expand region a bit
    x_min = min(x1, x2) - 20
    x_max = max(x1, x2) + 20
    y_min = min(y1, y2) - 10
    y_max = max(y1, y2) + 30

    h, w = frame.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    mouth = frame[y_min:y_max, x_min:x_max]

    if mouth.size == 0:
        return None

    mouth = cv2.resize(mouth, (24, 24))
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
    mouth = mouth.astype("float32") / 255.0
    mouth = mouth.reshape(1, 24, 24, 1)
    return mouth

print("Webcam opened. Starting fatigue + yawn detector...")

cap = cv2.VideoCapture(0)
eye_closed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    for face in faces:
        key = face['keypoints']

        # ----- EYE DETECTION -----
        left_eye = crop_eye(frame, key['left_eye'])
        right_eye = crop_eye(frame, key['right_eye'])

        if left_eye is not None and right_eye is not None:
            left_pred = np.argmax(eye_model.predict(left_eye)[0])
            right_pred = np.argmax(eye_model.predict(right_eye)[0])

            if left_pred == 0 and right_pred == 0:
                eye_closed_frames += 1
                cv2.putText(frame, "Eyes CLOSED", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                eye_closed_frames = 0
                cv2.putText(frame, "Eyes OPEN", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ----- MOUTH / YAWN DETECTION -----
        mouth = crop_mouth(frame, key['mouth_left'], key['mouth_right'])

        if mouth is not None:
            prob = float(yawn_model.predict(mouth)[0][0])

            if prob > 0.5:
                cv2.putText(frame, "YAWNING", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Mouth closed", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ----- DROWSINESS CONDITION -----
        if eye_closed_frames > 15:
            cv2.putText(frame, "DROWSY!", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    cv2.imshow("Fatigue Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

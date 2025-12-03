import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================================================
# LOAD MODELS (CHANGE THESE IF YOUR PATHS ARE DIFFERENT)
# =========================================================
eye_model = load_model(r"C:\Users\asus\Desktop\fatigue\models\eye_model.h5")
mouth_model = load_model(r"C:\Users\asus\Desktop\fatigue\models\mouth_model.h5")

# =========================================================
# LOAD HAAR CASCADES
# =========================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img = eye_img.reshape(1, 24, 24, 1) / 255.0
    return eye_img

def preprocess_mouth(mouth_img):
    mouth_img = cv2.resize(mouth_img, (24, 24))   # MATCH mouth_model.h5 INPUT SIZE
    mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
    mouth_img = mouth_img.reshape(1, 24, 24, 1) / 255.0
    return mouth_img

# =========================================================
# START CAMERA
# =========================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not opening!")
    exit()

print("Webcam opened. Starting fatigue + yawn detector...")

closed_eye_frames = 0

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error: frame not received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # ---------------- EYES ----------------
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        eye_states = []

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            eye_input = preprocess_eye(eye_img)
            pred = np.argmax(eye_model.predict(eye_input)[0])
            eye_states.append(pred)

        if len(eye_states) >= 2:   # both eyes detected
            if eye_states[0] == 0 and eye_states[1] == 0:
                closed_eye_frames += 1
                cv2.putText(frame, "Eyes CLOSED", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                closed_eye_frames = 0
                cv2.putText(frame, "Eyes OPEN", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---------------- MOUTH ----------------
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.6, 20)
        for (mx, my, mw, mh) in mouths:
            mouth_img = roi_color[my:my+mh, mx:mx+mw]
            mouth_input = preprocess_mouth(mouth_img)
            pred = int(round(float(mouth_model.predict(mouth_input)[0][0])))

            if pred == 1:
                cv2.putText(frame, "YAWNING", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            break  # first mouth only

        # ---------------- DROWSINESS ----------------
        if closed_eye_frames > 15:
            cv2.putText(frame, "DROWSY!", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    cv2.imshow("Fatigue Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

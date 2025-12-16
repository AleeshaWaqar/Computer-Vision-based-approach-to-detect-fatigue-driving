import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESH = 0.20
EAR_CONSEC_FRAMES = 3

def euclid(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def landmark_to_xy(lm, w, h):
    return (lm.x * w, lm.y * h)

def ear_from_landmarks(pts):
    p1, p2, p3, p4, p5, p6 = pts
    A = euclid(p2, p6)
    B = euclid(p3, p5)
    C = euclid(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Webcam not opened.")
        return

    blink_count = 0
    closed_streak = 0

    with mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            ear = 0.0
            if res.multi_face_landmarks:
                fl = res.multi_face_landmarks[0]
                left_pts = [landmark_to_xy(fl.landmark[i], w, h) for i in LEFT_EYE]
                right_pts = [landmark_to_xy(fl.landmark[i], w, h) for i in RIGHT_EYE]
                ear = (ear_from_landmarks(left_pts) + ear_from_landmarks(right_pts)) / 2.0

                if ear < EAR_THRESH:
                    closed_streak += 1
                else:
                    if closed_streak >= EAR_CONSEC_FRAMES:
                        blink_count += 1
                    closed_streak = 0

                cv2.putText(frame, f"EAR: {ear:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Blink Detector (EAR)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

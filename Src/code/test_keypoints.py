from mtcnn import MTCNN
import cv2

detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector.detect_faces(frame)
    for face in faces:
        print(face['keypoints'])  # PRINT ALL KEYPOINTS

    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

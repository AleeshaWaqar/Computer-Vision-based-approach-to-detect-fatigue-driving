import cv2

# Try all Windows webcam backends
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print("Frame:", ret)

    if not ret:
        break

    cv2.imshow("TEST CAMERA - press Q to close", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

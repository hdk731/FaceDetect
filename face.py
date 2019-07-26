# coding: UTF-8

import cv2

cascade_path = "./haarcascades/haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('not captured.')

cascade = cv2.CascadeClassifier(cascade_path)
color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        print('not captured.')
        break

    # facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))
    face_rect = cascade.detectMultiScale(frame)

    if len(face_rect) > 0:
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display frame
    cv2.imshow('camera capture', frame)

    # 50ms wait(20fps)
    k = cv2.waitKey(50)
    if k == 27:
        break

# finish
cap.release()
cv2.destroyAllWindows()

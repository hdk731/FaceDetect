# coding: UTF-8

import cv2
import threading
from time import sleep
from datetime import datetime


class OutputCapture(threading.Thread):
    def __init__(self, frame):
        super(OutputCapture, self).__init__()
        self._path = "./output/"
        self._frame = frame

    def run(self):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        image_path = self._path + now + '.jpg'
        cv2.imwrite(image_path, frame)


cascade_path = "./haarcascades/haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('not captured.')

ref_now = ''
cascade = cv2.CascadeClassifier(cascade_path)
color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        print('not captured.')
        break

    # Grayscale conversion
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # face_rect = cascade.detectMultiScale(frame)
    face_rect = cascade.detectMultiScale(frame_gray)

    if len(face_rect) > 0:
        for (x, y, w, h) in face_rect:
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            if threading.activeCount() == 1:
                if ref_now != now:
                    th = OutputCapture(frame)
                    th.start()

                    ref_now = now

            sleep(0.05)
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

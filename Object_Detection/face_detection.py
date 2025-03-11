import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "../DATA/haarcascades/haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(BASE_DIR, "../DATA/haarcascades/haarcascade_eye.xml")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

def face_and_eye_detection(img):
    face_img = img.copy()

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(64,64))

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = face_img[y:y+h, x:x+w]

        eye_rects = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30,30))

        for (ex, ey, ew, eh) in eye_rects:
            cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return face_img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = face_and_eye_detection(frame)
    cv2.imshow('Video Face Detection', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

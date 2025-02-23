import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Automatically determine the correct path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
CASCADE_PATH = os.path.join(BASE_DIR, "../DATA/haarcascades/haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def face_detection(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,255,255), 10)
    return face_img        

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = face_detection(frame)
    cv2.imshow('Video Face Detection', frame)
    
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
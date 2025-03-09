import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
CASCADE_PATH = os.path.join(BASE_DIR, "../DATA/haarcascades/haarcascade_eye.xml")

eye_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def eye_detection(img):
    eye_img = img.copy()
    eye_rects = eye_cascade.detectMultiScale(eye_img)
    
    for (x,y,w,h) in eye_rects:
        cv2.rectangle(eye_img, (x,y), (x+w, y+h), (255,255,255), 10)
    return eye_img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = eye_detection(frame)
    cv2.imshow('Video Eye Detection', frame)
    
    k = cv2.waitKeyEx(1)
    if k == 27:
        break
cap.release()
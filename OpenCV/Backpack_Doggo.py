import cv2
import numpy as np
import matplotlib.pyplot as plt

######
def draw_circle(event,x,y,flags,params):
    
    if event == cv2.EVENT_LBUTTONDOWN:
       cv2.circle(img_resized, (x,y), radius = 50, color=(0,255,0), thickness = 2)
    
#####
img = cv2.imread('./DATA/dog_backpack.jpg')
img_resized = cv2.resize(img, (400,400))
cv2.namedWindow(winname='Dog')
cv2.setMouseCallback('Dog', draw_circle)

while True:
    cv2.imshow('Dog',img_resized)
    
    if
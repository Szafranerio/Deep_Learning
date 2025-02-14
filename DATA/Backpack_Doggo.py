import cv2
import numpy as np
import matplotlib.pyplot as plt

######
def draw_circle(event,x,y,flags,params):
    
    if event == cv2.EVENT_LBUTTONDOWN:
       cv2.circle(img, (x,y), (0,255,0), -1)
    
#####
img = cv2.imread('../DATA/dog_backpack.jpg')
cv2.namedWindow(winname='Dog')
cv2.setMouseCallback('Dog', draw_circle)

while True:
    cv2.imshow('Dog',img)
    
    if cv2.waitKey(20) & 0xff == 27:
        break
    
cv2.destroyAllWindows()
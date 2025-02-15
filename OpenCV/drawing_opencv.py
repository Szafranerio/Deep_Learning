import cv2
import numpy as np

#VAR
drawing = False
ix = -1
iy = -1

######
def draw_rectangle(event,x,y,flags,params):
    global ix,iy, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x,y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy), (x,y),(0,255,0), -1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
 
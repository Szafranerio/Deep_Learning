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
        cv2.rectangle(img,(x,y))
    
#####
img = np.zeros((512,512,3))
cv2.namedWindow(winname='my_drawing_2')
cv2.setMouseCallback('my_drawing_2', draw_rectangle)

while True:
    cv2.imshow('my_drawing_2', img)
    
    if cv2.waitKey(20) & 0xff == 27:
        break
    
cv2.destroyAllWindows()
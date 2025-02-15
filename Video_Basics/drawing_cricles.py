import cv2

cap = cv2.VideoCapture(0)

#Callback
def draw_circle(event,x,y, flags,param):
    
    global pt1, Left_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if Left_clicked == True: 
            pt1=(0,0)
            Left_clicked = False     
            
        elif Left_clicked == False:
            pt1 = (x,y)
            Left_clicked = True
            
            

#Global Var
pt1=(0,0)
Left_clicked = False

#Connect to call back
cap = cv2.VideoCapture(0)
cv2.namedWindow('Test')
cv2.setMouseCallback('Test', draw_circle)

while True:
    
    ret, frame = cap.read()
    
    if Left_clicked:
        cv2.circle(frame, center=pt1, radius=25, color=(0,0,255), thickness = 5)

    cv2.imshow('Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
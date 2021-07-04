from tensorflow.keras.models import load_model
model1 = load_model("./models/detection_model.h5")
model2 = load_model("./models/classification_model.h5")
signs = ["interdit de depasser","vitesse max 70","obligation de tourner","danger","stop"]
def decision1(model,objct): 
    prediction = model.predict(objct)
    l = prediction.tolist()
    return(l[0][0])
def decision2(img,model,objct):  
    prediction = model.predict(objct)
    l = prediction.tolist()
    X = l[0].index(max(l[0]))  
    img = cv2.putText(img,signs[X],(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1,cv2.LINE_AA) 
def decision_final(img,objct):
    objct = cv2.resize(objct, dsize=(40,40), interpolation=cv2.INTER_CUBIC)
    objct = cv2.cvtColor(objct, cv2.COLOR_BGR2GRAY)
    objct = objct / 255
    objct = np.reshape(objct,(1,40,40,1))
    if decision1(model1,objct)>0.5:
        cv2.rectangle(img, (y1, x1), (y2, x2), (255, 0, 0), 2) 
        decision2(img,model2,objct)
import cv2 
import numpy as np
cap = cv2.VideoCapture(0)
while(1):
    _, frame = cap.read()   
    
    Height, Width = frame.shape[:2]   
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    lower_red1 = np.array([0,100,100])            
    upper_red1 = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([160,100,100])           
    upper_red2 = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  
    lower_blue = np.array([100,150,0])             
    upper_blue = np.array([140,255,255])
    mask3 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = mask1 + mask2 + mask3      
    
    contours, _ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    if len(contours) > 0:
        
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)  
        M = cv2.moments(c)   
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) 

        
        except:
            center =   int(Height/2), int(Width/2)
        

        try :
            if int(radius) > 50:
                x1,y1 = center[1]-int(radius) -10,center[0]-int(radius) -10
                d = int(radius)*2
                x2 , y2 = x1 + d + 20  , y1 + d + 20
                objct = frame[x1:x2,y1:y2,:]
                decision_final(frame,objct)
        
        except :
            
            continue
    
    text = "Cliquer sur 'ECHAP' pour quitter ... "
    img1 = cv2.putText(frame,text,(5,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        
        
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) == 27 :
        break

cap.release()
cv2.destroyAllWindows()

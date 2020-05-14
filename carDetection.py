import cv2
import time
import numpy as np
car_classifier=cv2.CascadeClassifier('haarcascade_car.xml')

#video capture
cap=cv2.VideoCapture('cars.avi')

while cap.isOpened():
    time.sleep(0.05)
    ret,frame=cap.read()
    #captures each frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars=car_classifier.detectMultiScale(gray,1.03,5)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(170,255,201),2)
        cv2.imshow('Pedestrians',frame)
    if cv2.waitKey(1)==13: #13 is the enter key
        break
cap.release()
cv2.destroyAllWindows()


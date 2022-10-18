#import pacakage for computer vision
import cv2

#import random number and range generator
from random import randrange

#giving access for the pre trained Algorithm
trained_model=cv2.CascadeClassifier('Algorithm.xml')

#giving access to the live camera feed and the storing in the variable
livefeed=cv2.VideoCapture(0)


while True:

    successful_vidobject_read, vid_object = livefeed.read()

    grayscale_livefeed = cv2.cvtColor(vid_object, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_model.detectMultiScale(grayscale_livefeed)

    for (x,y,w,h) in face_coordinates:

        cv2.rectangle(vid_object, (x,y), (x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)

    cv2.imshow('WELCOME TO THE AGE OF AI' , vid_object)
    
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break


livefeed.release()
print("process completed")
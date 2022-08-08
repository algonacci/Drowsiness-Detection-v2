import cv2
import mediapipe as mp
import time
import numpy as np
# import RPi.GPIO as GPIO
import time
import module as m

# GPIO.cleanup()
# GPIO.setmode(GPIO.BCM)
# buzzer = 21
# GPIO.setup(buzzer,GPIO.OUT)
# GPIO.output(buzzer, GPIO.HIGH)

cap = cv2.VideoCapture("vid_002.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter("result.mp4", fourcc, 29.0, (1280,720))
out2 = cv2.VideoWriter("result_2.mp4", fourcc, 29.0, (1280,720))
idx_frame = 0
th_frame_eye = 0
th_frame_mouth = 0
list_ear_mouth = []
list_ear_eye = []
total_frame = 0
total_speed = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    idx_frame+=1
    print("frame",idx_frame)
    if ret==True:
        tic = time.time()
        frame = m.resize_to_small(frame)
        frame, image2, ear_eye, ear_mouth, is_closed, is_yawn = m.drowsiness_pipeline(frame)
        frame = m.resize_to_hd(frame)
        list_ear_eye.append(ear_eye)
        list_ear_mouth.append(ear_mouth)
        if is_closed==1:
            th_frame_eye+=1
        else:
            th_frame_eye=0
            
        if is_yawn==1:
            th_frame_mouth+=1
        else:
            th_frame_mouth=0
            
        frame = m.plot_text(frame, ear_eye, ear_mouth, is_closed, is_yawn, th_frame_eye, th_frame_mouth)
        out.write(frame)
        out2.write(image2)
#         print("Execution Time", time.time() - tic)
        total_frame+=1
        total_speed+=time.time() - tic
#         if total_frame>50:
#             break
    else:
        break
cap.release()
out.release()
out2.release()
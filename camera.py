# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:17:07 2018

@author: siwanghu
"""

import cv2
import dlib
import os
import sys

save_dir = "./data/test"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

index,count=0,100
camera = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()

while(True):
    if index < count:
        ret,frame = camera.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("window",frame)
        result = detector(gray_frame, 1)
        for i, d in enumerate(result):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = gray_frame[x1:y1,x2:y2]
            cv2.imwrite(save_dir+'/'+str(index)+'.jpg', face)
            print("保存成功"+str(index)+".jpg")
            index += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        sys.exit()
        
        
        
        
        
        
        
    
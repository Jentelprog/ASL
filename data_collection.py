import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)

offset=20
imgSize=300

folder="DataCollection/test"
counter=0

while(True):
    success,img=cap.read()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255

        try:
            imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
            aspectRatio=h/w

            if aspectRatio > 1:
                k=imgSize/h
                wcal=math.ceil(w*k)
                hcal=300
                imgResize=cv2.resize(imgCrop,(wcal,hcal))
                wGap=math.ceil((300-wcal)/2)
                imgWhite[:, wGap:wcal+wGap] = imgResize
            else:
                k = imgSize / w
                hcal = math.ceil(h * k)
                wcal = 300
                imgResize = cv2.resize(imgCrop, (wcal, hcal))
                hGap = math.ceil((300 - hcal) / 2)
                imgWhite[ hGap:hcal+ hGap,:] = imgResize
            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)
        except Exception as e:
            print(e)
            continue
    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key==ord('s'):
        counter+=1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgWhite)
        print(counter)
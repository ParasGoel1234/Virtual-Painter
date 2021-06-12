import cv2
import mediapipe as mp
import HandTrackingModule as htm
import os
import numpy as np
import time

folderPath = "VirtualPainterImg"
mylist = os.listdir(folderPath)
detector = htm.HandDetector(detectionCon=0.85)
# print(mylist)
overLayList = []
drawColor = (0, 255, 0)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)
# print(len(overLayList))
header = overLayList[0]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    # import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmLimits = detector.findPosition(img, draw=False)
    # Tip of index finger and middle finger
    if len(lmLimits) != 0:
        x1, y1 = lmLimits[8][1:]
        x2, y2 = lmLimits[12][1:]

        # check which finger is up
        fingers = detector.fingerUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("Selection Mode")
            if y1 < 125:
                if 250< x1< 450:
                    header = overLayList[0]
                    drawColor = (0, 255, 0)
                elif 550< x1< 750:
                    header = overLayList[1]
                    drawColor = (255, 0, 0)
                elif 800< x1< 950:
                    header = overLayList[2]
                    drawColor = (0, 0, 255)
                elif 1050< x1< 1200:
                    header = overLayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 12, drawColor, cv2.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:

                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, 50)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 50)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, 10)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 10)
            xp, yp = x1, y1

    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    cv2.imshow("image", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)

import cvzone
import cv2
import os
import math
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

menuImages = []
path = "filters"
pathList = os.listdir(path)

pathList.sort()

for pathImg in pathList:
    img = (cv2.imread(path+"/"+pathImg, cv2.IMREAD_UNCHANGED))
    menuImages.append(img)

menuCount = len(menuImages)
detector = HandDetector(detectionCon=0.8)
menuChoice = -1
isImageSelected = False

while True:
    success, cameraFeedImg = cap.read()
    cameraFeedImg = cv2.flip(cameraFeedImg, 1)

    wHeight, wWidth, wChannel = cameraFeedImg.shape

    x = 0
    y = 0
    xIncrement = math.floor(wWidth / menuCount)
    yIncrement = math.floor(wHeight / menuCount)

    handsDetector = detector.findHands(cameraFeedImg, flipType=False)
    hands = handsDetector[0]
    cameraFeedImg = handsDetector[1]

    indexFingerTop = 0
    try:
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            indexFingerTop = lmList1[8]
            indexFingerBottom = lmList1[6]

            # Check x axis i.e index[0] i.e finger is on the vertical menu
            if (indexFingerTop[0] < 100):
                i = 0
                # Replace xIncrement with yIncrement and change wWidth to w Height so that you can pick the right image
                while (yIncrement*i <= wHeight):
                    # Check if indexFingerTop[1] is less then yIncrement*i
                    if (indexFingerTop[1] < yIncrement*i):
                        menuChoice = i-1
                        isImageSelected = True
                        break
                    i = i+1

            if (indexFingerTop[1] > indexFingerBottom[1]):
                isImageSelected = False

        if (isImageSelected):
            image = cv2.resize(menuImages[menuChoice], (100, 100))
            cameraFeedImg = cvzone.overlayPNG(
                cameraFeedImg, image, [int(indexFingerTop[0]), int(indexFingerTop[1])])

    except Exception as e:
        print(e)

    try:
        for image in menuImages:
            margin = 20
            image = cv2.resize(
                image, (yIncrement - margin, yIncrement - margin))
            cameraFeedImg = cvzone.overlayPNG(cameraFeedImg, image, [0, y])
            y = y + yIncrement - 4

    except:
        print("Image out of bounds")

    cv2.imshow("Face Filter App", cameraFeedImg)
    cv2.waitKey(1)

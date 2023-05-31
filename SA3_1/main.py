import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import math
import cvzone


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

menuImages = []
path = "filters"
pathList = os.listdir(path)
pathList.sort()

for pathImg in pathList:
    img = (cv2.imread(path + "/" + pathImg, cv2.IMREAD_UNCHANGED))
    menuImages.append(img)

menuCount = len(menuImages)

detector = HandDetector(detectionCon=0.8)

# Flag variable for item selected from the filters menu
menuChoice = -1

while True:
    success, cameraFeedImg = cap.read()
    cameraFeedImg = cv2.flip(cameraFeedImg, 1)

    wHeight, wWidth, wChannel = cameraFeedImg.shape
    x = 0
    xIncrement = math.floor(wWidth / menuCount)


    handsDetector = detector.findHands(cameraFeedImg, flipType=False)
    hands = handsDetector[0]
    cameraFeedImg = handsDetector[1]

    try:
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  
            indexFingerTop = lmList1[8]
            indexFingerBottom = lmList1[6]

            # Find which image is selected by the pointer finger
            if (indexFingerTop[1] < xIncrement):
                i = 0
                while (xIncrement*i <= wWidth):
                    if (indexFingerTop[0] < xIncrement*i):
                        menuChoice = i-1
                        # Set isImageSelected flag variable to true to show that image can be dragged now
                        isImageSelected = True 
                        break
                    i = i+1

           
    except Exception as e:
        print(e)


    try:
        for image in menuImages:
            margin = 20
            image = cv2.resize(image, (xIncrement - margin, xIncrement - margin))
            cameraFeedImg = cvzone.overlayPNG(cameraFeedImg, image, [x, 0])
            x = x + xIncrement
    except:
        print("out of bounds")
   
    cv2.imshow("Face Filter App", cameraFeedImg)
    cv2.waitKey(1)
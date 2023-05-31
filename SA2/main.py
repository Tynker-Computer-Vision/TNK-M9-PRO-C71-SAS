import cv2
import os
from cvzone.HandTrackingModule import HandDetector
# Import math library
import math
# Import cvzone library
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

# Count the number of images in the menuList
menuCount = len(menuImages)

detector = HandDetector(detectionCon=0.8)

while True:
    success, cameraFeedImg = cap.read()
    cameraFeedImg = cv2.flip(cameraFeedImg, 1)

    # Get width and height of final output screen
    wHeight, wWidth, wChannel = cameraFeedImg.shape

    # Set initial position to 0 (menu will start displaying at 0)
    x = 0
    # Calculate increments to display next menuImage
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

           
    except Exception as e:
        print(e)


    try:
        # Overlay menu images to camera feed
        for image in menuImages:
            margin = 20
            image = cv2.resize(image, (xIncrement - margin, xIncrement - margin))
            cameraFeedImg = cvzone.overlayPNG(cameraFeedImg, image, [x, 0])
            x = x + xIncrement
    except:
        print("out of bounds")
   
    cv2.imshow("Face Filter App", cameraFeedImg)
    cv2.waitKey(1)
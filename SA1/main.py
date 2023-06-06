import cv2
import os
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Create list to hold menu images and pathList
menuImages = []
path = "filters"
pathList = os.listdir(path)

# Sort all the images in the list
pathList.sort()
print(type(pathList))

# Print the pathList
print(pathList)


# Load all the images in the list
for pathImg in pathList:
    img = (cv2.imread(path + "/" + pathImg, cv2.IMREAD_UNCHANGED))
    menuImages.append(img)


detector = HandDetector(detectionCon=0.8)

while True:
    success, cameraFeedImg = cap.read()
    cameraFeedImg = cv2.flip(cameraFeedImg, 1)

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

   
    # Show the loaded image
    cv2.imshow("Face Filter App", menuImages[2])
    cv2.waitKey(1)
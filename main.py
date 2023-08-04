import cv2
import time
import handDetectModule as hdm
import numpy as np
import os

## NOTE run this file from the current directory as img files
# are addressed that way


# utility functions ##############################

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

###########################################

########## global vars ####################

window_name = "Virtual Painter"
previousTime = 0
currentTime = 0
mode = "None"
tool = "white brush"
drawColor = (255,255,255) #white by default
brushSize = 30
eraserSize = 100
xp, yp = 0,0
stroke = brushSize
toolChange = False 
############################################


################# UI ######################
folderPath = 'UI'
print("file exists?", os.path.exists(folderPath))
imgList = os.listdir(folderPath)
overlayList = []
for imgPath in imgList:
    img = cv2.imread(f'{folderPath}\{imgPath}')
    overlayList.append(img)
   
header = overlayList[0]
##########################################


############### cv2 starts ################
cap = cv2.VideoCapture(0)
detector = hdm.handDetector(detectionConf=0.85)
imgCanvas = np.zeros((864, 1152,3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 1. display header and cam screen correctly
    header = cv2.resize(header, (1152,100),interpolation =cv2.INTER_AREA)
    img = rescale_frame(img, percent=180)
    #img/cam-screen size is 1152x864 
    img[0:100, 0:1152] = header # header resized to 1152 x 100
   
    # 2. get media-pipe model to identify hand & fingers
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, 0, False)
    # lmList is list of all 21 points for that handIdx=0
    # and each point is a triplet (index, x, y)

    if len(lmList)!=0:
        x1,y1 = lmList[8][1:] #tip of index finger
        x2,y2 = lmList[12][1:] #tip of middle finger

        # 3. get which fingers are up
        fingers = detector.fingersUp()

        # 4. selection mode-> 2 fingers Up mode
        if fingers[1] and fingers[2]:
            mode = "Setection mode"
            toolChange = True
            if y1 < 100:
                if 50 < x1 <150:
                    header = overlayList[0] 
                    tool = "white brush"
                    drawColor = (255,255,255)
                    
                elif 250 < x1 <350:
                    header = overlayList[1]
                    tool = "cyan brush"
                    drawColor = (255,255,0)
                    
                elif 450 < x1 <550:
                    header = overlayList[2]
                    tool = "green brush"
                    drawColor = (0,200,0)
                    
                elif 650 < x1 <750:
                    header = overlayList[3]
                    tool = "orange brush"
                    
                    drawColor = (0,153,255)
                elif 850 < x1 <950:
                    header = overlayList[4]
                    tool = "Eraser"
                   
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-30), (x2,y2+30), drawColor, 2)



        # 5. Drawing mode-> 1 fingers Up mode
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            mode = "Drawing mode"
            if xp==0 and yp==0:
                xp, yp =  x1,y1
            elif toolChange:
                xp, yp = x1, y1
                toolChange = False

            if drawColor == (0,0,0):
                stroke = eraserSize
            else: 
                stroke = brushSize

            cv2.line(img, (xp,yp), (x1,y1), drawColor, stroke)
            cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, stroke)

            xp, yp = x1, y1
    
    ################## inserting the canvas over cam-screen

    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    # above is one way to add img & canvas but looks dark & bad

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50,255, cv2.THRESH_BINARY_INV)
    # its like creating a mask, the area where we didn't draw is masked out
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv) #drawn line is removed from img
    img = cv2.bitwise_or(img, imgCanvas) #drawn line is superimposed over removed area of img



    ########## dispplay content ################################
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img=img, text="FPS : "+str(int(fps)), 
                org= (20,150), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=1.8, color=(0,255,0), thickness=3)
    cv2.putText(img=img, text=f"Mode : {mode}", 
                org= (800,150), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=1.5, color=(255,25,0), thickness=2)
    cv2.putText(img=img, text=f"Tool : {tool}", 
                org= (800,180), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=1.5, color=(255,25,0), thickness=2)
    ############################################################
   
    cv2.imshow(window_name, img)
    # cv2.imshow("Canvas", imgCanvas)

    key = cv2.waitKey(1)
    if key == ord("q") :
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackConf)
        self.hands = self.mpHands.Hands(static_image_mode=False,
               max_num_hands=self.maxHands,
               model_complexity=1,
               min_detection_confidence=self.detectionConf,
               min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img


    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        #below code is to fetch each point as required
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, channels = img.shape
                centerX, centerY = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, centerX, centerY])

                if draw:
                    if id==0 or id==4: #will draw big-dot on palm point and thumb
                        cv2.circle(img, (centerX, centerY), 15, (255,0,255), cv2.FILLED)


        return self.lmList

    def fingersUp(self):
        fingers = []
        tipIds = [4,8,12,16,20] #thumb, first finger, ...
        if len(self.lmList)!=0:
            if self.lmList[0][1] < self.lmList[2][1]: # thumb is on right side of screen
                if self.lmList[3][1] < self.lmList[4][1]: #open thumb
                    fingers.append(1)
                else :
                    fingers.append(0)
            else: # thumb is on left side of screen
                if self.lmList[3][1] < self.lmList[4][1]: #closed thumb
                    fingers.append(0)
                else :
                    fingers.append(1)           

            for id in range(1,5): #4 fingers, thumb dealt above
                t = tipIds[id]
                if self.lmList[t][2] > self.lmList[t-2][2]:
                    fingers.append(0) #finger is closed
                else:
                    fingers.append(1)    
        
        return fingers



def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        l = detector.findPosition(img, 0, True)
        # l is list of all 21 points for that handIdx=0
        # if len(l)!=0:
        #     print(l[4])
        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img=img, text=str(int(fps)), org= (10,70), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, 
                    color=(255,0,255), thickness=3)
        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == ord("q") :
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
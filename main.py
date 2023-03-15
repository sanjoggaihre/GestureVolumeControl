import numpy as np
import cv2
import time
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def handTracking():
    cap = cv2.VideoCapture(0)

    #defining mediapipe library
    mpHands = mp.solutions.hands
    print(type(mpHands))    #module type
    # print(dir(mpHands))
    #Hands() fxn has parameters, which have default value
    hands = mpHands.Hands()
    # drawing the palm landmarks
    mpDraw = mp.solutions.drawing_utils

    #activating the pycaw
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    p_time = time.time()
    c_time = 0
    dist = 0
    vol_bar = 200
    min_hand = 25
    max_hand = 180
    vol_per = 100
    hand_positions_array = []
    while True:
        success, img = cap.read()
        #we have image of (480*640)
        # mediapipe hands() only take rgb color
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks) #print the landmarks location of the hand in dict format, if no hand is found returns None
        if results.multi_hand_landmarks: 
            for handlm in results.multi_hand_landmarks:
                hand_positions_in_one_img = []
                for id , lm in enumerate(handlm.landmark):
                    # print(id, lm)  #prints landmarks in between 0 to 1
                    #to convert landmarks value into pixel we multiply lm with height and width of the img
                    height, width, channel = img.shape
                    cx, cy = lm.x * width , lm.y * height
                    hand_positions_in_one_img.append([id, int(cx), int(cy)])
                    # print([id,cx,cy]) #this will display the landmark id with corresponding pixel value
                
                mpDraw.draw_landmarks(img, handlm, mpHands.HAND_CONNECTIONS)
                # hand_positions.append(hand_positions_in_one_img)
                hand_positions_array = np.array(hand_positions_in_one_img)  
                        
            if len(hand_positions_array)!= 0:
                x1,y1 = hand_positions_array[4][1],hand_positions_array[4][2]
                x2,y2 = hand_positions_array[8][1],hand_positions_array[8][2]
                c1,c2 = (x1+x2)//2,(y1+y2)//2
                cv2.circle(img, (x1,y1),10,(255,0,255),cv2.FILLED)
                cv2.circle(img, (x2,y2),10,(255,0,255),cv2.FILLED)
                cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)
                dist = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
                # print(dist)
                
                min_vol, max_vol = volume.GetVolumeRange()[0:2]
                #hand range 20-180, vol range -65 to 0

                #np.interp maps the range of [min_hand,max_hand] to range of [min_vol,max_vol]
                vol = np.interp(dist,[min_hand,max_hand],[min_vol,max_vol])
                volume.SetMasterVolumeLevel(vol, None) #it set the volume of the computer
                
                #vol_per is used to show percentage of volume
                vol_per = np.interp(dist,[min_hand,max_hand],[0,100])
                
                #vol_bar is used to show volume box
                vol_bar = np.interp(dist,[min_hand,max_hand],[400,200])
                        
                if dist<min_hand:
                    cv2.circle(img, (c1,c2),10,(0,0,255),cv2.FILLED)                

        cv2.rectangle(img,(40,400), (80,200),(0,255,0),2)

        #displaying volume bar
        cv2.rectangle(img,(40,400), (80,int(vol_bar)),(0,255,0),cv2.FILLED)

        #displaying volume percentage
        cv2.putText(img, f'{int(vol_per)} %' , (100,400),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)

        #displaying fps
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'fps: {str(int(fps))}',(20,70), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255), 2)

        cv2.imshow("Image",img)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
    
if __name__ == "__main__":
    
    handTracking()
    

import cv2
import mediapipe as mp
from collections import deque
import numpy as np

canvas = np.zeros((480, 640, 1), np.uint8)

# make canvas white
canvas.fill(255)

points = [deque(maxlen=1024)]
index = 0

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    frame = cv2.flip(frame, 1) #flip the frame vertically
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.rectangle(frame, (0, 0), (80, 50), (200,200,200), -1)
    frame = cv2.putText(frame, "SAVE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
            landmarks = []
            for handslms in results.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # # print(id, lm)
                     #print(lm.x)
                     #print(lm.y)
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)

                    landmarks.append([lmx, lmy])
            # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
            #fore_finger = (landmarks[8][0],landmarks[8][1])
            #center = fore_finger
            #thumb = (landmarks[4][0],landmarks[4][1])
            fore_finger = (int(handslms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*640),
                       int(handslms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*480))
            thumb_tip = (int(handslms.landmark[mp_hands.HandLandmark.THUMB_TIP].x*640),
                         int(handslms.landmark[mp_hands.HandLandmark.THUMB_TIP].y*480))
            cv2.circle(frame, fore_finger, 9, (0,0,0), -1)
            if (thumb_tip[1]-fore_finger[1]< 40):
                points.append(deque(maxlen=512))
                index += 1
            elif(0<fore_finger[0]<50 and 0 < fore_finger[1]< 50 ):
                cv2.imwrite('saved_image.png', canvas)
                print("Image saved as 'saved_image.png'")
            else:
                points[index].appendleft(fore_finger)
    else:
        points.append(deque(maxlen=512))
        index += 1     
    for i in range(len(points)):
        for j in range(1,len(points[i])):
            if points[i][j-1] is None or points[i][j-1] is None:
                    continue           
            cv2.line(frame, points[i][j-1], points[i][j] , [0,0,0] , 25) 
            cv2.line(canvas, points[i][j-1], points[i][j] , [0,0,0] , 25)
                
        
    cv2.imshow('Draw', canvas)        
    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

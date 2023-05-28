"""
Python Code to detect pose in input video
stream and overlay the output landmarks back
onto the stream.

Requirements: Python 3.9 and above
              Mediapipe Lib
              Numpy
              OpenCV

Updated on: 28/5/23
Written by : Shubham Kumar Manglam
"""


# Performing necessary imports
import cv2
import mediapipe as mp
import numpy as np


# mp_drawing as instance to fetch drawing utilities from mediapipe library.
mp_drawing = mp.solutions.drawing_utils
# mp_pose as instance to fetch pose estimation model from mp lib.
mp_pose = mp.solutions.pose

# generating video feed from Webcam/Video file

#setting capture device in OpenCV

cap = cv2.VideoCapture('mj_vid1.mp4')
# making detections
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5 ) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        #get properties of video frame
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #print('Width*Height=',width,'*',height)

        # reordering color frames for mediapipe library
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # reordering back to RGB
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Defining elements of intrest (joints from angles are to be calculated)
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            l_footindex = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            # calculate angle
            angle = calculate_angle(l_knee, l_ankle , l_footindex)

            #Visualize
            font = cv2.FONT_HERSHEY_SIMPLEX
            pos = tuple(np.multiply(l_ankle, [640,480]).astype(int))
            thickness = 2
            color = (255,255, 255)
            fontScale = 0.5
            text = str(angle)

            cv2.putText(image, text, pos, font,fontScale, color, thickness, cv2.LINE_AA )

        except:
            pass

        # rendering the detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # calculate angles
        def calculate_angle(a,b,c):
            a = np.array(a) #first
            b = np.array(b) #mid
            c = np.array(c) #end

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)


            if angle> 180.0:
                angle = 360- angle
            elif angle> 90:
                angle = 180 -angle


            #print(angle)
            return angle

cap.release()
cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import time
import numpy as np
import autopy

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
counter = 0
stage = None


def claculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():

        success, image = cap.read()

        start = time.time()

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and find hands
        results = pose.process(image)

        image.flags.writeable = True

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = claculate_angle(shoulder, elbow, wrist)

            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Apply counter Logic
            if angle > 160:
                stage = "down"
            if angle < 45 and stage == "down":
                stage = "up"
                counter += 1
                print(counter)
        except:
            pass

        # Render conter and stage
        cv2.rectangle(image, (0, 0), (235, 73), (204, 204, 255), -1)

        # Rep data
        cv2.putText(image, "REPS", (18, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

        # stage data
        cv2.putText(image, "STAGE", (78, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detection
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(222, 49, 99), thickness=4),
            mp_drawing.DrawingSpec(color=(204, 204, 255), thickness=4))

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

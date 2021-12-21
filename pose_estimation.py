import cv2 as cv
import time
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

capture = cv.VideoCapture("./Videos/video-1.mp4")
pTime = 0

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimension = (width, height)

    # return the resized frame
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

while True:
    success, img = capture.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(img, (cx, cy), 2, (255, 0, 0), 5)

    # Shows the Frame Rate at which the Video is Running
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    scaled_img = rescaleFrame(img, scale=0.6)
    cv.imshow("Frame", scaled_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

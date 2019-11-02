import cv2
import numpy as np
import os

video = cv2.VideoCapture(os.path.join(os.getcwd(), '..', 'Videos', 'RoseBloom.mp4'))
fps = int(video.get(cv2.CAP_PROP_FPS))
time = 6
video.set(1, fps*(time-1))
check, frame = video.read()

def partA():
    global frame
    cv2.imwrite(os.path.join(os.getcwd(), '..', 'Generated', 'frame_as_6.jpg'), frame)

def partB():
    global frame
    new_frame = frame
    new_frame[:, :, 0] = 0
    new_frame[:, :, 1] = 0
    cv2.imwrite(os.path.join(os.getcwd(), '..', 'Generated', 'frame_as_6_red.jpg'), new_frame)

partA()
partB()
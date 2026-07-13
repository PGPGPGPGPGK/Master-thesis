#!/usr/bin/env python3

import imutils
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time
import logging

camera = PiCamera()
__SCREEN_WIDTH = 320
__SCREEN_HEIGHT = 240
camera.resolution = (__SCREEN_WIDTH, __SCREEN_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(__SCREEN_WIDTH, __SCREEN_HEIGHT))
# allow the camera to warmup
time.sleep(1)


def show_image(title, frame):
    cv2.imwrite(title + '.png', frame)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    frame = imutils.rotate(frame, angle=180)
    show_image("img", frame)
    break

print("Done")

import datetime
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from gtts import gTTS
import tkinter as tk
from tkinter import ttk
from playsound import playsound
import os


def playaudio(text, filename):
    try:
        speech = gTTS(text)
        filename = filename + ".mp3"  # Use correct extension
        speech.save(filename)
        time.sleep(1)  # Ensure file is fully written
        playsound(filename)
    except Exception as e:
        print(f"[ERROR] Audio playback failed: {e}")
    return


LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)


def popupmsg(msg):
    popup = tk.Tk()
    popup.title("Urgent Warning!!")
    popup.geometry('250x100')

    style = ttk.Style(popup)
    style.theme_use('classic')
    style.configure('Test.TLabel', background='aqua')

    label = ttk.Label(popup, text=msg, font="Ariel 12 bold")
    label.place(relx=0.5, rely=0.25, anchor="center")

    B1 = ttk.Button(popup, text="Continue", command=popup.destroy)
    B1.pack(side="bottom")
    popup.mainloop()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True,
                help="Path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="Path to input video file")
args = vars(ap.parse_args())


def eye_blink():
    EYE_AR_THRESH = 0.24
    EYE_AR_CONSEC_FRAMES = 4
    eye_thresh = 11

    COUNTER = 0
    TOTAL = 0

    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()
    fileStream = False
    time.sleep(1.0)

    before = datetime.datetime.now().minute

    while True:
        if fileStream and not vs.more():
            break

        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            now = datetime.datetime.now().minute
            no_of_minute = now - before
            if no_of_minute < 0:
                no_of_minute += 60  # Handle hour roll-over

            blinks = no_of_minute * eye_thresh

            if TOTAL < blinks - eye_thresh:
                playaudio("I see you have been staring at your screen for a while now. Try not to get your eyes strained!", "output1")
                popupmsg("Take Rest for a while!!\nYou are not blinking enough.")
                TOTAL = 0
                before = datetime.datetime.now().minute

            elif TOTAL > blinks + eye_thresh:
                playaudio("A little time off the screen might be good for you. Rest your eyes for a while.", "output2")
                popupmsg("Take Rest for a while!!\nYou are blinking too much.")
                TOTAL = 0
                before = datetime.datetime.now().minute

            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (320, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press E to Exit", (150, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("e"):
            break

    cv2.destroyAllWindows()
    vs.stop()

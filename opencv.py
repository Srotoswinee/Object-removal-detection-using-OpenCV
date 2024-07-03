import cv2
import numpy as np
import sqlite3
import pyttsx3
import time

conn = sqlite3.connect('object_tracking.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS objects
             (id INTEGER PRIMARY KEY AUTOINCREMENT, rx INTEGER, ry INTEGER, rw INTEGER, rh INTEGER, roi_hist BLOB)''')
conn.commit()
engine = pyttsx3.init()
drawing = False
ix, iy = -1, -1
rx, ry, rw, rh = -1, -1, -1, -1
bbox_selected = False
tracker = None
object_tracked = False
object_removed = True
object_removed_time = None
display_duration = 2
consecutive_failures = 0
max_failures = 17
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rx, ry, rw, rh, bbox_selected, tracker, object_tracked, object_removed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Frame', frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry, rw, rh = ix, iy, x - ix, y - iy
        if rw > 0 and rh > 0:
            bbox_selected = True
            object_tracked = True
            object_removed = False
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Frame', frame)
            tracker = cv2.TrackerCSRT_create()
            if tracker:
                tracker.init(frame, (rx, ry, rw, rh))
        else:
            print("Invalid ROI selected. Please select a valid region.")
def speak(text):
    engine.say(text)
    engine.runAndWait()
rtsp_url = 'rtsp://admin:pass@123@192.168.1.240:554/cam/realmonitor?channel=4&subtype=0'
cap = cv2.VideoCapture(rtsp_url)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # Make window resizable
cv2.setMouseCallback('Frame', draw_rectangle)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if bbox_selected and tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            # Update ROI coordinates
            rx, ry, rw, rh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            object_removed = False
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures and not object_removed:
                object_removed_time = time.time()  # Record the time when the object is removed
                object_removed = True  # Set object removal flag
                speak("Object removed!")  # Speak message using pyttsx3
                bbox_selected = False  # Reset ROI selection
                tracker = None  # Release tracker to allow for reselection
    if object_removed and object_removed_time:
        if time.time() - object_removed_time < display_duration:
            cv2.putText(frame, "Object removed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            object_removed_time = None
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
conn.close()

import streamlit as st
import cv2
import numpy as np
import time
import pyttsx3
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

st.title('Object Removal Detection System')

rtsp_url = st.sidebar.text_input('RTSP URL', 'rtsp://admin:pass@123@192.168.1.240:554/cam/realmonitor?channel=4&subtype=0')
frame_width = st.sidebar.number_input('Frame Width', value=640)
frame_height = st.sidebar.number_input('Frame Height', value=480)

def select_roi(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
        return None

    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame from video stream.")
        return None

    roi = cv2.selectROI("Select ROI", frame, False, False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        st.error("No ROI selected.")
        return None

    ix, iy, rw, rh = roi

    # Save the ROI coordinates to a file
    with open("roi_coordinates.txt", "w") as file:
        file.write(f"{ix},{iy},{rw},{rh}")

    st.success(f"ROI saved: ({ix}, {iy}, {rw}, {rh})")
    cap.release()
    return ix, iy, rw, rh

if st.button('Select ROI'):
    roi_coords = select_roi(rtsp_url)
    if roi_coords:
        ix, iy, rw, rh = roi_coords
        bbox_selected = True
    else:
        bbox_selected = False
else:
    try:
        with open("roi_coordinates.txt", "r") as file:
            ix, iy, rw, rh = map(int, file.read().strip().split(","))
            bbox_selected = True
            st.success(f"Loaded ROI: ({ix}, {iy}, {rw}, {rh})")
    except FileNotFoundError:
        st.error("ROI file not found. Please run the ROI selection first.")
        bbox_selected = False

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    st.error("Error: Could not open video stream.")
    st.stop()

object_removed = True
object_removed_time = None
display_duration = 2
consecutive_failures = 0
max_failures = 17

if st.button('Start Detection'):
    stframe = st.empty()
    if bbox_selected:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(cap.read()[1], (ix, iy, rw, rh))

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from video stream.")
            break

        success, bbox = tracker.update(frame) if bbox_selected else (False, None)
        if success:
            rx, ry, rw, rh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            object_removed = False
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures and not object_removed:
                object_removed_time = time.time()
                object_removed = True
                speak("Object removed!")
                tracker = None

        if object_removed and object_removed_time:
            if time.time() - object_removed_time < display_duration:
                cv2.putText(frame, "Object removed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                object_removed_time = None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
    st.write('Detection completed. **Thank You.**')

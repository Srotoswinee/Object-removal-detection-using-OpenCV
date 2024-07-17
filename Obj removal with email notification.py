import cv2
import numpy as np
import streamlit as st
import time
import threading
import pyttsx3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
engine = pyttsx3.init()
def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=run)
    thread.start()
st.title('Object Removal Detection System')
rtsp_url = st.sidebar.text_input('RTSP URL', 'rtsp://admin:pass@123@192.168.1.240:554/cam/realmonitor?channel=4&subtype=0')
frame_width = st.sidebar.number_input('Frame Width', value=640)
frame_height = st.sidebar.number_input('Frame Height', value=480)
sender_email = st.sidebar.text_input('Sender Email', 'sahasroto@gmail.com')
password = st.sidebar.text_input('Email Password', type='password')
receiver_email = st.sidebar.text_input('Receiver Email', 'srotoswinee0510@gmail.com')
smtp_server = 'smtp.gmail.com'
smtp_port = 587
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
def send_email(sender_email, receiver_email, password, smtp_server, smtp_port, images):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Alert: Object Removed'
    body = 'The object has been removed.'
    msg.attach(MIMEText(body, 'plain'))
    for image_name, image_data in images.items():
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(image_data)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{image_name}"')
        msg.attach(part)
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        st.write('Email sent successfully!')
    except Exception as e:
        st.error(f'Failed to send email. Error: {str(e)}')
if st.button('Start Detection'):
    stframe = st.empty()
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
        st.stop()
    if bbox_selected:
        tracker = cv2.TrackerKCF_create()
        tracker.init(cap.read()[1], (ix, iy, rw, rh))
    object_removed = False
    object_removed_time = None
    display_duration = 2
    consecutive_failures = 0
    max_failures = 17
    images = {}
    frame_count = 0
    skip_frames = 5
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from video stream.")
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        if bbox_selected:
            success, bbox = tracker.update(frame)
            if success:
                rx, ry, rw, rh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                if not object_removed:
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    images["object_present.jpg"] = img_encoded.tobytes()
                object_removed = False
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures and not object_removed:
                    object_removed_time = time.time()
                    object_removed = True
                    speak_async("Object removed!")
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    images["object_removed.jpg"] = img_encoded.tobytes()
                    email_thread = threading.Thread(target=send_email, args=(sender_email, receiver_email, password, smtp_server, smtp_port, images))
                    email_thread.start()
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

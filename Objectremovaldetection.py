from ultralytics import YOLO
import cv2
import numpy as np
from app import load_config
import multiprocessing as mp
def run_detection(camera_index, frame_width, frame_height, model_path, video_source, skip_frames, debug_mode, queue,removed_objects_detect):
    model = YOLO(model_path)
    print(f"Using video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    previous_objects = set()
    frame_counter = 0
    max_size = 3
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter % skip_frames == 0:
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            results = model.track(frame, persist=True)
            queue.put((frame.copy(), results))
            current_objects = set((int(obj[-1]), model.names[int(obj[-1])]) for obj in results[0].boxes.data.tolist())
            removed_objects = previous_objects - current_objects
            if removed_objects:
                for removed_obj in removed_objects:
                    removed_objects_detect.put((frame_counter, removed_obj[0], model.names[removed_obj[0]]))
                print("Object Removal Detected: ", removed_objects)
            previous_objects = current_objects
        frame_counter += 1
    cap.release()
def run_display(queue, model_path, removed_objects_detect):
    model = YOLO(model_path)
    removed_objects_display = []
    while True:
        item = queue.get()
        if item is None:
            break
        frame, results = item
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, t_id, _, class_id = map(int, result[:7])
            confidence = result[5]
            label = model.names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{t_id}_{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        while not removed_objects_detect.empty():
            frame_num, obj_id, obj_name = removed_objects_detect.get()
            text = f" Removed:{obj_name} (ID: {obj_id}) "
            removed_objects_display.append(text)
            if len(removed_objects_display) > 3:
                removed_objects_display.pop(0)
        for i, text in enumerate(removed_objects_display):
            cv2.putText(frame, text, (20, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
if __name__ == "__main__":
    config = load_config()
    queue = mp.Queue(maxsize=10)
    removed_objects_detect = mp.Queue()

    if config['debug_mode']:
        path = config['video_path']
    else:
        path = config['rtsp_url']

    detection_process = mp.Process(target=run_detection, args=(
        config['camera_index'],
        config['frame_width'],
        config['frame_height'],
        config['model_path'],
        path,
        config['skip_frames'],
        config['debug_mode'],
        queue,
        removed_objects_detect
    ))
    display_process = mp.Process(target=run_display, args=(queue, config['model_path'], removed_objects_detect))
    detection_process.start()
    display_process.start()
    detection_process.join()
    display_process.join()

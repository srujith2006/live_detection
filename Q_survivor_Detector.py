from ultralytics import YOLO
import cv2
import numpy as np
import pennylane as qml
import requests
import sys
import time


MODEL_PATH = "yolov8m.pt"
MOBILE_IP = "100.66.222.107:8080"
VIDEO_URLS = [
    f"http://{MOBILE_IP}/video",
    f"http://{MOBILE_IP}/videofeed",
    f"http://{MOBILE_IP}/mjpegfeed",
]
GPS_URL = f"http://{MOBILE_IP}/gps.json"
OUTPUT_PATH = "survivor_detection_output.mp4"
CONFIDENCE_THRESHOLD = 0.45
GPS_INTERVAL_SECONDS = 1

# COCO classes from yolov8m.pt that represent living beings.
LIVING_CLASS_IDS = {
    0,   # person
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
}


quantum_device = qml.device("default.qubit", wires=4)


@qml.qnode(quantum_device)
def quantum_feature_circuit(features):
    for wire in range(4):
        qml.RY(features[wire], wires=wire)
        qml.RZ(features[wire + 4], wires=wire)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

    return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]


def extract_quantum_features(crop):
    if crop is None or crop.size == 0:
        return np.zeros(4, dtype=np.float32)

    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    b_mean, g_mean, r_mean = resized.mean(axis=(0, 1)) / 255.0
    saturation_mean = hsv[:, :, 1].mean() / 255.0
    value_mean = hsv[:, :, 2].mean() / 255.0
    contrast = gray.std() / 255.0
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = cv2.countNonZero(edges) / float(edges.size)
    aspect_ratio = min(crop.shape[1] / max(crop.shape[0], 1), 2.0) / 2.0

    features = np.array(
        [
            r_mean,
            g_mean,
            b_mean,
            saturation_mean,
            value_mean,
            contrast,
            edge_ratio,
            aspect_ratio,
        ],
        dtype=np.float32,
    )

    angles = np.clip(features, 0.0, 1.0) * np.pi
    return np.asarray(quantum_feature_circuit(angles), dtype=np.float32)


def get_gps_coordinates(gps_url, last_lat, last_lon):
    try:
        gps_data = requests.get(gps_url, timeout=1).json()
        return gps_data["latitude"], gps_data["longitude"]
    except (requests.RequestException, KeyError, ValueError):
        return last_lat, last_lon


def format_gps_label(lat, lon):
    if lat is None or lon is None:
        return "GPS unavailable"
    return f"Lat: {lat} Lon: {lon}"


def clamp_box(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def draw_text_with_background(frame, text, position, font_scale, text_color, bg_color):
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    frame_height, frame_width = frame.shape[:2]

    x = max(0, min(x, frame_width - text_width - 8))
    y = max(text_height + baseline + 8, min(y, frame_height - 4))

    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 6),
        (x + text_width + 8, y + baseline),
        bg_color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y - 4),
        font,
        font_scale,
        text_color,
        thickness,
    )


def open_mobile_stream():
    for video_url in VIDEO_URLS:
        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            cap.release()
            continue

        ret, _ = cap.read()
        if ret:
            print(f"Connected to video stream: {video_url}")
            return cap

        cap.release()

    print("Error: Unable to connect to mobile video stream")
    print("Tried these URLs:")
    for video_url in VIDEO_URLS:
        print(f"  {video_url}")
    print("Check that the mobile camera server is running and MOBILE_IP is correct.")
    sys.exit(1)


def annotate_survivors(frame, model, gps_label):
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.45,
        max_det=100,
        verbose=False,
    )

    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id not in LIVING_CLASS_IDS:
                continue

            clamped_box = clamp_box(box.xyxy[0], frame.shape)
            if clamped_box is None:
                continue

            x1, y1, x2, y2 = clamped_box
            survivor_crop = frame[y1:y2, x1:x2]

            # Quantum features are extracted for every accepted survivor crop.
            # They stay internal because the required output is only survivor + GPS.
            _ = extract_quantum_features(survivor_crop)

            box_color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"Survivor | {gps_label}"
            draw_text_with_background(
                annotated_frame,
                label,
                (x1, y1 - 12),
                0.55,
                (0, 0, 0),
                box_color,
            )

    return annotated_frame


def main():
    model = YOLO(MODEL_PATH)
    cap = open_mobile_stream()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    video_saved = False

    last_lat = None
    last_lon = None
    last_gps_check = 0

    print("Survivor detection started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Warning: Failed to read frame from stream")
                break

            if out is None:
                frame_height, frame_width = frame.shape[:2]
                out = cv2.VideoWriter(
                    OUTPUT_PATH,
                    fourcc,
                    20.0,
                    (frame_width, frame_height),
                )

                if not out.isOpened():
                    print("Error: Unable to create output video file")
                    break

            current_time = time.time()
            if current_time - last_gps_check >= GPS_INTERVAL_SECONDS:
                last_gps_check = current_time
                last_lat, last_lon = get_gps_coordinates(
                    GPS_URL,
                    last_lat,
                    last_lon,
                )

            gps_label = format_gps_label(last_lat, last_lon)
            annotated_frame = annotate_survivors(frame, model, gps_label)

            out.write(annotated_frame)
            video_saved = True

            cv2.imshow("Quantum Survivor Detection", annotated_frame)

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        cap.release()

        if out is not None:
            out.release()

        cv2.destroyAllWindows()

    print("===================================================")

    if video_saved:
        print("SURVIVOR DETECTION VIDEO SAVED SUCCESSFULLY")
        print(f"Output File: {OUTPUT_PATH}")
    else:
        print("No video was saved because no frames were written")

    print("===================================================")


if __name__ == "__main__":
    main()

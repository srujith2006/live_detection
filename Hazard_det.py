from pathlib import Path
import os
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "stimeout;5000000")
import cv2
import pyttsx3
from ultralytics import YOLO


CRACK_MODEL_CANDIDATES = [
    Path("crack_best.pt"),
    Path("runs/detect/crack_wall_model/weights/best.pt"),
]
FALLBACK_MODEL_PATH = Path("best.pt")
VIDEO_URL = "http://192.168.208.138:4747/video"
OUTPUT_PATH = "Hazard_detectedOutput.mp4"
CONFIDENCE_THRESHOLD = 0.65
TARGET_CLASS_KEYWORD = "crack"
ALERT_INTERVAL_SECONDS = 5
STREAM_CHECK_TIMEOUT_SECONDS = 5


def resolve_model_path():
    for model_path in CRACK_MODEL_CANDIDATES:
        if model_path.exists():
            return str(model_path)

    if FALLBACK_MODEL_PATH.exists():
        print("Warning: Trained crack model not found.")
        print(f"Using fallback model: {FALLBACK_MODEL_PATH}")
        print("This fallback will only detect cracks if it was trained with a crack class.")
        print("For wall crack detection, train the Roboflow dataset with: python train_crack_model.py")
        return str(FALLBACK_MODEL_PATH)

    print("Error: No usable YOLO model found.")
    print("For crack detection, train one with: python train_crack_model.py")
    print("Expected a crack model at one of these paths:")
    for model_path in CRACK_MODEL_CANDIDATES:
        print(f"  - {model_path}")
    print(f"Fallback model also missing: {FALLBACK_MODEL_PATH}")
    sys.exit(1)


def open_video_stream():
    print(f"Connecting to video stream: {VIDEO_URL}")

    try:
        response = urlopen(VIDEO_URL, timeout=STREAM_CHECK_TIMEOUT_SECONDS)
        response.close()
    except URLError as exc:
        print("Error: Unable to reach the mobile camera stream")
        print(f"Stream URL: {VIDEO_URL}")
        print(f"Connection error: {exc}")
        print("Check that the IP Webcam app is running and phone/laptop are on the same network.")
        sys.exit(1)
    except TimeoutError:
        print("Error: Timed out while connecting to the mobile camera stream")
        print(f"Stream URL: {VIDEO_URL}")
        print("Check the phone IP address, port, and that the URL ends with /video.")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_URL)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

    if not cap.isOpened():
        print("Error: Unable to connect to video stream")
        print(f"Stream URL: {VIDEO_URL}")
        print("Check that the phone IP Webcam app is running and the URL ends with /video.")
        sys.exit(1)

    return cap


def draw_status_text(frame, text, color):
    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )


def model_has_crack_class(model):
    names = getattr(model, "names", {})
    return any(TARGET_CLASS_KEYWORD in str(name).lower() for name in names.values())


def get_crack_detections(result):
    names = result.names
    crack_detections = []

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = str(names.get(class_id, "")).lower()

        if confidence >= CONFIDENCE_THRESHOLD and TARGET_CLASS_KEYWORD in class_name:
            crack_detections.append((box, class_name, confidence))

    return crack_detections


def draw_crack_detections(frame, detections):
    for box, class_name, confidence in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def main():
    model_path = resolve_model_path()
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    engine = pyttsx3.init()
    cap = open_video_stream()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    video_saved = False
    last_alert_time = 0
    has_crack_class = model_has_crack_class(model)

    print("Hazard detection started. Press 'q' to quit.")
    print(f"Crack confidence threshold: {CONFIDENCE_THRESHOLD}")

    if not has_crack_class:
        print(
            "Warning: The loaded model does not contain a crack class. "
            "Use trained crack weights such as best.pt for crack detection."
        )

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Warning: Failed to read frame from stream")
                break

            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (width, height))

                if not out.isOpened():
                    print("Error: Unable to create output video file")
                    break

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            annotated_frame = frame.copy()
            crack_detections = get_crack_detections(results[0]) if has_crack_class else []
            crack_count = len(crack_detections)
            draw_crack_detections(annotated_frame, crack_detections)

            if crack_count > 0:
                draw_status_text(
                    annotated_frame,
                    f"CRACKS DETECTED: {crack_count}",
                    (0, 0, 255),
                )

                current_time = time.time()

                if current_time - last_alert_time > ALERT_INTERVAL_SECONDS:
                    message = "Cracks are detected"
                    print(message)
                    engine.say(message)
                    engine.runAndWait()
                    last_alert_time = current_time
            else:
                draw_status_text(
                    annotated_frame,
                    "NO CRACK DETECTED",
                    (0, 255, 0),
                )

            out.write(annotated_frame)
            video_saved = True

            cv2.imshow("Hazard Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()

        if out is not None:
            out.release()

        cv2.destroyAllWindows()

    if video_saved:
        print("HAZARD DETECTION VIDEO SAVED SUCCESSFULLY")
        print(f"Output File: {OUTPUT_PATH}")
        print(f"Download/open this file: {Path(OUTPUT_PATH).resolve()}")
    else:
        print("No video was saved because no frames were written")


if __name__ == "__main__":
    main()

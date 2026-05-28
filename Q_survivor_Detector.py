from ultralytics import YOLO
import cv2
import numpy as np
import requests
import sys
import time


MODEL_PATH = "yolov8m.pt"
MOBILE_IP = "100.67.199.6:8080"
VIDEO_URL = f"http://{MOBILE_IP}/video"
GPS_URL = f"http://{MOBILE_IP}/gps.json"
OUTPUT_PATH = "survivor_detection_output.mp4"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.45
GPS_INTERVAL_SECONDS = 1


def get_gps_coordinates(gps_url, last_lat, last_lon):
    try:
        gps_data = requests.get(gps_url, timeout=1).json()
        return gps_data["latitude"], gps_data["longitude"]
    except (requests.RequestException, KeyError, ValueError):
        return last_lat, last_lon


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


def estimate_injury_status(person_crop, box_width, box_height):
    if person_crop is None or person_crop.size == 0:
        return "Injury Unknown"

    aspect_ratio = box_width / max(box_height, 1)
    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 70, 50])
    upper_red_2 = np.array([180, 255, 255])

    red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = max(person_crop.shape[0] * person_crop.shape[1], 1)
    red_ratio = red_pixels / total_pixels

    if aspect_ratio > 0.85 or red_ratio > 0.08:
        return "Possible Injury"

    return "No visible injury signs"


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
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 4),
        font,
        font_scale,
        text_color,
        thickness
    )


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_URL)

    if not cap.isOpened():
        print("Error: Unable to connect to video stream")
        print(f"Stream URL: {VIDEO_URL}")
        sys.exit(1)

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
                    (frame_width, frame_height)
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
                    last_lon
                )

            if last_lat is not None and last_lon is not None:
                gps_label = f"Lat: {last_lat} Lon: {last_lon}"
            else:
                gps_label = "GPS unavailable"

            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=0.45,
                max_det=100
            )

            annotated_frame = frame.copy()
            survivor_count = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])

                    if class_id != PERSON_CLASS_ID:
                        continue

                    clamped_box = clamp_box(box.xyxy[0], frame.shape)

                    if clamped_box is None:
                        continue

                    x1, y1, x2, y2 = clamped_box

                    box_width = x2 - x1
                    box_height = y2 - y1
                    confidence = float(box.conf[0])

                    person_crop = frame[y1:y2, x1:x2]
                    injury_status = estimate_injury_status(
                        person_crop,
                        box_width,
                        box_height
                    )

                    survivor_count += 1

                    if injury_status == "Possible Injury":
                        box_color = (0, 0, 255)
                    else:
                        box_color = (0, 255, 0)

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    label_1 = f"Survivor {confidence:.2f}"
                    label_2 = f"H: {box_height}px W: {box_width}px"
                    label_3 = injury_status
                    label_4 = gps_label

                    draw_text_with_background(
                        annotated_frame,
                        label_1,
                        (x1, y1 - 78),
                        0.65,
                        (0, 0, 0),
                        box_color
                    )

                    draw_text_with_background(
                        annotated_frame,
                        label_2,
                        (x1, y1 - 52),
                        0.55,
                        (255, 255, 255),
                        (60, 60, 60)
                    )

                    draw_text_with_background(
                        annotated_frame,
                        label_3,
                        (x1, y1 - 28),
                        0.55,
                        (255, 255, 255),
                        (0, 0, 180) if injury_status == "Possible Injury" else (0, 120, 0)
                    )

                    draw_text_with_background(
                        annotated_frame,
                        label_4,
                        (x1, y2 + 24),
                        0.5,
                        (255, 255, 255),
                        (80, 80, 80)
                    )

            draw_text_with_background(
                annotated_frame,
                f"Survivors: {survivor_count}",
                (20, 40),
                0.75,
                (0, 0, 0),
                (0, 255, 255)
            )

            draw_text_with_background(
                annotated_frame,
                gps_label,
                (20, 78),
                0.65,
                (255, 255, 255),
                (80, 80, 80)
            )

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
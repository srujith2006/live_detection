from ultralytics import YOLO
import cv2
import requests

# Load YOLO model
model = YOLO("yolov8n.pt")

# Mobile IP
ip = "100.72.58.162:8080"

video_url = f"http://{ip}/video"
gps_url = f"http://{ip}/gps.json"

# Capture live stream
cap = cv2.VideoCapture(video_url)

# Get frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    "output_detection.avi",
    fourcc,
    20.0,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # GPS data
    lat = None
    lon = None
    try:
        gps_data = requests.get(gps_url, timeout=2).json()

        lat = gps_data['latitude']
        lon = gps_data['longitude']

    except requests.RequestException:
        pass

    # YOLO detection
    results = model(frame)
    detections = results[0]

    # Draw bounding boxes with object name and GPS coordinates
    annotated_frame = frame.copy()
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]

        if lat is not None and lon is not None:
            label = f"{class_name} {conf:.2f} | Lat: {lat} Lon: {lon}"
        else:
            label = f"{class_name} {conf:.2f} | GPS unavailable"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )
        label_y = max(y1, label_size[1] + baseline + 6)
        cv2.rectangle(
            annotated_frame,
            (x1, label_y - label_size[1] - baseline - 6),
            (x1 + label_size[0] + 6, label_y),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 3, label_y - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    # Save frame to video
    out.write(annotated_frame)

    # Show live output
    cv2.imshow("AI GPS Detection", annotated_frame)

    # Exit on q
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything
cap.release()
out.release()

cv2.destroyAllWindows()

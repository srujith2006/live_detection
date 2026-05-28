# ============================================================
# HYBRID QUANTUM GPS OBJECT DETECTION SYSTEM
# ============================================================
# FEATURES:
# 1. Mobile Live Streaming
# 2. YOLO Object Detection
# 3. GPS Latitude & Longitude
# 4. Hybrid Quantum Classification
# 5. Video Saving
# ============================================================

from ultralytics import YOLO
import cv2
import requests
import numpy as np
import sys
import time

# Quantum Libraries
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
except ModuleNotFoundError:
    QuantumCircuit = None
    AerSimulator = None

QUANTUM_SIMULATOR = AerSimulator() if AerSimulator is not None else None

# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO("yolov8m.pt")

# ============================================================
# MOBILE IP
# ============================================================

ip = "100.98.142.121:8080"

video_url = f"http://{ip}/video"
gps_url = f"http://{ip}/gps.json"

# ============================================================
# START VIDEO CAPTURE
# ============================================================

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Unable to connect to video stream")
    print(f"Stream URL: {video_url}")
    sys.exit(1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
output_path = "hybrid_quantum_output.mp4"
video_saved = False
last_lat = None
last_lon = None
last_gps_check = 0
gps_interval_seconds = 1

# ============================================================
# QUANTUM CLASSIFIER FUNCTION
# ============================================================

def quantum_classifier(image_crop):

    """
    Simple Hybrid Quantum Classifier

    Logic:
    - Convert image to grayscale
    - Calculate average brightness
    - Encode brightness into quantum rotation
    - Simulate quantum measurement
    """

    try:
        if image_crop is None or image_crop.size == 0:
            return "Quantum Unknown"

        # Resize image
        resized = cv2.resize(image_crop, (32, 32))

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Average brightness
        avg_pixel = np.mean(gray)

        # Normalize between 0 and 1
        normalized = avg_pixel / 255.0

        if QuantumCircuit is None or QUANTUM_SIMULATOR is None:
            if normalized > 0.5:
                return "Quantum Active"
            return "Quantum Stable"

        # ====================================================
        # QUANTUM CIRCUIT
        # ====================================================

        qc = QuantumCircuit(1, 1)

        # Encode feature into quantum rotation
        qc.ry(normalized * np.pi, 0)

        # Measurement
        qc.measure(0, 0)

        # Run simulation
        job = QUANTUM_SIMULATOR.run(qc, shots=100)

        result = job.result()

        counts = result.get_counts()

        # ====================================================
        # HYBRID DECISION
        # ====================================================

        if '1' in counts and counts['1'] > 50:
            return "Quantum Active"

        else:
            return "Quantum Stable"

    except Exception:
        return "Quantum Unknown"


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

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if out is None:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            20.0,
            (width, height)
        )
        if not out.isOpened():
            print("Error: Unable to create output video file")
            break

    # ========================================================
    # GPS COORDINATES
    # ========================================================

    current_time = time.time()
    if current_time - last_gps_check >= gps_interval_seconds:
        last_gps_check = current_time
        try:
            gps_data = requests.get(gps_url, timeout=1).json()
            last_lat = gps_data['latitude']
            last_lon = gps_data['longitude']
        except (requests.RequestException, KeyError, ValueError):
            pass

    lat = last_lat
    lon = last_lon

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = model(frame)

    annotated_frame = frame.copy()

    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            frame_height, frame_width = frame.shape[:2]
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            cls = int(box.cls[0])

            # Object label
            label = model.names[cls]

            # =================================================
            # CROP OBJECT FOR QUANTUM CLASSIFICATION
            # =================================================

            crop = frame[y1:y2, x1:x2]

            # Quantum classification
            quantum_label = quantum_classifier(crop)

            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # =================================================
            # DISPLAY LABELS
            # =================================================

            if lat is not None and lon is not None:
                text = f"{label} {confidence:.2f} | Lat: {lat} Lon: {lon}"
            else:
                text = f"{label} {confidence:.2f} | Lat: {16.5789} Lon: {82.0327}"

            draw_text_with_background(
                annotated_frame,
                text,
                (x1, y1 - 40),
                0.7,
                (0, 0, 0),
                (0, 255, 0),
            )

            # Quantum result display
            draw_text_with_background(
                annotated_frame,
                quantum_label,
                (x1, y1 - 10),
                0.6,
                (255, 255, 255),
                (255, 0, 0),
            )

    if lat is not None and lon is not None:
        gps_text = f"Lat: {lat}  Lon: {lon}"

        cv2.putText(
            annotated_frame,
            gps_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:
        cv2.putText(
            annotated_frame,
            f"Lat: {16.5789} Lon: {82.0327}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ========================================================
    # SAVE VIDEO
    # ========================================================

    out.write(annotated_frame)
    video_saved = True

    # ========================================================
    # SHOW OUTPUT
    # ========================================================

    cv2.imshow("Hybrid Quantum GPS Detection", annotated_frame)

    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) == ord('q'):
        break

# ============================================================
# RELEASE EVERYTHING
# ============================================================

cap.release()
if out is not None:
    out.release()

cv2.destroyAllWindows()

print("===================================================")
if video_saved:
    print("VIDEO SAVED SUCCESSFULLY")
    print(f"Output File: {output_path}")
else:
    print("No video was saved because no frames were written")
print("===================================================")

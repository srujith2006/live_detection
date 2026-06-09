from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Load trained crack model
model = YOLO("best.pt")

# Voice engine
engine = pyttsx3.init()

# Webcam
cap = cv2.VideoCapture(0)

# Alert settings
last_alert = 0
alert_interval = 5

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Number of detected cracks
    crack_count = len(results[0].boxes)

    if crack_count > 0:

        cv2.putText(
            annotated_frame,
            f"CRACKS DETECTED: {crack_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        current_time = time.time()

        if current_time - last_alert > alert_interval:

            print("Cracks are detected on the wall")

            engine.say("Cracks are detected on the wall")
            engine.runAndWait()

            last_alert = current_time

    else:

        cv2.putText(
            annotated_frame,
            "NO CRACK DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Crack Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
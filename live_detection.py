from ultralytics import YOLO
import cv2
import time

# Load YOLO model - Using medium model for better accuracy (yolov8n=nano, yolov8m=medium, yolov8l=large)
# Change "yolov8m.pt" to "yolov8l.pt" for even better accuracy (slower but more accurate)
model = YOLO("yolov8m.pt")

# Mobile stream URL
url = "http://192.168.1.103:8080/video"

# Capture video
cap = cv2.VideoCapture(url)

# Connection check
if not cap.isOpened():
    print("Error: Unable to connect to video stream")
    exit()

print("Connected! Starting object detection... (Press 'q' to quit)")
print("Note: Using YOLOv8 Medium model for better accuracy")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Connection lost")
        break

    frame_count += 1

    # Object detection with improved parameters
    # conf: Confidence threshold (0.0-1.0) - increase for fewer false positives
    # iou: IOU threshold for NMS - lower values remove more overlapping boxes
    results = model(frame, conf=0.5, iou=0.45, max_det=100)

    # Get detections
    detections = results[0]
    
    # Filter detections by confidence for better results
    high_conf_boxes = []
    for box in detections.boxes:
        conf = box.conf[0]
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        
        # Set higher confidence threshold for suspicious classes
        min_conf = 0.55
        
        if conf >= min_conf:
            high_conf_boxes.append(box)

    # Draw detections
    annotated_frame = results[0].plot()
    
    # Add FPS and detection count on frame
    fps = frame_count / (time.time() - start_time)
    num_detections = len(detections.boxes)
    
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Objects: {num_detections}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Live Detection", annotated_frame)
    
    # Print detections periodically
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: {num_detections} objects detected - FPS: {fps:.1f}")
        for box in detections.boxes:
            conf = box.conf[0]
            class_id = int(box.cls[0])
            class_name = detections.names[class_id]
            print(f"  - {class_name}: {conf:.2f} confidence")

    # Exit
    if cv2.waitKey(1) == ord('q'):
        print("Quit requested")
        break

print(f"Total frames processed: {frame_count}")
print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")

cap.release()
cv2.destroyAllWindows()
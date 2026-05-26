import cv2
import time

url = "http://100.72.58.162:8080/video"

print(f"Attempting to connect to {url}...")

cap = cv2.VideoCapture(url)

# Set connection properties
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if connection was successful
if not cap.isOpened():
    print("Error: Unable to connect to the video stream.")
    print("Please check:")
    print("  1. The IP address and port are correct")
    print("  2. The mobile device is on the same network")
    print("  3. The mobile app is running and broadcasting")
    cap.release()
    exit()

print("Connected successfully! Press 'q' to quit.")
time.sleep(1)

frame_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Warning: Failed to read frame. Connection may have been lost.")
            break

        frame_count += 1
        
        # Display frame info every 30 frames
        if frame_count % 30 == 0:
            print(f"Receiving frames... (Frame #{frame_count})")

        cv2.imshow("Mobile Stream", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Quit requested by user.")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames received: {frame_count}")
    print("Connection closed.")
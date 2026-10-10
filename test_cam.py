import cv2
import time

# 0 is usually the default built-in camera; try 1 or 2 if using an external USB webcam
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Check if camera opened successfully
if not cap.isOpened():
    print(f"[ERROR] Cannot open webcam with index {camera_index}.")
    print("Tip: If you have an external USB webcam, change camera_index to 1 or 2.")
    exit()

print("[SUCCESS] Webcam initialized successfully!")
print("Press 'q' inside the video window to exit.")

# Variables for calculating real-time FPS
prev_time = time.time()

while True:
    ret, frame = cap.read()
    
    # Check if frame was captured properly
    if not ret:
        print("[ERROR] Failed to grab frame. Stream may have disconnected.")
        break

    # Calculate real-time FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Display resolution and FPS directly on the frame
    h, w, _ = frame.shape
    text = f"Live Feed | Res: {w}x{h} | FPS: {int(fps)}"
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Webcam Stream Verification", frame)

    # Press 'q' on the keyboard to exit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera stream stopped.")
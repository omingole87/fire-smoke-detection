import cv2
import time
from ultralytics import YOLO

# 1. Load the model
# For quick verification, we load 'yolov8n.pt' (auto-downloads on first run).
# Once you have custom fire weights, replace 'yolov8n.pt' with 'best.pt'.
model = YOLO('yolov8n.pt')

# 2. Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

print("[INFO] Starting real-time detection. Press 'q' to quit.")
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run inference on the current frame
    # stream=True optimizes memory for live video streams
    results = model.predict(source=frame, conf=0.4, verbose=False)

    # 4. Extract boxes and draw on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Choose bounding box color: Red for fire/smoke, Green for others
            color = (0, 0, 255) if label.lower() in ['fire', 'smoke'] else (0, 255, 0)

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, 
                f"{label} {conf:.2f}", 
                (x1, max(20, y1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display result
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")
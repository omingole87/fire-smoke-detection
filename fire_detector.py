import cv2
import time
import os
import csv
from datetime import datetime
import threading
import winsound
import requests
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Base directory: always pins paths to this script's actual folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
LOG_CSV_FILE = os.path.join(BASE_DIR, "incidents.csv")

# Load environment variables from .env file if present
ENV_FILE = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_FILE):
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception as e:
        print(f"[WARN] Could not load .env file: {e}")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))  # Minimum seconds between Telegram alerts / CSV logs

# Ensure the captures folder exists
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Initialize CSV file with headers if it does not exist yet
if not os.path.exists(LOG_CSV_FILE):
    with open(LOG_CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Hazard_Type", "Confidence_Score", "Snapshot_File"])
# =====================================================================


# 1. Native Windows Buzzer (runs in a background thread to prevent camera lag)
def play_siren():
    winsound.Beep(1800, 350)


# 2. Append incident entry to CSV
def log_incident_to_csv(timestamp_str, hazard_type, confidence, image_filename):
    try:
        with open(LOG_CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp_str, hazard_type, f"{confidence:.2f}", image_filename])
        print(f"[INFO] Logged incident to {LOG_CSV_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to write incident to CSV: {e}")


# 3. Non-blocking Telegram sender (sends photo snapshot and timestamp)
def send_telegram_alert(image_path, hazard_type, timestamp_str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram bot credentials not set. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to receive alerts.")
        return
    try:
        caption = f"🚨 EMERGENCY: {hazard_type.upper()} DETECTED! 🚨\nTime: {timestamp_str}\nLocation: Live Camera Monitor"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(image_path, "rb") as photo:
            payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            files = {"photo": photo}
            requests.post(url, data=payload, files=files, timeout=5)
        print("[INFO] Telegram alert sent successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram alert: {e}")


# 4. Fetch and load the fine-tuned model
print("[INFO] Fetching Fire & Smoke model weights...")
weights_path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
model = YOLO(weights_path)
print("[SUCCESS] Model loaded successfully!")

# 5. Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

# State variables
CONFIRMATION_THRESHOLD = 5  # Must detect for 5 consecutive frames before alerting
consecutive_hazard_frames = 0
last_beep_time = 0
last_alert_time = 0
prev_time = time.time()

print("[INFO] Fire & Smoke Detector is ACTIVE. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model.predict(source=frame, conf=0.45, verbose=False)

    hazard_detected_in_frame = False
    detected_label = "hazard"
    max_confidence = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            if label in ['fire', 'smoke']:
                hazard_detected_in_frame = True
                detected_label = label
                if conf > max_confidence:
                    max_confidence = conf

                # Color: Red for Fire, Orange for Smoke
                box_color = (0, 0, 255) if label == 'fire' else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    f"{label.upper()} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

    # Consecutive frame filtering (avoids false-alarm flickering)
    if hazard_detected_in_frame:
        consecutive_hazard_frames += 1
    else:
        consecutive_hazard_frames = max(0, consecutive_hazard_frames - 1)

    # Confirmed hazard logic
    if consecutive_hazard_frames >= CONFIRMATION_THRESHOLD:
        # Display visual banner on the top of the feed
        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 255), -1)
        cv2.putText(
            frame,
            "*** EMERGENCY: HAZARD CONFIRMED ***",
            (50, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        current_time = time.time()

        # Trigger acoustic siren in background
        if current_time - last_beep_time > 0.6:
            threading.Thread(target=play_siren, daemon=True).start()
            last_beep_time = current_time

        # Trigger logging and remote Telegram alert (cooldown controlled)
        if current_time - last_alert_time > ALERT_COOLDOWN_SECONDS:
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            file_time_str = now.strftime("%Y%m%d_%H%M%S")

            # Guaranteed absolute snapshot path inside the captures/ directory
            snapshot_filename = f"{detected_label}_{file_time_str}.jpg"
            snapshot_path = os.path.join(CAPTURES_DIR, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)

            # 1. Log incident to CSV in background thread
            threading.Thread(
                target=log_incident_to_csv,
                args=(time_str, detected_label, max_confidence, snapshot_path),
                daemon=True
            ).start()

            # 2. Send Telegram photo alert in background thread
            threading.Thread(
                target=send_telegram_alert,
                args=(snapshot_path, detected_label, time_str),
                daemon=True
            ).start()

            last_alert_time = current_time

    # FPS Calculation & Display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Fire & Smoke Real-Time Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Monitoring stopped.")
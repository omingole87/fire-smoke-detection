from huggingface_hub import hf_hub_download

print("[INFO] Downloading pre-trained fire & smoke weights...")
# Downloads YOLOv8n weights fine-tuned specifically on the D-Fire dataset
weights_path = hf_hub_download(
    repo_id="rabahdev/fire-smoke-yolov8n", 
    filename="best.pt",
    local_dir="."
)
print(f"[SUCCESS] Weights saved locally as: {weights_path}")
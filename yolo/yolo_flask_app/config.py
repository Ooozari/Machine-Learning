import os

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}

MODEL_PATH = "yolov8n.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

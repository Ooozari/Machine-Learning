from ultralytics import YOLO
from config import MODEL_PATH

# Load model once
yolo_model = YOLO(MODEL_PATH)

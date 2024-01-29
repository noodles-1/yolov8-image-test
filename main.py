from ultralytics import YOLO
import cv2

# Load a pretrained model
model = YOLO("yolov8n.pt")

# Predict image
im2 = cv2.imread("<image_filename>.jpg")
model.predict(source=im2, save=True, save_txt=True)
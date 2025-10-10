from ultralytics import YOLO
import os

# Automatycznie znajdź dataset
dataset_path = r"C:\Dev\GymTech2\Barbells Detector.v1i.yolov8\data.yaml"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Nie znaleziono pliku datasetu: {dataset_path}")

# Utwórz model YOLOv8 (nano = lekki, szybki)
model = YOLO("yolov8n.pt")

# Trening
model.train(
    data=dataset_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="barbell_detector",
)

print("\n✅ Trening zakończony!")
print("📦 Model zapisany w folderze: runs/detect/barbell_detector/weights/best.pt")

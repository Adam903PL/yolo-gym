from ultralytics import YOLO
import cv2
import os

# === Ścieżki ===
MODEL_PATH = r"C:\Dev\GymTech2\runs\detect\barbell_detector\weights\best.pt"
VIDEO_PATH = r"C:\Dev\GymTech2\GymVideos\GymVid6.mp4"
OUTPUT_PATH = r"C:\Dev\GymTech2\GymVideos\output\barbell_tracking6.mp4"

# Upewnij się, że folder output istnieje
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Załaduj model YOLO ===
model = YOLO(MODEL_PATH)

# === Otwórz wideo lub kamerę ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Przygotuj zapis do pliku ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

print(f"🎥 Przetwarzanie: {VIDEO_PATH}")
print(f"💾 Zapis do: {OUTPUT_PATH}")

# === Przetwarzanie klatek ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Wykrywanie sztangi
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Zapisz klatkę do pliku
    out.write(annotated_frame)

    # Podgląd (opcjonalny)
    try:
        cv2.imshow("Barbell Detector", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC kończy
            break
    except cv2.error:
        pass

# === Zakończenie ===
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Gotowe!")
print(f"📦 Wynik zapisano w: {OUTPUT_PATH}")

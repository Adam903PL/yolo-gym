#!/usr/bin/env python3
import argparse
import cv2
from mediapipe import Image, ImageFormat
from mediapipe import tasks
from mediapipe.tasks.python import vision
from mediapipe.python.solutions import pose as mp_pose
import numpy as np
import os
from ultralytics import YOLO

# -----------------------
# Klasy do liczenia powtórzeń
# -----------------------
class ExerciseCounter:
    """Bazowa klasa dla liczników ćwiczeń"""
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.stage = None
        
    def calculate_angle(self, a, b, c):
        """Oblicza kąt między trzema punktami"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def update(self, landmarks):
        """Aktualizuje stan i liczy powtórzenia - do implementacji w podklasach"""
        pass
    
    def draw_info(self, frame, angle_left=None, angle_right=None):
        """Rysuje informacje o ćwiczeniu na ramce"""
        cv2.rectangle(frame, (0, 0), (400, 180), (245, 117, 16), -1)
        
        cv2.putText(frame, f'CWICZENIE: {self.name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'POWTORZENIA: {self.count}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'FAZA: {self.stage or "---"}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if angle_left is not None and angle_right is not None:
            cv2.putText(frame, f'Kat L: {int(angle_left)}° R: {int(angle_right)}°', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class BenchPressCounter(ExerciseCounter):
    """Licznik wyciskania sztangi z kontrolą ruchu sztangi"""
    def __init__(self):
        super().__init__('BENCH PRESS')
        self.angle_left = 0
        self.angle_right = 0
        self.top_threshold = 150
        self.bottom_threshold = 100
        self._frames_required = 2
        self._frames_top = 0
        self._frames_bottom = 0
        
        # System aktywacji bazujący na ruchu sztangi
        self.is_active = False
        self.barbell_positions = []
        self.movement_threshold = 30  # Minimalne przesunięcie w pikselach aby aktywować
        
    def check_barbell_movement(self, barbell_y):
        """Sprawdza czy sztanga wystarczająco się poruszyła"""
        if barbell_y is None:
            return
            
        self.barbell_positions.append(barbell_y)
        
        # Zachowaj ostatnie 10 pozycji
        if len(self.barbell_positions) > 10:
            self.barbell_positions.pop(0)
        
        # Aktywuj jeśli jest ruch większy niż threshold
        if len(self.barbell_positions) >= 5:
            max_pos = max(self.barbell_positions)
            min_pos = min(self.barbell_positions)
            movement = abs(max_pos - min_pos)
            
            if movement > self.movement_threshold:
                self.is_active = True

    def update(self, landmarks, barbell_y=None):
        """Aktualizacja z kontrolą aktywacji przez ruch sztangi"""
        # Sprawdź ruch sztangi
        self.check_barbell_movement(barbell_y)
        
        # Nie licz repów jeśli system nie jest aktywny
        if not self.is_active:
            return
        
        # Left arm: shoulder (11), elbow (13), wrist (15)
        shoulder_left = landmarks[11]
        elbow_left = landmarks[13]
        wrist_left = landmarks[15]

        # Right arm: shoulder (12), elbow (14), wrist (16)
        shoulder_right = landmarks[12]
        elbow_right = landmarks[14]
        wrist_right = landmarks[16]

        self.angle_left = self.calculate_angle(shoulder_left, elbow_left, wrist_left)
        self.angle_right = self.calculate_angle(shoulder_right, elbow_right, wrist_right)

        in_bottom = (self.angle_left <= self.bottom_threshold or
                     self.angle_right <= self.bottom_threshold)
        in_top = (self.angle_left >= self.top_threshold and
                  self.angle_right >= self.top_threshold)

        if in_bottom:
            self._frames_bottom = min(self._frames_bottom + 1, self._frames_required)
        else:
            self._frames_bottom = 0

        if in_top:
            self._frames_top = min(self._frames_top + 1, self._frames_required)
        else:
            self._frames_top = 0

        if self.stage is None:
            if self._frames_top:
                self.stage = 'GORA'
            elif self._frames_bottom:
                self.stage = 'DOL'
            else:
                self.stage = 'GORA'
            return

        if self.stage == 'GORA':
            if self._frames_bottom >= self._frames_required:
                self.stage = 'DOL'
                self._frames_bottom = 0
                self._frames_top = 0
        elif self.stage == 'DOL':
            if self._frames_top >= self._frames_required:
                self.stage = 'GORA'
                self.count += 1
                self._frames_top = 0
                self._frames_bottom = 0

    def draw_info(self, frame, angle_left=None, angle_right=None):
        """Wyświetl info z statusem aktywacji"""
        super().draw_info(frame, self.angle_left, self.angle_right)
        
        # Dodaj status aktywacji
        status_text = "AKTYWNY" if self.is_active else "CZEKA NA RUCH..."
        status_color = (0, 255, 0) if self.is_active else (0, 165, 255)
        cv2.putText(frame, status_text, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


class SquatCounter(ExerciseCounter):
    """Licznik dla przysiadów"""
    def __init__(self):
        super().__init__("PRZYSIADY")
        
    def update(self, landmarks, barbell_y=None):
        hip = landmarks[24]
        knee = landmarks[26]
        ankle = landmarks[28]
        
        angle = self.calculate_angle(hip, knee, ankle)
        
        if angle < 100:
            self.stage = "DOL"
        if angle > 160 and self.stage == "DOL":
            self.stage = "GORA"
            self.count += 1


class PullUpCounter(ExerciseCounter):
    """Licznik dla podciągnięć"""
    def __init__(self):
        super().__init__("PODCIAGNIECIA")
        
    def update(self, landmarks, barbell_y=None):
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if angle > 160:
            self.stage = "DOL"
        if angle < 90 and self.stage == "DOL":
            self.stage = "GORA"
            self.count += 1


class PushUpCounter(ExerciseCounter):
    """Licznik dla pompek"""
    def __init__(self):
        super().__init__("POMPKI")
        
    def update(self, landmarks, barbell_y=None):
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if angle < 90:
            self.stage = "DOL"
        if angle > 160 and self.stage == "DOL":
            self.stage = "GORA"
            self.count += 1


class BicepCurlCounter(ExerciseCounter):
    """Licznik dla uginania ramion z hantlami"""
    def __init__(self):
        super().__init__("BICEPS CURL")
        
    def update(self, landmarks, barbell_y=None):
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if angle > 160:
            self.stage = "DOL"
        if angle < 50 and self.stage == "DOL":
            self.stage = "GORA"
            self.count += 1


class ShoulderPressCounter(ExerciseCounter):
    """Licznik dla wyciskania nad głowę"""
    def __init__(self):
        super().__init__("BARKI - WYCISKANIE")
        
    def update(self, landmarks, barbell_y=None):
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
    
        angle = self.calculate_angle(shoulder, elbow, wrist)
    
        if self.stage is None:
            self.stage = "GORA"

        if angle < 100:
            self.stage = "DOL"

        if angle > 160 and self.stage == "DOL":
            self.stage = "GORA"
            self.count += 1


# -----------------------
# Słownik dostępnych ćwiczeń
# -----------------------
EXERCISES = {
    'benchpress': BenchPressCounter,
    'squat': SquatCounter,
    'pullup': PullUpCounter,
    'pushup': PushUpCounter,
    'bicep': BicepCurlCounter,
    'shoulder': ShoulderPressCounter,
}

# -----------------------
# Argumenty
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", default="./GymVideos/GymVid6.mp4", help="Ścieżka do pliku wideo")
parser.add_argument("--pose_model", default="models/pose_landmarker_heavy.task", help="Model MediaPipe Pose")
parser.add_argument("--yolo_model", default=r"C:\Dev\GymTech2\runs\detect\barbell_detector\weights\best.pt", 
                    help="Model YOLO do detekcji sztangi")
parser.add_argument("--output", default="./GymVideos/outputs/GymVid6_integrated.mp4", help="Ścieżka do zapisu")
parser.add_argument("--exercise", default="benchpress", 
                    choices=list(EXERCISES.keys()),
                    help="Rodzaj ćwiczenia do zliczania")
args = parser.parse_args()

if not os.path.isfile(args.video):
    raise SystemExit(f"❌ Nie znaleziono pliku wideo: {args.video}")

# -----------------------
# Inicjalizacja MediaPipe
# -----------------------
BaseOptions = tasks.BaseOptions
VisionRunningMode = vision.RunningMode
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=args.pose_model),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

# -----------------------
# Inicjalizacja YOLO
# -----------------------
yolo_model = YOLO(args.yolo_model)

# Utworzenie licznika dla wybranego ćwiczenia
counter = EXERCISES[args.exercise]()

# -----------------------
# Odczyt wideo
# -----------------------
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

rotation_code = None
orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
if orientation_prop is not None:
    orientation_value = int(cap.get(orientation_prop) or 0)
    rotation_lookup = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotation_code = rotation_lookup.get(orientation_value)
    if rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) and w <= h:
        rotation_code = None
    if rotation_code is not None:
        print(f"🔄 Wykryto orientację wideo: {orientation_value} stopni")

frame_size = (w, h)
if rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
    frame_size = (h, w)
    w, h = frame_size

out_dir = os.path.dirname(args.output)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output, fourcc, fps, frame_size)

print(f"▶️  Przetwarzanie: {args.video}")
print(f"🏋️  Ćwiczenie: {counter.name}")
print(f"🧠 Model Pose: {args.pose_model}")
print(f"🎯 Model YOLO: {args.yolo_model}")
print(f"💾 Zapis: {args.output}")

# -----------------------
# Przetwarzanie
# -----------------------
with PoseLandmarker.create_from_options(pose_options) as landmarker:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        # -----------------------
        # YOLO: Detekcja sztangi
        # -----------------------
        yolo_results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
        annotated = yolo_results[0].plot()  # Ramka z bounding boxami
        
        # Znajdź pozycję Y sztangi (środek bbox)
        barbell_y = None
        if len(yolo_results[0].boxes) > 0:
            # Weź pierwszy wykryty obiekt (sztangę)
            box = yolo_results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            barbell_y = (y1 + y2) / 2  # Środek bbox w osi Y

        # -----------------------
        # MediaPipe: Pose tracking
        # -----------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * (1000 / fps))
        pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if pose_result.pose_landmarks:
            for landmark_list in pose_result.pose_landmarks:
                # Rysowanie szkieletu
                for connection in mp_pose.POSE_CONNECTIONS:
                    start = landmark_list[connection[0]]
                    end = landmark_list[connection[1]]
                    x1_pose, y1_pose = int(start.x * w), int(start.y * h)
                    x2_pose, y2_pose = int(end.x * w), int(end.y * h)
                    cv2.line(annotated, (x1_pose, y1_pose), (x2_pose, y2_pose), (0, 255, 0), 2)
                
                # Aktualizacja licznika z pozycją sztangi
                counter.update(landmark_list, barbell_y)
        
        # Rysowanie informacji o powtórzeniach
        counter.draw_info(annotated)
        
        out.write(annotated)
        cv2.imshow("Integrated: Pose + Barbell + Rep Counter", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Gotowe! Wynik zapisano do: {args.output}")
print(f"📊 Policzono powtórzeń: {counter.count}")
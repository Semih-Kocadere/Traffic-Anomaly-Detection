import os
import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO

# --- Ayarlar ---
CAPTION_JSON = "aligned_caption.json"
MODEL_PATH = "./runs/vehicle_selected_yolo/weights/best.pt"
VIDEO_PATH = "test_video2.mp4"

RISKY_KEYWORDS = ["sudden", "fast", "crash", "hit", "crossing", "fall", "run", "danger"]
DANGER_DISTANCE = 60
TARGET_SIZE = 224
INFERENCE_INTERVAL = 5  # her 5 frame'de bir tespit yap

print("✅ GPU kullanılabilir mi:", torch.cuda.is_available())

# --- YOLO Model ve Sınıf ID'leri ---
model = YOLO(MODEL_PATH)
print("📦 Model sınıfları:", model.names)
class_map = {name.lower(): idx for idx, name in model.names.items()}
PEDESTRIAN_ID = class_map.get("pedestrian", 0)
VEHICLE_ID = class_map.get("vehicle", 1)

# --- Yardımcı Fonksiyonlar ---
def is_caption_risky(caption):
    caption = caption.lower()
    return any(word in caption for word in RISKY_KEYWORDS)

def calculate_distance(box1, box2):
    x1, y1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2, y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def is_too_close(peds, vehicles):
    for ped in peds:
        for veh in vehicles:
            if calculate_distance(ped, veh) < DANGER_DISTANCE:
                return True
    return False

def letterbox_resize(image, target_size=224):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, left, top

# --- Caption verisi yükle ---
with open(CAPTION_JSON, "r", encoding="utf-8") as f:
    aligned_data = {os.path.basename(entry["frame_path"]): entry for entry in json.load(f)}

# --- Video işle ---
def detect_anomalies_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        return

    frame_index = 0
    results_cache = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]
        resized, scale, left, top = letterbox_resize(frame, target_size=TARGET_SIZE)

        # --- Tespit sadece her 5 frame’de bir yapılır ---
        if frame_index % INFERENCE_INTERVAL == 0:
            results = model.predict(resized, conf=0.1, verbose=False)[0]
            pedestrians, vehicles = [], []

            for box in results.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Geri ölçekleme (224 → orijinal)
                x1 = (x1 - left) / scale
                x2 = (x2 - left) / scale
                y1 = (y1 - top) / scale
                y2 = (y2 - top) / scale

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                w, h = x2 - x1, y2 - y1
                box_data = [x1, y1, w, h]

                if cls == PEDESTRIAN_ID:
                    pedestrians.append(box_data)
                elif cls == VEHICLE_ID:
                    vehicles.append(box_data)

            results_cache = (pedestrians, vehicles)
        else:
            pedestrians, vehicles = results_cache

        # --- Görselleştirme ---
        for box in pedestrians:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Pedestrian", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for box in vehicles:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Vehicle", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # --- Caption & Mesafe analiz ---
        frame_name = f"frame_{frame_index:05d}.jpg"
        caption_data = aligned_data.get(frame_name, {})
        caption_text = caption_data.get("caption_pedestrian", "") + " " + caption_data.get("caption_vehicle", "")

        danger = is_too_close(pedestrians, vehicles) or is_caption_risky(caption_text)

        if danger:
            cv2.putText(frame, "🚨 TEHLIKE TESPIT EDILDI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "✅ NORMAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        cv2.imshow("Anomaly Detection", frame)

        # 30 FPS'e yakın göstermek için bekleme süresi: 33 ms
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

# --- Başlat ---
detect_anomalies_on_video(VIDEO_PATH)
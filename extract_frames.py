import os
import cv2
import json
import random
from tqdm import tqdm


NUM_VIDEOS = 200
ANNOTATION_DIR = "./annotations/annotations/bbox_annotated/vehicle/train"
VIDEO_ROOT = "./videos/videos/train"
OUTPUT_IMAGE_ROOT = "./yolo_dataset/images/train"
OUTPUT_MAPPING_JSON = "frame_video_mapping_vehicle_selected.json"
RESIZE_TO = (224, 224)

frame_video_mapping = {}

def extract_frames_from_video(video_path, output_dir, scene, video_id, resize_to=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Video açılamadı: {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_to:
            frame = cv2.resize(frame, resize_to)

        frame_name = f"frame_{frame_idx:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)

        unique_id = f"{scene}overhead_view{video_id}frame{frame_idx:05d}"
        frame_video_mapping[unique_id] = frame_path.replace("\\", "/")

        frame_idx += 1

    cap.release()
    return frame_idx

def main():
    # --- 1. Annotasyonu olan video listesini çıkar ---
    scene_video_ids = set()

    for scene in os.listdir(ANNOTATION_DIR):
        scene_path = os.path.join(ANNOTATION_DIR, scene, "overhead_view")
        if not os.path.isdir(scene_path):
            continue

        for file in os.listdir(scene_path):
            if not file.endswith(".json"):
                continue

            video_id = file.replace("_bbox.json", "")
            json_path = os.path.join(scene_path, file)

            with open(json_path, "r", encoding="utf-8") as f_json:
                data = json.load(f_json)

            for ann in data.get("annotations", []):
                if ann.get("bbox"):
                    scene_video_ids.add((scene, video_id))
                    break

    # Rastgele 25 tanesini al
    selected = random.sample(list(scene_video_ids), min(NUM_VIDEOS, len(scene_video_ids)))

    # --- 2. Frame çıkar ---
    for scene, video_id in tqdm(selected, desc="🎬 Frame çıkarımı yapılıyor"):
        video_path = os.path.join(VIDEO_ROOT, scene, "overhead_view", f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"[UYARI] Video bulunamadı: {video_path}")
            continue

        output_dir = os.path.join(OUTPUT_IMAGE_ROOT, f"{scene}overhead_view{video_id}")
        extract_frames_from_video(video_path, output_dir, scene, video_id)

    # Mapping kaydet
    with open(OUTPUT_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(frame_video_mapping, f, indent=2)

    print(f"\n✅ {len(frame_video_mapping)} frame çıkarıldı.")
    print(f"📁 Mapping dosyası: {OUTPUT_MAPPING_JSON}")

if __name__ == "__main__":
    main()
import os
import json
from tqdm import tqdm

FRAME_MAPPING_FILE = "frame_video_mapping_vehicle_selected.json"

ANNOTATION_SOURCES = {
    0: "./annotations/annotations/bbox_annotated/pedestrian/train",
    1: "./annotations/annotations/bbox_annotated/vehicle/train"
}

with open(FRAME_MAPPING_FILE, "r", encoding="utf-8") as f:
    frame_mapping = json.load(f)

mapping_keys = set(frame_mapping.keys())
total_written = 0
skipped = 0

for class_id, root_path in ANNOTATION_SOURCES.items():
    for scene in tqdm(os.listdir(root_path), desc=f"🔍 Class {class_id} - {('Pedestrian' if class_id == 0 else 'Vehicle')}"):
        scene_path = os.path.join(root_path, scene, "overhead_view")
        if not os.path.isdir(scene_path):
            continue

        for file in os.listdir(scene_path):
            if not file.endswith(".json"):
                continue

            video_id = file.replace("_bbox.json", "").replace(".json", "")
            json_path = os.path.join(scene_path, file)

            with open(json_path, "r", encoding="utf-8") as f_json:
                data = json.load(f_json)

            for ann in data.get("annotations", []):
                image_id = ann["image_id"]
                bbox = ann["bbox"]
                unique_id = f"{scene}overhead_view{video_id}frame{int(image_id):05d}"

                if unique_id not in mapping_keys:
                    skipped += 1
                    continue

                image_path = frame_mapping[unique_id]
                label_path = image_path.replace("images/train", "labels/train").replace(".jpg", ".txt")
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                x, y, w, h = bbox
                x_center = (x + w / 2) / 1920
                y_center = (y + h / 2) / 1080
                w_norm = w / 1920
                h_norm = h / 1080

                with open(label_path, "a", encoding="utf-8") as out_f:
                    out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    total_written += 1

print(f"\n✅ Toplam {total_written} bbox etiketi yazıldı.")
print(f"⚠️ {skipped} frame mapping içinde bulunamadı.")
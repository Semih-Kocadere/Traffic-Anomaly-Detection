import os
import json
from tqdm import tqdm

FRAME_MAPPING_FILE = "frame_video_mapping.json"

# 🔧 Hem annotated hem generated kaynaklarını tanımla
ANNOTATION_SOURCES = {
    "annotated": {
        0: "./DatasetVision/wts_dataset/wts_dataset_zip/annotations/annotations/bbox_annotated/pedestrian/train",
        1: "./DatasetVision/wts_dataset/wts_dataset_zip/annotations/annotations/bbox_annotated/vehicle/train"
    },
    "generated": {
        0: "./DatasetVision/wts_dataset/wts_dataset_zip/annotations/annotations/bbox_generated/pedestrian/train",
        # araçlar için generated yoksa sadece pedestrian alınır
    }
}

# Mapping dosyasını yükle
with open(FRAME_MAPPING_FILE, "r", encoding="utf-8") as f:
    frame_mapping = json.load(f)

mapping_keys = list(frame_mapping.keys())
total_written = 0
skipped = 0

for source_type, sources in ANNOTATION_SOURCES.items():
    for class_id, root_path in sources.items():
        for scene in tqdm(os.listdir(root_path), desc=f"📦 {source_type.upper()} - Class {class_id}"):
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
                    unique_id = f"{scene}_overhead_view_{video_id}_frame_{int(image_id):05d}"

                    if unique_id not in frame_mapping:
                        skipped += 1
                        continue

                    image_path = frame_mapping[unique_id]

                    if "/images/train/" in image_path:
                        label_path = image_path.replace("images/train", "labels/train")
                    elif "/images/val/" in image_path:
                        label_path = image_path.replace("images/val", "labels/val")
                    else:
                        skipped += 1
                        continue

                    label_path = label_path.replace(".jpg", ".txt")
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
print(f"⚠️ {skipped} frame eşleşmedi (mapping dışında kaldı).")

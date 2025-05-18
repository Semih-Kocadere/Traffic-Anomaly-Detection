import os
import cv2
import json
from tqdm import tqdm
import random

VIDEO_ROOT = "./DatasetVision/videos-001/videos"
OUTPUT_IMAGE_ROOT = "./yolo_dataset/images"
OUTPUT_MAPPING_JSON = "frame_video_mapping.json"

SPLITS = ["train", "val"]
VIEWS = ["overhead_view", "vehicle_view"]
VAL_TRAIN_SPLIT_RATIO = 0.8

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

        unique_id = f"{scene}_overhead_view_{video_id}_frame_{frame_idx:05d}"
        frame_video_mapping[unique_id] = frame_path.replace("\\", "/")

        frame_idx += 1

    cap.release()
    return frame_idx

def collect_scenes(split):
    path = os.path.join(VIDEO_ROOT, split)
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def process_all():
    # Val sahnelerinden %80'i ekstra train'e
    val_scenes = collect_scenes("val")
    random.seed(42)
    extra_train_scenes = set(random.sample(val_scenes, int(len(val_scenes) * VAL_TRAIN_SPLIT_RATIO)))
    pure_val_scenes = set(val_scenes) - extra_train_scenes

    for split in SPLITS:
        scenes = collect_scenes(split)
        for scene in tqdm(scenes, desc=f"🎬 {split.upper()} sahneleri işleniyor"):
            scene_path = os.path.join(VIDEO_ROOT, split, scene)
            if not os.path.isdir(scene_path):
                continue

            for view in VIEWS:
                view_path = os.path.join(scene_path, view)
                if not os.path.exists(view_path):
                    continue

                for video_file in sorted(os.listdir(view_path)):
                    if not video_file.endswith(".mp4"):
                        continue

                    video_path = os.path.join(view_path, video_file)
                    video_id = video_file.replace(".mp4", "")
                    output_subdir = f"{scene}_overhead_view_{video_id}"

                    if split == "train" or scene in extra_train_scenes:
                        output_split = "train"
                    elif split == "val" and scene in pure_val_scenes:
                        output_split = "val"
                    else:
                        continue

                    output_dir = os.path.join(OUTPUT_IMAGE_ROOT, output_split, output_subdir)
                    extract_frames_from_video(video_path, output_dir, scene, video_id)

    # mapping yaz
    with open(OUTPUT_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(frame_video_mapping, f, indent=2)

    print(f"\n✅ Toplam {len(frame_video_mapping)} frame çıkarıldı.")

if __name__ == "__main__":
    process_all()

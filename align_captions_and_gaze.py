import os
import json
from tqdm import tqdm

CAPTION_ROOT = "./annotations/caption/train"
GAZE_ROOT = "./annotations/3D_gaze/train"
MAPPING_FILE = "frame_video_mapping.json"
OUTPUT_JSON = "aligned_caption.json"


def load_gaze(gaze_file_path):
    if not os.path.exists(gaze_file_path):
        return {}
    with open(gaze_file_path, "r") as f:
        data = json.load(f)
    return {ann["image_id"]: ann["gaze"] for ann in data.get("annotations", [])}


def extract_matching_frames():
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        frame_mapping = json.load(f)

    aligned = []
    missing_frames = 0

    for unique_id, frame_path in tqdm(frame_mapping.items(), desc="🔍 Eşleşme işlemi yapılıyor"):
        # unique_id: scene_overhead_view_videoID_frame_00000
        # scene = ilk parça, video klasör adından alınabilir
        parts = unique_id.split("_overhead_view_")
        if len(parts) != 2:
            continue
        scene = parts[0]
        video_id = parts[1].rsplit("_frame_", 1)[0]
        frame_index = int(unique_id.split("_frame_")[-1])

        cap_file = os.path.join(CAPTION_ROOT, scene, "overhead_view", f"{scene}_caption.json")
        gaze_scene_dir = os.path.join(GAZE_ROOT, scene)
        gaze_dict = {}

        # Caption dosyası yoksa atla
        if not os.path.exists(cap_file):
            continue

        # Gaze varsa yükle
        if os.path.isdir(gaze_scene_dir):
            for gaze_json in os.listdir(gaze_scene_dir):
                if gaze_json.endswith(".json"):
                    gaze_dict = load_gaze(os.path.join(gaze_scene_dir, gaze_json))
                    break

        with open(cap_file, "r") as f:
            cap_data = json.load(f)

        matched = False
        for event in cap_data.get("event_phase", []):
            start_frame = int(float(event.get("start_time", 0)) * 30)
            end_frame = int(float(event.get("end_time", 0)) * 30)

            if start_frame <= frame_index <= end_frame:
                label = event.get("labels", [""])[0]
                caption_ped = event.get("caption_pedestrian", "")
                caption_veh = event.get("caption_vehicle", "")

                item = {
                    "frame_path": frame_path.replace("\\", "/"),
                    "frame_id": unique_id,
                    "label": label,
                    "caption_pedestrian": caption_ped,
                    "caption_vehicle": caption_veh
                }

                if frame_index in gaze_dict:
                    item["gaze"] = gaze_dict[frame_index]

                aligned.append(item)
                matched = True
                break

        if not matched:
            missing_frames += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(aligned, f, indent=2)

    print(f"\n✅ Toplam eşleşmiş frame: {len(aligned)}")
    print(f"⚠️ Karşılığı bulunamayan frame sayısı: {missing_frames}")


if __name__ == "__main__":
    extract_matching_frames()

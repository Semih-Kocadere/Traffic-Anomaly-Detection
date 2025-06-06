import os
from glob import glob

def clean_unlabeled_frames(image_root, label_root):
    removed = 0
    for split in ["train"]:
        img_dir = os.path.join(image_root, split)
        lbl_dir = os.path.join(label_root, split)

        for root, _, files in os.walk(img_dir):
            for f in files:
                if not f.endswith(".jpg"):
                    continue
                frame_path = os.path.join(root, f)
                rel_path = os.path.relpath(frame_path, img_dir)  # alt klasör + dosya

                label_path = os.path.join(lbl_dir, rel_path).replace(".jpg", ".txt")

                if not os.path.exists(label_path):
                    os.remove(frame_path)
                    removed += 1

    print(f"✅ {removed} etiketsiz frame silindi.")

if __name__ == "__main__":
    clean_unlabeled_frames("./yolo_dataset/images", "./yolo_dataset/labels")
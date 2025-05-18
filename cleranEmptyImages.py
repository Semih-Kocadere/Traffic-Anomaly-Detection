import os
import shutil

IMAGE_TRAIN_DIR = "./yolo_dataset/images/train"

empty_dirs = []

for root, dirs, files in os.walk(IMAGE_TRAIN_DIR, topdown=False):
    if not files and not dirs:
        empty_dirs.append(root)

for d in empty_dirs:
    shutil.rmtree(d)
    print(f"🗑️ Silindi: {d}")

print(f"\n✅ Toplam {len(empty_dirs)} boş klasör silindi.")

from ultralytics import YOLO
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    model = YOLO("yolov8n.pt")  # Hafif model: hızlı eğitim
    model.train(
        data="./yolo_dataset/data.yaml",
        epochs=200,            # Küçük dataset için yeterli
        imgsz=224,
        batch=16,
        name="vehicle_selected_yolo",
        project="runs",
        exist_ok=True,
        patience=5,
        val=False,
        device=0  # GPU için 0, CPU için "cpu"
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
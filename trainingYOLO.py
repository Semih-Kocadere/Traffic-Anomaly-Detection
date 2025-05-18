from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    model = YOLO("yolov8n.pt")  # veya yolov8m.pt, best.pt vb.
    model.train(
        data="./yolo_dataset/data.yaml",
        epochs=100,
        imgsz=224,
        batch=16,
        name="anomaly_yolo_train",
        project="runs",
        exist_ok=True,
        patience=10,
        device=0 # GPU kullanıyorsan 0, CPU için "cpu",
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

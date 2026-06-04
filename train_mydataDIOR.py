from ultralytics import YOLO
import os
import torch
from pathlib import Path

DATASET_NAME = "DIOR"
AUX_EXPERIMENT = "exp2_sum"  # off, head_only, exp1_mean, exp1_sum, exp2_mean, exp2_sum
AUX_LAYERS = "all"  # all, b1, b2, b3, b4, b5, or comma-separated like b1,b3,b5

CLOUD_DATA_YAML = Path("/workspace/datasets/DIOR_YOLO/data.yaml")
if CLOUD_DATA_YAML.exists():
    DATA_YAML = str(CLOUD_DATA_YAML)
    RUN_PROJECT = "/workspace/runs/detect_DIOR"
else:
    DATA_YAML = r"C:\Users\14288\OneDrive\Desktop\DIOR_YOLO\data.yaml"
    RUN_PROJECT = r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\detect_DIOR"

def main():
    os.environ.setdefault("YOLO_AUX_EXPERIMENT", AUX_EXPERIMENT)
    os.environ.setdefault("YOLO_AUX_LAYERS", AUX_LAYERS)
    run_name = f"{DATASET_NAME}_{AUX_EXPERIMENT}_{AUX_LAYERS.replace(',', '-')}"

    print("Using CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    
    model = YOLO("yolov12n.pt")  


    results = model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=4,
        device=0,       
        workers=0,      
        amp=True,
        project=RUN_PROJECT,
        name=run_name,
        exist_ok=True,
        save_period=1,
    )

    print("train done")

if __name__ == "__main__":
    main()

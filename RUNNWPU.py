from ultralytics import YOLO

model = YOLO(r"runs\detect\train2\weights\best.pt")
model.val(data="data.yaml", device=0)
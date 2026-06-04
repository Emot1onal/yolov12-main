from ultralytics import YOLO

model = YOLO("yolov12n.pt")
model.predict(source=r"C:\Users\14288\OneDrive\Desktop\563.jpg", imgsz=640, device="cpu")
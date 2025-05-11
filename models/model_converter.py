from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n-pose.pt")

# Export the model to CoreML format
model.export(format="coreml")

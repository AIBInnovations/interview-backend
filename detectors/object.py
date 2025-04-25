import os
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
TARGETS = {"cell phone", "book", "laptop", "notebook"}

def analyze_objects(frame):
    """
    Returns list of dicts:
      [ { name, confidence, bbox, warning: bool }, … ]
    """
    results = model.predict(source=frame, verbose=False)[0]
    out = []
    for box, cls, conf in zip(results.boxes.xyxy,
                              results.boxes.cls,
                              results.boxes.conf):
        name = model.model.names[int(cls)]
        x1,y1,x2,y2 = map(int, box)
        warn = name in TARGETS
        out.append({
            "name": name,
            "confidence": float(conf),
            "bbox": [x1,y1,x2,y2],
            "warning": warn
        })
    return out

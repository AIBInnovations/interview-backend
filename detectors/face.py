# import cv2
# from facenet_pytorch import MTCNN

# mtcnn = MTCNN(keep_all=True, device="cpu")

# def analyze_face(frame):
#     """
#     Returns dict:
#       { flag: 0|1|2, boxes: [ [x1,y1,x2,y2], … ] }
#     """
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     boxes, _ = mtcnn.detect(rgb)
#     if boxes is None:
#         boxes = []
#     else:
#         boxes = boxes.tolist()
#     if len(boxes) == 0:
#         flag = 0
#     elif len(boxes) == 1:
#         flag = 1
#     else:
#         flag = 2
#     return {"flag": flag, "boxes": boxes}

import cv2
import numpy as np
from facenet_pytorch import MTCNN

# Initialize MTCNN with stricter thresholds to reduce false positives
mtcnn = MTCNN(keep_all=True, device="cpu", thresholds=[0.8, 0.9, 0.9])


def analyze_face(frame, prob_thresh=0.90):
    """
    Runs face detection and filters by confidence.

    Returns dict:
      { flag: 0|1|2, boxes: [ [x1,y1,x2,y2], … ] }
    """
    # Convert BGR (OpenCV) to RGB (MTCNN)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces: boxes and probabilities
    boxes, probs = mtcnn.detect(rgb)

    # Ensure lists for iteration
    if boxes is None or probs is None:
        boxes, probs = [], []

    filtered = []
    for box, p in zip(boxes, probs):
        if p is not None and p >= prob_thresh:
            # Convert numpy array to list if needed
            if isinstance(box, np.ndarray):
                filtered.append(box.tolist())
            else:
                filtered.append(box)

    # Determine flag based on number of good detections
    count = len(filtered)
    if count == 0:
        flag = 0  # no face
    elif count == 1:
        flag = 1  # exactly one face
    else:
        flag = 2  # multiple faces

    return {"flag": flag, "boxes": filtered}

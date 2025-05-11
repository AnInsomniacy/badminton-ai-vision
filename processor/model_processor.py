import numpy as np
from ultralytics import YOLO


def load_yolo_model(model_path):
    """Load YOLO model

    Args:
        model_path: Path to the YOLO model file

    Returns:
        model: Loaded YOLO model
    """
    return YOLO(model_path)


def detect_with_model(model, image, confidence_threshold, stream=False):
    """Run detection on image using model

    Args:
        model: YOLO model
        image: PIL Image to process
        confidence_threshold: Detection confidence threshold
        stream: Whether to use streaming mode (returns a generator)

    Returns:
        results: Detection results (list or generator)
    """
    # return model(image, conf=confidence_threshold, stream=stream)
    return model.predict(image, conf=confidence_threshold, stream=stream, verbose=False)


def extract_detection_data(results):
    """Extract boxes, confidence scores and keypoints from results

    Args:
        results: Detection results from YOLO model

    Returns:
        boxes: Bounding boxes or None
        confs: Confidence scores or None
        keypoints: Keypoints or None
    """
    boxes, confs, keypoints = None, None, None

    for result in results:
        # Extract boxes
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

        # Extract keypoints
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            if hasattr(result.keypoints, 'xy'):
                keypoints = result.keypoints.xy.cpu().numpy()
            else:
                keypoints = np.array(result.keypoints)

    return boxes, confs, keypoints

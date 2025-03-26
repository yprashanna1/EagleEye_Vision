import cv2
import math
from ultralytics import YOLO

# helper functions for accident detection module
def compute_centroid(bbox):
    # bbox format: {"x1": int, "y1": int, "x2": int, "y2": int}
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (cx, cy)

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def compute_iou(bbox1, bbox2):
    # Compute Intersection over Union for two bounding boxes.
    x_left = max(bbox1["x1"], bbox2["x1"])
    y_top = max(bbox1["y1"], bbox2["y1"])
    x_right = min(bbox1["x2"], bbox2["x2"])
    y_bottom = min(bbox1["y2"], bbox2["y2"])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
    bbox2_area = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

# Accident Detection Class
class AccidentDetector:
    def __init__(self, distance_threshold=50, acceleration_threshold=15):
        """
        distance_threshold: maximum allowed distance (in pixels) to match detections to an existing track.
        acceleration_threshold: if the difference in consecutive velocities (in pixels/frame) exceeds this value,
                                we flag an accident.
        """
        self.tracks = {}  # track_id -> {"centroid_history": [(x, y), ...], "label": str, "bbox": dict}
        self.next_track_id = 0
        self.distance_threshold = distance_threshold
        self.acceleration_threshold = acceleration_threshold

    def update(self, detections):
        """
        Update tracker with current detections.
        Returns a list of accident flags (True/False) corresponding to each detection in the order provided.
        """
        detection_centroids = [compute_centroid(det["bounding_box"]) for det in detections]
        assigned_tracks = [-1] * len(detections)
        updated_track_ids = set()

        # for each existing track, try to find matching detection
        for track_id, track_data in self.tracks.items():
            last_centroid = track_data["centroid_history"][-1]
            best_match_idx = None
            best_distance = float("inf")
            for i, centroid in enumerate(detection_centroids):
                if assigned_tracks[i] != -1:
                    continue  # already assigned
                dist = euclidean_distance(last_centroid, centroid)
                if dist < best_distance and dist < self.distance_threshold:
                    best_distance = dist
                    best_match_idx = i
            if best_match_idx is not None:
                assigned_tracks[best_match_idx] = track_id
                updated_track_ids.add(track_id)
                self.tracks[track_id]["centroid_history"].append(detection_centroids[best_match_idx])
                self.tracks[track_id]["bbox"] = detections[best_match_idx]["bounding_box"]
                self.tracks[track_id]["label"] = detections[best_match_idx]["label"]

        # For detections not assigned to any track, create new tracks
        for i, track_id in enumerate(assigned_tracks):
            if track_id == -1:
                new_track_id = self.next_track_id
                self.next_track_id += 1
                assigned_tracks[i] = new_track_id
                self.tracks[new_track_id] = {
                    "centroid_history": [detection_centroids[i]],
                    "label": detections[i]["label"],
                    "bbox": detections[i]["bounding_box"]
                }
                updated_track_ids.add(new_track_id)

        # compute accident flags based on sudden acceleration or collision
        accident_flags = [False] * len(detections)
        # 1. check acceleration for each detection
        for i, track_id in enumerate(assigned_tracks):
            history = self.tracks[track_id]["centroid_history"]
            if len(history) >= 3:
                v1 = euclidean_distance(history[-2], history[-3])
                v2 = euclidean_distance(history[-1], history[-2])
                acceleration = abs(v2 - v1)
                if acceleration > self.acceleration_threshold:
                    accident_flags[i] = True

        # 2. check for collision with bounding box overlap (IoU)
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                iou = compute_iou(detections[i]["bounding_box"], detections[j]["bounding_box"])
                if iou > 0.5:
                    accident_flags[i] = True
                    accident_flags[j] = True

        return accident_flags

# create a global instance of the AccidentDetector to track objects across frames
ACCIDENT_DETECTOR = AccidentDetector()

class ModelCache:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = YOLO("yolov8n.pt")
        return cls._model

def process_frame(frame):
    model = ModelCache.get_model()
    results = model(frame)
    detections = []
    for box in results[0].boxes:
        class_idx = int(box.cls[0])
        label = model.names[class_idx] if class_idx < len(model.names) else f"class_{class_idx}"
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": conf,
            "bounding_box": {
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3])
            }
        })
    return detections

def classify_detections(detections):
    """
    Classifies detections for both suspicious objects and accidents.
    Returns a list of tuples: (detection, color, label_text)
    Normal objects are marked in green.
    Suspicious objects (e.g., gun, knife, backpack, etc.) with confidence > 0.6 are marked in red.
    Objects flagged for an accident are marked in blue.
    """
    suspicious_labels = ["gun", "knife", "backpack", "suspicious_bag", "explosive_device"]
    
    # Get accident flags from the full-fledged accident detector
    accident_flags = ACCIDENT_DETECTOR.update(detections)
    labeled_detections = []
    
    for i, detection in enumerate(detections):
        label = detection["label"]
        confidence = detection["confidence"]
        # Default: green for normal objects
        color = (0, 255, 0)
        label_text = f"{label} {confidence:.2f}"
        
        # Mark suspicious objects in red if above threshold
        if label in suspicious_labels and confidence > 0.6:
            color = (0, 0, 255)
            label_text += " (SUSPICIOUS)"
        
        # Mark as accident (blue) if flagged by the accident detector
        if accident_flags[i]:
            color = (255, 0, 0)
            label_text += " (ACCIDENT)"
        
        labeled_detections.append((detection, color, label_text))
    
    return labeled_detections

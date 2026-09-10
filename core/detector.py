import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO

@dataclass
class WorkerCompliance:
    """
    Compliance status of an individual worker on frame.
    """
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    confidence: float
    detected_ppe: List[str] = field(default_factory=list)
    explicit_violations: List[str] = field(default_factory=list)
    missing_mandatory: List[str] = field(default_factory=list)
    is_compliant: bool = True
    ppe_boxes: List[Dict[str, Any]] = field(default_factory=list)  # list of {class_name, bbox, conf}

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Returns (x, y) ground contact point at the bottom center of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


@dataclass
class DetectionResult:
    """Container for full frame inference results."""
    workers: List[WorkerCompliance]
    all_detections: List[Dict[str, Any]]  # Raw detections
    inference_time_ms: float = 0.0


class PPEDetector:
    """
    High-performance Industrial PPE Detection & Anatomical Association Engine.
    Uses fine-tuned 19-class YOLOv8 model with automatic fallback to standard YOLOv8.
    """

    # Anatomical region multipliers relative to worker bounding box [x1, y1, x2, y2]
    REGION_HEAD_Y_MAX = 0.35  # Upper 35% is head region (Helmets, Glasses, Masks)
    REGION_TORSO_Y_MIN = 0.15 # 15% to 75% is torso region (Safety Vests)
    REGION_TORSO_Y_MAX = 0.75
    REGION_FEET_Y_MIN = 0.65  # Lower 35% is feet region (Boots)

    def __init__(
        self,
        model_path: str = "models/ppe_yolov8.pt",
        fallback_model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        mandatory_ppe: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.fallback_model_path = fallback_model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.mandatory_ppe = mandatory_ppe or ["Hard_hat", "Vest"]
        
        self.model = None
        self.is_custom_ppe_model = False
        self._load_model()

    def _load_model(self):
        """Loads the specialized PPE model, or falls back to standard YOLOv8."""
        if not os.path.exists(self.model_path) and "ppe_yolov8.pt" in self.model_path:
            try:
                print(f"[PPEDetector] '{self.model_path}' not found. Downloading fine-tuned weights from Hugging Face...")
                os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
                import urllib.request
                url = "https://huggingface.co/killuminati1/construction-ppe-yolov8/resolve/main/best.pt"
                urllib.request.urlretrieve(url, self.model_path)
                print(f"[PPEDetector] Download complete: {self.model_path}")
            except Exception as dl_err:
                print(f"[PPEDetector] Could not auto-download PPE model ({dl_err}). Falling back to standard YOLOv8.")

        if os.path.exists(self.model_path):
            try:
                print(f"[PPEDetector] Loading fine-tuned PPE model from {self.model_path}...")
                self.model = YOLO(self.model_path)
                self.is_custom_ppe_model = True
                print(f"[PPEDetector] PPE Model loaded with {len(self.model.names)} classes: {self.model.names}")
                return
            except Exception as e:
                print(f"[PPEDetector] Warning: Failed to load {self.model_path}: {e}")

        # Fallback to standard YOLOv8n
        print(f"[PPEDetector] Loading fallback model {self.fallback_model_path}...")
        self.model = YOLO(self.fallback_model_path)
        self.is_custom_ppe_model = False
        print("[PPEDetector] Fallback model loaded.")

    def set_mandatory_ppe(self, items: List[str]):
        """Dynamically updates the enforced PPE checklist."""
        self.mandatory_ppe = items

    def _compute_overlap(self, box_a: List[float], box_b: List[float]) -> float:
        """Computes intersection-over-box_a (containment ratio)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        a_area = max((ax2 - ax1) * (ay2 - ay1), 1.0)
        return inter_area / a_area

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Runs object detection on the frame, performs anatomical PPE association,
        and returns worker compliance statuses.
        """
        import time
        t_start = time.time()

        h, w = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        inference_time_ms = (time.time() - t_start) * 1000.0

        raw_detections: List[Dict[str, Any]] = []
        worker_boxes: List[Dict[str, Any]] = []
        ppe_items: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy())
                    cls_id = int(b.cls[0].cpu().numpy())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    det = {
                        "bbox": xyxy,
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    }
                    raw_detections.append(det)

                    # Worker/Person class detection
                    # Custom PPE model uses 'Worker', standard COCO model uses 'person'
                    if cls_name.lower() in ["worker", "person"]:
                        worker_boxes.append(det)
                    else:
                        ppe_items.append(det)

        # In case the PPE model detected PPE items but missed person bbox (e.g. tight crop or partial view),
        # or if standard model is used with heuristic color analysis:
        workers_compliance = self._associate_ppe_with_workers(worker_boxes, ppe_items, frame)

        return DetectionResult(
            workers=workers_compliance,
            all_detections=raw_detections,
            inference_time_ms=round(inference_time_ms, 1)
        )

    def _associate_ppe_with_workers(
        self,
        worker_boxes: List[Dict[str, Any]],
        ppe_items: List[Dict[str, Any]],
        frame: np.ndarray
    ) -> List[WorkerCompliance]:
        """
        Maps detected PPE and violation items to individual worker bounding boxes
        based on anatomical spatial containment.
        """
        workers: List[WorkerCompliance] = []

        # If no explicit worker was detected but PPE items exist,
        # group PPE items into a pseudo worker bbox
        if not worker_boxes and ppe_items:
            # Check if there are PPE items with significant presence
            # Create a bounding box enclosing the items
            all_pts = [item["bbox"] for item in ppe_items]
            min_x = min(p[0] for p in all_pts)
            min_y = min(p[1] for p in all_pts)
            max_x = max(p[2] for p in all_pts)
            max_y = max(p[3] for p in all_pts)
            # Expand to roughly person height if only helmet is visible
            height = max_y - min_y
            if height < (frame.shape[0] * 0.25):
                max_y = min(frame.shape[0], min_y + height * 3.5)
            worker_boxes.append({
                "bbox": [min_x, min_y, max_x, max_y],
                "conf": 0.70,
                "class_name": "Worker"
            })

        for w_det in worker_boxes:
            wx1, wy1, wx2, wy2 = w_det["bbox"]
            w_h = max(wy2 - wy1, 1.0)
            w_w = max(wx2 - wx1, 1.0)

            # Define anatomical sub-zones
            head_zone = [wx1 - 0.1 * w_w, wy1 - 0.1 * w_h, wx2 + 0.1 * w_w, wy1 + self.REGION_HEAD_Y_MAX * w_h]
            torso_zone = [wx1 - 0.1 * w_w, wy1 + self.REGION_TORSO_Y_MIN * w_h, wx2 + 0.1 * w_w, wy1 + self.REGION_TORSO_Y_MAX * w_h]
            feet_zone = [wx1 - 0.1 * w_w, wy1 + self.REGION_FEET_Y_MIN * w_h, wx2 + 0.1 * w_w, wy2 + 0.1 * w_h]

            detected_ppe_set = set()
            explicit_violations_set = set()
            associated_ppe_boxes = []

            for item in ppe_items:
                ix1, iy1, ix2, iy2 = item["bbox"]
                item_cx = (ix1 + ix2) / 2.0
                item_cy = (iy1 + iy2) / 2.0
                item_name = item["class_name"]

                # Check if item center is within or near worker horizontally
                if not (wx1 - 0.2 * w_w <= item_cx <= wx2 + 0.2 * w_w):
                    continue

                # Check anatomical match
                is_associated = False

                # Helmet / Headwear / Glasses / Mask
                if item_name in ["Hard_hat", "No-Helmet", "Glass", "No-Glass", "Mask", "No-Mask", "Ear-Protection", "No-Ear-Protection"]:
                    if self._compute_overlap(item["bbox"], head_zone) > 0.15 or (head_zone[1] <= item_cy <= head_zone[3]):
                        is_associated = True

                # Vest
                elif item_name in ["Vest", "No-Vest"]:
                    if self._compute_overlap(item["bbox"], torso_zone) > 0.15 or (torso_zone[1] <= item_cy <= torso_zone[3]):
                        is_associated = True

                # Boots
                elif item_name in ["Boots", "No-Boots"]:
                    if self._compute_overlap(item["bbox"], feet_zone) > 0.15 or (feet_zone[1] <= item_cy <= feet_zone[3]):
                        is_associated = True

                # Gloves
                elif item_name in ["Glove", "No-Glove"]:
                    if self._compute_overlap(item["bbox"], [wx1 - 0.3 * w_w, wy1 + 0.3 * w_h, wx2 + 0.3 * w_w, wy1 + 0.9 * w_h]) > 0.15:
                        is_associated = True
                
                # General fallback containment
                elif self._compute_overlap(item["bbox"], w_det["bbox"]) > 0.20:
                    is_associated = True

                if is_associated:
                    associated_ppe_boxes.append(item)
                    if item_name.startswith("No-"):
                        explicit_violations_set.add(item_name)
                    else:
                        detected_ppe_set.add(item_name)

            # If standard COCO model was used (no custom PPE classes), apply color-based high-vis heuristic
            if not self.is_custom_ppe_model:
                color_ppe = self._heuristic_ppe_check(frame, w_det["bbox"])
                detected_ppe_set.update(color_ppe)

            # Determine missing mandatory equipment
            missing_mandatory = []
            for req in self.mandatory_ppe:
                # Direct check
                has_item = req in detected_ppe_set
                # Negative class check (e.g. 'No-Helmet' violates 'Hard_hat')
                negative_alias = f"No-{req.replace('_', '')}"
                is_explicitly_violating = any(
                    v.lower() == negative_alias.lower() or 
                    v.lower() == f"no-{req.lower()}"
                    for v in explicit_violations_set
                )

                if not has_item or is_explicitly_violating:
                    missing_mandatory.append(req)

            is_compliant = (len(missing_mandatory) == 0 and len(explicit_violations_set) == 0)

            workers.append(WorkerCompliance(
                bbox=w_det["bbox"],
                confidence=w_det["conf"],
                detected_ppe=sorted(list(detected_ppe_set)),
                explicit_violations=sorted(list(explicit_violations_set)),
                missing_mandatory=sorted(missing_mandatory),
                is_compliant=is_compliant,
                ppe_boxes=associated_ppe_boxes
            ))

        return workers

    def _heuristic_ppe_check(self, frame: np.ndarray, bbox: List[float]) -> List[str]:
        """
        Color-space fallback heuristic: detects vibrant safety vest colors
        (neon yellow/green, safety orange) and bright hardhats in the respective regions.
        """
        detected = []
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return detected

        person_crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = person_crop.shape[:2]
        if crop_h < 20 or crop_w < 10:
            return detected

        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

        # Torso crop (vest check)
        t_y1, t_y2 = int(crop_h * 0.20), int(crop_h * 0.65)
        torso_hsv = hsv[t_y1:t_y2, :]

        if torso_hsv.size > 0:
            # High-visibility fluorescent yellow/green mask (Hue ~25 to 65, high Sat & Val)
            yellow_mask = cv2.inRange(torso_hsv, np.array([20, 100, 100]), np.array([65, 255, 255]))
            # High-visibility orange mask (Hue ~5 to 20, high Sat & Val)
            orange_mask = cv2.inRange(torso_hsv, np.array([5, 120, 120]), np.array([20, 255, 255]))
            vest_pixels = cv2.countNonZero(yellow_mask) + cv2.countNonZero(orange_mask)
            total_torso_pixels = torso_hsv.shape[0] * torso_hsv.shape[1]

            if total_torso_pixels > 0 and (vest_pixels / total_torso_pixels) > 0.12:
                detected.append("Vest")

        # Head crop (hardhat check)
        head_y2 = int(crop_h * 0.30)
        head_hsv = hsv[0:head_y2, :]
        if head_hsv.size > 0:
            yellow_head = cv2.inRange(head_hsv, np.array([20, 100, 120]), np.array([45, 255, 255]))
            white_head = cv2.inRange(head_hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
            hard_hat_pixels = cv2.countNonZero(yellow_head) + cv2.countNonZero(white_head)
            total_head_pixels = head_hsv.shape[0] * head_hsv.shape[1]
            if total_head_pixels > 0 and (hard_hat_pixels / total_head_pixels) > 0.15:
                detected.append("Hard_hat")

        return detected


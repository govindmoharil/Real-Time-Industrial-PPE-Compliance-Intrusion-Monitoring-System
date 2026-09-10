import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class TrackedWorker:
    """
    State and historical trajectory of an identified worker.
    """
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    detected_ppe: List[str] = field(default_factory=list)
    missing_mandatory: List[str] = field(default_factory=list)
    explicit_violations: List[str] = field(default_factory=list)
    is_compliant: bool = True
    ppe_boxes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trajectory trail of foot contact points [(x, y), ...]
    trail: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Tracking lifecycle
    disappeared_count: int = 0
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)

    # Debounce / Violation state
    non_compliant_streak: int = 0
    compliant_streak: int = 0
    violation_start_time: Optional[float] = None
    alert_triggered: bool = False
    active_zone_violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Ground contact point (bottom-center)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


class WorkerTracker:
    """
    Robust Multi-Object Worker Tracker with IoU matching, Centroid smoothing,
    and temporal violation debouncing.
    """

    def __init__(
        self,
        max_disappeared: int = 20,
        iou_match_threshold: float = 0.30,
        debounce_seconds: float = 1.0,
        debounce_frames: int = 5
    ):
        self.max_disappeared = max_disappeared
        self.iou_match_threshold = iou_match_threshold
        self.debounce_seconds = debounce_seconds
        self.debounce_frames = debounce_frames

        self.next_id: int = 1
        self.tracks: Dict[int, TrackedWorker] = {}

    def _compute_iou(self, box_a: List[float], box_b: List[float]) -> float:
        """Computes intersection-over-union between two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detected_workers: List[Any], current_time: Optional[float] = None) -> List[TrackedWorker]:
        """
        Updates tracks with newly detected workers from the current frame.
        """
        if current_time is None:
            current_time = time.time()

        # If no tracks exist yet, initialize all detections as new tracks
        if len(self.tracks) == 0:
            for dw in detected_workers:
                self._register_worker(dw, current_time)
            return list(self.tracks.values())

        # If no detections in current frame, increment disappeared counts
        if len(detected_workers) == 0:
            for worker_id in list(self.tracks.keys()):
                self.tracks[worker_id].disappeared_count += 1
                if self.tracks[worker_id].disappeared_count > self.max_disappeared:
                    self._deregister_worker(worker_id)
            return list(self.tracks.values())

        # Match existing tracks with current detections using IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detected_workers)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            for j, dw in enumerate(detected_workers):
                iou_matrix[i, j] = self._compute_iou(self.tracks[tid].bbox, dw.bbox)

        # Greedy bipartite matching
        matched_tracks = set()
        matched_detections = set()

        if iou_matrix.size > 0:
            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_match_threshold:
                    break
                row, col = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[row]

                if row not in matched_tracks and col not in matched_detections:
                    matched_tracks.add(row)
                    matched_detections.add(col)
                    self._update_worker(tid, detected_workers[col], current_time)

                # Invalidate matched row and column
                iou_matrix[row, :] = -1.0
                iou_matrix[:, col] = -1.0

        # Handle unmatched existing tracks
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid].disappeared_count += 1
                if self.tracks[tid].disappeared_count > self.max_disappeared:
                    self._deregister_worker(tid)

        # Handle unmatched new detections
        for j, dw in enumerate(detected_workers):
            if j not in matched_detections:
                self._register_worker(dw, current_time)

        return list(self.tracks.values())

    def _register_worker(self, detection: Any, current_time: float):
        """Creates a new tracked worker entry."""
        worker = TrackedWorker(
            id=self.next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            detected_ppe=detection.detected_ppe,
            missing_mandatory=detection.missing_mandatory,
            explicit_violations=detection.explicit_violations,
            is_compliant=detection.is_compliant,
            ppe_boxes=detection.ppe_boxes,
            first_seen_time=current_time,
            last_seen_time=current_time
        )
        worker.trail.append(worker.foot_point)

        if not detection.is_compliant:
            worker.non_compliant_streak = 1
            worker.violation_start_time = current_time
        else:
            worker.compliant_streak = 1

        self.tracks[self.next_id] = worker
        self.next_id += 1

    def _update_worker(self, worker_id: int, detection: Any, current_time: float):
        """Updates an existing tracked worker with new frame detections."""
        worker = self.tracks[worker_id]
        worker.bbox = detection.bbox
        worker.confidence = (worker.confidence * 0.7) + (detection.confidence * 0.3)
        worker.detected_ppe = detection.detected_ppe
        worker.missing_mandatory = detection.missing_mandatory
        worker.explicit_violations = detection.explicit_violations
        worker.ppe_boxes = detection.ppe_boxes
        worker.is_compliant = detection.is_compliant
        worker.disappeared_count = 0
        worker.last_seen_time = current_time
        worker.trail.append(worker.foot_point)

        # Debounce logic
        if not detection.is_compliant:
            worker.non_compliant_streak += 1
            worker.compliant_streak = 0
            if worker.violation_start_time is None:
                worker.violation_start_time = current_time
        else:
            worker.compliant_streak += 1
            # Reset violation streak if compliant for several frames
            if worker.compliant_streak >= 3:
                worker.non_compliant_streak = 0
                worker.violation_start_time = None
                worker.alert_triggered = False

    def _deregister_worker(self, worker_id: int):
        """Removes a lost worker track."""
        if worker_id in self.tracks:
            del self.tracks[worker_id]

    def should_trigger_ppe_alert(self, worker: TrackedWorker, current_time: float) -> bool:
        """
        Determines whether a worker's PPE non-compliance has exceeded the debounce duration
        and has not yet been alerted.
        """
        if worker.is_compliant:
            return False
        if worker.alert_triggered:
            return False

        # Check frame count threshold and duration threshold
        if worker.non_compliant_streak >= self.debounce_frames:
            if worker.violation_start_time is not None:
                duration = current_time - worker.violation_start_time
                if duration >= self.debounce_seconds:
                    return True
        return False

    def reset(self):
        """Resets tracker state."""
        self.tracks.clear()
        self.next_id = 1


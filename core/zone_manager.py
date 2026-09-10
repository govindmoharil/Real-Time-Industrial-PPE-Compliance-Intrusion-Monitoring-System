import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from shapely.geometry import Point, Polygon

@dataclass
class Zone:
    """
    Represents an industrial geofence / virtual hazardous polygon zone.
    Points are stored as normalized coordinates [[x1, y1], [x2, y2], ...] in range [0.0, 1.0].
    """
    id: str
    name: str
    zone_type: str  # 'DANGER_EXCLUSION', 'PPE_MANDATORY', 'SAFE_TRANSIT'
    points: List[List[float]]  # Normalized polygon vertices [[x, y], ...]
    color_rgb: Tuple[int, int, int] = (255, 0, 0)
    required_ppe: List[str] = field(default_factory=lambda: ["Hard_hat", "Vest"])
    max_dwell_seconds: float = 30.0  # Max allowed time before loitering alert
    description: str = ""

    def get_pixel_points(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Converts normalized vertices to integer pixel coordinates for OpenCV."""
        pts = np.array([
            [int(p[0] * frame_width), int(p[1] * frame_height)]
            for p in self.points
        ], dtype=np.int32)
        return pts

    def contains_point(self, norm_x: float, norm_y: float) -> bool:
        """Checks if normalized point (x, y) lies inside this zone polygon."""
        if len(self.points) < 3:
            return False
        try:
            poly = Polygon(self.points)
            point = Point(norm_x, norm_y)
            return poly.contains(point) or poly.touches(point)
        except Exception:
            return False

    def contains_footprint(self, bbox: List[float], frame_width: int, frame_height: int) -> bool:
        """
        Determines whether the worker's foot contact point (bottom-center of bbox)
        is inside this zone polygon.
        bbox is [x1, y1, x2, y2] in pixel coordinates.
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / (2.0 * frame_width)
        foot_y = y2 / float(frame_height)
        return self.contains_point(foot_x, foot_y)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "zone_type": self.zone_type,
            "points": self.points,
            "color_rgb": list(self.color_rgb),
            "required_ppe": self.required_ppe,
            "max_dwell_seconds": self.max_dwell_seconds,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Zone":
        return cls(
            id=data["id"],
            name=data["name"],
            zone_type=data.get("zone_type", "DANGER_EXCLUSION"),
            points=data.get("points", []),
            color_rgb=tuple(data.get("color_rgb", [255, 0, 0])),
            required_ppe=data.get("required_ppe", ["Hard_hat", "Vest"]),
            max_dwell_seconds=float(data.get("max_dwell_seconds", 30.0)),
            description=data.get("description", "")
        )


class ZoneManager:
    """
    Manages active geofence zones, monitors worker intrusions, and tracks dwell times.
    """

    def __init__(self, config_path: str = "configs/zones.json"):
        self.config_path = config_path
        self.zones: Dict[str, Zone] = {}
        # worker_id -> {zone_id: entry_timestamp}
        self.worker_zone_entries: Dict[int, Dict[str, float]] = {}
        self.load_zones()

    def load_zones(self):
        """Loads zones from JSON file or generates factory defaults if missing."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.zones = {
                        item["id"]: Zone.from_dict(item)
                        for item in data.get("zones", [])
                    }
                return
            except Exception as e:
                print(f"Error loading zones config: {e}. Loading defaults.")

        self.zones = self._create_default_zones()
        self.save_zones()

    def _create_default_zones(self) -> Dict[str, Zone]:
        """Creates sample factory zones (Restricted Crane Zone, Chemical Storage, Assembly Lane)."""
        defaults = [
            Zone(
                id="zone_danger_crane",
                name="Zone 1: Heavy Machinery Danger",
                zone_type="DANGER_EXCLUSION",
                points=[
                    [0.05, 0.45],
                    [0.42, 0.45],
                    [0.42, 0.95],
                    [0.05, 0.95]
                ],
                color_rgb=(230, 45, 45),  # Red
                required_ppe=["Hard_hat", "Vest"],
                max_dwell_seconds=0.0,  # 0 = Any presence is immediate violation
                description="Heavy robotic arm & overhead crane sweep radius. Zero unauthorized entry permitted."
            ),
            Zone(
                id="zone_ppe_mandatory",
                name="Zone 2: High-Risk Assembly",
                zone_type="PPE_MANDATORY",
                points=[
                    [0.55, 0.40],
                    [0.95, 0.40],
                    [0.95, 0.92],
                    [0.55, 0.92]
                ],
                color_rgb=(240, 160, 20),  # Amber / Orange
                required_ppe=["Hard_hat", "Vest", "Boots"],
                max_dwell_seconds=600.0,
                description="Assembly line cell. Hard hat and high-vis vest strictly required."
            )
        ]
        return {z.id: z for z in defaults}

    def save_zones(self):
        """Persists current zones to JSON config."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        data = {
            "version": "1.0",
            "zones": [z.to_dict() for z in self.zones.values()]
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_zone(self, zone: Zone):
        self.zones[zone.id] = zone
        self.save_zones()

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.save_zones()
            return True
        return False

    def evaluate_worker(
        self,
        worker_id: int,
        bbox: List[float],
        detected_ppe: List[str],
        frame_width: int,
        frame_height: int,
        current_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a worker's location against all defined zones.
        Returns a list of violation events (if any).
        """
        if current_time is None:
            current_time = time.time()

        if worker_id not in self.worker_zone_entries:
            self.worker_zone_entries[worker_id] = {}

        violations = []
        active_zone_ids = set()

        for zone_id, zone in self.zones.items():
            is_inside = zone.contains_footprint(bbox, frame_width, frame_height)

            if is_inside:
                active_zone_ids.add(zone_id)
                # Record entry time if newly entered
                if zone_id not in self.worker_zone_entries[worker_id]:
                    self.worker_zone_entries[worker_id][zone_id] = current_time

                entry_time = self.worker_zone_entries[worker_id][zone_id]
                dwell_time = current_time - entry_time

                # 1. Check Strict Danger / Exclusion Zone
                if zone.zone_type == "DANGER_EXCLUSION":
                    violations.append({
                        "type": "ZONE_INTRUSION",
                        "severity": "CRITICAL",
                        "zone_id": zone.id,
                        "zone_name": zone.name,
                        "dwell_time": round(dwell_time, 1),
                        "message": f"CRITICAL: Intrusion into {zone.name}!"
                    })

                # 2. Check PPE Mandatory Zone
                elif zone.zone_type == "PPE_MANDATORY":
                    missing = [
                        item for item in zone.required_ppe
                        if item not in detected_ppe
                    ]
                    if missing:
                        violations.append({
                            "type": "ZONE_PPE_VIOLATION",
                            "severity": "HIGH",
                            "zone_id": zone.id,
                            "zone_name": zone.name,
                            "missing_ppe": missing,
                            "dwell_time": round(dwell_time, 1),
                            "message": f"Non-compliant in {zone.name}: Missing {', '.join(missing)}"
                        })

                # 3. Check Loitering in Safe Transit / Hazard Zone
                elif zone.zone_type == "SAFE_TRANSIT" and zone.max_dwell_seconds > 0:
                    if dwell_time > zone.max_dwell_seconds:
                        violations.append({
                            "type": "ZONE_LOITERING",
                            "severity": "MEDIUM",
                            "zone_id": zone.id,
                            "zone_name": zone.name,
                            "dwell_time": round(dwell_time, 1),
                            "message": f"Loitering alert in {zone.name} ({round(dwell_time)}s > {zone.max_dwell_seconds}s)"
                        })

        # Cleanup entries for zones the worker has exited
        current_recorded = list(self.worker_zone_entries[worker_id].keys())
        for z_id in current_recorded:
            if z_id not in active_zone_ids:
                del self.worker_zone_entries[worker_id][z_id]

        return violations

    def prune_old_workers(self, active_worker_ids: List[int]):
        """Cleans up internal tracking state for workers no longer in frame."""
        tracked = list(self.worker_zone_entries.keys())
        for wid in tracked:
            if wid not in active_worker_ids:
                del self.worker_zone_entries[wid]


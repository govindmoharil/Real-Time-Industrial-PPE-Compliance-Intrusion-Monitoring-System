import os
import time
import wave
import struct
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from .database import IncidentDatabase
from .tracker import TrackedWorker

@dataclass
class IncidentAlert:
    """Represents an active notification event."""
    id: int
    timestamp: str
    camera_id: str
    worker_id: int
    violation_type: str
    severity: str
    zone_name: Optional[str]
    message: str
    snapshot_path: Optional[str]
    confidence: float


class AlertManager:
    """
    Coordinates violation alerts, debouncing cooldowns, forensic snapshot capture,
    and persistent database logging.
    """

    def __init__(
        self,
        db: IncidentDatabase,
        snapshot_dir: str = "data/snapshots",
        sound_dir: str = "data/audio",
        alert_cooldown_seconds: float = 5.0
    ):
        self.db = db
        self.snapshot_dir = snapshot_dir
        self.sound_dir = sound_dir
        self.alert_cooldown_seconds = alert_cooldown_seconds

        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.sound_dir, exist_ok=True)

        # Cache of (worker_id, violation_type) -> last_alert_time
        self.last_alert_times: Dict[Tuple[int, str], float] = {}
        
        # Recent live alerts for HUD / Web Ticker
        self.recent_alerts: List[IncidentAlert] = []
        self.max_recent_alerts = 25

        self.alarm_sound_path = self._generate_alarm_sound()

    def _generate_alarm_sound(self) -> str:
        """
        Generates a 0.5s synthesized dual-tone industrial warning chime (.wav)
        so no external audio files are required.
        """
        sound_file = os.path.join(self.sound_dir, "alert_chime.wav")
        if os.path.exists(sound_file):
            return sound_file

        sample_rate = 22050
        duration = 0.45  # seconds
        num_samples = int(sample_rate * duration)

        with wave.open(sound_file, "w") as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)

            # Two-tone beep (880 Hz -> 660 Hz)
            frames = bytearray()
            for i in range(num_samples):
                t = float(i) / sample_rate
                freq = 880.0 if t < 0.22 else 660.0
                envelope = math.sin(math.pi * (t / duration))  # Fade in/out
                val = int(32767.0 * 0.5 * envelope * math.sin(2.0 * math.pi * freq * t))
                val = max(-32768, min(32767, val))
                frames.extend(struct.pack("<h", val))

            wav.writeframes(frames)

        return sound_file

    def process_worker_violations(
        self,
        worker: TrackedWorker,
        zone_violations: List[Dict[str, Any]],
        annotated_frame: np.ndarray,
        camera_id: str = "CAM-01",
        current_time: Optional[float] = None
    ) -> List[IncidentAlert]:
        """
        Evaluates a worker's PPE compliance and zone violations.
        Triggers snapshots, DB logs, and live alerts if cooldown allows.
        """
        if current_time is None:
            current_time = time.time()

        triggered_alerts = []

        # 1. Process Zone Violations (Intrusions take highest priority)
        for zv in zone_violations:
            v_type = zv["type"]
            cooldown_key = (worker.id, f"{v_type}_{zv.get('zone_id', '')}")
            last_alert = self.last_alert_times.get(cooldown_key, 0.0)

            if current_time - last_alert >= self.alert_cooldown_seconds:
                self.last_alert_times[cooldown_key] = current_time
                alert = self._record_violation(
                    worker=worker,
                    violation_type=v_type,
                    severity=zv.get("severity", "CRITICAL"),
                    zone_name=zv.get("zone_name"),
                    message=zv.get("message", "Restricted zone violation!"),
                    details=zv,
                    annotated_frame=annotated_frame,
                    camera_id=camera_id,
                    current_time=current_time
                )
                triggered_alerts.append(alert)

        # 2. Process General PPE Non-Compliance (Debounced)
        if not worker.is_compliant:
            cooldown_key = (worker.id, "PPE_NON_COMPLIANT")
            last_alert = self.last_alert_times.get(cooldown_key, 0.0)

            # Check if debounced duration met
            if (current_time - last_alert >= self.alert_cooldown_seconds) and (worker.non_compliant_streak >= 5):
                missing_str = ", ".join(worker.missing_mandatory)
                self.last_alert_times[cooldown_key] = current_time
                worker.alert_triggered = True

                details = {
                    "missing_mandatory": worker.missing_mandatory,
                    "explicit_violations": worker.explicit_violations,
                    "detected_ppe": worker.detected_ppe
                }

                alert = self._record_violation(
                    worker=worker,
                    violation_type="PPE_NON_COMPLIANCE",
                    severity="HIGH",
                    zone_name=None,
                    message=f"Worker #{worker.id} missing mandatory PPE: {missing_str}",
                    details=details,
                    annotated_frame=annotated_frame,
                    camera_id=camera_id,
                    current_time=current_time
                )
                triggered_alerts.append(alert)

        return triggered_alerts

    def _record_violation(
        self,
        worker: TrackedWorker,
        violation_type: str,
        severity: str,
        zone_name: Optional[str],
        message: str,
        details: Dict[str, Any],
        annotated_frame: np.ndarray,
        camera_id: str,
        current_time: float
    ) -> IncidentAlert:
        """Captures snapshot, saves to SQLite DB, and updates recent alerts."""
        dt_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S_%f")[:19]
        iso_str = datetime.fromtimestamp(current_time).isoformat()

        # Save snapshot
        snapshot_filename = f"incident_{dt_str}_{camera_id}_W{worker.id}_{violation_type}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)

        try:
            # Stamp watermark on snapshot
            stamped_frame = annotated_frame.copy()
            cv2.putText(
                stamped_frame,
                f"INCIDENT: {violation_type} | Worker #{worker.id} | {iso_str[:19]}",
                (20, stamped_frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imwrite(snapshot_path, stamped_frame)
        except Exception as e:
            print(f"[AlertManager] Error saving snapshot: {e}")
            snapshot_path = ""

        # Log into SQLite DB
        db_id = self.db.log_incident(
            camera_id=camera_id,
            worker_id=worker.id,
            violation_type=violation_type,
            severity=severity,
            confidence=worker.confidence,
            zone_name=zone_name,
            details=details,
            snapshot_path=snapshot_path,
            timestamp=iso_str
        )

        alert = IncidentAlert(
            id=db_id,
            timestamp=iso_str,
            camera_id=camera_id,
            worker_id=worker.id,
            violation_type=violation_type,
            severity=severity,
            zone_name=zone_name,
            message=message,
            snapshot_path=snapshot_path,
            confidence=worker.confidence
        )

        # Add to recent alerts queue
        self.recent_alerts.insert(0, alert)
        if len(self.recent_alerts) > self.max_recent_alerts:
            self.recent_alerts.pop()

        return alert

    def get_recent_alerts(self) -> List[IncidentAlert]:
        """Returns the most recent active alerts."""
        return self.recent_alerts


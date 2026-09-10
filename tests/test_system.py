import os
import shutil
import pytest
import numpy as np
import cv2

from core.database import IncidentDatabase
from core.zone_manager import Zone, ZoneManager
from core.detector import PPEDetector, WorkerCompliance
from core.tracker import WorkerTracker, TrackedWorker
from core.alert_manager import AlertManager
from core.visualizer import Visualizer

TEST_DIR = "data/test_tmp"

@pytest.fixture(autouse=True)
def setup_teardown():
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)


def test_database_crud():
    db_path = os.path.join(TEST_DIR, "test_db.sqlite")
    db = IncidentDatabase(db_path=db_path)

    # 1. Log incident
    inc_id = db.log_incident(
        camera_id="CAM-TEST",
        worker_id=1,
        violation_type="ZONE_INTRUSION",
        severity="CRITICAL",
        confidence=0.92,
        zone_name="Robot Zone A",
        details={"reason": "Hazard perimeter crossed"}
    )
    assert inc_id > 0

    # 2. Query incidents
    records = db.get_incidents(limit=10)
    assert len(records) == 1
    assert records[0]["worker_id"] == 1
    assert records[0]["violation_type"] == "ZONE_INTRUSION"
    assert records[0]["details"]["reason"] == "Hazard perimeter crossed"

    # 3. Update status
    updated = db.update_incident_status(inc_id, "RESOLVED", notes="Worker notified and evacuated")
    assert updated is True
    records_after = db.get_incidents(status="RESOLVED")
    assert len(records_after) == 1
    assert records_after[0]["notes"] == "Worker notified and evacuated"

    # 4. Statistics
    stats = db.get_statistics(hours=24)
    assert stats["total_incidents"] == 1
    assert stats["critical_incidents"] == 1
    assert "ZONE_INTRUSION" in stats["violation_breakdown"]

    # 5. Export CSV
    csv_file = os.path.join(TEST_DIR, "report.csv")
    out = db.export_to_csv(csv_file)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_zone_manager_and_geofencing():
    # Define a 100x100 zone in normalized coordinates [0.1, 0.1] to [0.5, 0.5]
    zone = Zone(
        id="test_danger_zone",
        name="Danger Zone",
        zone_type="DANGER_EXCLUSION",
        points=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
        color_rgb=(255, 0, 0)
    )

    # Point-in-polygon tests
    assert zone.contains_point(0.2, 0.2) is True
    assert zone.contains_point(0.8, 0.8) is False

    # Footprint containment test (frame size 1000x1000)
    # Worker bbox from (150, 100) to (250, 300) -> foot point is (200, 300) -> norm (0.2, 0.3)
    bbox_inside = [150, 100, 250, 300]
    assert zone.contains_footprint(bbox_inside, frame_width=1000, frame_height=1000) is True

    # Worker bbox outside: foot point is (700, 800) -> norm (0.7, 0.8)
    bbox_outside = [650, 600, 750, 800]
    assert zone.contains_footprint(bbox_outside, frame_width=1000, frame_height=1000) is False

    # ZoneManager evaluation
    zm = ZoneManager(config_path=os.path.join(TEST_DIR, "zones.json"))
    zm.zones.clear()
    zm.add_zone(zone)

    # Evaluate worker inside danger zone
    violations = zm.evaluate_worker(
        worker_id=10,
        bbox=bbox_inside,
        detected_ppe=["Hard_hat", "Vest"],
        frame_width=1000,
        frame_height=1000
    )
    assert len(violations) == 1
    assert violations[0]["type"] == "ZONE_INTRUSION"
    assert violations[0]["severity"] == "CRITICAL"


def test_tracker_and_debounce():
    tracker = WorkerTracker(max_disappeared=5, debounce_seconds=0.5, debounce_frames=3)

    # Mock detection
    det1 = WorkerCompliance(
        bbox=[100, 100, 200, 300],
        confidence=0.88,
        detected_ppe=[],
        missing_mandatory=["Hard_hat", "Vest"],
        is_compliant=False
    )

    # Frame 1
    tracked = tracker.update([det1], current_time=100.0)
    assert len(tracked) == 1
    w = tracked[0]
    assert w.id == 1
    assert w.is_compliant is False
    assert w.non_compliant_streak == 1
    assert tracker.should_trigger_ppe_alert(w, current_time=100.0) is False

    # Frame 2
    tracked = tracker.update([det1], current_time=100.2)
    assert tracked[0].non_compliant_streak == 2

    # Frame 3 - Met frames count, check time window
    tracked = tracker.update([det1], current_time=100.6)
    assert tracked[0].non_compliant_streak == 3
    # 100.6 - 100.0 = 0.6s >= 0.5s debounce duration
    assert tracker.should_trigger_ppe_alert(tracked[0], current_time=100.6) is True


def test_detector_inference_on_sample_image():
    detector = PPEDetector(model_path="models/ppe_yolov8.pt", conf_threshold=0.25)
    
    # Create synthetic test frame
    test_frame = np.full((640, 640, 3), 120, dtype=np.uint8)
    # Draw yellow hardhat oval and orange vest box
    cv2.circle(test_frame, (320, 200), 40, (0, 220, 245), -1) # Yellow
    cv2.rectangle(test_frame, (280, 250), (360, 450), (0, 140, 255), -1) # Orange vest
    
    res = detector.detect(test_frame)
    assert res is not None
    assert isinstance(res.workers, list)
    assert res.inference_time_ms >= 0.0


def test_alert_manager_and_snapshot():
    db_path = os.path.join(TEST_DIR, "alert_db.sqlite")
    snap_dir = os.path.join(TEST_DIR, "snaps")
    sound_dir = os.path.join(TEST_DIR, "audio")
    db = IncidentDatabase(db_path=db_path)
    alert_mgr = AlertManager(db=db, snapshot_dir=snap_dir, sound_dir=sound_dir, alert_cooldown_seconds=1.0)

    # Verify chime was synthesized
    assert os.path.exists(alert_mgr.alarm_sound_path)
    assert os.path.getsize(alert_mgr.alarm_sound_path) > 0

    worker = TrackedWorker(
        id=7,
        bbox=[50, 50, 150, 250],
        confidence=0.90,
        is_compliant=False,
        missing_mandatory=["Hard_hat"],
        non_compliant_streak=10
    )
    worker.violation_start_time = 100.0

    mock_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    zone_violations = [{
        "type": "ZONE_INTRUSION",
        "severity": "CRITICAL",
        "zone_id": "z1",
        "zone_name": "Danger Cell",
        "message": "Intrusion detected"
    }]

    alerts = alert_mgr.process_worker_violations(
        worker=worker,
        zone_violations=zone_violations,
        annotated_frame=mock_frame,
        camera_id="CAM-TEST",
        current_time=102.0
    )

    assert len(alerts) >= 1
    assert alerts[0].worker_id == 7
    assert alerts[0].severity == "CRITICAL"
    # Verify snapshot image was saved
    assert os.path.exists(alerts[0].snapshot_path)

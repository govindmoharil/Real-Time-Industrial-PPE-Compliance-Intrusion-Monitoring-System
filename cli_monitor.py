"""
Industrial Safety AI - Headless & Real-Time CLI Video Monitor.
Can be executed as a background daemon, 24/7 CCTV monitor, or batch video analyzer.
"""

import argparse
import os
import sys
import time
import cv2

from core.database import IncidentDatabase
from core.zone_manager import ZoneManager
from core.detector import PPEDetector
from core.tracker import WorkerTracker
from core.alert_manager import AlertManager
from core.visualizer import Visualizer

def run_monitor(
    source: str = "data/demo/industrial_sample.mp4",
    model_path: str = "models/ppe_yolov8.pt",
    zones_config: str = "configs/zones.json",
    db_path: str = "data/database.sqlite",
    camera_id: str = "CAM-01",
    conf_threshold: float = 0.30,
    max_frames: int = -1,
    display: bool = False,
    output_video_path: str = ""
):
    print("=" * 65)
    print("  INDUSTRIAL PPE COMPLIANCE & INTRUSION MONITORING SYSTEM")
    print("=" * 65)
    print(f"[*] Source:           {source}")
    print(f"[*] Camera ID:        {camera_id}")
    print(f"[*] Model Weights:    {model_path}")
    print(f"[*] Confidence:       {conf_threshold}")
    print(f"[*] Database:         {db_path}")

    # Check if source is webcam index
    if source.isdigit():
        video_src = int(source)
    else:
        video_src = source

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source '{source}'")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Stream Info:      {frame_width}x{frame_height} @ {fps:.1f} FPS (Total: {total_frames} frames)")

    # Initialize Core Components
    db = IncidentDatabase(db_path=db_path)
    zone_mgr = ZoneManager(config_path=zones_config)
    detector = PPEDetector(model_path=model_path, conf_threshold=conf_threshold)
    tracker = WorkerTracker(debounce_seconds=1.0, debounce_frames=5)
    alert_mgr = AlertManager(db=db)
    visualizer = Visualizer()

    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        print(f"[*] Recording output to: {output_video_path}")

    frame_count = 0
    total_violations_logged = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            now = time.time()

            # 1. Detection
            det_res = detector.detect(frame)

            # 2. Tracking
            tracked_workers = tracker.update(det_res.workers, current_time=now)

            # 3. Zone evaluation & Intrusion detection
            active_intrusions = {}
            breached_zone_ids = set()

            for worker in tracked_workers:
                zv = zone_mgr.evaluate_worker(
                    worker_id=worker.id,
                    bbox=worker.bbox,
                    detected_ppe=worker.detected_ppe,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    current_time=now
                )
                if zv:
                    active_intrusions[worker.id] = zv
                    for v in zv:
                        breached_zone_ids.add(v["zone_id"])

            zone_mgr.prune_old_workers([w.id for w in tracked_workers])

            # 4. Rendering overlays
            annotated = frame.copy()
            annotated = visualizer.draw_zones(annotated, list(zone_mgr.zones.values()), breached_zone_ids)
            annotated = visualizer.draw_workers(annotated, tracked_workers, active_intrusions)
            annotated = visualizer.draw_hud(annotated, tracked_workers, len(breached_zone_ids), camera_id=camera_id)

            # 5. Alerting & Snapshot recording
            for worker in tracked_workers:
                worker_zv = active_intrusions.get(worker.id, [])
                new_alerts = alert_mgr.process_worker_violations(
                    worker=worker,
                    zone_violations=worker_zv,
                    annotated_frame=annotated,
                    camera_id=camera_id,
                    current_time=now
                )
                if new_alerts:
                    total_violations_logged += len(new_alerts)
                    for a in new_alerts:
                        print(f"  [ALERT] Frame {frame_count:04d} | {a.severity} | W#{a.worker_id} | {a.violation_type} | {a.message}")

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Industrial Safety AI Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] User requested exit.")
                    break

            if max_frames > 0 and frame_count >= max_frames:
                print(f"[*] Reached max frames limit ({max_frames}). Stopping.")
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    print("=" * 65)
    print(f"[*] Processing Complete.")
    print(f"[*] Processed Frames:         {frame_count}")
    print(f"[*] Total Violations Logged:  {total_violations_logged}")
    print(f"[*] Database Path:            {db_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Safety PPE & Intrusion Monitoring CLI")
    parser.add_argument("--source", type=str, default="data/demo/industrial_sample.mp4", help="Video file, RTSP URL, or webcam ID (0)")
    parser.add_argument("--model", type=str, default="models/ppe_yolov8.pt", help="Path to YOLO weights")
    parser.add_argument("--zones", type=str, default="configs/zones.json", help="Path to zones JSON config")
    parser.add_argument("--db", type=str, default="data/database.sqlite", help="Path to SQLite DB")
    parser.add_argument("--camera-id", type=str, default="CAM-01-FACTORY-FLOOR", help="Camera identifier")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process (-1 for all)")
    parser.add_argument("--display", action="store_true", help="Display video in OpenCV window")
    parser.add_argument("--output", type=str, default="", help="Save annotated output video path")

    args = parser.parse_args()
    run_monitor(
        source=args.source,
        model_path=args.model,
        zones_config=args.zones,
        db_path=args.db,
        camera_id=args.camera_id,
        conf_threshold=args.conf,
        max_frames=args.max_frames,
        display=args.display,
        output_video_path=args.output
    )


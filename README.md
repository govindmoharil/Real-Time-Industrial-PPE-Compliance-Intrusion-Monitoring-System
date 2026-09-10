# 🛡️ Real-Time Industrial PPE Compliance & Intrusion Monitoring System

An enterprise-grade, computer vision-based safety platform designed for 24/7 autonomous monitoring of manufacturing plants, construction sites, chemical refineries, and automated warehouses. 

The system leverages deep learning (**YOLOv8**), multi-object tracking, and geometric geofencing to verify Personal Protective Equipment (PPE) compliance in real-time and immediately flag unauthorized personnel intrusions into hazardous exclusion zones.

---

## 🌟 Key Features

- **Multi-Class PPE Detection & Anatomical Association**:
  - Detects **Hard Hats / Helmets**, **High-Visibility Safety Vests**, **Safety Boots**, **Gloves**, **Protective Glasses**, and **Face Masks**.
  - Distinguishes both presence and explicit absence cues (`No-Helmet`, `No-Vest`, `No-Boots`, etc.).
  - Anatomical association maps detected equipment to specific worker body parts (head, torso, feet).
- **Hazardous Zone Geofencing & Intrusion Monitoring**:
  - Virtual polygon hazard zones with normalized coordinates (resolution-independent).
  - Zone types:
    - `DANGER_EXCLUSION`: Restricted operating envelope (e.g. robotic arms, crane sweep, high voltage). Any worker triggers immediate high-priority alarm.
    - `PPE_MANDATORY`: High-risk zones requiring strict PPE combinations (e.g., Helmet + Vest).
    - `SAFE_TRANSIT`: Monitored pedestrian walkways with loitering/obstruction alerts.
  - **Ground-Plane Contact Anchor**: Uses bottom-center bounding box point \(((x_1 + x_2)/2, y_2)\) to test actual feet contact on the ground, preventing false alarms when hands or heads reach past the border.
- **Multi-Object Worker Tracking & Debouncing**:
  - Assigns persistent IDs across frames with motion trajectory trails.
  - Temporal debounce engine suppresses single-frame detector flicker and prevents alarm fatigue.
- **Forensic Evidence Vault & Incident Logging**:
  - Automatically captures and watermarks high-resolution violation frames into `data/snapshots/`.
  - SQLite database indexes incidents with timestamps, worker IDs, camera IDs, violation types, and review status (`NEW`, `ACKNOWLEDGED`, `RESOLVED`, `FALSE_POSITIVE`).
  - One-click CSV export and safety audit reporting.
- **Multi-Source Video Ingestion**:
  - Real-time webcam (`0`, `1`), RTSP/HTTP IP camera feeds, uploaded video files (`.mp4`, `.avi`), static snapshots, and built-in synthetic industrial simulation.
- **Modern Industrial Web Operations Dashboard**:
  - Built with **Streamlit** with dark SCADA aesthetics, live HUD stats, interactive geofence configurator, forensic incident viewer, and Plotly analytics.
- **Headless CLI Monitor**:
  - Standalone daemon runner for 24/7 background CCTV recording and edge device deployment.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Video Stream / Camera Ingestion\nWebcam / RTSP / MP4 / Image] --> B[Frame Preprocessor & Grabber]
    B --> C[AI Detection Engine\nFine-Tuned YOLOv8 PPE + Fallback COCO]
    C --> D[Worker Tracking Engine\nPersistent IDs & Foot Ground Anchor]
    D --> E[Spatial PPE Association\nWorker Box ↔ Head / Torso / Feet Matching]
    D --> F[Geofencing & Zone Engine\nPoint-in-Polygon Feet Analysis]
    E --> G[Compliance Evaluation\nHelmet / Vest / Gloves / Boots Verification]
    F --> H[Intrusion Detection\nDanger Zone / PPE-Mandatory / Loitering]
    G & H --> I[Alert & Debounce Manager\nNoise Filter / Cooldown / Snapshot Trigger]
    I --> J[(SQLite Incident DB\nEvidence Vault & Audit Logs)]
    I --> K[Audio & Visual Alarm Signals]
    J & K --> L[Streamlit Industrial Operations Dashboard\nLive HUD, Forensic Vault, Analytics, Zone Editor]
```

---

## 📁 Repository Structure

```
├── app.py                      # Interactive Streamlit Web Operations Dashboard
├── cli_monitor.py              # Headless CLI Runner for 24/7 background CCTV monitoring
├── generate_demo_assets.py     # Synthetic industrial video & test images generator
├── requirements.txt            # Python dependencies
├── core/
│   ├── __init__.py             # Package initializer
│   ├── detector.py             # YOLOv8 PPE detector with anatomical association
│   ├── tracker.py              # Multi-object worker tracker with debounce logic
│   ├── zone_manager.py         # Polygon geofencing and point-in-polygon intrusion checks
│   ├── alert_manager.py        # Snapshot capture, audio chime, and cooldown manager
│   ├── database.py             # SQLite persistence layer and analytics queries
│   └── visualizer.py           # High-FPS OpenCV HUD and overlay renderer
├── configs/
│   ├── settings.json           # System confidence, debounce, and camera settings
│   └── zones.json              # Configurable polygon hazard zones
├── models/
│   └── ppe_yolov8.pt           # Fine-tuned 19-class industrial PPE YOLO weights
├── data/
│   ├── snapshots/              # Captured violation evidence images
│   ├── demo/                   # Demo video and test images
│   └── database.sqlite         # SQLite incident audit logs
└── tests/
    └── test_system.py          # Pytest automated test suite
```

---

## 🚀 Quick Start

### 1. Installation

Clone or open the repository and install dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Generate Synthetic Demo Assets (Optional)

Generate realistic simulated factory video and test images for instant evaluation:

```powershell
python generate_demo_assets.py
```

### 3. Launch the Interactive Web Dashboard

Start the industrial safety operations center:

```powershell
streamlit run app.py
```

The web dashboard will open in your browser at `http://localhost:8501`.

#### Web Dashboard Features:
1. **🔴 Live Video Monitor**: Live video feed with toggleable overlays (bounding boxes, hazard zones, motion trails, HUD bar) and real-time alert ticker.
2. **📐 Hazard Zone Configurator**: Inspect, create, edit, or delete polygonal hazard zones directly on a reference image.
3. **📋 Incident Evidence Vault**: Searchable audit log with high-res snapshot inspection, status resolution workflow, and CSV export.
4. **📊 Safety Analytics & KPIs**: Site compliance rate, violation category breakdown, zone intrusion hotspots, and hourly incident trends.
5. **⚙️ System Settings**: Adjust confidence thresholds, debounce parameters, and mandatory PPE checklist.

---

### 4. Headless CLI Monitoring (Edge / Background Daemon)

To run the system in headless mode on an RTSP stream, camera, or video file:

```powershell
# Run on demo video
python cli_monitor.py --source data/demo/industrial_sample.mp4 --conf 0.30

# Run on live USB webcam with display window
python cli_monitor.py --source 0 --display

# Run on RTSP IP Camera and save annotated recording
python cli_monitor.py --source "rtsp://admin:password@192.168.1.100:554/stream1" --output data/recordings/cctv_annotated.mp4
```

---

## 🧪 Running Automated Tests

Run the comprehensive pytest suite:

```powershell
python -m pytest tests/test_system.py -v
```

All 5 test suites cover:
- SQLite database logging, filtering, and stats aggregation
- Polygon geofencing and feet point-in-polygon containment
- Worker tracking, persistence, and temporal debouncing
- Deep learning inference on test frames
- Alert triggering, synthesized warning audio, and snapshot generation

---

## ⚙️ Configuration

### Hazardous Zones (`configs/zones.json`)
Zones are defined using normalized \([x, y]\) coordinates (ranging from `0.0` to `1.0`), ensuring compatibility across 720p, 1080p, and 4K cameras:

```json
{
  "id": "zone_danger_robotics",
  "name": "Zone 1: Automated Robotics Cell",
  "zone_type": "DANGER_EXCLUSION",
  "points": [
    [0.05, 0.45],
    [0.42, 0.45],
    [0.42, 0.95],
    [0.05, 0.95]
  ],
  "color_rgb": [235, 45, 45],
  "required_ppe": ["Hard_hat", "Vest"],
  "max_dwell_seconds": 0.0
}
```

### System Settings (`configs/settings.json`)
Customize detection thresholds, debounce windows, and enforced equipment:

```json
{
  "confidence_threshold": 0.30,
  "debounce_seconds": 1.0,
  "debounce_frames": 5,
  "alert_cooldown_seconds": 5.0,
  "mandatory_ppe": ["Hard_hat", "Vest"],
  "camera_id": "CAM-01-FACTORY-FLOOR"
}
```

---

## 📜 License
This project is released under the MIT License.


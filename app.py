"""
Industrial PPE Compliance & Intrusion Monitoring System - Web Dashboard.
Built with Streamlit, OpenCV, and Ultralytics YOLOv8.
"""

import os
import time
import json
import base64
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.database import IncidentDatabase
from core.zone_manager import Zone, ZoneManager
from core.detector import PPEDetector
from core.tracker import WorkerTracker
from core.alert_manager import AlertManager, IncidentAlert
from core.visualizer import Visualizer

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Industrial Safety AI | PPE & Intrusion Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial Dark Theme CSS
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1f26;
        border: 1px solid #2d3748;
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    /* Alert badge pill */
    .alert-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        margin-right: 6px;
    }
    .pill-critical { background-color: #991b1b; color: #fee2e2; border: 1px solid #ef4444; }
    .pill-high { background-color: #9a3412; color: #ffedd5; border: 1px solid #f97316; }
    .pill-compliant { background-color: #065f46; color: #d1fae5; border: 1px solid #10b981; }
    /* Section headers */
    .section-header {
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 6px;
        margin-top: 10px;
        margin-bottom: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Initialization & State Management
# ---------------------------------------------------------
@st.cache_resource
def get_database():
    return IncidentDatabase(db_path="data/database.sqlite")

@st.cache_resource
def get_zone_manager():
    return ZoneManager(config_path="configs/zones.json")

@st.cache_resource
def get_detector(conf_threshold: float, mandatory_ppe_tuple: tuple):
    return PPEDetector(
        model_path="models/ppe_yolov8.pt",
        conf_threshold=conf_threshold,
        mandatory_ppe=list(mandatory_ppe_tuple)
    )

def load_settings():
    settings_file = "configs/settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "confidence_threshold": 0.30,
        "debounce_seconds": 1.0,
        "debounce_frames": 5,
        "alert_cooldown_seconds": 5.0,
        "mandatory_ppe": ["Hard_hat", "Vest"],
        "camera_id": "CAM-01-FACTORY-FLOOR",
        "enable_audio_alarm": True,
        "display": {"show_boxes": True, "show_zones": True, "show_trails": True, "show_hud": True, "show_ppe_boxes": False}
    }

def save_settings(data):
    with open("configs/settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

settings = load_settings()
db = get_database()
zone_mgr = get_zone_manager()


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ **Safety AI Controls**")
    st.caption("Real-Time PPE Compliance & Geofence Intrusion System")
    st.divider()

    # Video Source Selector
    st.markdown("##### 📹 **Video Ingestion Source**")
    source_type = st.radio(
        "Source Type:",
        ["Synthetic Industrial Simulation", "Live Webcam", "Upload Video File", "Static Test Image", "RTSP / IP Stream URL"],
        index=0
    )

    video_source_path = "data/demo/industrial_sample.mp4"
    uploaded_image_file = None

    if source_type == "Synthetic Industrial Simulation":
        video_source_path = "data/demo/industrial_sample.mp4"
        if not os.path.exists(video_source_path):
            st.warning("Sample demo video not found. Click below to generate it.")
            if st.button("Generate Demo Assets"):
                with st.spinner("Generating demo simulation..."):
                    import subprocess
                    subprocess.run(["python", "generate_demo_assets.py"])
                st.rerun()

    elif source_type == "Live Webcam":
        cam_index = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
        video_source_path = int(cam_index)

    elif source_type == "Upload Video File":
        uploaded_video = st.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            os.makedirs("data/uploads", exist_ok=True)
            video_source_path = os.path.join("data/uploads", uploaded_video.name)
            with open(video_source_path, "wb") as f:
                f.write(uploaded_video.read())
            st.success(f"Loaded: {uploaded_video.name}")

    elif source_type == "Static Test Image":
        image_choice = st.selectbox(
            "Select Test Image",
            ["Demo: Intrusion & Violation", "Demo: Compliant Worker", "Upload Custom Image"]
        )
        if image_choice == "Demo: Intrusion & Violation":
            video_source_path = "data/demo/test_violation.jpg"
        elif image_choice == "Demo: Compliant Worker":
            video_source_path = "data/demo/test_compliant.jpg"
        else:
            uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_img is not None:
                os.makedirs("data/uploads", exist_ok=True)
                video_source_path = os.path.join("data/uploads", uploaded_img.name)
                with open(video_source_path, "wb") as f:
                    f.write(uploaded_img.read())

    elif source_type == "RTSP / IP Stream URL":
        rtsp_url = st.text_input("RTSP / HTTP Stream URL", value="rtsp://127.0.0.1:8554/live")
        video_source_path = rtsp_url

    st.divider()

    # Overlay Toggles
    st.markdown("##### 👁️ **Visual Overlays**")
    show_boxes = st.checkbox("Worker Bounding Boxes", value=settings["display"].get("show_boxes", True))
    show_zones = st.checkbox("Hazard Zones (Geofences)", value=settings["display"].get("show_zones", True))
    show_trails = st.checkbox("Worker Motion Trails", value=settings["display"].get("show_trails", True))
    show_hud = st.checkbox("Industrial HUD Banner", value=settings["display"].get("show_hud", True))
    show_ppe_boxes = st.checkbox("PPE Item Boxes (Helmets/Vests)", value=settings["display"].get("show_ppe_boxes", False))

    st.divider()

    # Detection Parameters
    st.markdown("##### ⚙️ **Inference Parameters**")
    conf_thresh = st.slider("Detection Confidence", 0.15, 0.85, float(settings.get("confidence_threshold", 0.30)), 0.05)
    
    st.markdown("##### 📋 **Mandatory PPE Rules**")
    all_ppe_options = ["Hard_hat", "Vest", "Boots", "Glove", "Glass", "Mask"]
    selected_mandatory = st.multiselect(
        "Enforced Gear:",
        all_ppe_options,
        default=settings.get("mandatory_ppe", ["Hard_hat", "Vest"])
    )

    if st.button("Save Settings as Default"):
        settings["confidence_threshold"] = conf_thresh
        settings["mandatory_ppe"] = selected_mandatory
        settings["display"]["show_boxes"] = show_boxes
        settings["display"]["show_zones"] = show_zones
        settings["display"]["show_trails"] = show_trails
        settings["display"]["show_hud"] = show_hud
        settings["display"]["show_ppe_boxes"] = show_ppe_boxes
        save_settings(settings)
        st.toast("Settings saved successfully!", icon="✅")


# ---------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown("## 🛡️ Industrial Safety AI — PPE Compliance & Intrusion System")
st.markdown(
    "Automated Computer Vision Platform for Real-Time PPE Verification, Hazard Zone Intrusion Detection, and Incident Forensics."
)

tabs = st.tabs([
    "🔴 Live Video Monitor",
    "📐 Hazard Zone Configurator",
    "📋 Incident Evidence Vault",
    "📊 Safety Analytics & KPIs",
    "⚙️ System Settings"
])


# ---------------------------------------------------------
# TAB 1: Live Video Monitor
# ---------------------------------------------------------
with tabs[0]:
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 4])
    with col_ctrl1:
        run_stream = st.toggle("⚡ **Active Stream Processing**", value=True)
    with col_ctrl2:
        camera_label = st.text_input("Camera Label", value=settings.get("camera_id", "CAM-01-FACTORY-FLOOR"))

    # Live Metrics HUD Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    metric_workers = m_col1.metric("Active Workers", "0")
    metric_compliant = m_col2.metric("Compliant", "0")
    metric_violations = m_col3.metric("PPE Violations", "0")
    metric_intrusions = m_col4.metric("Zone Breaches", "0")
    metric_compliance_rate = m_col5.metric("Compliance Rate", "100%")

    col_video, col_alerts = st.columns([7, 3])

    with col_video:
        video_placeholder = st.empty()

    with col_alerts:
        st.markdown("#### 🚨 **Live Incident Ticker**")
        alerts_placeholder = st.empty()

    # Stream Processing Loop
    if run_stream:
        # Initialize components for this run
        detector = get_detector(conf_thresh, tuple(selected_mandatory))
        detector.set_mandatory_ppe(selected_mandatory)
        
        tracker = WorkerTracker(
            debounce_seconds=float(settings.get("debounce_seconds", 1.0)),
            debounce_frames=int(settings.get("debounce_frames", 5))
        )
        alert_mgr = AlertManager(db=db, alert_cooldown_seconds=float(settings.get("alert_cooldown_seconds", 5.0)))
        visualizer = Visualizer(
            show_boxes=show_boxes,
            show_zones=show_zones,
            show_trails=show_trails,
            show_hud=show_hud,
            show_ppe_boxes=show_ppe_boxes
        )

        # Check if source is image or video
        is_static_image = isinstance(video_source_path, str) and video_source_path.lower().endswith(('.jpg', '.jpeg', '.png'))

        if is_static_image:
            if os.path.exists(video_source_path):
                frame = cv2.imread(video_source_path)
                h, w = frame.shape[:2]
                now = time.time()

                det_res = detector.detect(frame)
                tracked_workers = tracker.update(det_res.workers, current_time=now)

                active_intrusions = {}
                breached_zone_ids = set()
                for worker in tracked_workers:
                    zv = zone_mgr.evaluate_worker(worker.id, worker.bbox, worker.detected_ppe, w, h, current_time=now)
                    if zv:
                        active_intrusions[worker.id] = zv
                        for v in zv:
                            breached_zone_ids.add(v["zone_id"])

                annotated = frame.copy()
                annotated = visualizer.draw_zones(annotated, list(zone_mgr.zones.values()), breached_zone_ids)
                annotated = visualizer.draw_workers(annotated, tracked_workers, active_intrusions)
                annotated = visualizer.draw_hud(annotated, tracked_workers, len(breached_zone_ids), camera_id=camera_label)

                # Process alerts for image
                for worker in tracked_workers:
                    worker_zv = active_intrusions.get(worker.id, [])
                    alert_mgr.process_worker_violations(worker, worker_zv, annotated, camera_id=camera_label, current_time=now)

                # Update metrics
                tot = len(tracked_workers)
                comp = sum(1 for w in tracked_workers if w.is_compliant and not active_intrusions.get(w.id))
                viol = tot - comp
                breaches = len(breached_zone_ids)
                rate = int(comp / tot * 100) if tot > 0 else 100

                metric_workers.metric("Active Workers", str(tot))
                metric_compliant.metric("Compliant", str(comp))
                metric_violations.metric("PPE Violations", str(viol))
                metric_intrusions.metric("Zone Breaches", str(breaches))
                metric_compliance_rate.metric("Compliance Rate", f"{rate}%")

                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                recent_alerts = alert_mgr.get_recent_alerts()
                if recent_alerts:
                    with alerts_placeholder.container():
                        for a in recent_alerts[:5]:
                            p_cls = "pill-critical" if a.severity == "CRITICAL" else "pill-high"
                            st.markdown(f"<span class='alert-pill {p_cls}'>{a.severity}</span> **W#{a.worker_id}** - {a.violation_type}", unsafe_allow_html=True)
                            st.caption(f"{a.message} ({a.timestamp[11:19]})")
                            st.divider()
                else:
                    alerts_placeholder.info("No active violations detected.")
            else:
                st.error(f"Test image not found at: {video_source_path}")

        else:
            # Video or Camera Stream loop
            cap = cv2.VideoCapture(video_source_path)
            if not cap.isOpened():
                st.error(f"Could not open video source: {video_source_path}")
            else:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                frame_counter = 0

                # Process loop
                while run_stream:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video if pre-recorded file
                        if isinstance(video_source_path, str) and not video_source_path.startswith("rtsp"):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                break
                        else:
                            break

                    frame_counter += 1
                    now = time.time()

                    # Detection
                    det_res = detector.detect(frame)

                    # Tracking
                    tracked_workers = tracker.update(det_res.workers, current_time=now)

                    # Zone Intrusions
                    active_intrusions = {}
                    breached_zone_ids = set()
                    for worker in tracked_workers:
                        zv = zone_mgr.evaluate_worker(
                            worker.id, worker.bbox, worker.detected_ppe,
                            frame_width, frame_height, current_time=now
                        )
                        if zv:
                            active_intrusions[worker.id] = zv
                            for v in zv:
                                breached_zone_ids.add(v["zone_id"])

                    zone_mgr.prune_old_workers([w.id for w in tracked_workers])

                    # Visualization
                    annotated = frame.copy()
                    annotated = visualizer.draw_zones(annotated, list(zone_mgr.zones.values()), breached_zone_ids)
                    annotated = visualizer.draw_workers(annotated, tracked_workers, active_intrusions)
                    annotated = visualizer.draw_hud(annotated, tracked_workers, len(breached_zone_ids), camera_id=camera_label)

                    # Alerts & Incident recording
                    for worker in tracked_workers:
                        worker_zv = active_intrusions.get(worker.id, [])
                        alert_mgr.process_worker_violations(
                            worker, worker_zv, annotated,
                            camera_id=camera_label, current_time=now
                        )

                    # Update UI every 2 frames for smooth web rendering
                    if frame_counter % 2 == 0:
                        tot = len(tracked_workers)
                        comp = sum(1 for w in tracked_workers if w.is_compliant and not active_intrusions.get(w.id))
                        viol = tot - comp
                        breaches = len(breached_zone_ids)
                        rate = int(comp / tot * 100) if tot > 0 else 100

                        metric_workers.metric("Active Workers", str(tot))
                        metric_compliant.metric("Compliant", str(comp))
                        metric_violations.metric("PPE Violations", str(viol))
                        metric_intrusions.metric("Zone Breaches", str(breaches))
                        metric_compliance_rate.metric("Compliance Rate", f"{rate}%")

                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Update Live Alert Ticker
                        recent_alerts = alert_mgr.get_recent_alerts()
                        if recent_alerts:
                            with alerts_placeholder.container():
                                for a in recent_alerts[:5]:
                                    p_cls = "pill-critical" if a.severity == "CRITICAL" else "pill-high"
                                    st.markdown(f"<span class='alert-pill {p_cls}'>{a.severity}</span> **W#{a.worker_id}** - {a.violation_type}", unsafe_allow_html=True)
                                    st.caption(f"{a.message} ({a.timestamp[11:19]})")
                                    st.divider()
                        else:
                            alerts_placeholder.info("No active violations detected.")

                    time.sleep(0.015)

                cap.release()
    else:
        video_placeholder.info("Stream processing paused. Toggle 'Active Stream Processing' above to start monitoring.")


# ---------------------------------------------------------
# TAB 2: Hazard Zone Configurator
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("### 📐 **Hazard Zone & Geofence Configurator**")
    st.caption("Define, inspect, and adjust polygon exclusion zones and mandatory PPE zones on the factory floor.")

    z_col_preview, z_col_editor = st.columns([6, 4])

    with z_col_preview:
        st.markdown("#### **Active Geofences Overlay Preview**")
        # Generate preview frame with zones
        preview_bg = cv2.imread("data/demo/test_compliant.jpg")
        if preview_bg is None:
            preview_bg = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        preview_annotated = preview_bg.copy()
        vis_preview = Visualizer(show_zones=True, show_hud=False)
        preview_annotated = vis_preview.draw_zones(preview_annotated, list(zone_mgr.zones.values()))
        rgb_preview = cv2.cvtColor(preview_annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_preview, channels="RGB", use_container_width=True)

    with z_col_editor:
        st.markdown("#### **Zone Management & Editor**")
        zone_ids = list(zone_mgr.zones.keys())

        action = st.radio("Action:", ["Inspect / Edit Existing Zone", "Add New Zone", "Delete Zone"], horizontal=True)

        if action == "Inspect / Edit Existing Zone":
            selected_zone_id = st.selectbox("Select Zone to Edit", zone_ids, format_func=lambda x: zone_mgr.zones[x].name)
            zone = zone_mgr.zones[selected_zone_id]

            e_name = st.text_input("Zone Name", value=zone.name)
            e_type = st.selectbox(
                "Zone Type",
                ["DANGER_EXCLUSION", "PPE_MANDATORY", "SAFE_TRANSIT"],
                index=["DANGER_EXCLUSION", "PPE_MANDATORY", "SAFE_TRANSIT"].index(zone.zone_type)
            )
            e_desc = st.text_area("Description / Safety Directive", value=zone.description)
            e_dwell = st.number_input("Max Allowed Dwell (seconds)", value=float(zone.max_dwell_seconds), min_value=0.0, step=5.0)

            st.markdown("**Polygon Vertices (Normalized [x, y] in range 0.0 - 1.0):**")
            points_json = st.text_area(
                "Vertices JSON",
                value=json.dumps(zone.points, indent=2),
                height=140
            )

            if st.button("💾 Save Zone Changes", type="primary"):
                try:
                    parsed_pts = json.loads(points_json)
                    zone.name = e_name
                    zone.zone_type = e_type
                    zone.description = e_desc
                    zone.max_dwell_seconds = e_dwell
                    zone.points = parsed_pts
                    zone_mgr.save_zones()
                    st.success(f"Zone '{e_name}' updated successfully!")
                    st.rerun()
                except Exception as err:
                    st.error(f"Invalid points JSON: {err}")

        elif action == "Add New Zone":
            new_id = st.text_input("Zone ID (Unique identifier)", value=f"zone_{int(time.time())}")
            new_name = st.text_input("Zone Name", value="Zone X: Chemical Hazard Area")
            new_type = st.selectbox("Zone Type", ["DANGER_EXCLUSION", "PPE_MANDATORY", "SAFE_TRANSIT"])
            new_desc = st.text_input("Description", value="Chemical storage hazard area. Strict PPE required.")
            default_pts = [[0.2, 0.5], [0.5, 0.5], [0.5, 0.8], [0.2, 0.8]]
            new_pts_str = st.text_area("Polygon Vertices JSON", value=json.dumps(default_pts, indent=2), height=130)

            if st.button("➕ Create Zone", type="primary"):
                try:
                    parsed = json.loads(new_pts_str)
                    new_zone = Zone(
                        id=new_id,
                        name=new_name,
                        zone_type=new_type,
                        points=parsed,
                        color_rgb=(240, 50, 50) if new_type == "DANGER_EXCLUSION" else (50, 150, 240),
                        description=new_desc
                    )
                    zone_mgr.add_zone(new_zone)
                    st.success(f"Zone '{new_name}' added successfully!")
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed to add zone: {err}")

        elif action == "Delete Zone":
            del_zone_id = st.selectbox("Select Zone to Delete", zone_ids, format_func=lambda x: zone_mgr.zones[x].name)
            if st.button("🗑️ Confirm Delete Zone", type="secondary"):
                zone_mgr.remove_zone(del_zone_id)
                st.warning(f"Zone removed.")
                st.rerun()


# ---------------------------------------------------------
# TAB 3: Incident Evidence Vault
# ---------------------------------------------------------
with tabs[2]:
    st.markdown("### 📋 **Forensic Incident Vault & Safety Audit Log**")
    st.caption("Searchable audit log of safety non-compliance, evidence snapshots, and review status.")

    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        f_violation = st.selectbox("Violation Type", ["ALL", "ZONE_INTRUSION", "PPE_NON_COMPLIANCE", "ZONE_PPE_VIOLATION"])
    with f_col2:
        f_status = st.selectbox("Review Status", ["ALL", "NEW", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"])
    with f_col3:
        f_limit = st.slider("Max Records", 10, 200, 50)
    with f_col4:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Records"):
            st.rerun()

    incidents = db.get_incidents(
        limit=f_limit,
        violation_type=f_violation,
        status=f_status
    )

    if not incidents:
        st.info("No safety incident records match the selected filters.")
    else:
        df = pd.DataFrame(incidents)
        display_df = df[["id", "timestamp", "camera_id", "worker_id", "violation_type", "zone_name", "severity", "status"]].copy()
        
        t_col1, t_col2 = st.columns([6, 4])

        with t_col1:
            st.dataframe(display_df, use_container_width=True, height=420)
            
            # Export CSV
            csv_path = db.export_to_csv("data/incident_report.csv")
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📥 Export Full Incident Audit Log (CSV)",
                    data=f.read(),
                    file_name=f"Safety_Incidents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

        with t_col2:
            st.markdown("#### **Evidence Inspection & Review**")
            selected_inc_id = st.selectbox("Inspect Incident ID #:", df["id"].tolist())
            selected_record = next(item for item in incidents if item["id"] == selected_inc_id)

            st.markdown(f"**Timestamp:** `{selected_record['timestamp']}`")
            st.markdown(f"**Worker ID:** `#{selected_record['worker_id']}` | **Camera:** `{selected_record['camera_id']}`")
            st.markdown(f"**Violation:** `{selected_record['violation_type']}` | **Severity:** `{selected_record['severity']}`")
            if selected_record.get("zone_name"):
                st.markdown(f"**Hazard Zone:** `{selected_record['zone_name']}`")

            # Snapshot Image
            snap_path = selected_record.get("snapshot_path")
            if snap_path and os.path.exists(snap_path):
                st.image(snap_path, caption=f"Snapshot Evidence #{selected_record['id']}", use_container_width=True)
            else:
                st.warning("No snapshot image file found on disk.")

            # Status Management Workflow
            st.markdown("##### **Incident Resolution Workflow**")
            new_stat = st.selectbox(
                "Update Status:",
                ["NEW", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"],
                index=["NEW", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"].index(selected_record.get("status", "NEW"))
            )
            notes = st.text_input("Reviewer Notes", value=selected_record.get("notes", ""))

            if st.button("Update Incident Status"):
                db.update_incident_status(selected_inc_id, new_stat, notes)
                st.success(f"Incident #{selected_inc_id} marked as {new_stat}!")
                st.rerun()


# ---------------------------------------------------------
# TAB 4: Safety Analytics & KPIs
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("### 📊 **Safety Analytics & Compliance Intelligence**")
    st.caption("Aggregated risk metrics, hazard zone hotspots, and shift safety trends.")

    stats = db.get_statistics(hours=72)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Incidents (72h)", str(stats["total_incidents"]))
    kpi2.metric("Pending Review", str(stats["pending_incidents"]))
    kpi3.metric("Critical Intrusions", str(stats["critical_incidents"]))
    
    # Calculate synthetic compliance score
    safety_score = max(50, 100 - (stats["total_incidents"] * 3))
    kpi4.metric("Site Safety Score", f"{safety_score}/100", delta="-5%" if stats["total_incidents"] > 5 else "+2%")

    st.divider()

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### **Violations by Category**")
        vb = stats.get("violation_breakdown", {})
        if vb:
            df_v = pd.DataFrame(list(vb.items()), columns=["Violation Type", "Count"])
            fig_pie = px.pie(
                df_v,
                names="Violation Type",
                values="Count",
                color_discrete_sequence=px.colors.sequential.Sunsetdark,
                hole=0.45
            )
            fig_pie.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3f4f6")
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No violation category data available.")

    with c_chart2:
        st.markdown("#### **Hazard Zone Intrusion Hotspots**")
        zb = stats.get("zone_breakdown", {})
        if zb:
            df_z = pd.DataFrame(list(zb.items()), columns=["Zone Name", "Incidents"])
            fig_bar = px.bar(
                df_z,
                x="Incidents",
                y="Zone Name",
                orientation="h",
                color="Incidents",
                color_continuous_scale="Reds"
            )
            fig_bar.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3f4f6")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No zone hotspot data recorded.")

    # Timeline Chart
    st.markdown("#### **Incident Hourly Frequency Trend**")
    ht = stats.get("hourly_trend", [])
    if ht:
        df_ht = pd.DataFrame(ht)
        fig_trend = px.area(
            df_ht,
            x="hour",
            y="count",
            labels={"hour": "Time Bucket", "count": "Incidents Logged"},
            color_discrete_sequence=["#ef4444"]
        )
        fig_trend.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6")
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Insufficient timeline trend data.")


# ---------------------------------------------------------
# TAB 5: System Settings
# ---------------------------------------------------------
with tabs[4]:
    st.markdown("### ⚙️ **System Configuration & Maintenance**")
    st.caption("Fine-tune computer vision parameters, alert thresholds, and database maintenance.")

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.markdown("#### **Temporal Debounce & Alarm Thresholds**")
        db_frames = st.slider("Debounce Consecutive Frames", 1, 20, int(settings.get("debounce_frames", 5)))
        db_seconds = st.slider("Debounce Time Window (sec)", 0.2, 5.0, float(settings.get("debounce_seconds", 1.0)), 0.2)
        cooldown = st.slider("Alert Incident Cooldown (sec)", 1.0, 30.0, float(settings.get("alert_cooldown_seconds", 5.0)), 1.0)
        audio_alert = st.checkbox("Enable Audio Warning Siren / Chime", value=settings.get("enable_audio_alarm", True))

    with cfg_col2:
        st.markdown("#### **Model Weights & Hardware**")
        st.text_input("Primary PPE Model Weights", value="models/ppe_yolov8.pt", disabled=True)
        st.text_input("Fallback COCO Model", value="yolov8n.pt", disabled=True)
        st.info("Model: Fine-Tuned YOLOv8 SafeSight Architecture (19 classes: Worker, Hard_hat, Vest, Boots, Gloves, Masks, Glasses, and negative violation cues).")

        st.markdown("#### **Database Maintenance**")
        if st.button("⚠️ Clear All Incident History"):
            db.clear_all_incidents()
            st.warning("All incident records have been purged from the SQLite database.")
            st.rerun()


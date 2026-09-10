import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from .tracker import TrackedWorker
from .zone_manager import Zone

class Visualizer:
    """
    Renders professional industrial overlays, color-coded bounding boxes,
    polygonal hazard zones, worker tracking trails, and Heads-Up Display (HUD).
    """

    # Industrial UI Color Palette (BGR format)
    COLOR_COMPLIANT = (60, 210, 80)      # Neon Emerald Green
    COLOR_VIOLATION = (40, 40, 235)      # Bright Red
    COLOR_INTRUSION = (0, 0, 255)        # Pure Red
    COLOR_WARNING = (30, 160, 255)       # Industrial Amber/Orange
    COLOR_HUD_BG = (20, 24, 30)          # Dark Slate
    COLOR_HUD_TEXT = (240, 245, 250)     # Off-white

    def __init__(
        self,
        show_boxes: bool = True,
        show_zones: bool = True,
        show_trails: bool = True,
        show_hud: bool = True,
        show_ppe_boxes: bool = False
    ):
        self.show_boxes = show_boxes
        self.show_zones = show_zones
        self.show_trails = show_trails
        self.show_hud = show_hud
        self.show_ppe_boxes = show_ppe_boxes

        # FPS calculation
        self.prev_time = time.time()
        self.fps_rolling = 30.0

    def draw_zones(
        self,
        frame: np.ndarray,
        zones: List[Zone],
        breached_zone_ids: Optional[set] = None
    ) -> np.ndarray:
        """
        Draws semi-transparent polygonal hazard zones with pulsing borders on breach.
        """
        if not self.show_zones or not zones:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()
        breached_zone_ids = breached_zone_ids or set()

        for zone in zones:
            pts = zone.get_pixel_points(w, h)
            if len(pts) < 3:
                continue

            # Base color in BGR
            r, g, b = zone.color_rgb
            bgr_color = (b, g, r)

            is_breached = zone.id in breached_zone_ids
            alpha = 0.35 if is_breached else 0.20

            # Fill zone polygon
            cv2.fillPoly(overlay, [pts], bgr_color)

            # Draw polygon perimeter outline
            border_thickness = 3 if is_breached else 2
            border_color = (0, 0, 255) if is_breached else bgr_color
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=border_thickness)

            # Label banner
            centroid = np.mean(pts, axis=0).astype(int)
            label = f"{zone.name} [{'BREACH!' if is_breached else zone.zone_type}]"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lx, ly = max(10, centroid[0] - tw // 2), max(20, centroid[1])
            cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (15, 15, 20), -1)
            cv2.putText(
                frame,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255) if is_breached else (240, 240, 240),
                1,
                cv2.LINE_AA
            )

        # Blend semi-transparent fill
        return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    def draw_workers(
        self,
        frame: np.ndarray,
        workers: List[TrackedWorker],
        active_intrusions: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ) -> np.ndarray:
        """
        Draws worker bounding boxes, status pills, trajectory trails, and feet contact points.
        """
        active_intrusions = active_intrusions or {}

        for worker in workers:
            x1, y1, x2, y2 = [int(v) for v in worker.bbox]
            intrusions = active_intrusions.get(worker.id, [])

            is_intruder = len(intrusions) > 0
            is_compliant = worker.is_compliant and not is_intruder

            # Color selection
            if is_intruder:
                box_color = self.COLOR_INTRUSION
            elif not worker.is_compliant:
                box_color = self.COLOR_VIOLATION
            else:
                box_color = self.COLOR_COMPLIANT

            # 1. Motion trail
            if self.show_trails and len(worker.trail) > 1:
                pts = list(worker.trail)
                for i in range(1, len(pts)):
                    alpha = float(i) / len(pts)
                    thickness = int(1 + alpha * 2)
                    pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                    pt2 = (int(pts[i][0]), int(pts[i][1]))
                    cv2.line(frame, pt1, pt2, box_color, thickness)

            # 2. Foot contact point (ground anchor)
            fx, fy = int(worker.foot_point[0]), int(worker.foot_point[1])
            cv2.circle(frame, (fx, fy), 5, box_color, -1)
            cv2.circle(frame, (fx, fy), 8, (255, 255, 255), 1)

            if not self.show_boxes:
                continue

            # 3. Worker Bounding Box with styled corners
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            corner_len = min(15, (x2 - x1) // 4)
            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 4)
            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 4)
            # Bottom corners
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 4)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 4)

            # 4. Status Badge Header
            if is_intruder:
                header_text = f"W#{worker.id} | INTRUSION BREACH!"
            elif not worker.is_compliant:
                missing = ", ".join(worker.missing_mandatory) or "MISSING PPE"
                header_text = f"W#{worker.id} | NO {missing}"
            else:
                header_text = f"W#{worker.id} | COMPLIANT"

            (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), box_color, -1)
            cv2.putText(
                frame,
                header_text,
                (x1 + 5, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # 5. PPE Detail Tags (under box)
            if not is_compliant and worker.detected_ppe:
                wearing_str = "Wearing: " + ", ".join(worker.detected_ppe)
                (dtw, dth), _ = cv2.getTextSize(wearing_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(frame, (x1, y2), (x1 + dtw + 6, y2 + dth + 6), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    wearing_str,
                    (x1 + 3, y2 + dth + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (180, 220, 180),
                    1,
                    cv2.LINE_AA
                )

            # 6. Optional PPE Bounding Boxes
            if self.show_ppe_boxes and worker.ppe_boxes:
                for pbox in worker.ppe_boxes:
                    px1, py1, px2, py2 = [int(v) for v in pbox["bbox"]]
                    pname = pbox["class_name"]
                    pcolor = (0, 255, 255) if "hat" in pname.lower() else (255, 180, 0)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), pcolor, 1)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        workers: List[TrackedWorker],
        breached_count: int = 0,
        camera_id: str = "CAM-01"
    ) -> np.ndarray:
        """
        Renders an industrial Heads-Up Display (HUD) banner at the top of the video feed.
        """
        if not self.show_hud:
            return frame

        h, w = frame.shape[:2]
        hud_height = 42

        # Draw semi-transparent HUD background
        hud_overlay = frame[:hud_height, :].copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_height), self.COLOR_HUD_BG, -1)
        frame[:hud_height, :] = cv2.addWeighted(hud_overlay, 0.85, frame[:hud_height, :], 0.15, 0)

        # FPS calculation
        now = time.time()
        dt = max(now - self.prev_time, 1e-4)
        current_fps = 1.0 / dt
        self.fps_rolling = (self.fps_rolling * 0.9) + (current_fps * 0.1)
        self.prev_time = now

        total_workers = len(workers)
        compliant_workers = sum(1 for w in workers if w.is_compliant and not w.active_zone_violations)
        compliance_pct = int((compliant_workers / total_workers * 100)) if total_workers > 0 else 100

        # Elements formatting
        left_margin = 15
        y_pos = 26

        # 1. Camera & Status Indicator
        status_color = (0, 0, 255) if breached_count > 0 else (60, 220, 80)
        cv2.circle(frame, (left_margin + 6, y_pos - 6), 6, status_color, -1)
        cam_text = f"SAFETY-AI | {camera_id}"
        cv2.putText(frame, cam_text, (left_margin + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Metric Pills in center
        metrics_x = max(left_margin + 220, int(w * 0.35))
        m_text = f"WORKERS: {total_workers}  |  COMPLIANT: {compliant_workers}/{total_workers} ({compliance_pct}%)  |  BREACHES: {breached_count}"
        cv2.putText(frame, m_text, (metrics_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 230, 240), 1)

        # 3. FPS counter on right
        fps_text = f"{self.fps_rolling:.1f} FPS"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(frame, fps_text, (w - tw - 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 200), 1)

        # 4. Critical Warning Strip if Breaches Active
        if breached_count > 0:
            warning_height = 28
            warn_overlay = frame[h - warning_height:h, :].copy()
            cv2.rectangle(warn_overlay, (0, 0), (w, warning_height), (0, 0, 220), -1)
            frame[h - warning_height:h, :] = cv2.addWeighted(warn_overlay, 0.90, frame[h - warning_height:h, :], 0.10, 0)
            
            w_msg = "CRITICAL SAFETY ALERT: RESTRICTED ZONE INTRUSION OR PPE VIOLATION DETECTED!"
            (wtw, _), _ = cv2.getTextSize(w_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            cv2.putText(
                frame,
                w_msg,
                (max(10, (w - wtw) // 2), h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2
            )

        return frame


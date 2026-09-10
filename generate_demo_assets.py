"""
Generates synthetic industrial test videos and snapshot test images
for immediate end-to-end evaluation of the PPE Compliance & Intrusion Monitoring System.
"""

import os
import math
import numpy as np
import cv2

def draw_industrial_background(w: int = 1280, h: int = 720) -> np.ndarray:
    """Draws a clean industrial factory floor perspective."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Factory wall (upper 40%) - Slate grey with industrial gradient
    for y in range(int(h * 0.42)):
        c = int(45 + (y / (h * 0.42)) * 30)
        img[y, :] = (c, c + 5, c + 10)
        
    # Factory floor (lower 58%) - Polished concrete with perspective lines
    for y in range(int(h * 0.42), h):
        c = int(65 + ((y - h * 0.42) / (h * 0.58)) * 40)
        img[y, :] = (c, c + 2, c - 5)
        
    # Perspective grid lines on floor
    vp_x, vp_y = w // 2, int(h * 0.40)
    for x in range(0, w, 120):
        cv2.line(img, (vp_x, vp_y), (x, h), (90, 95, 90), 1)
        
    for fy in range(int(h * 0.48), h, 45):
        cv2.line(img, (0, fy), (w, fy), (85, 90, 85), 1)

    # Overhead warning beams / trusses
    cv2.line(img, (0, int(h * 0.42)), (w, int(h * 0.42)), (120, 125, 130), 2)
    
    # Background machinery silhouettes / control panels
    cv2.rectangle(img, (60, int(h * 0.20)), (280, int(h * 0.42)), (30, 35, 40), -1)
    cv2.rectangle(img, (60, int(h * 0.20)), (280, int(h * 0.42)), (80, 85, 90), 2)
    cv2.putText(img, "ROBOTIC ARM CELL A", (80, int(h * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    cv2.rectangle(img, (w - 320, int(h * 0.22)), (w - 80, int(h * 0.42)), (35, 40, 45), -1)
    cv2.rectangle(img, (w - 320, int(h * 0.22)), (w - 80, int(h * 0.42)), (80, 85, 90), 2)
    cv2.putText(img, "CNC MACHINING STATION", (w - 305, int(h * 0.27)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)

    return img


def draw_human_figure(
    frame: np.ndarray,
    cx: int,
    feet_y: int,
    height: int,
    has_helmet: bool = True,
    has_vest: bool = True,
    worker_label: str = ""
):
    """
    Renders a realistic 2D human worker silhouette with safety gear (Helmet, High-Vis Vest, Boots).
    """
    scale = height / 180.0
    head_radius = int(14 * scale)
    head_cy = int(feet_y - height + head_radius + 6 * scale)
    
    torso_top = head_cy + head_radius + 4
    torso_bottom = int(feet_y - 65 * scale)
    torso_width = int(44 * scale)
    
    leg_l_x = cx - int(12 * scale)
    leg_r_x = cx + int(12 * scale)
    feet_bottom = feet_y
    
    # Legs (Navy industrial trousers)
    trousers_color = (60, 45, 35)  # Dark blue / slate
    cv2.line(frame, (cx - 8, torso_bottom), (leg_l_x, feet_bottom - 10), trousers_color, int(12 * scale))
    cv2.line(frame, (cx + 8, torso_bottom), (leg_r_x, feet_bottom - 10), trousers_color, int(12 * scale))

    # Safety boots (Steel toe work boots)
    boot_color = (20, 20, 25)
    cv2.rectangle(frame, (leg_l_x - int(8 * scale), feet_bottom - 10), (leg_l_x + int(8 * scale), feet_bottom), boot_color, -1)
    cv2.rectangle(frame, (leg_r_x - int(8 * scale), feet_bottom - 10), (leg_r_x + int(8 * scale), feet_bottom), boot_color, -1)

    # Torso
    if has_vest:
        # High-Vis Fluorescent Safety Vest (Safety Orange / Fluorescent Yellow with silver retro-reflective stripes)
        vest_color = (0, 140, 255)  # Vibrant orange in BGR
        cv2.rectangle(frame, (cx - torso_width // 2, torso_top), (cx + torso_width // 2, torso_bottom), vest_color, -1)
        # Silver reflective stripes
        cv2.rectangle(frame, (cx - torso_width // 2, torso_top + int(12 * scale)), (cx + torso_width // 2, torso_top + int(18 * scale)), (220, 220, 220), -1)
        cv2.rectangle(frame, (cx - torso_width // 2, torso_top + int(32 * scale)), (cx + torso_width // 2, torso_top + int(38 * scale)), (220, 220, 220), -1)
        cv2.line(frame, (cx - int(10 * scale), torso_top), (cx - int(10 * scale), torso_bottom), (220, 220, 220), int(3 * scale))
        cv2.line(frame, (cx + int(10 * scale), torso_top), (cx + int(10 * scale), torso_bottom), (220, 220, 220), int(3 * scale))
    else:
        # Regular dark work shirt (No safety vest)
        shirt_color = (90, 80, 75)
        cv2.rectangle(frame, (cx - torso_width // 2, torso_top), (cx + torso_width // 2, torso_bottom), shirt_color, -1)

    # Arms
    cv2.line(frame, (cx - torso_width // 2, torso_top + 5), (cx - torso_width // 2 - int(12 * scale), torso_bottom - 10), (90, 80, 75), int(8 * scale))
    cv2.line(frame, (cx + torso_width // 2, torso_top + 5), (cx + torso_width // 2 + int(12 * scale), torso_bottom - 10), (90, 80, 75), int(8 * scale))

    # Head (Skin tone)
    skin_color = (165, 195, 225)  # Light skin tone in BGR
    cv2.circle(frame, (cx, head_cy), head_radius, skin_color, -1)

    # Headwear / Hard Hat
    if has_helmet:
        # Industrial Safety Hard Hat (Vibrant OSHA Yellow with brim)
        helmet_color = (0, 220, 245)  # Bright yellow in BGR
        # Helmet dome
        cv2.ellipse(frame, (cx, head_cy - int(3 * scale)), (int(head_radius * 1.25), int(head_radius * 1.05)), 0, 180, 360, helmet_color, -1)
        # Helmet brim
        cv2.line(frame, (cx - int(head_radius * 1.4), head_cy - int(3 * scale)), (cx + int(head_radius * 1.4), head_cy - int(3 * scale)), helmet_color, int(4 * scale))
        # Top ridge
        cv2.line(frame, (cx, head_cy - int(head_radius * 1.5)), (cx, head_cy - int(3 * scale)), (0, 190, 210), int(3 * scale))
    else:
        # Natural hair (No helmet)
        hair_color = (30, 40, 50)
        cv2.ellipse(frame, (cx, head_cy - int(4 * scale)), (int(head_radius * 1.05), int(head_radius * 0.8)), 0, 180, 360, hair_color, -1)


def generate_synthetic_demo_video(
    output_path: str = "data/demo/industrial_sample.mp4",
    fps: int = 25,
    num_seconds: int = 10,
    width: int = 1280,
    height: int = 720
):
    """
    Generates a realistic 10-second industrial video simulation showcasing:
    - Worker 1: Fully compliant (Hard hat + Safety Vest) in safe walkway.
    - Worker 2: Violator (NO Hard hat, NO Vest) intruding directly into Robotic Danger Zone!
    - Worker 3: Partially compliant (Wearing Hard hat, but NO Safety Vest) in Assembly zone.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = fps * num_seconds
    base_bg = draw_industrial_background(width, height)

    print(f"[DemoGenerator] Generating {total_frames} frames into {output_path}...")

    for f in range(total_frames):
        t = f / float(fps)
        frame = base_bg.copy()

        # Worker 1: Compliant worker walking along central walkway
        # Moves smoothly from x=580 to x=640
        w1_x = int(580 + 35 * math.sin(t * 0.7))
        w1_y = int(height * 0.76 + 10 * math.sin(t * 1.4))
        draw_human_figure(frame, cx=w1_x, feet_y=w1_y, height=190, has_helmet=True, has_vest=True, worker_label="W#1")

        # Worker 2: Intruder without safety gear entering Danger Exclusion Zone (Left area, x=150 to x=320)
        # Moves from x=480 into Danger Zone (x=240)
        w2_progress = min(1.0, t / 7.0)
        w2_x = int(480 - (260 * w2_progress))
        w2_y = int(height * 0.82 + 8 * math.cos(t * 1.5))
        # Worker 2 has NO helmet and NO vest!
        draw_human_figure(frame, cx=w2_x, feet_y=w2_y, height=210, has_helmet=False, has_vest=False, worker_label="W#2")

        # Worker 3: In Staging / Assembly Area (Right area, x=950)
        # Standing in Assembly Zone wearing helmet but NO vest!
        w3_x = int(950 + 15 * math.sin(t * 0.5))
        w3_y = int(height * 0.72)
        draw_human_figure(frame, cx=w3_x, feet_y=w3_y, height=185, has_helmet=True, has_vest=False, worker_label="W#3")

        # Subtle camera noise/grain for realism
        noise = np.random.normal(0, 2.5, frame.shape).astype(np.int16)
        noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        out.write(noisy_frame)

    out.release()
    print(f"[DemoGenerator] Successfully generated video: {output_path} ({os.path.getsize(output_path)} bytes)")

    # Also save test images
    sample_frame_compliant = base_bg.copy()
    draw_human_figure(sample_frame_compliant, cx=640, feet_y=int(height * 0.75), height=200, has_helmet=True, has_vest=True)
    cv2.imwrite("data/demo/test_compliant.jpg", sample_frame_compliant)

    sample_frame_violation = base_bg.copy()
    draw_human_figure(sample_frame_violation, cx=220, feet_y=int(height * 0.80), height=210, has_helmet=False, has_vest=False)
    draw_human_figure(sample_frame_violation, cx=900, feet_y=int(height * 0.75), height=195, has_helmet=True, has_vest=False)
    cv2.imwrite("data/demo/test_violation.jpg", sample_frame_violation)
    print("[DemoGenerator] Generated sample test images: data/demo/test_compliant.jpg, data/demo/test_violation.jpg")

if __name__ == "__main__":
    generate_synthetic_demo_video()


"""
Real-Time HUD Layout for Driver Safety AI.
Renders header, telemetry panels, status indicator, DVI badge, and FPS metrics on OpenCV frames.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

def draw_hud(
    frame: np.ndarray,
    status_text: str,
    feature_vec: np.ndarray,
    dvi_score: float,
    dvi_level: str,
    fps: float,
    device_name: str = "CPU",
    warmup_counter: Optional[Tuple[int, int]] = None,
    decision_info: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Render real-time HUD layout overlay on OpenCV frame.
    Displays neural probabilities, state machine state (ALERT, SUSPECT, DROWSY, RECOVERY),
    confirmation timer, and telemetry metrics without overlapping panels.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top Header Banner
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 15), -1)
    cv2.line(overlay, (0, 36), (w, 36), (50, 50, 50), 1)
    cv2.putText(overlay, "DRIVER SAFETY AI", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Device & FPS Pill on Top Right
    fps_info = f"FPS: {fps:.1f} | Dev: {device_name}"
    cv2.putText(overlay, fps_info, (w - 200, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # 2. Decision Engine Info / State
    state = decision_info.get("state", status_text) if decision_info else status_text
    p_drowsy = decision_info.get("p_drowsy", 0.0) if decision_info else 0.0
    timer_cur = decision_info.get("confirmation_timer", 0.0) if decision_info else 0.0
    timer_target = decision_info.get("confirmation_target", 1.0) if decision_info else 1.0
    alarm_active = decision_info.get("alarm_triggered", False) if decision_info else False
    is_normal_blink = decision_info.get("is_normal_blink", False) if decision_info else False

    # Primary Status Badge
    custom_subtext = decision_info.get("status_subtext", None) if decision_info else None
    intervention_lvl = decision_info.get("intervention_level", 0) if decision_info else 0
    resp_status = decision_info.get("response_status", "NONE") if decision_info else "NONE"
    timeline_events = decision_info.get("event_timeline", []) if decision_info else []

    if state == "ALERT":
        status_color = (0, 220, 0)
        display_status = f"STATE: {state}"
        prob_subtext = custom_subtext or f"DROWSY Prob: {p_drowsy*100:.1f}%"
    elif state == "SUSPECTED_DROWSY":
        status_color = (0, 255, 255) # Yellow/Cyan
        display_status = "STATE: SUSPECTED DROWSY"
        prob_subtext = custom_subtext or f"Confirming... ({p_drowsy*100:.1f}%)"
    elif state == "CONFIRMED_DROWSY":
        status_color = (0, 140, 255) # Orange
        display_status = "STATE: CONFIRMED DROWSY"
        prob_subtext = custom_subtext or f"INTERVENTION L2 ACTIVE ({p_drowsy*100:.1f}%)"
    elif state == "PERSISTENT_DROWSY":
        status_color = (0, 0, 255)   # Red
        display_status = "STATE: PERSISTENT DROWSY"
        prob_subtext = custom_subtext or "CRITICAL: TAKE A SAFE BREAK"
    elif state == "RECOVERING":
        status_color = (255, 165, 0) # Orange/Blue
        display_status = f"STATE: {state}"
        prob_subtext = custom_subtext or "STABILITY CHECK - EYES OPEN"
    else:
        status_color = (0, 165, 255)
        display_status = f"STATE: {state}"
        prob_subtext = custom_subtext or f"DROWSY Prob: {p_drowsy*100:.1f}%"

    # Status Box (Top Left: x=12, y=42 to 86)
    cv2.rectangle(overlay, (12, 42), (265, 86), (25, 25, 25), -1)
    cv2.rectangle(overlay, (12, 42), (265, 86), status_color, 2)

    if warmup_counter is not None and warmup_counter[0] < warmup_counter[1]:
        display_status = f"WARMUP: {warmup_counter[0]}/{warmup_counter[1]}"
        cv2.putText(overlay, display_status, (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(overlay, display_status, (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)
        cv2.putText(overlay, prob_subtext, (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    # DVI Risk Badge (Top Right: x=w-200 to w-12, y=42 to 86)
    dvi_color = (0, 220, 0) if dvi_score < 25 else (0, 255, 255) if dvi_score < 50 else (0, 140, 255) if dvi_score < 75 else (0, 0, 255)
    cv2.rectangle(overlay, (w - 200, 42), (w - 12, 86), (25, 25, 25), -1)
    cv2.rectangle(overlay, (w - 200, 42), (w - 12, 86), dvi_color, 2)
    cv2.putText(overlay, f"DVI: {dvi_score:.1f}% ({dvi_level})", (w - 190, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, dvi_color, 2, cv2.LINE_AA)
    
    alarm_str = "ALARM: ACTIVE" if alarm_active else "ALARM: OFF"
    alarm_col = (0, 0, 255) if alarm_active else (0, 220, 0)
    cv2.putText(overlay, alarm_str, (w - 190, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, alarm_col, 1, cv2.LINE_AA)

    # 3. Telemetry Metrics Panel (Left Sidebar: x=12, y=94 to 285)
    cv2.rectangle(overlay, (12, 94), (265, 285), (20, 20, 20), -1)
    cv2.rectangle(overlay, (12, 94), (265, 285), (60, 60, 60), 1)

    interv_str = f"INTERVENTION: L{intervention_lvl}" if intervention_lvl > 0 else "INTERVENTION: NONE"
    metrics_text = [
        f"MEAN EAR:    {feature_vec[2]:.3f}",
        f"MAR:         {feature_vec[3]:.3f}",
        f"YAW:         {feature_vec[4]:.1f} deg",
        f"PITCH:       {feature_vec[5]:.1f} deg",
        f"PERCLOS:     {feature_vec[7]*100:.1f}%",
        f"BLINK/MIN:   {feature_vec[8]:.1f}",
        f"EYE CLOSE:   {feature_vec[9]:.2f}s",
        interv_str,
        f"RESPONSE:    {resp_status}",
    ]

    y_pos = 112
    for line in metrics_text:
        cv2.putText(overlay, line, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        y_pos += 18

    # 4. Mini Event Timeline Panel (Bottom Left: x=12, y=292 to 365)
    if timeline_events:
        cv2.rectangle(overlay, (12, 292), (320, 365), (15, 15, 15), -1)
        cv2.rectangle(overlay, (12, 292), (320, 365), (70, 70, 70), 1)
        cv2.putText(overlay, "EVENT TIMELINE", (20, 306),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        recent_3 = timeline_events[-3:]
        ev_y = 322
        for ev in recent_3:
            cv2.putText(overlay, str(ev), (20, ev_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1, cv2.LINE_AA)
            ev_y += 14

    # Blend with original frame for subtle transparency
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    return frame

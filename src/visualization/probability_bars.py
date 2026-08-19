"""
Softmax Class Probability Visualization Panel.
Renders clean real-time probability progress bars for the 4 driver states.
"""

import cv2
import numpy as np
from typing import List

def draw_probability_bars(
    frame: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    top_left: tuple = (12, 268),
    bar_width: int = 110,
    bar_height: int = 14
) -> np.ndarray:
    """
    Draw probability bars on the OpenCV frame cleanly without overlapping HUD components.
    probabilities: shape (4,) float array summing to 1.0 from Softmax.
    """
    x_start, y_start = top_left
    overlay = frame.copy()

    # Color palette for classes (BGR)
    # ALERT: Green (0,200,0), DROWSY: Red (0,0,220), YAWNING: Orange (0,140,255), DISTRACTED: Purple (180,0,180)
    colors = [
        (0, 200, 0),    # ALERT
        (0, 0, 220),    # DROWSY
        (0, 140, 255),  # YAWNING
        (180, 0, 180)   # DISTRACTED
    ]

    panel_w = 238
    panel_h = 135
    
    # Outer Panel Background (x: 12 to 250, y: 268 to 403)
    cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (60, 60, 60), 1)

    cv2.putText(overlay, "CLASS PROBABILITIES", (x_start + 10, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    y = y_start + 28
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        prob_val = float(np.clip(prob, 0.0, 1.0))
        filled_w = int(prob_val * bar_width)

        # Class Label
        cv2.putText(overlay, f"{name:<10}", (x_start + 10, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)

        # Background bar box
        cv2.rectangle(overlay, (x_start + 85, y), (x_start + 85 + bar_width, y + bar_height),
                      (40, 40, 40), -1)

        # Filled probability box
        if filled_w > 0:
            cv2.rectangle(overlay, (x_start + 85, y), (x_start + 85 + filled_w, y + bar_height),
                          colors[i % len(colors)], -1)

        # Border
        cv2.rectangle(overlay, (x_start + 85, y), (x_start + 85 + bar_width, y + bar_height),
                      (100, 100, 100), 1)

        # Percentage
        cv2.putText(overlay, f"{prob_val * 100:.1f}%", (x_start + 90 + bar_width, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        y += bar_height + 10

    # Blend with original frame for subtle transparency
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    return frame

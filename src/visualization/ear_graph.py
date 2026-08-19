"""
EAR Sparkline Oscilloscope Visualization Component.
Renders rolling 100-frame EAR graph with optional diagnostic threshold reference line (Section 35).
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional

class EARGraphOscilloscope:
    """
    Diagnostic rolling EAR oscilloscope sparkline overlay.
    """
    def __init__(
        self,
        maxlen: int = 100,
        ear_threshold: float = 0.21,
        width: int = 200,
        height: int = 60
    ):
        self.history = deque(maxlen=maxlen)
        self.ear_threshold = ear_threshold
        self.width = width
        self.height = height

    def update(self, mean_ear: float):
        """Append latest mean EAR value."""
        self.history.append(float(mean_ear))

    def draw(self, frame: np.ndarray, top_left: Optional[tuple] = None) -> np.ndarray:
        """Draw oscilloscope sparkline graph on frame dynamically at bottom right."""
        if len(self.history) < 2:
            return frame

        h, w, _ = frame.shape
        if top_left is None:
            # Dynamically position at bottom right corner
            top_left = (w - 215, h - 75)

        x_start, y_start = top_left
        overlay = frame.copy()

        # Graph background panel (Width: 200, Height: 60)
        cv2.rectangle(overlay, (x_start, y_start), (x_start + self.width, y_start + self.height),
                      (20, 20, 20), -1)
        cv2.rectangle(overlay, (x_start, y_start), (x_start + self.width, y_start + self.height),
                      (60, 60, 60), 1)

        cv2.putText(overlay, "EAR OSCILLOSCOPE", (x_start + 8, y_start + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        # Plot range mapping (EAR typically 0.0 to 0.45)
        min_val, max_val = 0.0, 0.45

        # Draw threshold line
        thresh_y = int(y_start + self.height - ((self.ear_threshold - min_val) / (max_val - min_val)) * self.height)
        thresh_y = np.clip(thresh_y, y_start + 18, y_start + self.height - 2)
        cv2.line(overlay, (x_start, thresh_y), (x_start + self.width, thresh_y), (0, 165, 255), 1, cv2.LINE_AA)

        # Plot points
        history_arr = np.array(self.history)
        n = len(history_arr)
        x_pts = np.linspace(x_start, x_start + self.width, n).astype(int)
        
        # Map EAR values to Y pixel coordinates (inverted for OpenCV)
        y_pts = y_start + self.height - ((history_arr - min_val) / (max_val - min_val) * self.height)
        y_pts = np.clip(y_pts, y_start + 18, y_start + self.height).astype(int)

        pts = np.column_stack((x_pts, y_pts))
        cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        return frame

"""
Head Pose Estimation Module using OpenCV solvePnP.
Calculates YAW, PITCH, ROLL in degrees and projects optional 3D coordinate axes onto the camera frame.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class HeadPoseEstimator:
    """
    Estimates 3D head pose angles (Yaw, Pitch, Roll) using OpenCV solvePnP and RQDecomp3x3.
    Ensures pitch, yaw, roll are physically interpretable in degrees centered around 0 for forward pose.
    """

    # 3D facial model reference points in OpenCV Camera Coordinate System
    # (+X points RIGHT, +Y points DOWN towards chin, +Z points INTO screen/camera)
    MODEL_3D_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
        (0.0, 330.0, -65.0),        # Chin (landmark 152)
        (-225.0, -170.0, -135.0),    # Left eye outer corner (landmark 33)
        (225.0, -170.0, -135.0),     # Right eye outer corner (landmark 263)
        (-150.0, 150.0, -125.0),     # Left mouth corner (landmark 61)
        (150.0, 150.0, -125.0)       # Right mouth corner (landmark 291)
    ], dtype=np.float64)

    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self):
        pass

    def estimate_pose(
        self,
        landmarks_3d: np.ndarray,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate Yaw, Pitch, Roll in degrees.
        Returns:
            yaw (float): Head rotation around Y-axis (-left, +right) in [-90, +90]
            pitch (float): Head rotation around X-axis (-down, +up) in [-90, +90]
            roll (float): Head rotation around Z-axis (-tilt right, +tilt left) in [-90, +90]
            rvec (np.ndarray): Rotation vector from solvePnP
            tvec (np.ndarray): Translation vector from solvePnP
            cam_matrix (np.ndarray): Camera focal length matrix
        """
        # Convert 2D pixel coordinates for the 6 key points
        image_points = np.array([
            (landmarks_3d[idx, 0] * img_width, landmarks_3d[idx, 1] * img_height)
            for idx in self.LANDMARK_INDICES
        ], dtype=np.float64)

        # Approximate camera focal length matrix based on image dimensions
        focal_length = float(img_width)
        center = (float(img_width / 2.0), float(img_height / 2.0))
        cam_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_3D_POINTS,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, None, None, None

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles using RQDecomp3x3
        angles, mtxR, mtxQ, qx, qy, qz = cv2.RQDecomp3x3(rmat)

        # angles contains (pitch_x, yaw_y, roll_z) in degrees
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize angles to [-180, 180] range cleanly
        pitch = float(((pitch + 180) % 360) - 180)
        yaw = float(((yaw + 180) % 360) - 180)
        roll = float(((roll + 180) % 360) - 180)

        return yaw, pitch, roll, rvec, tvec, cam_matrix

    def draw_3d_axes(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        cam_matrix: np.ndarray,
        nose_point_2d: Tuple[int, int],
        length: float = 50.0
    ) -> np.ndarray:
        """
        Draw 3D coordinate axes starting at nose tip.
        X-axis = Red, Y-axis = Green, Z-axis = Blue.
        """
        if rvec is None or tvec is None or cam_matrix is None:
            return frame

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Define 3D axis points in model space
        axis_3d = np.array([
            (length, 0.0, 0.0),    # X-axis (Red)
            (0.0, length, 0.0),    # Y-axis (Green)
            (0.0, 0.0, length)     # Z-axis (Blue)
        ], dtype=np.float64)

        # Project 3D points to 2D image plane
        axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist_coeffs)
        axis_2d = axis_2d.reshape(-1, 2).astype(int)

        p_nose = (int(nose_point_2d[0]), int(nose_point_2d[1]))
        p_x = tuple(axis_2d[0])
        p_y = tuple(axis_2d[1])
        p_z = tuple(axis_2d[2])

        cv2.line(frame, p_nose, p_x, (0, 0, 255), 2, cv2.LINE_AA) # X - Red
        cv2.line(frame, p_nose, p_y, (0, 255, 0), 2, cv2.LINE_AA) # Y - Green
        cv2.line(frame, p_nose, p_z, (255, 0, 0), 2, cv2.LINE_AA) # Z - Blue

        return frame

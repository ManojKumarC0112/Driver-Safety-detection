"""
Feature Extraction Script from Video Files for Driver Safety AI.
Processes input video file using MediaPipe Face Mesh and TemporalFeatureExtractor
to generate structured 12-feature CSV telemetry files in data/raw/ with subject & label provenance.
"""

import os
import argparse
import cv2
import pandas as pd
import numpy as np

from src.utils.paths import RAW_DATA_DIR, load_config
from src.features.face_landmarks import FaceLandmarkExtractor
from src.features.temporal_features import TemporalFeatureExtractor

def extract_features_from_video_file(
    video_path: str,
    subject_id: str,
    label_name: str,
    output_dir: str = str(RAW_DATA_DIR)
) -> str:
    """
    Extract 12 temporal features frame-by-frame from a video file.
    Saves CSV to output_dir with filename: {subject_id}_{label_name}_{timestamp}.csv
    """
    config = load_config()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = config["camera"]["fps"]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VideoExtract] Processing '{video_path}' | FPS: {fps:.1f} | Total Frames: {total_frames}")

    face_extractor = FaceLandmarkExtractor(
        max_num_faces=config["mediapipe"]["max_num_faces"],
        refine_landmarks=config["mediapipe"]["refine_landmarks"],
        min_detection_confidence=config["mediapipe"]["min_detection_confidence"],
        min_tracking_confidence=config["mediapipe"]["min_tracking_confidence"],
        target_width=config["camera"]["width"],
        target_height=config["camera"]["height"],
    )

    temporal_extractor = TemporalFeatureExtractor(
        ear_threshold=config["thresholds"]["ear_threshold"],
        mar_threshold=config["thresholds"]["mar_threshold"],
        perclos_window_frames=config["thresholds"]["perclos_window_frames"],
        fps=fps,
    )

    records = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_id += 1
        img_h, img_w, _ = frame.shape
        timestamp_sec = frame_id / fps

        is_valid_face, landmarks_3d, meta = face_extractor.process_frame(frame)
        feature_vec, is_valid_feat, _ = temporal_extractor.process_frame_landmarks(
            landmarks_3d=landmarks_3d,
            is_valid_face=is_valid_face,
            frame_width=img_w,
            frame_height=img_h,
            timestamp_sec=timestamp_sec
        )

        if is_valid_feat and is_valid_face:
            row = {
                "frame_id": frame_id,
                "timestamp": timestamp_sec,
                "EAR_LEFT": float(feature_vec[0]),
                "EAR_RIGHT": float(feature_vec[1]),
                "MEAN_EAR": float(feature_vec[2]),
                "MAR": float(feature_vec[3]),
                "YAW": float(feature_vec[4]),
                "PITCH": float(feature_vec[5]),
                "ROLL": float(feature_vec[6]),
                "PERCLOS": float(feature_vec[7]),
                "BLINK_RATE": float(feature_vec[8]),
                "EYE_CLOSURE_DURATION": float(feature_vec[9]),
                "MOUTH_OPEN_DURATION": float(feature_vec[10]),
                "HEAD_MOTION_MAGNITUDE": float(feature_vec[11]),
                "label": label_name,
                "subject_id": subject_id
            }
            records.append(row)

    cap.release()
    face_extractor.close()

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{subject_id}_{label_name}_video.csv")
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    print(f"[VideoExtract] Extracted {len(records)} valid frame vectors → {out_csv}")
    return out_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--subject_id", type=str, default="subject_video", help="Subject identifier")
    parser.add_argument("--label", type=str, required=True, choices=["ALERT", "DROWSY", "YAWNING", "DISTRACTED"])
    args = parser.parse_args()

    extract_features_from_video_file(args.video, args.subject_id, args.label)

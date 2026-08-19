"""
Webcam Data Collection Tool for Driver Safety AI.
Captures real-time facial feature telemetry (12 features) directly from webcam stream for a specified label & subject.
Stores structured tabular telemetry (CSV) without requiring raw video storage.
"""

import os
import time
import uuid
import argparse
import cv2
import pandas as pd
import numpy as np

from src.utils.paths import RAW_DATA_DIR, load_config
from src.features.face_landmarks import FaceLandmarkExtractor
from src.features.temporal_features import TemporalFeatureExtractor

def main():
    config = load_config()
    valid_classes = config["classes"]["names"]

    parser = argparse.ArgumentParser(description="Driver Safety AI Feature Collection Tool")
    parser.add_argument("--label", type=str, required=True, choices=valid_classes,
                        help=f"Class label to record: {valid_classes}")
    parser.add_argument("--subject_id", type=str, default="subject_01",
                        help="Identifier for the participant/driver (e.g., subject_01)")
    parser.add_argument("--camera_index", type=int, default=config["camera"]["index"],
                        help="Camera device index")
    parser.add_argument("--max_frames", type=int, default=1800,
                        help="Maximum frames to capture per session (e.g. 1800 = ~60 sec at 30 fps)")
    args = parser.parse_args()

    session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    output_csv = RAW_DATA_DIR / f"{args.subject_id}_{args.label}_{session_id}.csv"

    print("==================================================")
    print("      DRIVER SAFETY AI - DATA COLLECTION TOOL     ")
    print("==================================================")
    print(f" Label:        {args.label}")
    print(f" Subject ID:   {args.subject_id}")
    print(f" Session ID:   {session_id}")
    print(f" Output File:  {output_csv}")
    print(f" Target Limit: {args.max_frames} frames")
    print(" Press 'q' to stop recording and save dataset.")
    print("==================================================")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {args.camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

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
        fps=config["camera"]["fps"],
    )

    columns = [
        "timestamp", "session_id", "subject_id", "frame_id",
        "EAR_LEFT", "EAR_RIGHT", "MEAN_EAR", "MAR", "YAW", "PITCH", "ROLL",
        "PERCLOS", "BLINK_RATE", "EYE_CLOSURE_DURATION", "MOUTH_OPEN_DURATION", "HEAD_MOTION_MAGNITUDE",
        "label"
    ]

    telemetry_rows = []
    frame_count = 0
    start_time = time.time()

    print("\nStarting video capture in 3 seconds...")
    time.sleep(3)
    print("RECORDING... Perform designated action:", args.label)

    try:
        while cap.isOpened() and frame_count < args.max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to capture video frame.")
                break

            now_sec = time.time() - start_time
            is_valid, landmarks_3d, meta = face_extractor.process_frame(frame)

            feature_vec, is_valid_feat, extra = temporal_extractor.process_frame_landmarks(
                landmarks_3d=landmarks_3d,
                is_valid_face=is_valid,
                frame_width=meta.get("frame_width", config["camera"]["width"]),
                frame_height=meta.get("frame_height", config["camera"]["height"]),
                timestamp_sec=now_sec,
            )

            if is_valid_feat:
                frame_count += 1
                row = [now_sec, session_id, args.subject_id, frame_count] + feature_vec.tolist() + [args.label]
                telemetry_rows.append(row)

            # Draw status on camera view
            status_color = (0, 255, 0) if is_valid else (0, 0, 255)
            status_text = f"LABEL: {args.label} | FRAMES: {frame_count}/{args.max_frames}"
            cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"MEAN_EAR: {feature_vec[2]:.3f} | MAR: {feature_vec[3]:.3f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Driver Safety AI - Telemetry Data Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRecording stopped by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_extractor.close()

    if telemetry_rows:
        df = pd.DataFrame(telemetry_rows, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(df)} telemetry rows to {output_csv}")
    else:
        print("\nNo valid telemetry recorded.")

if __name__ == "__main__":
    main()

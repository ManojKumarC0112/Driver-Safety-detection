"""
Guided 12-Feature Telemetry Data Collection Application.
Collects feature telemetry ONLY (no raw video stored by default).
Requires Participant ID, Session ID, and Session Ground-Truth Label (ALERT or DROWSY).

Usage:
  python -m src.data.collector --participant P001 --session S001 --label ALERT --duration 30
"""

import os
import time
import argparse
import json
import csv
import cv2
import numpy as np
import yaml

from src.features.face_landmarks import FaceLandmarkExtractor
from src.features.eye_features import extract_eye_features
from src.features.mouth_features import calculate_mar
from src.features.head_pose import HeadPoseEstimator
from src.features.temporal_features import TemporalFeatureExtractor

CONFIG_PATH = os.path.join("configs", "proposed_12feature.yaml")

def parse_args():
    parser = argparse.ArgumentParser(description="Driver Safety AI - Guided 12-Feature Telemetry Data Collector")
    parser.add_argument("-p", "--participant", type=str, required=True, help="Participant ID (e.g. P001)")
    parser.add_argument("-s", "--session", type=str, required=True, help="Session ID (e.g. S001)")
    parser.add_argument("-l", "--label", type=str, required=True, choices=["ALERT", "DROWSY"], help="Ground-truth session label (ALERT or DROWSY)")
    parser.add_argument("-d", "--duration", type=int, default=30, help="Recording duration in seconds (default: 30)")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without displaying OpenCV window")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing session file if it already exists")
    return parser.parse_args()

def validate_features(feature_dict):
    """Validates 12 extracted features for NaN, Inf, and physically realistic ranges."""
    for key, val in feature_dict.items():
        if np.isnan(val) or np.isinf(val):
            return False, f"NaN or Inf detected in {key}"
    
    if not (0.0 <= feature_dict["MEAN_EAR"] <= 1.0):
        return False, f"Invalid MEAN_EAR value: {feature_dict['MEAN_EAR']}"
    if not (0.0 <= feature_dict["MAR"] <= 2.0):
        return False, f"Invalid MAR value: {feature_dict['MAR']}"
    if not (-180.0 <= feature_dict["YAW"] <= 180.0):
        return False, f"Invalid YAW angle: {feature_dict['YAW']}"
    if not (-180.0 <= feature_dict["PITCH"] <= 180.0):
        return False, f"Invalid PITCH angle: {feature_dict['PITCH']}"
    if not (-180.0 <= feature_dict["ROLL"] <= 180.0):
        return False, f"Invalid ROLL angle: {feature_dict['ROLL']}"
        
    return True, "Valid"

def run_collector():
    args = parse_args()
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config["collection"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = f"{args.participant}_{args.session}_{args.label}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    json_path = os.path.join(output_dir, f"{args.participant}_{args.session}_{args.label}.json")

    if os.path.exists(csv_path) and not args.overwrite:
        print(f"\n[Error] Session file '{csv_path}' already exists!")
        print("        Refusing to overwrite existing recording unless '--overwrite' flag is provided.")
        return

    print("\n==================================================================")
    print("   DRIVER SAFETY AI - GUIDED TELEMETRY DATA COLLECTOR             ")
    print("==================================================================")
    print(" [NOTICE] STATIONARY WEBCAM ONLY - PARTICIPANT MUST NOT OPERATE A ")
    print("          VEHICLE DURING DATA COLLECTION SESSIONS.                ")
    print("==================================================================")
    print(f" Participant ID : {args.participant}")
    print(f" Session ID     : {args.session}")
    print(f" Target Label   : {args.label}")
    print(f" Duration       : {args.duration} seconds")
    print(f" Output Path    : {csv_path}")
    print("==================================================================\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[Error] Cannot open webcam index {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    landmark_extractor = FaceLandmarkExtractor()
    pose_estimator = HeadPoseEstimator()
    temporal_extractor = TemporalFeatureExtractor(perclos_window_frames=90)

    # 1. Visual / Audio Countdown (3, 2, 1)
    countdown_secs = config["collection"]["countdown_seconds"]
    print(f"[Collector] Preparing recording. Get ready for {countdown_secs}-second countdown...")
    
    start_countdown = time.time()
    while time.time() - start_countdown < countdown_secs:
        remaining = int(np.ceil(countdown_secs - (time.time() - start_countdown)))
        ret, frame = cap.read()
        if not ret:
            break
        
        if not args.no_gui:
            display_frame = frame.copy()
            cv2.putText(display_frame, f"GET READY: {remaining}", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow("Guided Data Collector", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Collector] Cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

    print("\n[Collector] RECORDING STARTED! Maintain target pose/state...")

    records = []
    fieldnames = [
        "timestamp", "participant_id", "session_id", "frame_id",
        "EAR_LEFT", "EAR_RIGHT", "MEAN_EAR", "MAR", 
        "YAW", "PITCH", "ROLL", 
        "PERCLOS", "BLINK_RATE", "EYE_CLOSURE_DURATION", "MOUTH_OPEN_DURATION", "HEAD_MOTION_MAGNITUDE",
        "label"
    ]

    frame_id = 0
    valid_count = 0
    invalid_count = 0
    start_rec_time = time.time()

    while time.time() - start_rec_time < args.duration:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to read frame from webcam.")
            break

        frame_id += 1
        curr_time = time.time() - start_rec_time

        is_valid, landmarks_3d, meta = landmark_extractor.process_frame(frame)

        if is_valid and landmarks_3d is not None:
            # Extract 12-feature temporal vector
            feature_vec, is_valid_temp, extra_info = temporal_extractor.process_frame_landmarks(
                landmarks_3d=landmarks_3d,
                is_valid_face=is_valid,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                timestamp_sec=curr_time
            )

            feature_dict = {
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
                "HEAD_MOTION_MAGNITUDE": float(feature_vec[11])
            }

            is_valid, msg = validate_features(feature_dict)

            if is_valid:
                row = {
                    "timestamp": f"{curr_time:.4f}",
                    "participant_id": args.participant,
                    "session_id": args.session,
                    "frame_id": frame_id,
                    **feature_dict,
                    "label": args.label
                }
                records.append(row)
                valid_count += 1
            else:
                invalid_count += 1
        else:
            invalid_count += 1

        # GUI feedback
        if not args.no_gui:
            display_frame = frame.copy()
            rec_elapsed = curr_time
            time_left = max(0, args.duration - rec_elapsed)
            cv2.putText(display_frame, f"REC [{args.label}]: {time_left:.1f}s left", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Valid Frames: {valid_count} | Invalid: {invalid_count}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show red recording dot
            cv2.circle(display_frame, (600, 30), 12, (0, 0, 255), -1)

            cv2.imshow("Guided Data Collector", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Collector] Recording stopped prematurely by user.")
                break

    cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()

    print("\n[Collector] Recording completed!")
    print(f" Total Frames Processed : {frame_id}")
    print(f" Valid Telemetry Rows   : {valid_count}")
    print(f" Invalid / Missing Face : {invalid_count}")

    if valid_count < 30:
        print("[Error] Insufficient valid frames collected (< 30 frames). Session aborted without saving.")
        return

    # Save CSV output
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Save JSON metadata & summary
    num_sequences = max(0, valid_count - 30 + 1)
    metadata = {
        "participant_id": args.participant,
        "session_id": args.session,
        "label": args.label,
        "duration_seconds": args.duration,
        "total_frames": frame_id,
        "valid_frames": valid_count,
        "invalid_frames": invalid_count,
        "valid_ratio": float(valid_count / frame_id) if frame_id > 0 else 0.0,
        "num_30frame_sequences": num_sequences,
        "csv_path": csv_path,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Saved] Telemetry CSV  → {csv_path}")
    print(f"[Saved] Session Metadata → {json_path}")
    print(f"[Summary] Generated {num_sequences} sliding 30-frame temporal sequences!")

if __name__ == "__main__":
    run_collector()

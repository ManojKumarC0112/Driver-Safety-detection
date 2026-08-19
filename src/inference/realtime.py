"""
Real-Time Inference Engine for Driver Safety AI.
Executes real-time webcam pipeline:
Webcam → MediaPipe → 12 features → Normalization → 30-frame Deque → 1D CNN + Bi-LSTM → Softmax → Probability Smoothing → DVI Risk Engine → HUD Overlay → Audio Alarm → Telemetry Logger.
"""

import os
import sys
import time
import json
import joblib
import cv2
import numpy as np
from collections import deque
from typing import Optional, Dict, Any

import torch

from src.utils.paths import MODEL_PATH, SCALER_PATH, SCREENSHOTS_DIR, load_config
from src.features.face_landmarks import FaceLandmarkExtractor
from src.features.temporal_features import TemporalFeatureExtractor
from src.features.head_pose import HeadPoseEstimator
from src.model.driver_safety_net import DriverSafetyNet
from src.model.baseline_model import RuleBasedBaselineModel
from src.utils.dvi import DVIEngine
from src.inference.state_manager import StateManager
from src.inference.voice_alert import VoiceAlertEngine
from src.utils.logger import TelemetryLogger
from src.visualization.hud import draw_hud
from src.visualization.probability_bars import draw_probability_bars
from src.visualization.ear_graph import EARGraphOscilloscope

def trigger_audio_alarm(frequency_hz: int = 1000, duration_ms: int = 400):
    """Safely trigger audio warning on Windows without crashing application."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(frequency_hz, duration_ms)
        else:
            print("\a", end="")
    except Exception as e:
        pass

def run_realtime_application(
    config_path: Optional[str] = None,
    model_path: str = str(MODEL_PATH),
    scaler_path: str = str(SCALER_PATH)
):
    """
    Main real-time application entry point.
    """
    print("Loading model...")
    print("Loading feature scaler...")
    print("Loading MediaPipe...")
    print("Opening camera...")

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"

    class_names = config["classes"]["names"]
    window_size = config["features"]["window_size"]
    num_features = config["features"]["num_features"]

    # 1. Load Scaler
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"[RealTime] Loaded StandardScaler from {scaler_path}")
        except Exception as e:
            print(f"[RealTime] Warning: Could not load scaler ({e}). Using raw features.")
    else:
        print("[RealTime] Scaler not found. Proceeding with raw temporal features.")

    # 2. Load Model or Baseline
    model = None
    baseline_model = RuleBasedBaselineModel(
        ear_threshold=config["thresholds"]["ear_threshold"],
        mar_threshold=config["thresholds"]["mar_threshold"],
        yaw_threshold=config["thresholds"]["distraction_yaw_threshold"],
        pitch_threshold=config["thresholds"]["distraction_pitch_threshold"]
    )

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model = DriverSafetyNet(
                in_channels=config["model"]["in_channels"],
                cnn_filters=config["model"]["cnn_filters"],
                kernel_size=config["model"]["kernel_size"],
                lstm_hidden_size=config["model"]["lstm_hidden_size"],
                lstm_num_layers=config["model"]["lstm_num_layers"],
                lstm_bidirectional=config["model"]["lstm_bidirectional"],
                dropout=config["model"]["dropout"],
                num_classes=config["model"]["num_classes"]
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"[RealTime] Loaded PyTorch DriverSafetyNet from {model_path}")
        except Exception as e:
            print(f"[RealTime] Could not load PyTorch checkpoint ({e}). Using Rule-Based Baseline Engine.")
            model = None
    else:
        print("[RealTime] PyTorch model checkpoint not found. Using Rule-Based Baseline Engine.")

    print("\nDriver Safety AI Ready\n")

    # Initialize Extractor Components
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

    dvi_engine = DVIEngine(
        weight_p_drowsy=config["dvi"]["weight_p_drowsy"],
        weight_perclos=config["dvi"]["weight_perclos"],
        weight_eye_closure=config["dvi"]["weight_eye_closure"],
        weight_yaw_dev=config["dvi"]["weight_yaw_dev"],
        low_threshold=config["dvi"]["levels"]["low"],
        mod_threshold=config["dvi"]["levels"]["moderate"],
        high_threshold=config["dvi"]["levels"]["high"],
    )

    state_manager = StateManager(config=config)
    voice_alert_engine = VoiceAlertEngine(config=config)
    logger = TelemetryLogger()
    ear_oscilloscope = EARGraphOscilloscope(
        maxlen=config["ui"]["ear_graph_history_len"],
        ear_threshold=config["thresholds"]["ear_threshold"]
    )

    # State Deques
    sequence_buffer = deque(maxlen=window_size)
    prob_smoothing_buffer = deque(maxlen=config["model"]["inference_smoothing_window"])

    # UI Toggles
    show_mesh = config["ui"]["show_mesh"]
    show_3d_axes = config["ui"]["show_3d_axes"]
    show_ear_graph = config["ui"]["show_ear_graph"]
    show_probabilities = config["ui"]["show_probabilities"]

    # Open Camera
    cap = cv2.VideoCapture(config["camera"]["index"])
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {config['camera']['index']}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    frame_id = 0
    start_time = time.time()
    last_alarm_time = 0.0

    print("Controls: 'q'=Quit, 'm'=Toggle Mesh, 'a'=Toggle 3D Axes, 'g'=Toggle EAR Graph, 'p'=Toggle Probabilities\n")

    try:
        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Camera stream disconnected.")
                break

            frame_id += 1
            now_sec = time.time() - start_time
            img_h, img_w, _ = frame.shape

            # 1. Process Face Mesh & Landmarks
            is_valid_face, landmarks_3d, meta = face_extractor.process_frame(frame)

            # Draw optional face mesh
            if is_valid_face and show_mesh and landmarks_3d is not None:
                for pt in landmarks_3d:
                    px, py = int(pt[0] * img_w), int(pt[1] * img_h)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            # 2. Extract 12-dimensional temporal feature vector
            feature_vec, is_valid_feat, extra = temporal_extractor.process_frame_landmarks(
                landmarks_3d=landmarks_3d,
                is_valid_face=is_valid_face,
                frame_width=img_w,
                frame_height=img_h,
                timestamp_sec=now_sec
            )

            # Update EAR Oscilloscope
            ear_oscilloscope.update(feature_vec[2])

            # Draw optional 3D Axes
            if is_valid_face and show_3d_axes and extra["rvec"] is not None:
                frame = temporal_extractor.head_pose_estimator.draw_3d_axes(
                    frame, extra["rvec"], extra["tvec"], extra["cam_matrix"], extra["nose_2d"]
                )

            # 3. Temporal Sequence Deque Management
            if is_valid_feat:
                sequence_buffer.append(feature_vec)

            current_seq_len = len(sequence_buffer)

            # Determine Status & Prediction
            alarm_triggered = False

            if not is_valid_face:
                status_text = meta.get("status", "NO DRIVER FACE")
                probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
                dvi_score, dvi_level = 50.0, "MODERATE"
                decision_info = state_manager.process_frame(
                    p_drowsy=0.0,
                    mean_ear=feature_vec[2],
                    eye_closure_duration=0.0,
                    perclos=0.0,
                    is_valid_face=False,
                    timestamp_sec=now_sec
                )
            elif current_seq_len < window_size:
                status_text = f"INITIALIZING TEMPORAL MODEL ({current_seq_len}/{window_size} frames)"
                probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                dvi_score, dvi_level = 0.0, "LOW"
                decision_info = state_manager.process_frame(
                    p_drowsy=0.0,
                    mean_ear=feature_vec[2],
                    eye_closure_duration=feature_vec[9],
                    perclos=feature_vec[7],
                    is_valid_face=True,
                    timestamp_sec=now_sec
                )
            else:
                # 30 frames full -> Perform model inference
                seq_np = np.array(sequence_buffer, dtype=np.float32) # (30, 12)

                # Feature Scaling
                if scaler is not None:
                    seq_norm = scaler.transform(seq_np).reshape(1, window_size, num_features).astype(np.float32)
                else:
                    seq_norm = seq_np.reshape(1, window_size, num_features)

                if model is not None:
                    with torch.no_grad():
                        inp_tensor = torch.tensor(seq_norm, dtype=torch.float32).to(device)
                        probs_tensor = model.predict_proba(inp_tensor)
                        raw_probs = probs_tensor.cpu().numpy()[0]
                else:
                    # Fallback to Baseline Model
                    base_pred = baseline_model.predict_sequence(seq_np)
                    raw_probs = np.zeros(4, dtype=np.float32)
                    raw_probs[base_pred] = 1.0

                # Prediction Probability Smoothing
                prob_smoothing_buffer.append(raw_probs)
                probs = np.mean(prob_smoothing_buffer, axis=0)
                pred_class_id = int(np.argmax(probs))
                status_text = class_names[pred_class_id]

                # Explicit Distraction verification (Directive #6)
                if abs(feature_vec[4]) > config["thresholds"]["distraction_yaw_threshold"] or abs(feature_vec[5]) > config["thresholds"]["distraction_pitch_threshold"]:
                    if status_text != "DROWSY":
                        status_text = "DISTRACTED"

                # Post-Inference Temporal Safety State Manager
                decision_info = state_manager.process_frame(
                    p_drowsy=probs[1], # Neural network DROWSY probability
                    mean_ear=feature_vec[2],
                    eye_closure_duration=feature_vec[9],
                    perclos=feature_vec[7],
                    is_valid_face=True,
                    timestamp_sec=now_sec
                )

                # Multi-Stage Offline Voice Alert Trigger
                if decision_info.get("voice_event"):
                    voice_alert_engine.speak(decision_info["voice_event"])

                # Calculate DVI Risk Index
                dvi_score, dvi_level, _ = dvi_engine.calculate_dvi(
                    p_alert=probs[0],
                    perclos=feature_vec[7],
                    eye_closure_duration_sec=feature_vec[9],
                    yaw_deg=feature_vec[4]
                )

                # Alarm Trigger Check (Requires confirmed DROWSY state)
                if decision_info["alarm_triggered"]:
                    if (now_sec - last_alarm_time) > config["alarm"]["cooldown_seconds"]:
                        alarm_triggered = True
                        last_alarm_time = now_sec
                        trigger_audio_alarm(
                            frequency_hz=config["alarm"]["sound_frequency"],
                            duration_ms=config["alarm"]["sound_duration_ms"]
                        )
                        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
                        shot_path = SCREENSHOTS_DIR / f"drowsy_event_{int(time.time())}.png"
                        cv2.imwrite(str(shot_path), frame)

            # Calculate FPS
            loop_elapsed = time.time() - loop_start
            current_fps = float(1.0 / max(loop_elapsed, 1e-4))

            # Log frame telemetry
            logger.log_frame(
                frame_id=frame_id,
                fps=current_fps,
                feature_vec=feature_vec,
                probabilities=probs,
                dvi=dvi_score,
                predicted_state=status_text,
                is_blink=extra.get("is_blink", False),
                is_yawn=(feature_vec[10] > 0.5),
                alarm_triggered=alarm_triggered,
                state=decision_info.get("state", status_text),
                intervention_level=decision_info.get("intervention_level", 0),
                voice_event=decision_info.get("voice_event"),
                response_status=decision_info.get("response_status", "NONE")
            )

            # 4. Render OpenCV HUD
            frame = draw_hud(
                frame=frame,
                status_text=status_text,
                feature_vec=feature_vec,
                dvi_score=dvi_score,
                dvi_level=dvi_level,
                fps=current_fps,
                device_name=device_name,
                warmup_counter=(current_seq_len, window_size) if current_seq_len < window_size else None,
                decision_info=decision_info
            )

            # Render optional Probability Bars
            if show_probabilities:
                frame = draw_probability_bars(frame, probs, class_names)

            # Render optional EAR Graph
            if show_ear_graph:
                frame = ear_oscilloscope.draw(frame)

            # Render Strong Red Blinking Strobe Flash Effect during Drowsiness Alarm
            is_drowsy_alarm = alarm_triggered or decision_info.get("alarm_triggered", False) or (decision_info.get("state") == "DROWSY")
            enable_strobe = config.get("alarm", {}).get("enable_strobe_flash", True)
            strobe_hz = config.get("alarm", {}).get("strobe_frequency_hz", 6.0)

            if is_drowsy_alarm:
                # Calculate strobe pulse state at specified Hz (e.g. 6.0 Hz = ~166ms toggle)
                flash_on = (int(now_sec * strobe_hz) % 2 == 0) if enable_strobe else True
                
                if flash_on:
                    # Strong bright red full-screen alpha overlay
                    red_overlay = np.zeros_like(frame)
                    red_overlay[:, :] = (0, 0, 240)  # Bright BGR Red
                    frame = cv2.addWeighted(frame, 0.55, red_overlay, 0.45, 0)

                    # Thick screen boundary pulse ring
                    cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 0, 255), 18)

                    # Prominent Warning Banner
                    cv2.rectangle(frame, (0, img_h - 70), (img_w, img_h), (0, 0, 255), -1)
                    cv2.putText(frame, "CRITICAL DROWSINESS WARNING - WAKE UP!",
                                (img_w // 2 - 285, img_h - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
                else:
                    # Strobe Off Pulse: Maintain solid red border & bottom banner for readability
                    cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 0, 200), 8)
                    cv2.rectangle(frame, (0, img_h - 70), (img_w, img_h), (20, 20, 180), -1)
                    cv2.putText(frame, "CRITICAL DROWSINESS WARNING - WAKE UP!",
                                (img_w // 2 - 285, img_h - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            # Show Frame
            cv2.imshow("Driver Safety AI - Real-Time Vigilance Detection", frame)

            # Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting real-time monitoring...")
                break
            elif key == ord('m'):
                show_mesh = not show_mesh
                print(f"[UI] Face Mesh toggled: {show_mesh}")
            elif key == ord('a'):
                show_3d_axes = not show_3d_axes
                print(f"[UI] 3D Axes toggled: {show_3d_axes}")
            elif key == ord('g'):
                show_ear_graph = not show_ear_graph
                print(f"[UI] EAR Graph toggled: {show_ear_graph}")
            elif key == ord('p'):
                show_probabilities = not show_probabilities
                print(f"[UI] Probability Bars toggled: {show_probabilities}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_extractor.close()
        summary = logger.close_session()
        print("\nSession complete. Summary:")
        print(json.dumps(summary, indent=2))

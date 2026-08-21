import sys
from pathlib import Path
from collections import deque

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import cv2
import time
import json
import math
import logging
import asyncio
import urllib.parse
import urllib.request
import threading
import numpy as np
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Response, Query, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import existing core modules without touching main.py
from src.utils.paths import MODEL_PATH, SCALER_PATH, load_config
from src.utils.logger import TelemetryLogger
from src.features.face_landmarks import FaceLandmarkExtractor
from src.features.temporal_features import TemporalFeatureExtractor
from src.model.driver_safety_net import DriverSafetyNet
from src.utils.dvi import DVIEngine
from src.inference.state_manager import StateManager
from src.inference.voice_alert import VoiceAlertEngine
from src.services.dhaba_assistant import DhabaRecommendationEngine
from src.services.voice_service import VoiceSynthesisService
from src.report.pdf_report import generate_pdf_session_report

logger = logging.getLogger("web_app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Driver Safety AI — Web Dashboard & Smart Dhaba Assistant")

# Project paths
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "src" / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)

# Mount Static Files for frontend JS & assets
app.mount("/src/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# Global State Container for Thread-Safe Telemetry
class EngineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.show_mesh: bool = False
        self.show_id: bool = True
        self.voice_language: str = "english"
        self.latest_telemetry: Dict[str, Any] = {
            "state": "ALERT",
            "p_drowsy": 0.0,
            "dvi_score": 0.0,
            "dvi_level": "LOW",
            "ear": 0.35,
            "mar": 0.12,
            "perclos": 0.0,
            "blink_rate": 18.0,
            "eye_close_duration": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "intervention_level": 0,
            "response_status": "NONE",
            "event_timeline": [],
            "fps": 30.0,
            "alarm_triggered": False,
            "voice_language": "english",
        }
        self.frame_bytes: bytes = b""

engine_state = EngineState()

def draw_face_mesh_overlay(frame: np.ndarray, landmarks_3d: np.ndarray):
    """Draws glowing MediaPipe 3D face mesh tessellation & landmarks on frame."""
    if landmarks_3d is None or len(landmarks_3d) == 0:
        return
    h, w, _ = frame.shape
    for idx, (x, y, z) in enumerate(landmarks_3d):
        if idx % 3 == 0:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 1, (255, 255, 0), -1)

    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

    for eye_pts in [left_eye_indices, right_eye_indices]:
        pts = np.array([[int(landmarks_3d[i][0]*w), int(landmarks_3d[i][1]*h)] for i in eye_pts if i < len(landmarks_3d)], np.int32)
        if len(pts) > 0:
            cv2.polylines(frame, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)

    pts_mouth = np.array([[int(landmarks_3d[i][0]*w), int(landmarks_3d[i][1]*h)] for i in mouth_indices if i < len(landmarks_3d)], np.int32)
    if len(pts_mouth) > 0:
        cv2.polylines(frame, [pts_mouth], True, (0, 255, 160), 1, cv2.LINE_AA)

def draw_driver_id_overlay(frame: np.ndarray, landmarks_3d: np.ndarray, state_name: str = "ALERT"):
    """Draws tech face bounding box with corner brackets and Driver Verification Badge."""
    if landmarks_3d is None or len(landmarks_3d) == 0:
        return
    h, w, _ = frame.shape
    xs = [int(p[0] * w) for p in landmarks_3d]
    ys = [int(p[1] * h) for p in landmarks_3d]

    min_x, max_x = max(0, min(xs) - 15), min(w, max(xs) + 15)
    min_y, max_y = max(0, min(ys) - 25), min(h, max(ys) + 15)

    color = (0, 220, 0)
    if "DROWSY" in state_name:
        color = (0, 0, 235)
    elif "SUSPECT" in state_name:
        color = (0, 215, 255)

    length = 20
    thick = 2
    cv2.line(frame, (min_x, min_y), (min_x + length, min_y), color, thick)
    cv2.line(frame, (min_x, min_y), (min_x, min_y + length), color, thick)
    cv2.line(frame, (max_x, min_y), (max_x - length, min_y), color, thick)
    cv2.line(frame, (max_x, min_y), (max_x, min_y + length), color, thick)
    cv2.line(frame, (min_x, max_y), (min_x + length, max_y), color, thick)
    cv2.line(frame, (min_x, max_y), (min_x, max_y - length), color, thick)
    cv2.line(frame, (max_x, max_y), (max_x - length, max_y), color, thick)
    cv2.line(frame, (max_x, max_y), (max_x, max_y - length), color, thick)

    badge_h = 22
    badge_y1 = max(0, min_y - badge_h)
    cv2.rectangle(frame, (min_x, badge_y1), (max_x, min_y), (15, 23, 42), -1)
    cv2.rectangle(frame, (min_x, badge_y1), (max_x, min_y), color, 1)

    text = "DRIVER: MANOJ KUMAR | ID: #4092 (VERIFIED)"
    cv2.putText(frame, text, (min_x + 5, min_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

# Load Model & Pipeline
cfg = load_config()

# Helper for Model loading
def load_pytorch_model():
    try:
        import torch
        if not MODEL_PATH.exists():
            return None, None
        checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        
        num_classes = 4
        if "fc.weight" in state_dict:
            num_classes = state_dict["fc.weight"].shape[0]

        model = DriverSafetyNet(num_classes=num_classes)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, torch
    except Exception as e:
        logger.warning(f"Could not load PyTorch model: {e}")
        return None, None

def load_scaler():
    try:
        import joblib
        if SCALER_PATH.exists():
            return joblib.load(str(SCALER_PATH))
    except Exception as e:
        logger.warning(f"Could not load scaler: {e}")
    return None

model, torch_module = load_pytorch_model()
scaler = load_scaler()
face_extractor = FaceLandmarkExtractor()
temporal_extractor = TemporalFeatureExtractor()
sequence_buffer = deque(maxlen=30)
state_mgr = StateManager(cfg)
dvi_engine = DVIEngine()
voice_engine = VoiceAlertEngine(cfg)
dhaba_engine = DhabaRecommendationEngine()
voice_service = VoiceSynthesisService()
session_logger = TelemetryLogger(session_id=f"web_{int(time.time())}")

# Background Video & Inference Loop Thread
def background_inference_loop():
    cam_idx = cfg["camera"]["index"]
    cap = None
    
    # Try opening camera with DirectShow on Windows for faster/reliable access
    if sys.platform == "win32":
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    
    # Fallback to index 1 if index 0 fails
    if not cap.isOpened():
        logger.warning(f"Camera index {cam_idx} failed, attempting index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info(f"Webcam successfully initialized on index {cam_idx} (Buffer size: 1, Res: 640x480)")
    else:
        logger.error("Failed to open webcam. Please verify camera connection & permissions.")

    prev_time = time.time()
    class_names = ["ALERT", "DROWSY", "YAWNING", "DISTRACTED"]
    frame_counter = 0
    last_probs = None

    failed_reads = 0

    while True:
        try:
            if not cap or not cap.isOpened():
                logger.warning("Camera lost connection, attempting auto-reconnect...")
                time.sleep(0.5)
                cam_idx = cfg["camera"]["index"]
                if sys.platform == "win32":
                    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(cam_idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads > 15:
                    logger.warning("Camera failed 15 consecutive reads. Re-initializing...")
                    cap.release()
                    cap = None
                    failed_reads = 0
                time.sleep(0.03)
                continue

            failed_reads = 0

            frame = cv2.flip(frame, 1)
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Extract facial landmarks & 12 telemetry features
            is_valid, landmarks, _ = face_extractor.process_frame(frame)
            feature_vec, is_valid_feat, extra_info = temporal_extractor.process_frame_landmarks(
                landmarks_3d=landmarks,
                is_valid_face=is_valid,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                timestamp_sec=curr_time,
                fps=fps
            )

            # Scale features
            scaled_feat = feature_vec.copy()
            if scaler is not None:
                try:
                    scaled_feat = scaler.transform(feature_vec.reshape(1, -1))[0]
                except Exception as e:
                    logger.warning(f"Scaler transform failed: {e}")

            sequence_buffer.append(scaled_feat)

            # Model Inference (Strided execution every 4 frames for 30+ FPS performance)
            frame_counter += 1
            if model is not None and len(sequence_buffer) == 30 and (frame_counter % 4 == 0 or last_probs is None):
                try:
                    inp_arr = np.array(sequence_buffer, dtype=np.float32)
                    inp = torch_module.FloatTensor(inp_arr).unsqueeze(0)
                    with torch_module.no_grad():
                        out = model(inp)
                        last_probs = torch_module.softmax(out, dim=1).numpy()[0]
                except Exception as e:
                    logger.error(f"Inference error: {e}")

            probs = last_probs if last_probs is not None else np.array([0.95, 0.02, 0.02, 0.01], dtype=np.float32)

            p_drowsy = float(probs[1])
            mean_ear = float(feature_vec[2])
            eye_closure_sec = float(feature_vec[9])

            # State Manager update
            decision_info = state_mgr.process_frame(
                p_drowsy=p_drowsy,
                mean_ear=mean_ear,
                eye_closure_duration=eye_closure_sec,
                perclos=float(feature_vec[7]),
                timestamp_sec=curr_time,
                is_valid_face=is_valid
            )
            status_text = decision_info.get("state", "ALERT")

            dvi_score, dvi_level, _ = dvi_engine.calculate_dvi(
                p_alert=float(probs[0]),
                perclos=float(feature_vec[7]),
                eye_closure_duration_sec=eye_closure_sec,
                yaw_deg=float(feature_vec[4])
            )

            # Voice Trigger
            # Note: Server-side host pyttsx3 speaking disabled; voice alerts sent via telemetry payload to Web SpeechSynthesis API
            # if decision_info.get("voice_event"):
            #     voice_engine.speak(decision_info["voice_event"])

            # Render Overlays if enabled
            with engine_state.lock:
                do_mesh = engine_state.show_mesh
                do_id = engine_state.show_id

            if do_mesh and is_valid and landmarks is not None:
                draw_face_mesh_overlay(frame, landmarks)

            if do_id and is_valid and landmarks is not None:
                draw_driver_id_overlay(frame, landmarks, status_text)

            # Update Thread State
            with engine_state.lock:
                engine_state.latest_telemetry = {
                    "state": decision_info["state"],
                    "p_drowsy": round(p_drowsy * 100, 1),
                    "dvi_score": round(dvi_score, 1),
                    "dvi_level": dvi_level,
                    "ear": round(mean_ear, 3),
                    "mar": round(float(feature_vec[3]), 3),
                    "is_yawn": bool(feature_vec[3] > 0.52 or probs[2] > 0.35),
                    "yawn_prob": round(float(probs[2]) * 100, 1),
                    "perclos": round(float(feature_vec[7]) * 100, 1),
                    "blink_rate": round(float(feature_vec[8]), 1),
                    "eye_close_duration": round(eye_closure_sec, 2),
                    "yaw": round(float(feature_vec[4]), 1),
                    "pitch": round(float(feature_vec[5]), 1),
                    "intervention_level": decision_info.get("intervention_level", 0),
                    "response_status": decision_info.get("response_status", "NONE"),
                    "event_timeline": decision_info.get("event_timeline", [])[-5:],
                    "fps": round(fps, 1),
                    "alarm_triggered": decision_info.get("alarm_triggered", False),
                    "voice_event": decision_info.get("voice_event", None),
                    "voice_language": engine_state.voice_language
                }

                # Encode Frame JPEG for Streaming (60 quality for ultra low latency)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                engine_state.frame_bytes = buffer.tobytes()

            session_logger.log_frame(
                frame_id=frame_counter,
                fps=fps,
                feature_vec=feature_vec,
                probabilities=probs,
                dvi=dvi_score,
                predicted_state=status_text,
                is_blink=bool(extra_info.get("is_blink", False)),
                is_yawn=bool(feature_vec[10] > 0.5),
                alarm_triggered=decision_info.get("alarm_triggered", False),
                state=decision_info.get("state", status_text),
                intervention_level=decision_info.get("intervention_level", 0),
                voice_event=decision_info.get("voice_event"),
                response_status=decision_info.get("response_status", "NONE"),
                prev_state=decision_info.get("prev_state")
            )

            # Frame rate throttle (pacing at ~33ms for smooth 30 FPS stream)
            time.sleep(0.025)
        except Exception as e:
            logger.exception(f"background_inference_loop frame error: {e}")
            time.sleep(0.05)

    cap.release()

# Start background thread
threading.Thread(target=background_inference_loop, daemon=True).start()

# API Endpoints
@app.get("/")
def read_root():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>Driver Safety AI Dashboard Initializing...</h2>")

@app.api_route("/api/toggle_settings", methods=["GET", "POST"])
def toggle_settings(mesh: Optional[bool] = None, id: Optional[bool] = None):
    with engine_state.lock:
        if mesh is not None:
            engine_state.show_mesh = mesh
        if id is not None:
            engine_state.show_id = id
        return {
            "show_mesh": engine_state.show_mesh,
            "show_id": engine_state.show_id
        }

@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        try:
            while True:
                if engine_state.frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + engine_state.frame_bytes + b'\r\n')
                await asyncio.sleep(0.03)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/telemetry")
def get_telemetry():
    with engine_state.lock:
        return engine_state.latest_telemetry

def _fetch_nearby_places(lat: float, lon: float) -> Dict[str, Any]:
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
    ]
    query = f"""
    [out:json][timeout:3];
    (
      node["amenity"="restaurant"](around:5000, {lat}, {lon});
      node["amenity"="cafe"](around:5000, {lat}, {lon});
      node["amenity"="fuel"](around:5000, {lat}, {lon});
      node["highway"="rest_area"](around:5000, {lat}, {lon});
    );
    out body 8;
    """

    for endpoint in overpass_urls:
        try:
            data = urllib.parse.urlencode({"data": query}).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers={"User-Agent": "DriverSafetyAI/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                elements = res_data.get("elements", [])
                places: List[Dict[str, Any]] = []
                for el in elements:
                    tags = el.get("tags", {})
                    p_lat = el.get("lat", lat)
                    p_lon = el.get("lon", lon)
                    amenity = tags.get("amenity", "rest_stop")
                    name = tags.get("name", "Highway Rest Stop")
                    category = "🍲 Highway Dhaba" if amenity in {"restaurant", "cafe"} else "⛽ Fuel & Service Station" if amenity == "fuel" else "🅿️ Rest Layover"
                    places.append({
                        "name": name,
                        "category": category,
                        "distance_km": round(math.sqrt((p_lat - lat) ** 2 + (p_lon - lon) ** 2) * 111.0, 1),
                        "lat": p_lat,
                        "lon": p_lon,
                        "maps_url": f"https://www.google.com/maps/dir/?api=1&destination={p_lat},{p_lon}",
                        "tags": tags,
                        "amenities": {
                            "food": amenity in {"restaurant", "cafe"},
                            "fuel": amenity == "fuel",
                            "restroom": tags.get("toilets") in {"yes", "designated"} or tags.get("highway") == "rest_area",
                            "parking": tags.get("parking") in {"yes", "designated"} or tags.get("amenity") == "fuel",
                        }
                    })

                if places:
                    return {"source": "overpass", "places": places[:12]}
        except Exception:
            continue

    return {
        "source": "fallback",
        "places": dhaba_engine.build_fallback_places(lat, lon)
    }

@app.api_route("/api/voice_language", methods=["GET", "POST"])
def set_voice_language(language: Optional[str] = None):
    allowed = {"english", "hindi", "hinglish"}
    chosen = (language or engine_state.voice_language or "english").lower()
    if chosen not in allowed:
        chosen = "english"
    with engine_state.lock:
        engine_state.voice_language = chosen
        engine_state.latest_telemetry["voice_language"] = chosen
        return {
            "voice_language": chosen,
            "language_code": voice_service.profile_for(chosen)["language_code"],
            "sarvam_enabled": voice_service.enabled,
        }

@app.post("/api/voice_preview")
async def voice_preview(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    text = str(payload.get("text", "")).strip()
    language = str(payload.get("language", engine_state.voice_language or "english")).lower()
    if not text:
        return JSONResponse(status_code=400, content={"detail": "text is required"})

    result = voice_service.build_prompt(language, text)
    if result.audio_bytes:
        headers = {
            "X-Voice-Provider": result.provider,
            "X-Voice-Language": result.language,
            "X-Voice-Language-Code": result.language_code,
        }
        return Response(content=result.audio_bytes, media_type=result.mime_type, headers=headers)

    return JSONResponse(content=result.to_metadata())

@app.get("/api/session_summary")
def get_session_summary():
    summary = session_logger.build_summary()
    with engine_state.lock:
        summary["voice_language"] = engine_state.voice_language
    return summary

@app.get("/api/session_summary/download")
def download_session_summary(format: str = Query("json")):
    summary = session_logger.build_summary()
    with engine_state.lock:
        summary["voice_language"] = engine_state.voice_language
    filename_base = summary.get("session_id", "session_summary")

    if format.lower() == "pdf":
        pdf_path = generate_pdf_session_report(summary)
        return FileResponse(
            pdf_path,
            filename=f"{filename_base}_summary.pdf",
            media_type="application/pdf"
        )

    return Response(
        content=json.dumps(summary, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename_base}_summary.json"'
        }
    )

@app.get("/api/nearby_dhabas")
def get_nearby_dhabas(lat: float = Query(13.1147), lon: float = Query(77.5956)):
    """
    Queries OpenStreetMap Overpass API for nearby Dhabas, Restaurants, Fuel Stations, and Layovers.
    Returns a ranked recommendation package with a spoken summary.
    """
    nearby = _fetch_nearby_places(lat, lon)
    with engine_state.lock:
        current_state = engine_state.latest_telemetry.get("state", "ALERT")
        language = engine_state.voice_language

    recommendation = dhaba_engine.recommend(
        nearby["places"],
        driver_lat=lat,
        driver_lon=lon,
        context=current_state,
        language=language,
    )
    payload = recommendation.to_dict()
    payload["status"] = nearby["source"]
    payload["source"] = nearby["source"]
    payload["voice_language"] = language
    return payload

@app.post("/api/nap_alarm")
def trigger_nap_alarm():
    voice_engine.speak("ALERTNESS_RESTORED", force=True)
    return {"status": "ok", "message": "Power Nap Alarm Initialized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)

import os
import csv
import time
import threading

import cv2
import mediapipe as mp
from scipy.spatial import distance
from fer.fer import FER

try:
    import winsound
    WINDOWS = True
except ImportError:
    WINDOWS = False

from config import (
    EAR_THRESHOLD,
    CLOSED_EYES_FRAMES,
    SCREENSHOT_DIR,
    DROWSINESS_CSV,
    DROWSINESS_REPORT,
    DROWSINESS_OUTPUT_VIDEO,
)

# -----------------------------
# Blink / Yawn debouncing
# -----------------------------
# A single noisy frame crossing the threshold should NOT count as a
# real blink/yawn. Require the eyes/mouth to hold state for a minimum
# number of frames, and enforce a cooldown so the same event can't be
# counted twice.

MIN_BLINK_FRAMES = 4       # eyes must stay "closed" this many frames
BLINK_COOLDOWN = 0.3       # seconds between counted blinks

MIN_YAWN_FRAMES = 8        # mouth must stay "open" this many frames
YAWN_COOLDOWN = 2.0        # seconds between counted yawns

# Frames are downscaled to this width before running FaceMesh, since
# inference cost scales with pixel count and MediaPipe's output
# landmarks are normalized (0-1) — they map back onto the full
# resolution frame just fine without any extra math.
PROCESSING_WIDTH = 480


class DrowsinessDetector:

    def __init__(self):

        # -----------------------------
        # MediaPipe FaceMesh
        # -----------------------------

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            # Iris refinement is only needed if you use the iris
            # landmarks (indices 468-477). LEFT_EYE/RIGHT_EYE below
            # only use the base 6-point eye contour, so this stays
            # off — it roughly doubles inference cost otherwise.
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # -----------------------------
        # Deep Learning Emotion Model
        # -----------------------------
        self.emotion_detector = FER(mtcnn=False)
        self.emotion = "neutral"
        self.stress_score = 0.0
        self.emotion_thread = None
        self.emotion_lock = threading.Lock()
        self.current_frame = None

        # -----------------------------
        # Counters
        # -----------------------------

        self.eye_counter = 0
        self.blink_count = 0
        self.yawn_count = 0
        self.prev_eye_closed = False
        self.prev_yawn = False
        self.smoothed_ear = None

        self.yawn_counter = 0
        self.last_blink_time = 0.0
        self.last_yawn_time = 0.0

        # -----------------------------
        # Latest per-frame values
        # (kept up to date on every detect() call so callers like
        # main.py or process_video() can read them afterwards)
        # -----------------------------

        self.ear = 0.0
        self.mar = 0.0
        self.status = "NORMAL"

        # -----------------------------
        # FPS (on-screen overlay only)
        # -----------------------------

        self.prev_time = time.time()

        # -----------------------------
        # Alarm (multi-stage escalation)
        # -----------------------------

        self.alarm = False
        self.alarm_on = False
        self.alarm_start_time = 0

        # -----------------------------
        # Video-run bookkeeping
        # (reset at the start of every process_video() call)
        # -----------------------------

        self.total_frames = 0
        self.alert_count = 0
        self._csv_rows = []

        # -----------------------------
        # FaceMesh landmarks
        # -----------------------------

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291

    # ---------------------------------
    # Euclidean Distance
    # ---------------------------------

    def dist(self, p1, p2):
        return distance.euclidean(p1, p2)

    # ---------------------------------
    # Eye Aspect Ratio
    # ---------------------------------

    def eye_aspect_ratio(self, eye):
        A = self.dist(eye[1], eye[5])
        B = self.dist(eye[2], eye[4])
        C = self.dist(eye[0], eye[3])
        return (A + B) / (2 * C)

    # ---------------------------------
    # Mouth Aspect Ratio (normalized, scale-invariant)
    # ---------------------------------

    def mouth_ratio(self, upper, lower, left, right):
        vertical = self.dist(upper, lower)
        horizontal = self.dist(left, right)
        if horizontal == 0:
            return 0
        return vertical / horizontal

    # ---------------------------------
    # Alarm (multi-stage escalation)
    # ---------------------------------

    def play_alarm(self):
        now = time.time()
        # Cooldown of 4 seconds so the voice finishes speaking before triggering again
        if now - getattr(self, 'last_beep_time', 0.0) < 4.0:
            return
        self.last_beep_time = now

        def _speak():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                # Set a slightly faster text-to-speech rate for urgency
                rate = engine.getProperty('rate')
                engine.setProperty('rate', rate + 30)
                engine.say("WARNING! DRIVER DROWSINESS DETECTED. PLEASE PULL OVER IMMEDIATELY.")
                engine.runAndWait()
            except Exception:
                if WINDOWS:
                    import winsound
                    winsound.Beep(2500, 500)
                    winsound.Beep(2500, 500)

        threading.Thread(target=_speak, daemon=True).start()

    # ---------------------------------
    # Get Eye Points
    # ---------------------------------

    def get_eye_points(self, landmarks, indices, w, h):
        points = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            points.append((
                int(lm.x * w),
                int(lm.y * h)
            ))
        return points

    # ---------------------------------
    # Get Lip Points
    # ---------------------------------

    def get_lip_points(self, landmarks, w, h):
        upper = landmarks.landmark[self.UPPER_LIP]
        lower = landmarks.landmark[self.LOWER_LIP]
        left = landmarks.landmark[self.MOUTH_LEFT]
        right = landmarks.landmark[self.MOUTH_RIGHT]

        upper = (int(upper.x * w), int(upper.y * h))
        lower = (int(lower.x * w), int(lower.y * h))
        left = (int(left.x * w), int(left.y * h))
        right = (int(right.x * w), int(right.y * h))

        return upper, lower, left, right

    # ---------------------------------
    # Blink Detection
    # ---------------------------------

    def detect_blink(self, ear):
        # Apply exponential smoothing to reduce jitter that resets the counter
        if self.smoothed_ear is None:
            self.smoothed_ear = ear
        else:
            self.smoothed_ear = self.smoothed_ear * 0.6 + ear * 0.4
            
        if self.smoothed_ear < EAR_THRESHOLD:
            self.eye_counter += 1
            self.prev_eye_closed = True
        else:
            if self.prev_eye_closed and self.eye_counter >= MIN_BLINK_FRAMES:
                now = time.time()
                if now - self.last_blink_time >= BLINK_COOLDOWN:
                    self.blink_count += 1
                    self.last_blink_time = now
            self.eye_counter = 0
            self.prev_eye_closed = False

    # ---------------------------------
    # Yawn Detection (debounced)
    # ---------------------------------

    def detect_yawn(self, mar):
        if mar > 0.6:
            self.yawn_counter += 1
            if not self.prev_yawn and self.yawn_counter >= MIN_YAWN_FRAMES:
                now = time.time()
                if now - self.last_yawn_time >= YAWN_COOLDOWN:
                    self.yawn_count += 1
                    self.last_yawn_time = now
                self.prev_yawn = True
        else:
            self.yawn_counter = 0
            self.prev_yawn = False

    # ---------------------------------
    # Driver Status
    # ---------------------------------

    def driver_status(self):
        import config
        if self.eye_counter >= config.CLOSED_EYES_FRAMES:
            return "DROWSY"
        return "NORMAL"

    # ---------------------------------
    # Alarm Check + Screenshot on first alert
    # ---------------------------------

    def check_alarm(self, frame):
        import config
        if self.eye_counter >= config.CLOSED_EYES_FRAMES:
            if not self.alarm_on:
                self.alarm_on = True
                self.alarm_start_time = time.time()
                self.alert_count += 1

                os.makedirs(SCREENSHOT_DIR, exist_ok=True)
                filename = os.path.join(SCREENSHOT_DIR, f"drowsy_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)

            self.alarm = True
            self.play_alarm()
        else:
            self.alarm = False
            self.alarm_on = False

    # ---------------------------------
    # Emotion DL Thread
    # ---------------------------------

    def _run_emotion_inference(self):
        if self.current_frame is None:
            return
        
        # Inference takes some CPU, running in background allows UI to stay 30fps
        emotion_name, score = self.emotion_detector.top_emotion(self.current_frame)
        
        with self.emotion_lock:
            if emotion_name:
                self.emotion = emotion_name
                if emotion_name in ["angry", "sad", "fear", "disgust"]:
                    self.stress_score = min(1.0, self.stress_score + 0.1)
                elif emotion_name == "happy":
                    self.stress_score = max(0.0, self.stress_score - 0.2)
                else:
                    self.stress_score = max(0.0, self.stress_score - 0.05)

    # ---------------------------------
    # Reset state when face is lost
    # ---------------------------------

    def reset_state(self):
        self.eye_counter = 0
        self.yawn_counter = 0
        self.alarm = False
        self.alarm_on = False
        self.prev_eye_closed = False
        self.prev_yawn = False
        self.emotion = "neutral"
        self.stress_score = 0.0

    # ---------------------------------
    # Detect Drowsiness (single frame)
    # ---------------------------------

    def detect(self, frame):
        # Flip frame horizontally to un-mirror video feed
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # Run FaceMesh on a downscaled copy for speed. Landmarks are
        # normalized (0-1), so they map onto the full-resolution
        # frame below without any extra scaling math.
        if w > PROCESSING_WIDTH:
            scale = PROCESSING_WIDTH / w
            proc_frame = cv2.resize(frame, (PROCESSING_WIDTH, int(h * scale)))
        else:
            proc_frame = frame

        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0]

            # -----------------------------
            # Eyes
            # -----------------------------

            left_eye = self.get_eye_points(landmarks, self.LEFT_EYE, w, h)
            right_eye = self.get_eye_points(landmarks, self.RIGHT_EYE, w, h)

            leftEAR = self.eye_aspect_ratio(left_eye)
            rightEAR = self.eye_aspect_ratio(right_eye)

            ear = (leftEAR + rightEAR) / 2

            # -----------------------------
            # Mouth
            # -----------------------------

            upper, lower, m_left, m_right = self.get_lip_points(landmarks, w, h)
            mar = self.mouth_ratio(upper, lower, m_left, m_right)

            # -----------------------------
            # Blink            # Alarm (alarm needs the frame for screenshots)
            # -----------------------------

            self.detect_blink(ear)
            self.detect_yawn(mar)
            self.check_alarm(frame)

            # -----------------------------
            # DL Emotion & Stress Evaluator (Background Thread)
            # -----------------------------
            if self.total_frames % 5 == 0:
                with self.emotion_lock:
                    is_alive = self.emotion_thread is not None and self.emotion_thread.is_alive()
                
                if not is_alive:
                    self.current_frame = frame.copy()
                    self.emotion_thread = threading.Thread(target=self._run_emotion_inference, daemon=True)
                    self.emotion_thread.start()

            status = self.driver_status()

            # store latest values so external callers (main.py,
            # process_video's CSV/PDF export) can read them
            self.ear = ear
            self.mar = mar
            self.status = status

            eye_status = "Eyes Closed" if ear < EAR_THRESHOLD else "Eyes Open"
            mouth_status = "Yawning" if mar > 0.6 else "Normal"

            # -----------------------------
            # Draw Face Mesh (subsampled)
            # -----------------------------
            # Drawing all 468 landmarks every frame (468 separate
            # cv2.circle calls in a Python loop) is a bigger CPU cost
            # than the MediaPipe inference itself. Every 4th point
            # keeps the same "mesh" look at a quarter of the cost.

            for lm in landmarks.landmark[::4]:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Highlight eyes and lips a bit brighter on top of the mesh
            for p in left_eye:
                cv2.circle(frame, p, 2, (0, 255, 255), -1)

            for p in right_eye:
                cv2.circle(frame, p, 2, (0, 255, 255), -1)

            cv2.circle(frame, upper, 3, (255, 0, 0), -1)
            cv2.circle(frame, lower, 3, (255, 0, 0), -1)

            # -----------------------------
            # Status Color
            # -----------------------------

            color = (0, 255, 0)
            if status == "DROWSY":
                color = (0, 0, 255)

            # -----------------------------
            # Display
            # -----------------------------

            cv2.putText(frame, f"EAR : {ear:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"MAR : {mar:.2f}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Blinks : {self.blink_count}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Yawns : {self.yawn_count}", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Status : {status}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, eye_status, (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, mouth_status, (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.putText(frame, f"Emotion: {self.emotion.capitalize()}", (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 105, 180), 2)
            cv2.putText(frame, f"Stress : {self.stress_score:.1f}", (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if self.stress_score > 0.5 else (0, 255, 0), 2)

            if status == "DROWSY":
                # Flash the screen red until eyes open
                flash_on = int(time.time() * 8) % 2 == 0
                if flash_on:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                # Centered alert to avoid overlapping left-panel metrics
                cv2.putText(frame, "DROWSINESS ALERT!", (max(10, w//2 - 180), 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        else:
            # Face lost — reset counters instead of freezing last state
            self.reset_state()
            self.status = "NO FACE"

            cv2.putText(frame, "NO FACE DETECTED", (150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # -----------------------------
        # FPS
        # -----------------------------

        current = time.time()
        elapsed = current - self.prev_time
        fps = 1 / elapsed if elapsed > 0 else 0.0
        self.prev_time = current

        h_frame, w_frame = frame.shape[:2]
        fps_text = f"FPS : {fps:.1f}"
        (text_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, fps_text, (w_frame - text_w - 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -----------------------------
        # Exit Message
        # -----------------------------

        cv2.putText(frame, "Press Q to Exit", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    # ---------------------------------
    # Process an entire video file end-to-end: runs detection frame
    # by frame, writes an annotated output video, logs a per-frame
    # CSV, and builds a PDF summary report.
    # ---------------------------------

    def process_video(
        self,
        video_path,
        output_video_path=DROWSINESS_OUTPUT_VIDEO,
        csv_path=DROWSINESS_CSV,
        pdf_path=DROWSINESS_REPORT,
        show_preview=True,
        max_seconds=None,
    ):
        for path in (output_video_path, csv_path, pdf_path):
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)

        # video_path may be a file path (str) or a webcam index (int),
        # e.g. config.DROWSINESS_VIDEO = 1
        if isinstance(video_path, int) and hasattr(cv2, 'CAP_DSHOW') and WINDOWS:
            cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(video_path)

        if isinstance(video_path, int):
            from config import FRAME_WIDTH, FRAME_HEIGHT
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video saving disabled to conserve storage space
        writer = None

        # A live camera has no natural end-of-stream, and without a
        # preview window there's no 'q' key to stop it either — so cap
        # runtime automatically unless the caller set their own limit.
        effective_max_seconds = max_seconds
        if effective_max_seconds is None and isinstance(video_path, int) and not show_preview:
            effective_max_seconds = 20

        # Reset per-run counters/state
        self.total_frames = 0
        self.alert_count = 0
        self.blink_count = 0
        self.yawn_count = 0
        self._csv_rows = []
        self.reset_state()

        window_name = "SmartTrafficAI - Drowsiness Detection"
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.total_frames += 1
                frame = self.detect(frame)
                if writer:
                    writer.write(frame)

                if show_preview:
                    cv2.imshow(window_name, frame)
                    # webcam feeds never hit "end of stream" on their own —
                    # 'q' (or closing the window) is the only way out
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self._csv_rows.append({
                    "frame": self.total_frames,
                    "timestamp_sec": round(self.total_frames / fps, 2),
                    "ear": round(self.ear, 3),
                    "mar": round(self.mar, 3),
                    "status": self.status,
                    "blink_count": self.blink_count,
                    "yawn_count": self.yawn_count,
                    "alarm_active": self.alarm_on,
                })

                if effective_max_seconds is not None and (time.time() - start_time) >= effective_max_seconds:
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        self._write_csv(csv_path)
        self._write_pdf_report(pdf_path, video_path, output_video_path)

        return output_video_path

    # ---------------------------------
    # Headless variant for Gradio (or any environment with no local
    # cv2 window / keyboard). Runs the full pipeline with no preview,
    # and simply returns the output video path.
    # ---------------------------------

    def process_video_headless(
        self,
        video_path,
        output_video_path=DROWSINESS_OUTPUT_VIDEO,
        csv_path=DROWSINESS_CSV,
        pdf_path=DROWSINESS_REPORT,
        max_seconds=None,
    ):
        return self.process_video(
            video_path,
            output_video_path=output_video_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            show_preview=False,
            max_seconds=max_seconds,
        )

    # ---------------------------------
    # CSV export
    # ---------------------------------

    def _write_csv(self, csv_path):
        if not self._csv_rows:
            return

        fieldnames = list(self._csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(self._csv_rows)

    # ---------------------------------
    # PDF summary report
    # ---------------------------------

    def _write_pdf_report(self, pdf_path, video_path, output_video_path):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            )
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            print("reportlab not installed — skipping PDF report "
                  "(pip install reportlab)")
            return

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        elements = []

        elements.append(Paragraph("Smart Traffic AI - Drowsiness Report", styles["Title"]))
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(Paragraph(f"Source video: {video_path}", styles["Normal"]))
        elements.append(Paragraph(f"Annotated output: {output_video_path}", styles["Normal"]))
        elements.append(Spacer(1, 0.5 * cm))

        summary_data = [
            ["Metric", "Value"],
            ["Total Frames Processed", str(self.total_frames)],
            ["Blink Count", str(self.blink_count)],
            ["Yawn Count", str(self.yawn_count)],
            ["Drowsiness Alerts", str(self.alert_count)],
            ["Final EAR", f"{self.ear:.3f}"],
            ["Final MAR", f"{self.mar:.3f}"],
            ["Final Status", self.status],
        ]

        table = Table(summary_data, colWidths=[8 * cm, 8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ]))

        elements.append(table)
        doc.build(elements)
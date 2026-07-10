"""
====================================================
SmartTrafficAI Configuration File
====================================================
Edit paths and parameters here only.
"""

import os

# -------------------------------------------------
# Base Directory
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# Model
# -------------------------------------------------

YOLO_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n.pt"
)

# -------------------------------------------------
# Input Videos
# -------------------------------------------------

TRAFFIC_VIDEO = os.path.join(
    BASE_DIR,
    "input",
    "traffic.mp4"
)

DROWSINESS_VIDEO = 0

# Bundled sample clip for the Gradio "Use bundled sample video" option —
# kept separate from DROWSINESS_VIDEO, which is a live webcam index and
# has no natural end-of-stream (unsuitable for a one-shot demo/headless run).
DROWSINESS_SAMPLE_VIDEO = os.path.join(
    BASE_DIR,
    "input",
    "drowsiness_sample.mp4"
)

# -------------------------------------------------
# Output Directories
# -------------------------------------------------

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SCREENSHOT_DIR = os.path.join(
    OUTPUT_DIR,
    "screenshots"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

CSV_DIR = os.path.join(
    OUTPUT_DIR,
    "csv"
)

VIDEO_DIR = os.path.join(
    OUTPUT_DIR,
    "videos"
)

# Create folders automatically
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# -------------------------------------------------
# CSV Files
# -------------------------------------------------

TRAFFIC_CSV = os.path.join(
    CSV_DIR,
    "traffic.csv"
)

DROWSINESS_CSV = os.path.join(
    CSV_DIR,
    "drowsiness.csv"
)

# -------------------------------------------------
# PDF Reports
# -------------------------------------------------

TRAFFIC_REPORT = os.path.join(
    REPORT_DIR,
    "traffic_report.pdf"
)

DROWSINESS_REPORT = os.path.join(
    REPORT_DIR,
    "drowsiness_report.pdf"
)

# -------------------------------------------------
# Output Videos
# -------------------------------------------------

TRAFFIC_OUTPUT_VIDEO = os.path.join(
    VIDEO_DIR,
    "traffic_output.mp4"
)

DROWSINESS_OUTPUT_VIDEO = os.path.join(
    VIDEO_DIR,
    "drowsiness_output.mp4"
)

# -------------------------------------------------
# YOLO Parameters
# -------------------------------------------------

CONFIDENCE = 0.40

VEHICLE_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck"
]

# -------------------------------------------------
# Traffic Density Thresholds
# -------------------------------------------------

LOW_DENSITY = 5
MEDIUM_DENSITY = 15

# -------------------------------------------------
# Drowsiness Parameters
# -------------------------------------------------

EAR_THRESHOLD = 0.23

CLOSED_EYES_FRAMES = 20

# -------------------------------------------------
# Drawing
# -------------------------------------------------

FONT = 0

FONT_SCALE = 0.6

THICKNESS = 2

BOX_COLOR = (0, 255, 0)

TEXT_COLOR = (255, 255, 255)

WARNING_COLOR = (0, 0, 255)

BACKGROUND_COLOR = (0, 0, 0)

# -------------------------------------------------
# Recording
# -------------------------------------------------

FPS = 30

FRAME_WIDTH = 1280

FRAME_HEIGHT = 720

# -------------------------------------------------
# Streamlit
# -------------------------------------------------

PAGE_TITLE = "SmartTrafficAI"

PAGE_ICON = "🚦"

LAYOUT = "wide"
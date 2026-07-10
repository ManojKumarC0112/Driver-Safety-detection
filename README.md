# 🚗 Driver Drowsiness Detection System

> **Note:** This project was specifically developed and submitted as part of a comprehensive **Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) Training Program**. It demonstrates the real-world application of advanced computer vision pipelines, live deep-learning inference, and mathematical heuristics for robust video analysis.

An AI-powered real-time Driver Drowsiness Detection System built using **Python**, **OpenCV**, **MediaPipe FaceMesh**, and **Gradio**. 

## 📊 Core AI/ML Logic & Metrics
The foundation of this detection system relies on real-time spatial calculations derived from a deep-learning-based 468-point 3D facial mesh:
- **Eye Aspect Ratio (EAR):** Computes the Euclidean distance between precise horizontal and vertical eye landmarks to detect micro-sleeps, blinks, and prolonged eye closure (the primary fatigue indicator).
- **Mouth Aspect Ratio (MAR):** Analyzes the normalized, scale-invariant spatial distance of the driver's lips to algorithmically detect yawning (an early predictive drowsiness indicator).
- **Temporal Debouncing & State Buffering:** Employs temporal cooldown loops and moving frame-buffers (e.g. `MIN_BLINK_FRAMES`, `MIN_YAWN_FRAMES`) to significantly reduce false positive rates (FPR) and ensure high-accuracy ML alerting.

---

## 📌 Features

- 👁️ Real-time Eye Aspect Ratio (EAR) calculation
- 😮 Mouth Aspect Ratio (MAR) calculation
- 😴 Driver drowsiness detection
- 👀 Blink detection
- 🥱 Yawn detection
- 🔊 Automatic alert/alarm
- 📷 Screenshot capture during drowsiness
- 🎥 Video file and webcam support
- 📄 Automatic PDF report generation
- 📊 CSV logging
- 🌐 Interactive Gradio Web Interface
- ⚡ Real-time FPS display

---

## 🛠️ Technologies Used

- Python 3.10+
- OpenCV
- MediaPipe FaceMesh
- Gradio
- NumPy
- SciPy
- ReportLab
- Ultralytics (YOLOv8)

---

## 📂 Project Structure

```
Driver-Drowsiness-Detection-System/
│
├── models/
│   └── yolov8n.pt
│
├── modules/
│   ├── __init__.py
│   ├── camera.py
│   ├── drowsiness.py
│   ├── logger.py
│   ├── pdf_report.py
│   └── utils.py
│
├── output/
│   ├── csv/
│   ├── reports/
│   ├── screenshots/
│   └── videos/
│
├── app.py
├── main.py
├── config.py
├── test.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/karkiii23/Driver-Drowsiness-Detection-System.git

cd Driver-Drowsiness-Detection-System
```

### Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### Gradio Web Interface

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:7860
```

---

### Desktop Version

```bash
python main.py
```

---

## 📊 Outputs

The application automatically generates:

- Annotated output video
- CSV log
- PDF report
- Drowsiness screenshots

These are stored inside the **output/** folder.

---

## 🧠 Detection Workflow

1. Capture webcam/video frames
2. Detect facial landmarks using MediaPipe FaceMesh
3. Calculate Eye Aspect Ratio (EAR)
4. Calculate Mouth Aspect Ratio (MAR)
5. Detect blinks and yawns
6. Monitor prolonged eye closure
7. Trigger drowsiness alert
8. Save screenshot
9. Generate CSV and PDF report

---

## 📦 Requirements

```
opencv-python
mediapipe
numpy
scipy
gradio
ultralytics
reportlab
Pillow
```

Install using

```bash
pip install -r requirements.txt
```

---

## 📸 Screenshots

Add screenshots inside a `docs/` folder and display them here.

Example:

```
docs/
├── home.png
├── detection.png
├── alert.png
└── report.png
```

---

## 🚀 Future Improvements

- Email alert system
- SMS notification
- Driver identity recognition
- Cloud dashboard
- Mobile application
- Night vision support
- Face recognition
- Driver analytics
- GPS integration

---

## 👨‍💻 Author

**Shubham Singh Karki**

GitHub: https://github.com/karkiii23

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

```
⭐ Star the repository
🍴 Fork the project
## 📂 Project Structure

## 📂 Project Structure

```text
Driver-Drowsiness-Detection-System/
│
├── notebooks/
│   └── Drowsiness_Detection_Test.ipynb
│
├── models/
│   └── yolov8n.pt
│
├── modules/
│   ├── camera.py
│   ├── drowsiness.py
│   ├── logger.py
│   ├── pdf_report.py
│   └── utils.py
│
├── output/
│   ├── csv/
│   ├── reports/
│   ├── screenshots/
│   └── videos/
│
├── app.py
├── main.py
├── config.py
├── test.py
├── requirements.txt
├── README.md
└── LICENSE
```
## 📓 Jupyter Notebook

The project also includes a Jupyter Notebook for testing and experimenting with the drowsiness detection pipeline.

**Notebook Location**

```text
notebooks/Drowsiness_Detection_Test.ipynb
```

The notebook demonstrates:

- Loading the Driver Drowsiness Detection modules
- Testing webcam input
- Processing video files
- Eye Aspect Ratio (EAR) calculation
- Mouth Aspect Ratio (MAR) calculation
- Blink detection
- Yawn detection
- Drowsiness detection
- Visualizing real-time detection results

To run the notebook:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open:

```text
notebooks/Drowsiness_Detection_Test.ipynb
```

# Driver Safety AI: Full Project Context

This document is a compact but detailed handoff for an AI assistant that has not seen the project before.
It explains what the system is, how the data was collected, what models were used, why those choices were made, what the reported metrics are, and what caveats matter.

## 1. What the project is

Driver Safety AI is a real-time driver drowsiness, yawning, and distraction monitoring system built around webcam-based facial analysis.

The project has two closely related layers:

1. A proposed deep learning experiment that uses 12 temporal telemetry features over 30-frame windows to classify driver state.
2. A runtime safety application that turns those predictions into live alerts, a DVI risk score, a dashboard, session summaries, and a Smart Dhaba assistant.

There is also an older / comparison baseline path based on UTA-RLDD that is kept for academic contrast.

## 2. Main entry points

- `main.py`: launches the OpenCV-based real-time monitor.
- `web_app.py`: launches the FastAPI dashboard and webcam telemetry stream.
- `src/inference/realtime.py`: the core real-time inference loop for the OpenCV app.
- `src/web/index.html` and `src/web/app.js`: frontend for the web dashboard.

## 3. What data the project uses

The repo contains two data tracks:

### A. Proposed 12-feature telemetry dataset

This is the project-specific dataset and the main training path for the proposed deep model.

- Stored in `data/raw/proposed_telemetry/`
- Collected from webcam sessions with labeled ground truth
- Each session records structured telemetry, not raw video by default
- Labels are:
  - `ALERT`
  - `DROWSY`

The collection scripts show that each session captures:

- `participant_id`
- `session_id`
- `label`
- timestamped telemetry rows

Current files in `data/raw/proposed_telemetry/` show three participants:

- `P001`
- `P002`
- `P003`

Each participant has:

- one `ALERT` session
- one `DROWSY` session

The collection scripts indicate:

- 30-second recording sessions in the guided collector
- webcam capture at 30 FPS
- no raw video stored by default
- output is tabular telemetry CSV

### B. UTA-RLDD baseline path

The repo also includes a baseline/audit path for UTA-RLDD:

- `src/data/audit_uta_baseline.py`
- `src/training/train_baseline_uta.py`
- `src/training/diagnose_uta_baseline.py`

This is for comparison and failure analysis, not the main proposed runtime model.

## 4. What features are used

The proposed system uses 12 temporal features per frame:

1. `EAR_LEFT`
2. `EAR_RIGHT`
3. `MEAN_EAR`
4. `MAR`
5. `YAW`
6. `PITCH`
7. `ROLL`
8. `PERCLOS`
9. `BLINK_RATE`
10. `EYE_CLOSURE_DURATION`
11. `MOUTH_OPEN_DURATION`
12. `HEAD_MOTION_MAGNITUDE`

These are produced by `src/features/temporal_features.py` using:

- MediaPipe facial landmarks
- eye aspect ratio
- mouth aspect ratio
- head pose estimation
- rolling blink/closure statistics
- motion magnitude over time

## 5. Why these features were chosen

The codebase is built around the idea that drowsiness is not a single-frame event.
It is a temporal behavior.

The features were chosen because they capture the physical cues that usually precede or accompany fatigue:

- eye closure
- blink frequency
- prolonged closure
- yawning
- head movement / nodding
- gaze deviation
- face stability over a short time window

This is why the project does not rely on a raw image classifier alone.

## 6. Model architecture

The proposed model is `DriverSafetyNet` in `src/model/driver_safety_net.py`.

Architecture:

- input shape: `(batch, 30, 12)`
- 1D convolution over the temporal axis
- 2-layer bidirectional LSTM
- dropout
- final dense classifier

Concretely:

- `Conv1d(in_channels=12, out_channels=32, kernel_size=3, padding=1)`
- `BiLSTM(hidden_size=64, num_layers=2, bidirectional=True)`
- `Dropout(0.3)`
- `Linear(128, num_classes)`

Output classes in the proposed experiment:

- `ALERT`
- `DROWSY`

The web/runtime layer still supports a broader 4-class runtime state representation:

- `ALERT`
- `DROWSY`
- `YAWNING`
- `DISTRACTED`

That 4-class runtime is used for monitoring and HUD logic, but the trained proposed checkpoint in the config/eval flow is the 2-class model.

## 7. Why this algorithm was chosen

The code strongly suggests the model choice was made for these reasons:

### Why 1D CNN

- The input is already compact structured telemetry, not pixels.
- A 1D CNN is efficient for learning local temporal patterns across a short sequence.
- It is lighter than a 2D/3D video CNN and much more practical for real-time CPU-friendly inference.

### Why BiLSTM

- Drowsiness has sequence context.
- A BiLSTM can model patterns that unfold over several frames, not just the current frame.
- It helps the network understand persistence, buildup, and recovery.

### Why not only a rule-based system

The repo includes a rule-based baseline (`RuleBasedBaselineModel`) using EAR/MAR/head pose thresholds.
That baseline is useful for comparison and fallback, but it is brittle:

- it depends on hand-tuned thresholds
- it is sensitive to individual variation
- it can miss temporal persistence
- it does not learn from examples

### Why not a pure single-frame MLP

The pilot analysis includes a 1-frame MLP comparison.
The single-frame model is simpler, but it does not capture temporal dynamics as well as a sequence model in principle.

Important caveat:

The project’s own pilot analysis shows that, on the initial small dataset, a single-feature or single-frame model can look surprisingly strong because the dataset contains very static drowsy poses.
So the architecture choice is still sound, but the dataset quality strongly influences the apparent gain from temporal modeling.

## 8. How training data was prepared

The training pipeline is in `src/training/train.py`, `src/data/dataset_builder.py`, and `src/data/dataset.py`.

Key points:

- sequences are built from sliding 30-frame windows
- stride is 1
- each sequence is `(30, 12)`
- scaler is fit only on the training split
- participant-level isolation is enforced to avoid leakage
- class weights are used in `CrossEntropyLoss`
- training uses `AdamW`
- gradient clipping is enabled for LSTM stability
- early stopping is used

The dataset builder writes:

- `train_x.npy`
- `train_y.npy`
- `val_x.npy`
- `val_y.npy`
- `test_x.npy`
- `test_y.npy`
- `split_manifest.json`

The split manifest explicitly records that participant overlap is checked and rejected.

## 9. Training configuration

From `configs/proposed_12feature.yaml`:

- `window_size`: 30
- `num_features`: 12
- `num_classes`: 2
- `learning_rate`: 0.001
- `weight_decay`: 0.0001
- `batch_size`: 32
- `epochs`: 50
- `early_stopping_patience`: 10
- `dropout`: 0.3

The model checkpoint stores:

- model state dict
- class names
- feature names
- window size
- model config
- scaler path
- best epoch
- training curves
- timestamp

## 10. Reported accuracy and metrics

The project has multiple metric sources, and they should not be mixed casually.

### A. Main proposed test metrics

From `outputs/metrics/proposed_test_metrics.json`:

- Accuracy: `0.9477911646586346` = `94.78%`
- Macro precision: `95.27%`
- Macro recall: `94.78%`
- Macro F1: `94.77%`
- Weighted F1: `94.77%`
- Total test samples: `1743`

Per-class:

- `ALERT`
  - precision: `90.54%`
  - recall: `100.00%`
  - F1: `95.04%`
  - support: `871`
- `DROWSY`
  - precision: `100.00%`
  - recall: `89.56%`
  - F1: `94.49%`
  - support: `872`

Confusion matrix:

- `ALERT` predicted as `ALERT`: `871`
- `ALERT` predicted as `DROWSY`: `0`
- `DROWSY` predicted as `ALERT`: `91`
- `DROWSY` predicted as `DROWSY`: `781`

### B. Pilot / ablation analysis

From `outputs/metrics/proposed_pilot_analysis.md`:

- the initial pilot checkpoint reached about `89.67%` validation accuracy at epoch 1
- the same report warns that training can overfit quickly on the small participant set
- the report explicitly says not to claim the pilot metrics as final project accuracy

The same report shows feature ablations:

- all 12 features: `94.78%`
- eye-temporal-only: `97.42%`
- EAR + MAR only: `98.51%`
- MEAN_EAR only: `99.48%`
- MEAN_EAR single-frame MLP: `98.16%`

Interpretation:

The initial dataset is highly dominated by static eye-closure signals, so `MEAN_EAR` is extremely predictive.
That means the pilot data is useful, but it is not a perfect benchmark for real-world temporal drowsiness complexity.

### C. Baseline UTA-RLDD results

The baseline analysis files show lower performance on the imported UTA-RLDD path.

Examples from the saved reports:

- baseline_results.json: `36.25%` test accuracy
- failure analysis report:
  - majority-class baseline: `43.10%`
  - experiment A: `38.22%`
  - experiment C: `41.45%`

These numbers are for the baseline UTA-RLDD branch, not the proposed telemetry model.

## 11. What dataset was actually trained

Short answer:

- The proposed deep model was trained on your own collected 12-feature telemetry dataset under `data/raw/proposed_telemetry/`.
- The repo also keeps a separate UTA-RLDD baseline branch for comparison.

The strongest evidence for this is:

- the collector scripts that record your own participant/session telemetry
- the raw files named `P001`, `P002`, `P003`
- the participant-isolated split manifest
- the proposed test metrics file and pilot analysis report

So if a judge asks whether this is your own dataset:

Yes, the main proposed model is trained on project-collected telemetry sessions.
The codebase also uses UTA-RLDD as an academic baseline comparison, but that is separate.

## 12. Data collection protocol

Two collection scripts exist:

### `src/data/collector.py`

This is the guided collector:

- asks for participant ID
- asks for session ID
- asks for label (`ALERT` or `DROWSY`)
- records for a fixed duration
- validates features
- stores CSV + JSON metadata
- does not store raw video by default

### `src/data/collect_data.py`

This is a simpler webcam telemetry recorder:

- takes `--label`
- takes `--subject_id`
- captures telemetry rows to CSV
- intended for populating the proposed dataset folder

## 13. Validation and split policy

The project is careful about data leakage:

- participants are split at the subject level
- the scaler is fit only on the training split
- validation and test are transformed with the train scaler only
- NaN and Inf sequences are filtered out

The pilot analysis explicitly says:

- Train participant: `P003`
- Validation participant: `P001`
- Test participant: `P002`

This is a very important detail if you are defending the project, because it shows the system was evaluated across subjects instead of memorizing one person.

## 14. Runtime pipeline

At runtime, the OpenCV/web inference loop does the following:

1. capture a webcam frame
2. extract facial landmarks
3. compute the 12-dim telemetry vector
4. maintain a 30-frame sequence
5. scale features with the saved StandardScaler
6. run the CNN-BiLSTM model
7. smooth predictions
8. pass probabilities into the state manager
9. trigger voice/audio/visual alerts if needed
10. log telemetry and build session summaries

The realtime app uses a state manager to avoid noisy one-frame false alarms.

## 15. Safety / alert logic

The state manager in `src/inference/state_manager.py` handles:

- `ALERT`
- `SUSPECTED_DROWSY`
- `CONFIRMED_DROWSY`
- `PERSISTENT_DROWSY`
- `RECOVERING`

It uses:

- confirmation duration
- recovery duration
- voice cooldown
- post-recovery cooldown
- blink protection
- response monitoring

Why it exists:

Even a good classifier can be noisy frame-by-frame.
This logic makes the alerts more stable and realistic.

## 16. DVI score

The project computes a Driver Vigilance Index (DVI) as a normalized risk score.

It combines:

- `1 - P_ALERT`
- PERCLOS
- eye-closure duration
- yaw deviation

The DVI is mainly for visualization and risk communication.
It is not an automotive-certified safety measure.

## 17. Smart Dhaba Assistant

The Smart Dhaba Assistant is a web/dashboard feature that suggests nearby rest stops.

Earlier it was mostly hardcoded.
The current refactor makes it more of a recommendation engine:

- fetches nearby places from OpenStreetMap Overpass when available
- falls back to safe predefined rest stops if live lookup fails
- scores places by distance and amenities
- returns a spoken summary
- supports English / Hindi / Hinglish prompts

The reason this feature exists:

- if the driver is drowsy, the system should not only warn them
- it should also help them find a practical safe stop

## 18. Voice system

The voice stack is multi-layered:

- local browser speech synthesis for the dashboard
- Windows SAPI / pyttsx3 fallback on the Python side
- optional Sarvam TTS integration for natural English / Hindi / Hinglish speech

Sarvam is a good fit if you want better-sounding multilingual voice prompts.

From the code/docs:

- Sarvam Bulbul v3 supports `en-IN`, `hi-IN`, and other Indian languages
- the API returns base64 audio
- it is used here as an optional service, not a hard dependency

If `SARVAM_API_KEY` is absent, the system falls back to browser speech.

## 19. What the frontend shows

The web dashboard shows:

- live webcam feed
- current driver state
- DVI risk gauge
- EAR chart
- emergency overlay
- nearby dhaba suggestions
- session export controls
- language selector for voice prompts
- test alarm button

## 20. What a judge should hear if they ask “why is this deep learning?”

The best answer is:

This is deep learning because the system learns from 30-frame temporal sequences of facial telemetry, not just from hand-written thresholds. The CNN learns local temporal patterns and the BiLSTM learns longer dependencies in the sequence. That lets the model learn the progression of drowsiness over time, which a single-frame classifier or pure heuristic system cannot capture as well.

## 21. What you should be careful not to overclaim

Do not claim that the 94.78% test accuracy is proof of real-world robustness.

Why:

- the pilot dataset is small
- it currently has only 3 participants
- the pilot analysis shows very static drowsy pose patterns
- some ablation results are extremely high because `MEAN_EAR` is dominant in the current dataset

So the correct framing is:

- the system works and has strong measured results on the project’s held-out test split
- but it is still an academic prototype, not a production-certified driver safety device

## 22. Bottom line summary

If an AI assistant needs one clean summary:

Driver Safety AI is a real-time webcam-based driver monitoring system that extracts 12 facial-temporal features, builds 30-frame sequences, and classifies them with a 1D CNN + 2-layer BiLSTM. The main proposed model was trained on a project-collected telemetry dataset from participants P001-P003, with strict participant-level splits and scaler isolation. The reported held-out test accuracy is 94.78% with 94.77% macro F1 on 1743 test samples. The system wraps the classifier in a temporal safety state machine, DVI scoring, live audio/visual alerts, session reporting, and a Smart Dhaba assistant. A separate UTA-RLDD baseline branch is included for comparison and performs much worse than the proposed telemetry model.


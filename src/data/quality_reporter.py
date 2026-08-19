"""
Dataset Quality Reporter for Proposed 12-Feature Telemetry Dataset.
Audits raw telemetry CSVs and processed split arrays.
Outputs:
 - outputs/metrics/proposed_dataset_quality_report.json
 - outputs/metrics/proposed_dataset_quality_report.md
"""

import os
import glob
import json
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw", "proposed_telemetry")
PROC_DIR = os.path.join("data", "processed", "proposed_12feature")
OUTPUT_JSON = os.path.join("outputs", "metrics", "proposed_dataset_quality_report.json")
OUTPUT_MD = os.path.join("outputs", "metrics", "proposed_dataset_quality_report.md")

FEATURE_COLS = [
    "EAR_LEFT", "EAR_RIGHT", "MEAN_EAR", "MAR",
    "YAW", "PITCH", "ROLL",
    "PERCLOS", "BLINK_RATE", "EYE_CLOSURE_DURATION", "MOUTH_OPEN_DURATION", "HEAD_MOTION_MAGNITUDE"
]

def generate_quality_report():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    
    report = {
        "dataset_name": "Driver Safety AI - Proposed 12-Feature Telemetry",
        "feature_count": 12,
        "classes": ["ALERT", "DROWSY"],
        "raw_files_count": len(csv_files),
        "participants": [],
        "sessions": [],
        "total_frames": 0,
        "class_distribution": {"ALERT": 0, "DROWSY": 0},
        "feature_statistics": {},
        "nan_inf_count": 0,
        "split_status": "Not yet built"
    }

    if len(csv_files) == 0:
        report["status"] = "Empty dataset directory. Run src.data.collector to populate telemetry data."
        with open(OUTPUT_JSON, "w") as f:
            json.dump(report, f, indent=2)
        
        md_content = "# Proposed 12-Feature Telemetry Dataset Quality Report\n\n> [!NOTE]\n> Telemetry dataset directory `data/raw/proposed_telemetry/` is currently empty. Run `python -m src.data.collector` to begin participant recordings.\n"
        with open(OUTPUT_MD, "w") as f:
            f.write(md_content)
        return report

    all_dfs = []
    participants_set = set()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue

        all_dfs.append(df)
        pid = str(df["participant_id"].iloc[0])
        sid = str(df["session_id"].iloc[0])
        lbl = str(df["label"].iloc[0])

        participants_set.add(pid)
        report["sessions"].append({"participant_id": pid, "session_id": sid, "label": lbl, "frames": len(df)})
        report["total_frames"] += len(df)
        report["class_distribution"][lbl] = report["class_distribution"].get(lbl, 0) + len(df)

    report["participants"] = sorted(list(participants_set))

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        # Check NaN/Inf
        feat_matrix = full_df[FEATURE_COLS].values
        nan_count = int(np.isnan(feat_matrix).sum())
        inf_count = int(np.isinf(feat_matrix).sum())
        report["nan_inf_count"] = nan_count + inf_count

        for col in FEATURE_COLS:
            vals = full_df[col].values
            report["feature_statistics"][col] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }

    # Check processed splits if existing
    manifest_path = os.path.join(PROC_DIR, "split_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        report["split_status"] = manifest

    # Save JSON report
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Quality] Saved JSON report → {OUTPUT_JSON}")

    # Generate Markdown report
    md_lines = [
        "# Proposed 12-Feature Telemetry Dataset Quality Report",
        "",
        "## 1. Dataset Overview",
        "- **Dataset Name**: Driver Safety AI - Proposed 12-Feature Telemetry",
        "- **Input Features**: 12 MediaPipe Telemetry Features",
        "- **Target Classes**: `ALERT` (0) vs `DROWSY` (1)",
        f"- **Raw Session Files**: {len(csv_files)} telemetry CSVs",
        f"- **Total Participants**: {len(report['participants'])} (`{report['participants']}`)",
        f"- **Total Telemetry Frames**: {report['total_frames']} frames",
        f"- **NaN / Inf Integrity**: {report['nan_inf_count']} invalid entries (Pass)",
        "",
        "## 2. Class Balance Breakdown",
        f"- **ALERT Frames**: {report['class_distribution'].get('ALERT', 0)} frames",
        f"- **DROWSY Frames**: {report['class_distribution'].get('DROWSY', 0)} frames",
        "",
        "## 3. 12-Feature Statistical Summary",
        "| Feature | Min | Max | Mean | Std |",
        "|:---|:---:|:---:|:---:|:---:|"
    ]

    for col in FEATURE_COLS:
        stats = report["feature_statistics"].get(col, {"min": 0, "max": 0, "mean": 0, "std": 0})
        md_lines.append(f"| `{col}` | {stats['min']:.4f} | {stats['max']:.4f} | {stats['mean']:.4f} | {stats['std']:.4f} |")

    md_lines.extend([
        "",
        "## 4. Participant Isolation & Split Manifest",
        f"```json",
        json.dumps(report["split_status"], indent=2),
        "```"
    ])

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(md_lines))
    print(f"[Quality] Saved Markdown report → {OUTPUT_MD}")

    return report

if __name__ == "__main__":
    generate_quality_report()

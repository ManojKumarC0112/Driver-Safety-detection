"""
Dataset Builder & Sequence Generator for Proposed 12-Feature Telemetry Dataset.
Loads telemetry CSV files from data/raw/proposed_telemetry/, extracts 30-frame sliding sequences,
and performs STRICT participant-isolated splitting (Train/Val/Test).
Outputs: data/processed/proposed_12feature/
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = os.path.join("configs", "proposed_12feature.yaml")
FEATURE_COLS = [
    "EAR_LEFT", "EAR_RIGHT", "MEAN_EAR", "MAR",
    "YAW", "PITCH", "ROLL",
    "PERCLOS", "BLINK_RATE", "EYE_CLOSURE_DURATION", "MOUTH_OPEN_DURATION", "HEAD_MOTION_MAGNITUDE"
]
LABEL_MAP = {"ALERT": 0, "DROWSY": 1}

def validate_participant_splits(train_pids, val_pids, test_pids):
    """
    Hard assertion ensuring that a participant ID exists in EXACTLY ONE split.
    Raises ValueError if any participant overlap occurs across splits.
    """
    set_tr = set(train_pids)
    set_val = set(val_pids)
    set_te = set(test_pids)

    inter_tr_val = set_tr & set_val
    inter_tr_te = set_tr & set_te
    inter_val_te = set_val & set_te

    if inter_tr_val or inter_tr_te or inter_val_te:
        raise ValueError(
            "Participant-level isolation violation detected! "
            f"Train/Val overlap: {list(inter_tr_val)}, "
            f"Train/Test overlap: {list(inter_tr_te)}, "
            f"Val/Test overlap: {list(inter_val_te)}"
        )

def create_sequences_from_df(df, window_size=30, stride=1):
    """Creates sliding sequences of shape (N_seq, 30, 12) from a single session DataFrame."""
    feature_matrix = df[FEATURE_COLS].values
    n_frames = len(feature_matrix)
    
    if n_frames < window_size:
        return np.array([]), np.array([])

    seqs = []
    labels = []
    
    # Ground truth is constant per session (from session metadata / label column)
    session_label_str = df["label"].iloc[0]
    session_label_id = LABEL_MAP[session_label_str]

    for start_idx in range(0, n_frames - window_size + 1, stride):
        window = feature_matrix[start_idx : start_idx + window_size]
        seqs.append(window)
        labels.append(session_label_id)

    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)

def build_proposed_dataset(raw_dir=None, proc_dir=None, min_participants=None):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    if raw_dir is None:
        raw_dir = config["collection"]["output_dir"]
    if proc_dir is None:
        proc_dir = os.path.join("data", "processed", "proposed_12feature")
    if min_participants is None:
        min_participants = config.get("validation", {}).get("min_participants", 3)

    os.makedirs(proc_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    print("\n==================================================================")
    print("   PROPOSED 12-FEATURE DATASET BUILDER & SEQUENCE GENERATOR        ")
    print("==================================================================")
    print(f" Raw Telemetry Directory : {raw_dir}")
    print(f" Output Directory        : {proc_dir}")
    print(f" Telemetry CSVs Found    : {len(csv_files)}")
    print(f" Minimum Participants Req: {min_participants}")
    print("==================================================================")

    if len(csv_files) == 0:
        msg = f"No raw telemetry CSV files found in '{raw_dir}'."
        print(f"\n[Notice] {msg}")
        return None

    # Load and organize sessions by participant_id
    participant_data = {}
    total_valid_frames = 0

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty or len(df) < 30:
            print(f"[Skip] {os.path.basename(csv_file)} has insufficient rows (<30).")
            continue

        part_id = str(df["participant_id"].iloc[0])
        seqs, labels = create_sequences_from_df(df, window_size=30, stride=1)
        
        if len(seqs) == 0:
            continue

        if part_id not in participant_data:
            participant_data[part_id] = {"X": [], "y": [], "sessions": []}

        participant_data[part_id]["X"].append(seqs)
        participant_data[part_id]["y"].append(labels)
        participant_data[part_id]["sessions"].append(os.path.basename(csv_file))
        total_valid_frames += len(df)

    participants = sorted(list(participant_data.keys()))
    print(f"\n[Validation] Total Participants Collected: {len(participants)} ({participants})")
    print(f"[Validation] Total Valid Frames         : {total_valid_frames}")

    # Check minimum participant requirement
    if len(participants) < min_participants:
        err_msg = (
            "Insufficient participants for participant-isolated train/validation/test split. "
            f"Found {len(participants)} participant(s), but at least {min_participants} are required."
        )
        print(f"\n[ERROR] {err_msg}")
        raise ValueError(err_msg)

    # Concat sequences per participant
    for pid in participants:
        participant_data[pid]["X"] = np.concatenate(participant_data[pid]["X"], axis=0)
        participant_data[pid]["y"] = np.concatenate(participant_data[pid]["y"], axis=0)
        print(f" - Participant {pid}: {len(participant_data[pid]['X'])} sequences (Sessions: {participant_data[pid]['sessions']})")

    # Perform strict participant-level splitting BEFORE sequence assignment
    np.random.seed(42)
    shuffled_pids = list(np.random.permutation(participants))

    # Assign participants to splits
    val_pids = [shuffled_pids[0]]
    test_pids = [shuffled_pids[1]]
    train_pids = shuffled_pids[2:]

    # HARD ASSERTION: Zero overlap between splits
    validate_participant_splits(train_pids, val_pids, test_pids)

    def gather_split(pids):
        x_list = [participant_data[p]["X"] for p in pids]
        y_list = [participant_data[p]["y"] for p in pids]
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

    tr_x, tr_y = gather_split(train_pids)
    val_x, val_y = gather_split(val_pids)
    te_x, te_y = gather_split(test_pids)

    # Save array files
    np.save(os.path.join(proc_dir, "train_x.npy"), tr_x)
    np.save(os.path.join(proc_dir, "train_y.npy"), tr_y)
    np.save(os.path.join(proc_dir, "val_x.npy"), val_x)
    np.save(os.path.join(proc_dir, "val_y.npy"), val_y)
    np.save(os.path.join(proc_dir, "test_x.npy"), te_x)
    np.save(os.path.join(proc_dir, "test_y.npy"), te_y)

    manifest = {
        "train_participants": list(train_pids),
        "validation_participants": list(val_pids),
        "test_participants": list(test_pids),
        "train_sequence_count": len(tr_x),
        "validation_sequence_count": len(val_x),
        "test_sequence_count": len(te_x),
        "feature_dimensions": [30, 12],
        "zero_leakage_verified": True
    }

    manifest_path = os.path.join(proc_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Builder] Successfully built participant-isolated dataset splits!")
    print(f" Train Split: {tr_x.shape} (Participants: {train_pids})")
    print(f" Val Split  : {val_x.shape} (Participants: {val_pids})")
    print(f" Test Split : {te_x.shape} (Participants: {test_pids})")
    print(f"[Saved] Split manifest → {manifest_path}")

    return manifest

if __name__ == "__main__":
    build_proposed_dataset()

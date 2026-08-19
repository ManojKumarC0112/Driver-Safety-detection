"""
Diagnostic Script for UTA-RLDD Baseline Failure Analysis.
Evaluates:
 - Majority Class Baseline & Stratified Random Baseline
 - Exp A: Current Setup (CrossEntropy + StandardScaler)
 - Exp B: Class-Weighted CrossEntropy
 - Exp C: Original 2-Step Preprocessing + CrossEntropy
 - Exp D: Original Regression Formulation (Continuous [0, 10] output + MSE Loss + 3.34 Binning)
Saves confusion matrix plots and generates a comprehensive failure analysis report.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from src.model.driver_safety_net import DriverSafetyNet

DATA_DIR = os.path.join("data", "raw", "uta_rldd")
OUTPUT_DIR = os.path.join("outputs", "metrics", "failure_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP_CAT = {0.0: 0, 5.0: 1, 10.0: 2}
CLASS_NAMES = ["Alert (0.0)", "Low Vigilance (5.0)", "Drowsy (10.0)"]

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(fold: int = 1):
    train_x = np.load(os.path.join(DATA_DIR, f"Blinks_30_Fold{fold}.npy"))
    train_y_raw = np.load(os.path.join(DATA_DIR, f"Labels_30_Fold{fold}.npy")).flatten()
    test_x = np.load(os.path.join(DATA_DIR, f"BlinksTest_30_Fold{fold}.npy"))
    test_y_raw = np.load(os.path.join(DATA_DIR, f"LabelsTest_30_Fold{fold}.npy")).flatten()
    return train_x, train_y_raw, test_x, test_y_raw

def apply_original_normalization(train_x, test_x):
    """Applies the exact 2-step normalization from Training.py lines 420-431."""
    tr_norm = np.copy(train_x)
    te_norm = np.copy(test_x)
    for f in range(4):
        u_f = np.mean(tr_norm[:, :, f])
        std_f = np.std(tr_norm[:, :, f])
        if std_f == 0:
            std_f = 1e-7
        tr_norm[:, :, f] = (tr_norm[:, :, f] - u_f) / std_f
        te_norm[:, :, f] = (te_norm[:, :, f] - u_f) / std_f
    return tr_norm, te_norm

def plot_and_save_cm(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

# --- Regression Model Wrapper (Exp D) ---
class DriverSafetyNetRegression(nn.Module):
    """Adaptation of DriverSafetyNet for Continuous Drowsiness Regression in [0, 10]."""
    def __init__(self, in_channels=4, cnn_filters=32, lstm_hidden_size=64, dropout=0.3):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 30, 4) -> (B, 4, 30)
        x_cnn = self.relu(self.conv1d(x.permute(0, 2, 1)))
        lstm_out, _ = self.lstm(x_cnn.permute(0, 2, 1))
        last_step = lstm_out[:, -1, :]
        raw_out = self.fc(self.dropout(last_step))
        # 10 * Sigmoid output to map continuously to [0, 10]
        return 10.0 * torch.sigmoid(raw_out)

def run_diagnostics():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diagnostics] Running failure analysis on {device}...")

    train_x, train_y_raw, test_x, test_y_raw = load_data(fold=1)

    train_y_cat = np.array([LABEL_MAP_CAT[v] for v in train_y_raw], dtype=np.int64)
    test_y_cat = np.array([LABEL_MAP_CAT[v] for v in test_y_raw], dtype=np.int64)

    # 1. Calculate Majority Class Baseline
    test_counts = np.bincount(test_y_cat)
    maj_class = np.argmax(test_counts)
    maj_acc = test_counts[maj_class] / len(test_y_cat)
    
    # 2. Stratified Random Baseline
    tr_counts = np.bincount(train_y_cat) / len(train_y_cat)
    te_props = test_counts / len(test_y_cat)
    strat_random_acc = np.sum(tr_counts * te_props)

    print(f"\n--- BASELINE REFERENCE BENCHMARKS ---")
    print(f" Majority-Class Baseline Accuracy: {maj_acc*100:.2f}% (Class: {CLASS_NAMES[maj_class]})")
    print(f" Stratified Random Baseline Acc  : {strat_random_acc*100:.2f}%")
    print(f" Uniform Random Baseline Acc     : 33.33%")

    results = {
        "benchmarks": {
            "majority_class_accuracy": float(maj_acc),
            "stratified_random_accuracy": float(strat_random_acc),
            "uniform_random_accuracy": 0.3333333333333333
        },
        "experiments": {}
    }

    # Prepare datasets for Experiments A, B, C, D
    # Exp A & B: StandardScaler
    N_tr, T, F_dim = train_x.shape
    N_te, _, _ = test_x.shape
    scaler = StandardScaler()
    tr_sc = scaler.fit_transform(train_x.reshape(-1, F_dim)).reshape(N_tr, T, F_dim)
    te_sc = scaler.transform(test_x.reshape(-1, F_dim)).reshape(N_te, T, F_dim)

    # Exp C & D: Original 2-step Normalization
    tr_orig, te_orig = apply_original_normalization(train_x, test_x)

    def train_classifier(tr_x, tr_y, te_x, te_y, use_weights=False, epochs=15):
        tr_ds = TensorDataset(torch.tensor(tr_x, dtype=torch.float32), torch.tensor(tr_y, dtype=torch.long))
        te_ds = TensorDataset(torch.tensor(te_x, dtype=torch.float32), torch.tensor(te_y, dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

        model = DriverSafetyNet(in_channels=4, num_classes=3).to(device)
        
        if use_weights:
            class_counts = np.bincount(tr_y)
            weights = 1.0 / class_counts
            weights = weights / weights.sum()
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in te_loader:
                bx = bx.to(device)
                logits = model(bx)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        
        preds = np.array(preds)
        acc = float(accuracy_score(te_y, preds))
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(te_y, preds, average="macro")
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(te_y, preds, average="weighted")
        per_prec, per_rec, per_f1, per_supp = precision_recall_fscore_support(te_y, preds, average=None)
        cm = confusion_matrix(te_y, preds)

        return acc, float(macro_prec), float(macro_rec), float(macro_f1), float(w_f1), per_prec, per_rec, per_f1, per_supp, cm

    # --- Exp A: Current Setup ---
    print("\n[Exp A] Running Current Setup (CrossEntropy + StandardScaler)...")
    accA, mpA, mrA, mf1A, wf1A, ppA, prA, pf1A, psA, cmA = train_classifier(tr_sc, train_y_cat, te_sc, test_y_cat, use_weights=False)
    plot_and_save_cm(cmA, f"Exp A: Standard CrossEntropy (Acc: {accA*100:.2f}%)", "cm_exp_A.png")
    results["experiments"]["Exp_A_Current_Setup"] = {
        "accuracy": accA, "macro_f1": mf1A, "weighted_f1": wf1A,
        "per_class": {c: {"precision": float(ppA[i]), "recall": float(prA[i]), "f1": float(pf1A[i]), "support": int(psA[i])} for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": cmA.tolist()
    }

    # --- Exp B: Class-Weighted CrossEntropy ---
    print("\n[Exp B] Running Class-Weighted CrossEntropy...")
    accB, mpB, mrB, mf1B, wf1B, ppB, prB, pf1B, psB, cmB = train_classifier(tr_sc, train_y_cat, te_sc, test_y_cat, use_weights=True)
    plot_and_save_cm(cmB, f"Exp B: Weighted CrossEntropy (Acc: {accB*100:.2f}%)", "cm_exp_B.png")
    results["experiments"]["Exp_B_Class_Weighted"] = {
        "accuracy": accB, "macro_f1": mf1B, "weighted_f1": wf1B,
        "per_class": {c: {"precision": float(ppB[i]), "recall": float(prB[i]), "f1": float(pf1B[i]), "support": int(psB[i])} for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": cmB.tolist()
    }

    # --- Exp C: Original Normalization + CrossEntropy ---
    print("\n[Exp C] Running Original 2-Step Normalization + CrossEntropy...")
    accC, mpC, mrC, mf1C, wf1C, ppC, prC, pf1C, psC, cmC = train_classifier(tr_orig, train_y_cat, te_orig, test_y_cat, use_weights=False)
    plot_and_save_cm(cmC, f"Exp C: Orig Normalization + CE (Acc: {accC*100:.2f}%)", "cm_exp_C.png")
    results["experiments"]["Exp_C_Original_Norm_CE"] = {
        "accuracy": accC, "macro_f1": mf1C, "weighted_f1": wf1C,
        "per_class": {c: {"precision": float(ppC[i]), "recall": float(prC[i]), "f1": float(pf1C[i]), "support": int(psC[i])} for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": cmC.tolist()
    }

    # --- Exp D: Original Regression Formulation (Continuous [0, 10] Output + MSE Loss + 3.34 Binning) ---
    print("\n[Exp D] Running Original Regression Formulation (MSE Loss + 3.34 Binning)...")
    tr_dsD = TensorDataset(torch.tensor(tr_orig, dtype=torch.float32), torch.tensor(train_y_raw, dtype=torch.float32).unsqueeze(1))
    te_dsD = TensorDataset(torch.tensor(te_orig, dtype=torch.float32), torch.tensor(test_y_raw, dtype=torch.float32).unsqueeze(1))
    tr_loaderD = DataLoader(tr_dsD, batch_size=64, shuffle=True)
    te_loaderD = DataLoader(te_dsD, batch_size=64, shuffle=False)

    reg_model = DriverSafetyNetRegression(in_channels=4).to(device)
    mse_criterion = nn.MSELoss()
    reg_opt = torch.optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(15):
        reg_model.train()
        for bx, by in tr_loaderD:
            bx, by = bx.to(device), by.to(device)
            reg_opt.zero_grad()
            pred = reg_model(bx)
            loss = mse_criterion(pred, by)
            loss.backward()
            reg_opt.step()

    reg_model.eval()
    reg_preds_raw = []
    with torch.no_grad():
        for bx, _ in te_loaderD:
            bx = bx.to(device)
            p = reg_model(bx).cpu().numpy().flatten()
            reg_preds_raw.extend(p)

    reg_preds_raw = np.array(reg_preds_raw)
    
    # Original paper thresholding: predicts // 3.34 -> index 0, 1, 2
    labels_pool = np.array([0, 1, 2])
    clipped = np.clip(reg_preds_raw, 0.0, 10.0)
    pred_idx = (clipped // 3.34).astype(np.int8)
    pred_idx = np.clip(pred_idx, 0, 2)

    accD = float(accuracy_score(test_y_cat, pred_idx))
    mpD, mrD, mf1D, _ = precision_recall_fscore_support(test_y_cat, pred_idx, average="macro")
    w_precD, w_recD, wf1D, _ = precision_recall_fscore_support(test_y_cat, pred_idx, average="weighted")
    ppD, prD, pf1D, psD = precision_recall_fscore_support(test_y_cat, pred_idx, average=None)
    cmD = confusion_matrix(test_y_cat, pred_idx)

    plot_and_save_cm(cmD, f"Exp D: Regression + 3.34 Binning (Acc: {accD*100:.2f}%)", "cm_exp_D.png")
    results["experiments"]["Exp_D_Original_Regression"] = {
        "accuracy": accD, "macro_f1": float(mf1D), "weighted_f1": float(wf1D),
        "per_class": {c: {"precision": float(ppD[i]), "recall": float(prD[i]), "f1": float(pf1D[i]), "support": int(psD[i])} for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": cmD.tolist()
    }

    # Summary Print
    print(f"\n==========================================================")
    print(f"       SUMMARY OF BASELINE EXPERIMENTAL COMPARISONS       ")
    print(f"==========================================================")
    print(f" Majority-Class Baseline           : {maj_acc*100:.2f}%")
    print(f" Exp A (Current CrossEntropy)      : {accA*100:.2f}% (Macro F1: {mf1A:.4f})")
    print(f" Exp B (Class-Weighted CrossEnt)   : {accB*100:.2f}% (Macro F1: {mf1B:.4f})")
    print(f" Exp C (Orig Normalization + CE)   : {accC*100:.2f}% (Macro F1: {mf1C:.4f})")
    print(f" Exp D (Orig Regression + MSE)     : {accD*100:.2f}% (Macro F1: {mf1D:.4f})")
    print(f"==========================================================")

    # Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, "failure_analysis_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Diagnostics] Saved JSON summary → {summary_path}")

    return results

if __name__ == "__main__":
    run_diagnostics()

"""
Baseline Training and Evaluation Script for UTA-RLDD Preprocessed Dataset (Phase 0).
Trains 1D-CNN + 2-layer Bi-LSTM model on 4-feature temporal sequence blinks data.
Outputs:
 - outputs/metrics/baseline_results.json
 - outputs/metrics/baseline_confusion_matrix.png
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader

from src.model.driver_safety_net import DriverSafetyNet

CONFIG_PATH = os.path.join("configs", "baseline_uta.yaml")
DATA_DIR = os.path.join("data", "raw", "uta_rldd")
OUTPUT_JSON = os.path.join("outputs", "metrics", "baseline_results.json")
OUTPUT_CM_PNG = os.path.join("outputs", "metrics", "baseline_confusion_matrix.png")

LABEL_MAP = {0.0: 0, 5.0: 1, 10.0: 2}

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

    train_y = np.array([LABEL_MAP[val] for val in train_y_raw], dtype=np.int64)
    test_y = np.array([LABEL_MAP[val] for val in test_y_raw], dtype=np.int64)

    return train_x, train_y, test_x, test_y

def train_and_evaluate():
    set_seed(42)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    fold = config["training"]["primary_fold"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]

    print(f"\n==========================================================")
    print(f"   STARTING UTA-RLDD BASELINE EXPERIMENT (FOLD {fold})      ")
    print(f"==========================================================")

    train_x, train_y, test_x, test_y = load_data(fold=fold)
    print(f"[Dataset] Train samples: {len(train_x)} | Test samples: {len(test_x)}")
    print(f"[Dataset] Input sequence shape: {train_x.shape[1:]} (30 timestamps x 4 features)")

    # Standardize 4 features across time: fit on train_x, transform train_x and test_x
    N_tr, T, F_dim = train_x.shape
    N_te, _, _ = test_x.shape

    scaler = StandardScaler()
    train_x_reshaped = train_x.reshape(-1, F_dim)
    scaler.fit(train_x_reshaped)

    train_x_scaled = scaler.transform(train_x_reshaped).reshape(N_tr, T, F_dim)
    test_x_scaled = scaler.transform(test_x.reshape(-1, F_dim)).reshape(N_te, T, F_dim)

    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(train_x_scaled, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_x_scaled, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize DriverSafetyNet with in_channels=4, num_classes=3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n[Training] Training DriverSafetyNet (in_channels=4, num_classes=3) on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        tr_loss = running_loss / total
        tr_acc = correct / total

        # Validation on test loader
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                v_loss += loss.item() * len(by)
                preds = logits.argmax(dim=1)
                v_correct += (preds == by).sum().item()
                v_total += len(by)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f" Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Evaluation on Test Set
    print("\n[Evaluation] Computing detailed test set performance metrics...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted")
    per_class_prec, per_class_rec, per_class_f1, per_class_supp = precision_recall_fscore_support(all_targets, all_preds, average=None)

    class_names = config["classes"]["names"]
    cm = confusion_matrix(all_targets, all_preds)

    print(f"\n==========================================================")
    print(f"        FINAL UTA-RLDD BASELINE EVALUATION RESULTS        ")
    print(f"==========================================================")
    print(f" Test Accuracy        : {acc * 100:.2f}%")
    print(f" Macro Precision      : {macro_prec:.4f}")
    print(f" Macro Recall         : {macro_rec:.4f}")
    print(f" Macro F1-Score       : {macro_f1:.4f}")
    print(f" Weighted F1-Score    : {weighted_f1:.4f}")

    print("\nPer-Class Breakdowns:")
    for idx, cname in enumerate(class_names):
        print(f" - {cname:<25}: Precision={per_class_prec[idx]:.4f}, Recall={per_class_rec[idx]:.4f}, F1={per_class_f1[idx]:.4f}, Support={per_class_supp[idx]}")

    results_payload = {
        "fold": fold,
        "model_architecture": "1D-CNN + 2-layer Bi-LSTM (DriverSafetyNet)",
        "input_shape": [30, 4],
        "test_accuracy": acc,
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": {
            cname: {
                "precision": float(per_class_prec[idx]),
                "recall": float(per_class_rec[idx]),
                "f1_score": float(per_class_f1[idx]),
                "support": int(per_class_supp[idx])
            } for idx, cname in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "history": history
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[Results] Saved baseline JSON metrics → {OUTPUT_JSON}")

    # Plot & save confusion matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"UTA-RLDD Baseline Confusion Matrix (Fold {fold})\nAccuracy: {acc*100:.2f}%")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(OUTPUT_CM_PNG, dpi=300)
    plt.close()
    print(f"[Results] Saved confusion matrix plot → {OUTPUT_CM_PNG}")

if __name__ == "__main__":
    train_and_evaluate()

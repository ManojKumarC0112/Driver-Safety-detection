"""
Standalone Model Evaluation Module for Driver Safety AI.
Evaluates model on held-out test set, calculates macro/weighted metrics, generates confusion matrix plot,
classification report CSV, test metrics JSON, and training curve plots.
"""

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from src.utils.paths import METRICS_DIR, PLOTS_DIR, MODEL_PATH, SCALER_PATH, load_config
from src.model.driver_safety_net import DriverSafetyNet

def evaluate_model_on_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: list,
    device: torch.device,
    metrics_dir: str = str(METRICS_DIR),
    prefix: str = "proposed_"
) -> dict:
    """
    Perform evaluation on held-out test dataset and calculate all metrics without fabrication.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 1. Overall Accuracy
    acc = float(accuracy_score(all_targets, all_preds))

    # 2. Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(len(class_names)), zero_division=0
    )

    # 3. Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    # 4. Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    # 5. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1_score": float(f1_class[i]),
            "support": int(support_class[i])
        }

    results = {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(all_targets)
    }

    # Save metrics JSON
    os.makedirs(metrics_dir, exist_ok=True)
    json_path = os.path.join(metrics_dir, f"{prefix}test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[Evaluation] Saved test metrics to {json_path}")

    # Save Classification Report CSV
    report_dict = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(metrics_dir, f"{prefix}classification_report.csv")
    report_df.to_csv(csv_path)
    print(f"[Evaluation] Saved classification report CSV to {csv_path}")

    # Plot & Save Confusion Matrix
    plot_confusion_matrix(cm, class_names, os.path.join(metrics_dir, f"{prefix}confusion_matrix.png"))

    return results

def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    """Plot and save publication-ready confusion matrix figure."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Driver Safety AI - Test Set Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix plot to {output_path}")

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    output_dir: str = str(PLOTS_DIR),
    prefix: str = "proposed_"
):
    """Generate individual and combined academic training curve plots."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # 1. Combined Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_losses, "b-o", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-s", label="Validation Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, train_accs, "b-o", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, val_accs, "r-s", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}training_curves.png"), dpi=300)
    plt.close()

    print(f"[Evaluation] Saved training curve plots to {output_dir}")

def evaluate_checkpoint(
    model_path: str = "outputs/models/best_driver_safety_net.pt",
    proc_dir: str = "data/processed/proposed_12feature",
    scaler_path: str = "outputs/models/scaler_proposed_12feature.pkl"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==================================================================")
    print("   DRIVER SAFETY AI - PROPOSED MODEL TEST EVALUATION              ")
    print("==================================================================")
    print(f" Model Checkpoint : {model_path}")
    print(f" Processed Split  : {proc_dir}")
    print(f" Operating Device : {device}")
    print("==================================================================\n")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' not found.")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]
    class_names = checkpoint["class_names"]

    model = DriverSafetyNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test split
    te_x_path = os.path.join(proc_dir, "test_x.npy")
    te_y_path = os.path.join(proc_dir, "test_y.npy")
    te_x = np.load(te_x_path)
    te_y = np.load(te_y_path)

    # Scale test split
    if not os.path.exists(scaler_path):
        scaler_path = checkpoint.get("scaler_path", scaler_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    N_te, T, C = te_x.shape
    te_x_flat = te_x.reshape(-1, C)
    te_x_scaled_flat = scaler.transform(te_x_flat)
    te_x_scaled = te_x_scaled_flat.reshape(N_te, T, C)

    test_tensor_x = torch.tensor(te_x_scaled, dtype=torch.float32)
    test_tensor_y = torch.tensor(te_y, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(test_tensor_x, test_tensor_y), batch_size=32, shuffle=False)

    results = evaluate_model_on_test_set(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        metrics_dir="outputs/metrics",
        prefix="proposed_"
    )

    if "train_losses" in checkpoint and "val_losses" in checkpoint:
        plot_training_curves(
            train_losses=checkpoint["train_losses"],
            val_losses=checkpoint["val_losses"],
            train_accs=checkpoint["train_accs"],
            val_accs=checkpoint["val_accs"],
            output_dir="outputs/plots",
            prefix="proposed_"
        )

    print("\n==================================================================")
    print("   PROPOSED 12-FEATURE MODEL TEST RESULTS                         ")
    print("==================================================================")
    print(f" Test Accuracy        : {results['accuracy'] * 100:.2f}%")
    print(f" Macro Precision      : {results['macro_precision'] * 100:.2f}%")
    print(f" Macro Recall         : {results['macro_recall'] * 100:.2f}%")
    print(f" Macro F1-Score       : {results['macro_f1'] * 100:.2f}%")
    print(f" Weighted F1-Score    : {results['weighted_f1'] * 100:.2f}%")
    print("------------------------------------------------------------------")
    print(" Per-Class Breakdown:")
    for cname, cmetrics in results["per_class"].items():
        print(f"  - {cname:10s} | Precision: {cmetrics['precision']*100:.2f}% | Recall: {cmetrics['recall']*100:.2f}% | F1: {cmetrics['f1_score']*100:.2f}% (N={cmetrics['support']})")
    print("==================================================================\n")

    return results

def main():
    parser = argparse.ArgumentParser(description="Driver Safety AI Model Evaluation Script")
    parser.add_argument("--model-path", type=str, default="outputs/models/best_driver_safety_net.pt", help="Path to checkpoint")
    parser.add_argument("--proc-dir", type=str, default="data/processed/proposed_12feature", help="Path to processed split directory")
    parser.add_argument("--scaler-path", type=str, default="outputs/models/scaler_proposed_12feature.pkl", help="Path to scaler")
    args = parser.parse_args()

    evaluate_checkpoint(
        model_path=args.model_path,
        proc_dir=args.proc_dir,
        scaler_path=args.scaler_path
    )

if __name__ == "__main__":
    main()

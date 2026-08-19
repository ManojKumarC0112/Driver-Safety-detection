# UTA-RLDD Preprocessed Baseline Dataset Audit Report

## 1. Source & Provenance Information
- **Repository**: `rezaghoddoosian/Early-Drowsiness-Detection`
- **Research Paper**: A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection (CVPRW 2019)
- **Authors**: Reza Ghoddoosian, Marnim Galib, Vassilis Athitsos
- **Dataset**: UTA-RLDD (University of Texas at Arlington Real-Life Drowsiness Dataset)
- **License**: MIT License (as specified in repository LICENSE file)

## 2. Feature & Label Definitions
### Extracted Features (4 Features per Timestamp):
- `Feature 0`: **Normalized Blink Frequency** (`Freq`)
- `Feature 1`: **Normalized Blink Amplitude** (`Amp`)
- `Feature 2`: **Normalized Blink Duration** (`Dur`)
- `Feature 3`: **Normalized Eye Opening Velocity** (`Vel`)

### Raw Label Encoding in Repository:
- `0.0`: **Alert**
- `5.0`: **Low Vigilance / Semi-sleepy**
- `10.0`: **Drowsy / Sleepy**

## 3. Five-Fold Cross-Validation Audit
| Fold | Train Shape | Test Shape | Total Samples | Train Classes (0/5/10) | Test Classes (0/5/10) | NaN/Inf Count |
|------|-------------|------------|---------------|------------------------|-----------------------|---------------|
| Fold 1 | `[7379, 30, 4]` | `[1269, 30, 4]` | 8648 | 1419 / 2994 / 2966 | 228 / 494 / 547 | 0 |
| Fold 2 | `[6605, 30, 4]` | `[2043, 30, 4]` | 8648 | 1182 / 2653 / 2770 | 465 / 835 / 743 | 0 |
| Fold 3 | `[6465, 30, 4]` | `[2196, 30, 4]` | 8661 | 1199 / 2723 / 2543 | 449 / 771 / 976 | 0 |
| Fold 4 | `[7186, 30, 4]` | `[1462, 30, 4]` | 8648 | 1436 / 2904 / 2846 | 211 / 584 / 667 | 0 |
| Fold 5 | `[6957, 30, 4]` | `[1691, 30, 4]` | 8648 | 1352 / 2678 / 2927 | 295 / 810 / 586 | 0 |

## 4. Subject-Level Leakage & Fold Structure Audit
- **Fold Definition**: As documented in `Preprocessing.py` lines 147–155, each fold $X$ isolates all video sessions belonging to subject IDs in Fold $X$ into `BlinksTest_30_FoldX.npy`, while storing the remaining subjects in `Blinks_30_FoldX.npy`.
- **Subject Independence**: Preserved natively by the repository author. Samples are generated per subject before folding, ensuring zero subject-level overlap between train and test splits.
- **Sequence Dimensions**: Every sequence has exact dimension `(30, 4)` (30 timestamps $\times$ 4 blink features).
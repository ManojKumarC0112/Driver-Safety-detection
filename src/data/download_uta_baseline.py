"""
Downloader script for UTA-RLDD preprocessed baseline .npy datasets via GitHub zip archive.
"""

import os
import zipfile
import urllib.request
import shutil

ZIP_URL = "https://github.com/rezaghoddoosian/Early-Drowsiness-Detection/archive/refs/heads/master.zip"
TARGET_DIR = os.path.join("data", "raw", "uta_rldd")
TMP_ZIP = os.path.join("data", "raw", "uta_master.zip")

def download_and_extract_uta():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Check if all 20 files are already present
    expected_files = []
    for f in range(1, 6):
        expected_files.extend([
            f"Blinks_30_Fold{f}.npy",
            f"Labels_30_Fold{f}.npy",
            f"BlinksTest_30_Fold{f}.npy",
            f"LabelsTest_30_Fold{f}.npy"
        ])
    
    existing = [f for f in expected_files if os.path.exists(os.path.join(TARGET_DIR, f))]
    if len(existing) == 20:
        print(f"[UTA Downloader] All 20 .npy files already exist in '{TARGET_DIR}'.")
        return

    print(f"[UTA Downloader] Downloading GitHub master zip from {ZIP_URL}...")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(TMP_ZIP, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)
    print(f"[UTA Downloader] Downloaded zip ({os.path.getsize(TMP_ZIP)} bytes). Extracting...")

    with zipfile.ZipFile(TMP_ZIP, "r") as z:
        for member in z.namelist():
            if member.endswith(".npy"):
                fname = os.path.basename(member)
                if fname:
                    out_path = os.path.join(TARGET_DIR, fname)
                    with z.open(member) as source, open(out_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    print(f" -> Extracted {fname} ({os.path.getsize(out_path)} bytes)")

    if os.path.exists(TMP_ZIP):
        os.remove(TMP_ZIP)
    print(f"[UTA Downloader] Extraction complete. Files saved in '{TARGET_DIR}'.")

if __name__ == "__main__":
    download_and_extract_uta()

"""
Driver Safety AI - Main Application Entry Point.
Launches real-time driver drowsiness and vigilance monitoring application.
"""

import sys
import argparse
from src.utils.paths import MODEL_PATH, SCALER_PATH
from src.inference.realtime import run_realtime_application

def main():
    parser = argparse.ArgumentParser(description="Driver Safety AI - Real-Time Vigilance Detection System")
    parser.add_argument("--config", type=str, default=None, help="Optional path to custom config.yaml")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to PyTorch model checkpoint")
    parser.add_argument("--scaler", type=str, default=str(SCALER_PATH), help="Path to fitted StandardScaler")
    args = parser.parse_args()

    try:
        run_realtime_application(
            config_path=args.config,
            model_path=args.model,
            scaler_path=args.scaler
        )
    except KeyboardInterrupt:
        print("\n[Main] Driver Safety AI stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Main] Application Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

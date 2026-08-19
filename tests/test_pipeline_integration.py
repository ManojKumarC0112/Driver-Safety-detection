"""
Lightweight End-to-End Pipeline Integration Test (Section 53 & User Directive #7).
Verifies complete flow: Synthetic numerical 30x12 sequence tensor → Scaler → DriverSafetyNet → Softmax → DVI score → HUD text.
"""

import numpy as np
import torch
import pytest
from sklearn.preprocessing import StandardScaler

from src.model.driver_safety_net import DriverSafetyNet
from src.utils.dvi import DVIEngine
from src.visualization.hud import draw_hud

def test_full_pipeline_integration():
    # 1. Generate synthetic feature sequence (1, 30, 12)
    raw_seq = np.random.randn(1, 30, 12).astype(np.float32)

    # 2. Fit and transform using StandardScaler
    scaler = StandardScaler()
    scaler.fit(raw_seq.reshape(-1, 12))
    norm_seq = scaler.transform(raw_seq.reshape(-1, 12)).reshape(1, 30, 12).astype(np.float32)

    # 3. Instantiate DriverSafetyNet & run forward pass
    model = DriverSafetyNet(
        in_channels=12,
        cnn_filters=32,
        kernel_size=3,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        lstm_bidirectional=True,
        num_classes=4
    )
    model.eval()

    with torch.no_grad():
        inp_tensor = torch.tensor(norm_seq, dtype=torch.float32)
        probs = model.predict_proba(inp_tensor).numpy()[0]

    assert probs.shape == (4,)
    assert pytest.approx(float(np.sum(probs)), 0.01) == 1.0

    # 4. Run DVI Calculation Engine
    dvi_engine = DVIEngine()
    dvi_score, dvi_level, _ = dvi_engine.calculate_dvi(
        p_alert=probs[0],
        perclos=0.1,
        eye_closure_duration_sec=0.2,
        yaw_deg=5.0
    )
    assert 0.0 <= dvi_score <= 100.0

    # 5. Render HUD on mock OpenCV frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat_vec = raw_seq[0, -1, :] # Last frame features

    rendered_frame = draw_hud(
        frame=mock_frame,
        status_text="ALERT",
        feature_vec=feat_vec,
        dvi_score=dvi_score,
        dvi_level=dvi_level,
        fps=30.0,
        device_name="CPU"
    )

    assert rendered_frame.shape == (480, 640, 3)
    assert rendered_frame.dtype == np.uint8

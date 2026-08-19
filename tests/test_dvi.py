"""
Unit tests for Driver Vigilance Index (DVI) Calculation Engine.
"""

import pytest
from src.utils.dvi import DVIEngine

def test_dvi_score_bounds_and_risk_levels():
    engine = DVIEngine()

    # 1. Best case scenario: ALERT=1.0, PERCLOS=0, EyeClosure=0, Yaw=0 -> DVI = 0.0 (LOW)
    score_low, level_low, _ = engine.calculate_dvi(
        p_alert=1.0, perclos=0.0, eye_closure_duration_sec=0.0, yaw_deg=0.0
    )
    assert score_low == pytest.approx(0.0, abs=1e-5)
    assert level_low == "LOW"

    # 2. Worst case scenario: ALERT=0.0, PERCLOS=1.0, EyeClosure=5.0s, Yaw=50.0deg -> DVI = 100.0 (CRITICAL)
    score_high, level_high, _ = engine.calculate_dvi(
        p_alert=0.0, perclos=1.0, eye_closure_duration_sec=5.0, yaw_deg=50.0
    )
    assert score_high == pytest.approx(100.0, abs=1e-5)
    assert level_high == "CRITICAL"

    # 3. Moderate scenario
    score_mod, level_mod, _ = engine.calculate_dvi(
        p_alert=0.5, perclos=0.2, eye_closure_duration_sec=0.5, yaw_deg=10.0
    )
    assert 0.0 <= score_mod <= 100.0
    assert level_mod in ["LOW", "MODERATE", "HIGH", "CRITICAL"]

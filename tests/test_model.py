"""
Unit tests for DriverSafetyNet PyTorch Model Architecture (Section 3 & User Directive #4).
"""

import torch
import pytest
from src.model.driver_safety_net import DriverSafetyNet

def test_driver_safety_net_forward_shape():
    model = DriverSafetyNet(
        in_channels=12,
        cnn_filters=32,
        kernel_size=3,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        lstm_bidirectional=True,
        dropout=0.3,
        num_classes=4
    )
    model.eval()

    # Input tensor shape: (Batch=4, Sequence=30, Features=12)
    batch_size = 4
    x = torch.randn(batch_size, 30, 12, dtype=torch.float32)

    logits = model(x)

    # Output shape must be strictly (batch_size, 4)
    assert logits.shape == (batch_size, 4)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_driver_safety_net_predict_proba():
    model = DriverSafetyNet()
    x = torch.randn(2, 30, 12, dtype=torch.float32)

    probs = model.predict_proba(x)

    assert probs.shape == (2, 4)
    # Check probabilities sum to 1.0 per sample
    row_sums = torch.sum(probs, dim=1)
    assert torch.allclose(row_sums, torch.tensor([1.0, 1.0]), atol=1e-5)

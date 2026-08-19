"""
Unit tests for Sequence Generator Module.
"""

import numpy as np
import pytest
from src.data.create_sequences import create_sequences_from_array

def test_create_sequences_from_array():
    # 50 frames of 12 features
    feature_matrix = np.random.randn(50, 12).astype(np.float32)
    labels = np.zeros(50, dtype=np.int64)

    window_size = 30
    stride = 1

    X, y = create_sequences_from_array(feature_matrix, labels, window_size=window_size, stride=stride)

    # Number of sliding windows = 50 - 30 + 1 = 21
    assert X.shape == (21, 30, 12)
    assert y.shape == (21,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64

def test_sequence_generator_empty_short_input():
    # Input shorter than window size (20 frames < 30)
    feature_matrix = np.random.randn(20, 12).astype(np.float32)
    labels = np.zeros(20, dtype=np.int64)

    X, y = create_sequences_from_array(feature_matrix, labels, window_size=30, stride=1)
    assert len(X) == 0
    assert len(y) == 0

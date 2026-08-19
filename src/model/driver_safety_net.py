"""
DriverSafetyNet Architecture Implementation.
Hybrid 1D-CNN + 2-layer Bidirectional LSTM Classifier in PyTorch.
Input shape: (batch_size, 30, 12) -> Output shape: (batch_size, 4) raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DriverSafetyNet(nn.Module):
    """
    1D CNN + 2-layer Bi-LSTM Neural Network for Driver State Classification.
    Classes: 0: ALERT, 1: DROWSY, 2: YAWNING, 3: DISTRACTED
    """
    def __init__(
        self,
        in_channels: int = 12,
        cnn_filters: int = 32,
        kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 4
    ):
        super(DriverSafetyNet, self).__init__()

        self.in_channels = in_channels
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.num_classes = num_classes

        # 1D Convolutional Layer (operates over temporal sequence)
        # Input: (B, 12, 30) -> Output: (B, 32, 30)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2 # Keeps temporal dimension equal to 30
        )
        self.relu = nn.ReLU()

        # 2-layer Bidirectional LSTM Layer
        # Input: (B, 30, 32) -> Output: (B, 30, 64 * 2 = 128)
        self.lstm_out_dim = lstm_hidden_size * (2 if lstm_bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        # Classifier
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: Tensor of shape (B, 30, 12)
        Returns: raw logits tensor of shape (B, 4)
        """
        if x.dim() != 3 or x.shape[1] != 30 or x.shape[2] != self.in_channels:
            raise ValueError(f"Expected input shape (B, 30, {self.in_channels}), got {x.shape}")

        # 1. Permute for 1D CNN: (B, 30, 12) -> (B, 12, 30)
        x_cnn_in = x.permute(0, 2, 1)

        # 2. Apply 1D CNN + ReLU: (B, 12, 30) -> (B, 32, 30)
        x_cnn_out = self.relu(self.conv1d(x_cnn_in))

        # 3. Transpose back for Bi-LSTM: (B, 32, 30) -> (B, 30, 32)
        x_lstm_in = x_cnn_out.permute(0, 2, 1)

        # 4. Apply 2-layer Bi-LSTM: (B, 30, 32) -> (B, 30, 128)
        lstm_out, _ = self.lstm(x_lstm_in)

        # 5. Extract final temporal step representation: (B, 128)
        last_step_features = lstm_out[:, -1, :]

        # 6. Apply Dropout + Dense Classifier: (B, 128) -> (B, 4)
        logits = self.fc(self.dropout(last_step_features))

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference helper returning Softmax class probabilities.
        Returns: Tensor of shape (B, 4) summing to 1.0 per sample.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities

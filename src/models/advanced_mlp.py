"""Advanced tabular MLP models for HEALIX-FL.

The original project MLP is intentionally simple and kept unchanged for
backward compatibility with completed checkpoints. This module adds a stronger
centralized tabular model using BatchNorm, GELU, residual blocks, and AdamW.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual fully connected block for tabular features."""

    def __init__(self, width, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class AdvancedMLP(nn.Module):
    """BatchNorm residual MLP for binary tabular classification.

    The network returns a single logit, matching the existing MLP contract.
    """

    def __init__(self, input_dim, width=256, num_blocks=3, dropout=0.25):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(width=width, dropout=dropout) for _ in range(num_blocks)]
        )
        self.output = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

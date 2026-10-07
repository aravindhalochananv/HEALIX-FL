"""
Phase 5: Centralized MLP Baseline Model

This module defines a simple Multi-Layer Perceptron (MLP) for the
centralized baseline experiment in the HEALIX-FL project.

Architecture:
    Input -> Linear(128) -> ReLU -> Dropout(0.2) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1)

The final layer outputs a LOGIT (no sigmoid).
Loss: BCEWithLogitsLoss
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP for binary classification.
    
    Args:
        input_dim: Number of input features.
        hidden_dim1: Size of first hidden layer (default: 128).
        hidden_dim2: Size of second hidden layer (default: 64).
        dropout: Dropout probability (default: 0.2).
    """
    
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.2):
        super(MLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, x):
        """Forward pass returning logits.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            logits: Tensor of shape (batch_size, 1).
        """
        return self.network(x)


def count_parameters(model):
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

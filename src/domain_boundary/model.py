import os
from typing import Union

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

LABELS = {
    'NO_DOMAIN_REGION': 0,
    'DOMAIN_START': 1,
    'DOMAIN_MIDDLE': 2,
    'DOMAIN_END': 3,
}

class DomainBoundaryCNN(nn.Module):
    def __init__(self, embedding_dim: int = 1024, hidden_channels: int = 128, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(embedding_dim, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = x['embedding']  # (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.classifier(x)  # (B, C, L)
        x = x.transpose(1, 2)  # (B, L, C)
        return x
import torch.nn as nn
import torch.nn.functional as F

LABELS = {
    'NO_DOMAIN_REGION': 0,
    'DOMAIN_START': 1,
    'DOMAIN_MIDDLE': 2,
    'DOMAIN_END': 3,
}

class DomainBoundaryCNN(nn.Module):
    def __init__(self, embedding_dim: int = 1024, hidden_channels: int = 128, num_classes: int = 4, max_length=600):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x['embedding']
        x = x.transpose(1, 2)  # (B, L, 1024) → (B, 1024, L)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x
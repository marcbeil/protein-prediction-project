import torch
from torch import nn


class CathPred(nn.Module):
    def __init__(self, num_classes, in_channels=1024, out_channels=256):
        super(CathPred, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, 1024) → (B, 1024, L)
        x = self.conv(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

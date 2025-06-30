from torch import nn


class CathPredEnn2(nn.Module):
    def __init__(self, num_classes):
        super(CathPredEnn2, self).__init__()

        self.conv = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x["embedding"].permute(0, 2, 1)  # (B, L, 1024) → (B, 1024, L)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

"""Steering models.

PilotNet ("End to End Learning for Self-Driving Cars", Bojarski et al. 2016)
is the default. The video's "lines -> shapes -> one number" story maps to the
layers: early convs learn edges/lines, middle convs learn shapes (lane/road
regions), the dense layers turn that into an angle.

Shape walk-through for a 3x66x200 input (PilotNet):
  conv 5x5 s2 ->  24 x 31 x 98
  conv 5x5 s2 ->  36 x 14 x 47
  conv 5x5 s2 ->  48 x  5 x 22
  conv 3x3    ->  64 x  3 x 20
  conv 3x3    ->  64 x  1 x 18   => flatten = 1152

A ResNet-18-style regressor (built from scratch, no torchvision) is available
via ``build_model("resnet18")`` for the Phase 6 model comparison.
"""

import torch
import torch.nn as nn

from .config import CFG


class PilotNet(nn.Module):
    """NVIDIA PilotNet. Input (N, 3, 66, 200) in [0, 255] -> steering in [-1, 1]."""

    def __init__(self, dropout: float = CFG.dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,66,200) in [0,255] (YUV). Normalise inside the model so the
        # exported checkpoint is self-contained.
        x = x / 127.5 - 1.0
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (stride applied in the first 3x3 conv)."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.short(x))


class ResNetSteer(nn.Module):
    """ResNet-18 backbone (adapted stem for 66x200 input) + 1-unit regressor head.

    Input (N, 3, 66, 200) in [0, 255] -> steering in [-1, 1], same contract as
    PilotNet so the two architectures are drop-in comparable.
    """

    def __init__(self, dropout: float = CFG.dropout):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # 66x200 -> 33x100
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # -> 16x50
        )
        self.layers = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),  # 16x50
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),  # 8x25
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),  # 4x12
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),  # 2x6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone feature maps (used by the visualisation tools)."""
        h = x / 127.5 - 1.0
        h = self.stem(h)
        return self.layers(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).flatten(1)


ARCHS = {"pilotnet": PilotNet, "resnet18": ResNetSteer}


def build_model(arch: str = "pilotnet", dropout: float = CFG.dropout) -> nn.Module:
    """Factory used everywhere a checkpoint is loaded, so ``--arch`` round-trips."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}' (choose from {sorted(ARCHS)})")
    return ARCHS[arch](dropout=dropout)

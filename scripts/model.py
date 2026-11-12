import torch
import torch.nn as nn

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyCNN(nn.Module):
    """~70K-param CIFAR-10 CNN: 5x conv3x3 + 3x maxpool + GAP + FC."""

    def __init__(self, n=10):
        super().__init__()
        self.features = nn.Sequential(
            cbr(3, 16),
            cbr(16, 32), nn.MaxPool2d(2),   # 16x16
            cbr(32, 32),
            cbr(32, 64), nn.MaxPool2d(2),   # 8x8
            cbr(64, 64), nn.MaxPool2d(2),   # 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.fc(torch.flatten(x, 1))


def fuse_bn(model: TinyCNN) -> TinyCNN:
    """Return eval-mode copy with BN folded into conv (conv gets a bias)."""
    import copy
    m = copy.deepcopy(model).eval()
    for i, blk in enumerate(m.features):
        if isinstance(blk, nn.Sequential):
            conv, bn = blk[0], blk[1]
            w = conv.weight
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            fused = nn.Conv2d(conv.in_channels, conv.out_channels, 3, padding=1, bias=True)
            fused.weight.data = w * scale[:, None, None, None]
            fused.bias.data = bn.bias - bn.running_mean * scale
            m.features[i] = nn.Sequential(fused, nn.ReLU(inplace=True))
    return m

import numpy as np, torch
from pathlib import Path
from model import MEAN, STD
D = Path(__file__).resolve().parents[1] / "data"

def load(split):
    z = np.load(D / f"cifar10_{split}.npz")
    return z["x"], z["y"]  # uint8 NHWC, uint8 labels

def to_tensor(x_u8):
    """uint8 NHWC -> normalized float NCHW (same preprocessing as on-device)."""
    x = torch.from_numpy(x_u8).float().permute(0, 3, 1, 2) / 255.0
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)

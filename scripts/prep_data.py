# Convert HF parquet CIFAR-10 to uint8 numpy arrays (N,32,32,3) + labels.
import io, numpy as np, pyarrow.parquet as pq
from PIL import Image
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data"
for split in ("train", "test"):
    t = pq.read_table(D / f"{split}.parquet").to_pydict()
    imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
    labels = np.array(t["label"], dtype=np.uint8)
    np.savez(D / f"cifar10_{split}.npz", x=imgs, y=labels)
    print(split, imgs.shape, np.bincount(labels))

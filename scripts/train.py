import sys, time, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model import TinyCNN
from data import load, to_tensor

ROOT = Path(__file__).resolve().parents[1]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
xtr, ytr = load("train"); xte, yte = load("test")
Xtr, Ytr = to_tensor(xtr).to(dev), torch.from_numpy(ytr).long().to(dev)
Xte, Yte = to_tensor(xte).to(dev), torch.from_numpy(yte).long().to(dev)

def augment(x):
    n = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2,))
    x = x[:, :, i:i + 32, j:j + 32]  # per-batch crop offset (cheap); flip per-sample
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BS = 128; steps = len(Xtr) // BS
model = TinyCNN().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=EPOCHS * steps)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
for ep in range(EPOCHS):
    model.train(); t0 = time.time(); perm = torch.randperm(len(Xtr), device=dev)
    for s in range(steps):
        idx = perm[s * BS:(s + 1) * BS]
        opt.zero_grad(); lossf(model(augment(Xtr[idx])), Ytr[idx]).backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        acc = sum((model(Xte[k:k + 1000]).argmax(1) == Yte[k:k + 1000]).sum().item() for k in range(0, 10000, 1000)) / 100
    print(f"epoch {ep+1}/{EPOCHS} test acc {acc:.2f}% ({time.time()-t0:.0f}s)", flush=True)
torch.save(model.cpu().state_dict(), ROOT / "artifacts/tinycnn_cifar10.pt")
print("params", sum(p.numel() for p in model.parameters()))

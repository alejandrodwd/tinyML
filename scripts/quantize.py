"""Fold BN, export float weights for the on-device float32 reference, quantize to int8 with ESP-PPQ,
evaluate float vs int8 (PPQ simulation) on the full CIFAR-10 test set, and dump a test subset for the board."""
import sys, json, struct, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model import TinyCNN, fuse_bn
from data import load, to_tensor
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor import TorchExecutor

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT / "artifacts"; FW = ROOT / "firmware/main"
N_DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
torch.manual_seed(0)

model = TinyCNN(); model.load_state_dict(torch.load(ART / "tinycnn_cifar10.pt")); model.eval()
fused = fuse_bn(model).eval()
xte, yte = load("test"); Xte = to_tensor(xte); Yte = torch.from_numpy(yte).long()

def acc(fn, X, Y, bs=500):
    c = 0
    with torch.no_grad():
        for k in range(0, len(X), bs):
            c += (fn(X[k:k + bs]).argmax(1) == Y[k:k + bs]).sum().item()
    return 100 * c / len(X)

float_acc = acc(model, Xte, Yte)
assert abs(acc(fused, Xte, Yte) - float_acc) < 0.05, "BN folding changed accuracy"
print(f"float32 PyTorch acc (10k): {float_acc:.2f}%")

# ---- float32 weights for the on-device C reference (NHWC / [cout][kh][kw][cin]) ----
blobs = []
for blk in fused.features:
    if isinstance(blk, torch.nn.Sequential):
        conv = blk[0]
        blobs += [conv.weight.detach().permute(0, 2, 3, 1).contiguous().numpy().ravel(), conv.bias.detach().numpy()]
blobs += [fused.fc.weight.detach().numpy().ravel(), fused.fc.bias.detach().numpy()]
fw = np.concatenate(blobs).astype("<f4")
fw.tofile(FW / "float_weights.bin")
n_params = fw.size
print("float params (BN folded):", n_params, "bytes:", fw.nbytes)

# ---- ESP-PPQ int8 quantization ----
xtr, _ = load("train")
calib_idx = np.random.RandomState(0).choice(len(xtr), 1024, replace=False)
calib = torch.utils.data.DataLoader(to_tensor(xtr[calib_idx]), batch_size=32, shuffle=False)
espdl_path = ART / "tinycnn_int8.espdl"
graph = espdl_quantize_torch(
    model=fused, espdl_export_file=str(espdl_path), calib_dataloader=calib, calib_steps=32,
    input_shape=[1, 3, 32, 32], target="esp32p4", num_of_bits=8, device="cpu",
    error_report=True, export_test_values=False, verbose=0,
)
ex = TorchExecutor(graph=graph, device="cpu")
int8_acc = acc(lambda x: torch.cat([ex.forward(inputs=x[i:i + 1])[0] for i in range(len(x))]), Xte, Yte)
print(f"int8 ESP-PPQ simulated acc (10k): {int8_acc:.2f}%")
(FW / "tinycnn_int8.espdl").write_bytes(espdl_path.read_bytes())

# ---- device test subset: first N test images, class balanced enough ----
sub_x, sub_y = xte[:N_DEVICE], yte[:N_DEVICE]
with open(FW / "testset.bin", "wb") as f:
    f.write(struct.pack("<I", N_DEVICE)); f.write(sub_y.tobytes()); f.write(sub_x.tobytes())
sub_float = acc(model, Xte[:N_DEVICE], Yte[:N_DEVICE])
sub_int8 = acc(lambda x: torch.cat([ex.forward(inputs=x[i:i + 1])[0] for i in range(len(x))]), Xte[:N_DEVICE], Yte[:N_DEVICE])
json.dump(dict(float_acc_10k=float_acc, int8_sim_acc_10k=int8_acc, float_acc_subset=sub_float, int8_sim_acc_subset=sub_int8,
               n_subset=N_DEVICE, float_params=int(n_params), float_bytes=int(fw.nbytes),
               espdl_bytes=espdl_path.stat().st_size), open(ROOT / "results/host_metrics.json", "w"), indent=2)
print(open(ROOT / "results/host_metrics.json").read())

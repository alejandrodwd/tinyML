# TinyCNN on ESP32-P4 (FireBeetle 2, 360 MHz): float32 vs int8

Model: 5×conv3x3 + 3 maxpool + GAP + FC, 70,410 params (BN folded), CIFAR-10, trained here (30 epochs).
Quantization: ESP-PPQ 1.3.14, `espdl_quantize_torch`, int8 per-tensor power-of-2, 1,024 calibration images.
Runtime: ESP-DL 3.3.15, IDF 5.5.6. Float baseline is an optimized C float32 implementation on the same chip
(ESP-DL has no float32 path; see `firmware/main/float_ref.c`).

| Metric | float32 | int8 (ESP-DL) | Change |
|---|---|---|---|
| Accuracy, full 10k test set (host, PPQ-simulated int8) | 86.31% | 86.28% | −0.03 pts |
| Accuracy, 2,000 images **on device** | 86.20% | 86.00% | −0.20 pts |
| Weight storage | 281,640 B | 70,410 B | **4.0× smaller** |
| Deployed model file | 281,640 B (raw weights) | 84,800 B (.espdl) | 3.3× smaller |
| Latency per inference (on device) | 99.9 ms | **5.49 ms** (6.51 ms incl. input quantization) | 18× faster |
| ESP-DL runtime heap | — | 137,652 B | |

Float baseline optimization history (same accuracy, verified bit-close to PyTorch, max logit diff 2e-5):

| Float kernel | Latency |
|---|---|
| Naive loops, bounds checks per tap | 343 ms |
| Weight-stationary 4-channel blocks, branch-free contiguous row runs, `-O3 -ffast-math` | 158 ms |
| 8-channel interleaved weights: one input load feeds 8 `fmadd.s` chains | **99.9 ms** |

At ~14.6 M MACs/inference the float kernel now runs ~2.5 cycles/MAC; the hard floor for scalar FPU code is ~1 cycle/MAC
(~40 ms), so int8 + ESP-DL's SIMD kernels would still win by ≥7×. The remaining 18× gap reflects what int8 enables on the
P4 (16-lane int8 SIMD MACs, 4× less memory traffic), not an unoptimized baseline.

Raw log: `device_log.txt`; numbers: `summary.json`.

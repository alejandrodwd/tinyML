"""Combine host metrics + on-device log into results/summary.json and a printed table."""
import json, re
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
h = json.load(open(R / "host_metrics.json"))
log = (R / "device_log.txt").read_text()
kv = lambda line: dict(re.findall(r"(\w+)=([\-\d.\[\],]+)", line))
f = kv(re.search(r"RESULT float .*", log).group()); q = kv(re.search(r"RESULT int8 .*", log).group())
i8 = kv(re.search(r"INT8 espdl_bytes.*", log).group()); cpu = kv(re.search(r"BENCH_START .*", log).group())
n = int(f["n"])
s = dict(
    cpu_mhz=int(cpu["cpu_mhz"]), n_device=n,
    float_acc_host_10k=h["float_acc_10k"], int8_acc_host_10k=h["int8_sim_acc_10k"],
    float_acc_device=100 * int(f["correct"]) / n, int8_acc_device=100 * int(q["correct"]) / n,
    float_bytes=h["float_bytes"], int8_espdl_bytes=h["espdl_bytes"], int8_weight_bytes=h["float_params"],
    float_ms=float(f["latency_us"]) / 1000, int8_ms=float(q["latency_us"]) / 1000, int8_e2e_ms=float(q["e2e_us"]) / 1000,
    int8_runtime_heap=int(i8["runtime_heap_bytes"]),
)
s["size_ratio"] = s["float_bytes"] / s["int8_espdl_bytes"]
s["speedup"] = s["float_ms"] / s["int8_ms"]
s["acc_drop_10k"] = s["float_acc_host_10k"] - s["int8_acc_host_10k"]
s["acc_drop_device"] = s["float_acc_device"] - s["int8_acc_device"]
json.dump(s, open(R / "summary.json", "w"), indent=2)
print(json.dumps(s, indent=2))

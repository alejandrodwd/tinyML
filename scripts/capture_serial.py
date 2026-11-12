"""Reset the board and capture benchmark output until BENCH_DONE (+ ESP-DL profile)."""
import sys, time, serial
port, out = sys.argv[1], sys.argv[2]
s = serial.Serial(port, 115200, timeout=0.5)
s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False  # reset into app
lines, done_at, t0 = [], None, time.time()
while time.time() - t0 < 900:
    l = s.readline().decode(errors="replace").rstrip()
    if l:
        lines.append(l)
        if any(k in l for k in ("BENCH", "RESULT", "FLOAT", "INT8", "progress", "Guru", "abort", "E (")): print(l, flush=True)
        if "BENCH_DONE" in l: done_at = time.time()
    if done_at and time.time() - done_at > 4: break
open(out, "w").write("\n".join(lines) + "\n")

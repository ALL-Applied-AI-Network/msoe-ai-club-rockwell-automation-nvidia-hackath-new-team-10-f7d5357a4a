# UWM GB10 preflight (prepared 2026-10-16, NOT run)

Status: **no access yet.** Derek Riley: Shamar Webster administers the UWM GB10s, and access may come separately from
Brev. Nothing here is known about the machines beyond the name. Do not assume the model, OS, Isaac Sim version or
container setup. Run this only after Noor has authorized team access (never a personal account, D010).

Purpose: decide in a few minutes whether the frozen **gate-v2.0.2** can run there exactly as frozen. If it cannot,
fall back to Brev. Do not redesign or patch the gate (`MANIFEST.sha256` must stay 8 × OK).

## 1. Identify the machine (read-only)
```bash
id
uname -a
uname -m                                   # aarch64 expected for Grace-based GB10; verify, do not assume
cat /etc/os-release
nvidia-smi
df -h
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
free -h; nproc; lscpu | head -20
cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null
nvcc --version 2>/dev/null | tail -1      # CUDA toolkit, if installed
docker --version 2>/dev/null; docker info 2>/dev/null | grep -i -E "runtimes|nvidia"
```

## 2. Is Isaac Sim there, and which version?
```bash
for c in /isaac-sim ~/isaacsim ~/isaac-sim /opt/isaac-sim; do [ -x "$c/python.sh" ] && echo "$c" && cat "$c/VERSION" 2>/dev/null; done
find / -maxdepth 5 -name python.sh -path '*isaac*' 2>/dev/null
docker images 2>/dev/null | grep -i isaac
python3 -c "import isaacsim, importlib.metadata as m; print(m.version('isaacsim'))" 2>/dev/null
```
The twin README says it was tested with Isaac Sim 6.0. The gate records whatever version actually runs
(`run_log.txt`), so a different 6.x is a recorded fact, not a reason to change the kit.

## 3. Can the frozen `run_gate.sh` run unmodified?
It needs, in order:
- (a) internet for `gdown`, or a copied twin zip;
- (b) `python3 -m venv` plus `pip install usd-core numpy opencv-python-headless` (variant authoring);
- (c) an Isaac Sim `python.sh` or a pip `isaacsim`;
- (d) an RTX GPU for PathTracing.

Check (b) first. Per our earlier check, `usd-core` publishes no Linux aarch64 wheel; confirm on the machine:
```bash
python3 -m pip download usd-core --only-binary=:all: --no-deps -d /tmp/usdcheck && ls /tmp/usdcheck
python3 -m pip download opencv-python-headless numpy --only-binary=:all: --no-deps -d /tmp/usdcheck
curl -sI https://drive.google.com | head -1     # (a) network for the twin download
```

## 4. Decision
- **All of (a)–(d) pass:** follow the runbook in `GATE_v2.0.2.md` section 11. Use `ISAAC_PY=<path>` if the script
  cannot find Isaac, or run inside the Isaac container if that is how it is installed. One run, then copy back
  `results.tar.gz` and **STOP**. GPT reviews the evidence before any dataset generation.
- **(b) fails (no `usd-core` wheel) or Isaac Sim is missing or not 6.x-compatible:** STOP. Record the outputs of
  sections 1–3, and do not edit `run_gate.sh` or any hashed file. Use **Brev (x86, RTX PRO 6000, else L40S)** as the
  fallback. Any execution-only change (like the v2.0.1 precedent) needs GPT approval first.
- **Record in every case:** machine, CPU architecture, GPU, driver, CUDA, Isaac version, renderer settings (from
  `run_log.txt`), runtime, and cost or credits if any.

Never on the GB10 before the gate verdict: synthetic dataset generation, training, Cosmos, Nemotron.

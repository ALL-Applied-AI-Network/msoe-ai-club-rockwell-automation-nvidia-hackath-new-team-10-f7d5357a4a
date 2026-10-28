#!/usr/bin/env bash
# Offline installer/preflight for the staging5 GB10 payload.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

fail() {
    echo "BOOTSTRAP FAILED: $*" >&2
    exit 1
}

echo "== VIAL GUARD staging5 shadow-demo bootstrap =="
echo "SAFETY: evaluation only; this kit never releases product or actuates the line."

command -v sha256sum >/dev/null || fail "sha256sum is required"
command -v tar >/dev/null || fail "tar is required"
[ "$(uname -m)" = "aarch64" ] || fail "this payload is for NVIDIA GB10/aarch64; found $(uname -m)"

echo "[1/5] Verifying the immutable payload"
sha256sum --quiet -c SHA256SUMS || fail "payload checksum mismatch; copy the .run file again"

if [ -e .runtime-python ] || [ -e .venv ]; then
    fail "runtime already exists; install the .run file into a new empty directory"
fi

echo "[2/5] Installing bundled CPython (offline)"
mkdir .runtime-python
tar -xzf runtime/cpython-aarch64.tar.gz -C .runtime-python
PY="$ROOT/.runtime-python/python/bin/python3"
[ -x "$PY" ] || fail "bundled CPython did not extract as expected"
"$PY" -c 'import ensurepip, ssl, venv, zlib' || fail "bundled CPython self-check failed"

echo "[3/5] Installing bundled NumPy/OpenCV wheels (offline)"
"$PY" -m venv .venv
"$ROOT/.venv/bin/python" -m pip install \
    --disable-pip-version-check --no-index --find-links wheels/cp311 \
    -r wheels/cp311/requirements.txt

echo "[4/5] Checking runtime versions"
"$ROOT/.venv/bin/python" - <<'PY'
import cv2
import numpy
print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
if numpy.__version__ != "2.1.0":
    raise SystemExit("unexpected NumPy version")
if cv2.__version__ != "4.10.0":
    raise SystemExit("unexpected OpenCV version")
PY

echo "[5/5] Running model-aware golden preflight"
PREFLIGHT_OUT="$ROOT/preflight_result"
[ ! -e "$PREFLIGHT_OUT" ] || fail "preflight_result already exists"
(
    cd app/twin_to_line
    "$ROOT/.venv/bin/python" gb10/staging5_preflight.py \
        --frames "$ROOT/preflight/frames" \
        --model-dir "$ROOT/model/resnet18_staging5" \
        --expected "$ROOT/preflight/expected.csv" \
        --out "$PREFLIGHT_OUT"
) || fail "model-aware golden preflight failed; see preflight_result"

{
    echo "ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mode=shadow_evaluation_only"
    echo "model_onnx_sha256=$(sha256sum model/resnet18_staging5/model.onnx | cut -d' ' -f1)"
} > .READY

echo
echo "BOOTSTRAP OK"
echo "Run the bundled sample: bash $ROOT/run_sample.sh"
echo "Run a live folder:       bash $ROOT/live_demo.sh folder:/path/to/camera/images"


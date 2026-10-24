#!/usr/bin/env bash
# Twin-to-Line on a GB10 (aarch64 Linux): checksums -> code -> Python -> offline install -> preflight.
# Runs from the unpacked package folder; writes only inside it. LIVE_RUNBOOK.md section 7.
#   bash bootstrap.sh            offline only (the bundled Python and wheels)
#   bash bootstrap.sh --online   also allow pip to download the pinned versions if the offline install fails
set -uo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$PKG"
ONLINE=0; [ "${1:-}" = "--online" ] && ONLINE=1
T0=$(date +%s)
step() { echo "== $* ($(( $(date +%s) - T0 )) s)"; }
fail() { echo "BOOTSTRAP FAILED: $*"; exit 1; }

# Never erase evidence when somebody repeats the installation command.
for results in code/twin_to_line/results preflight_out; do
    if [ -e "$results" ] || [ -L "$results" ]; then
        fail "results already exist at $results; unpack the package into a fresh folder. Existing results are untouched"
    fi
done

step "1/5 checksums"
sha256sum --quiet -c SHA256SUMS || fail "SHA256SUMS mismatch: copy the package again"
echo "SHA256SUMS OK ($(wc -l < SHA256SUMS) files)"

step "2/5 code"
mkdir -p code && tar -xf code.tar -C code || fail "could not unpack code.tar"
echo "commit $(cat code/twin_to_line/ARCHIVE_COMMIT)"

step "3/5 python"
PY=""; TAG=""
if [ "$(uname -m)" = "aarch64" ]; then
    rm -rf pyhome && mkdir pyhome && tar -xzf runtime/cpython-*.tar.gz -C pyhome \
        && pyhome/python/bin/python3 -c "import venv, ensurepip, ssl, zlib" && PY="$PKG/pyhome/python/bin/python3" && TAG=cp311
fi
if [ -z "$PY" ]; then
    echo "bundled Python not usable here; trying the system Python"
    for v in 3.11 3.12 3.10; do
        if command -v "python$v" >/dev/null && "python$v" -c "import venv, ensurepip" 2>/dev/null; then
            PY="$(command -v "python$v")"; TAG="cp${v/./}"; break
        fi
    done
fi
[ -n "$PY" ] || fail "no usable Python: the bundled one did not run and there is no python3.10-3.12 with venv"
echo "python $PY ($("$PY" -V 2>&1)), wheels/$TAG"

step "4/5 venv + install"
rm -rf venv && "$PY" -m venv venv || fail "could not create the venv"
if ! venv/bin/python -m pip install --quiet --no-index --find-links "wheels/$TAG" -r "wheels/$TAG/requirements.txt"; then
    [ "$ONLINE" = 1 ] || fail "offline install failed; with internet: bash bootstrap.sh --online"
    echo "offline install failed; downloading the pinned versions"
    venv/bin/python -m pip install --quiet -r "wheels/$TAG/requirements.txt" || fail "online install failed"
fi
venv/bin/python -m pip list --disable-pip-version-check 2>/dev/null | grep -i -E "numpy|opencv"

step "5/5 preflight"
(cd code/twin_to_line && ../../venv/bin/python live_inspect.py --preflight "$PKG/preflight" --out "$PKG/preflight_out") \
    || fail "preflight (see preflight_out/preflight.json)"
echo "BOOTSTRAP OK in $(( $(date +%s) - T0 )) s"
echo "run from: cd $PKG/code/twin_to_line   with   ../../venv/bin/python live_inspect.py ..."

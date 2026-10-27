#!/usr/bin/env bash
# Team Ignite sim-to-real gate: 6 tiny Isaac Sim renders through CapSealCam + labels + crops + timing.
# Run ON the Brev Isaac Sim instance:   RATE_PER_HR=<$/hr shown in Brev> bash run_gate.sh   (auto-detects Isaac Sim)
#   or                                 ISAAC_PY=/path/to/python.sh bash run_gate.sh
# Output: ~/ignite_gate/results.tar.gz  (copy it back with: brev copy <instance>:ignite_gate/results.tar.gz .)
set -uo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
WORK="$HOME/ignite_gate"; mkdir -p "$WORK"; cd "$WORK"
LOG="$WORK/run_log.txt"; : > "$LOG"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
T_START=$(date +%s)
log "host $(hostname); $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
log "instance uptime: $(uptime -p 2>/dev/null)   (use this + console start time for total instance runtime)"
log "kit version: $(cat "$KIT/KIT_VERSION" 2>/dev/null || echo unknown)"
( cd "$KIT" && sha256sum -c MANIFEST.sha256 --quiet ) >> "$LOG" 2>&1 && log "kit files match MANIFEST.sha256" || log "WARNING: kit files differ from MANIFEST.sha256 (not the frozen gate kit)"

# 1) Twin: download straight from the MAIC Drive (public link) unless already here
TWIN="${TWIN:-$WORK/csi_digital_twin_v1}"
if [ ! -f "$TWIN/stations/tray_inspection/tray_inspection.usdc" ]; then
  log "downloading twin (213 MB) from Drive"
  FID=1nXRN_VZ_DKGS33jtELwn6z8WR8dbQfKe
  python3 -m pip install -q --user gdown >/dev/null 2>&1
  python3 -m gdown -q "$FID" -O twin.zip 2>/dev/null \
    || curl -sSL -o twin.zip "https://drive.usercontent.google.com/download?id=$FID&export=download&confirm=t" \
    || { log "download failed: run  brev copy ~/Downloads/'Copy of csi_digital_twin.zip' <instance>:ignite_gate/twin.zip  and rerun"; exit 1; }
  [ "$(stat -c %s twin.zip 2>/dev/null || echo 0)" -gt 200000000 ] || { log "twin.zip too small (Drive returned a page, not the file): brev copy it up instead"; exit 1; }
  unzip -q -o twin.zip && rm -f twin.zip
fi
[ -f "$TWIN/stations/tray_inspection/tray_inspection.usdc" ] || { log "twin not found at $TWIN"; exit 1; }
log "twin OK: $TWIN"

# 2) Plain-Python env for USD authoring (does not depend on Isaac's python exposing pxr)
if [ ! -x "$WORK/venv/bin/python" ]; then
  log "creating venv with usd-core"
  if python3 -m venv "$WORK/venv" 2>>"$LOG" && "$WORK/venv/bin/pip" install -q usd-core numpy opencv-python-headless 2>>"$LOG"; then :
  else
    log "venv unavailable; falling back to pip --user"; rm -rf "$WORK/venv"
    python3 -m pip install -q --user usd-core numpy opencv-python-headless 2>>"$LOG" || { log "could not install usd-core"; exit 1; }
  fi
fi
PY="$WORK/venv/bin/python"; [ -x "$PY" ] || PY="python3"
"$PY" -c "from pxr import Usd; import cv2, numpy" || { log "USD/OpenCV import failed in $PY"; exit 1; }
"$PY" "$KIT/make_variants.py" "$TWIN" "$WORK/variants" 2>&1 | tee -a "$LOG" || exit 1

# 3) Find Isaac Sim's python
if [ -z "${ISAAC_PY:-}" ]; then
  for c in /isaac-sim/python.sh "$HOME/isaacsim/python.sh" "$HOME/isaac-sim/python.sh" /opt/isaac-sim/python.sh \
           $(ls -d "$HOME"/.local/share/ov/pkg/isaac*/python.sh 2>/dev/null); do
    [ -x "$c" ] && ISAAC_PY="$c" && break
  done
fi
if [ -z "${ISAAC_PY:-}" ]; then          # pip-installed Isaac Sim (isaacsim package) in some python on PATH
  for p in python3 python; do
    command -v $p >/dev/null && $p -c "import isaacsim" 2>/dev/null && ISAAC_PY="$(command -v $p)" && break
  done
fi
if [ -z "${ISAAC_PY:-}" ]; then
  log "Isaac Sim python not found. Candidates on disk:"; find / -maxdepth 5 -name python.sh -path '*isaac*' 2>/dev/null | tee -a "$LOG"
  docker ps 2>/dev/null | tee -a "$LOG"
  log "Rerun with ISAAC_PY=<path to python.sh>. If Isaac Sim runs in a container, run this script inside it."; exit 2
fi
log "Isaac python: $ISAAC_PY"
[ -f "$(dirname "$ISAAC_PY")/VERSION" ] && log "Isaac Sim VERSION: $(cat "$(dirname "$ISAAC_PY")/VERSION")"

# 4) Renders: baseline first (it also warms the shader cache), then the variants
SUB="${SUBFRAMES:-64}"
mkdir -p renders
for v in baseline t00 t10 t20 t30 liquid; do
  t0=$(date +%s)
  log "render $v (subframes $SUB)"
  "$ISAAC_PY" "$KIT/render_keep.py" --twin "$TWIN" --camera CapSealCam --stage "$WORK/variants/$v/scene.usda" \
      --out "$WORK/renders/$v.png" --subframes "$SUB" >> "$WORK/renders/$v.log" 2>&1
  rc=$?; t1=$(date +%s)
  log "  $v exit=$rc wall=$((t1-t0))s"
  if [ $rc -ne 0 ] || [ ! -f "$WORK/renders/$v.png" ]; then
    log "  FAILED: last lines of renders/$v.log:"; tail -25 "$WORK/renders/$v.log" | tee -a "$LOG"
    [ "$v" = baseline ] && { log "baseline failed; stopping before spending more time"; break; }
  fi
done

# 5) Crops for the four test positions
for v in t00 t10 t20 t30 liquid; do
  [ -f "renders/$v.png" ] && "$PY" "$KIT/cut_crops.py" "renders/$v.png" "variants/$v/labels.json" "crops/$v" | tee -a "$LOG"
done
[ -f renders/baseline.png ] && "$PY" "$KIT/cut_crops.py" renders/baseline.png variants/baseline/labels.json crops/baseline G01,G02,G03,G04,G13,G30 | tee -a "$LOG"

T_END=$(date +%s)
log "script wall time: $((T_END-T_START))s"
if [ -n "${RATE_PER_HR:-}" ]; then   # e.g. RATE_PER_HR=6.0 from the Brev console for the chosen GPU
  log "approx cost of this script: \$$(python3 -c "print(round(($T_END-$T_START)/3600*$RATE_PER_HR,2))") at \$$RATE_PER_HR/hr (add provisioning + idle time from the console)"
fi
cp "$KIT/MANIFEST.sha256" "$KIT/KIT_VERSION" "$WORK/" 2>/dev/null
tar czf results.tar.gz run_log.txt variants renders crops MANIFEST.sha256 KIT_VERSION
log "DONE -> $WORK/results.tar.gz   (now STOP the instance in the Brev console)"

#!/usr/bin/env bash
# D040 v2 render on cc-isaac-gate-2, inside the Isaac container (where /isaac-sim/python.sh is). V2_RENDER.md has the
# Mac side. Writes only under /root/train_v2 and adds /root/train_v1/twin_fix2 next to work1.
#   bash v2_render.sh            check render (<= 3 tries, configs/v2_check.json) -> batch -> crops -> pack
# Exit 0 = pack ready; 2 = check render failed after 3 tries (no batch); 3 = budget guard; 1 = any other stop.
# Dry runs only (never on the instance): W, T1, ISAAC and V2_CRITERIA can be overridden.
set -euo pipefail
W=${W:-/root/train_v2}; T1=${T1:-/root/train_v1}; ISAAC=${ISAAC:-/isaac-sim/python.sh}
CRIT=${V2_CRITERIA:+--criteria $V2_CRITERIA}
RATE=4.38; BUDGET=10.0; BATCH_HOURS=1.45            # 568 scenes x 7.1 s + generation, crops, pack, copy-back, stop
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$W/v2_render.log"; }
cost_now() { awk -v r=$RATE '{printf "%.2f", $1 / 3600 * r}' /proc/uptime; }     # instance up-time x hourly rate

cd "$W"
log "== 0 environment: $(uname -m), $(nproc) CPUs; instance cost so far \$$(cost_now)"
test -x "$ISAAC" || { log "STOP: $ISAAC not found (not in the Isaac container)"; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | tee -a "$W/v2_render.log" || log "nvidia-smi not available"

log "== 1 upload checksums"
(cd upload && sha256sum -c upload.sha256) | tee -a "$W/v2_render.log"

log "== 2 code and work2"
rm -rf code && mkdir code && tar -xzf upload/code_*.tar.gz -C code
C=$W/code/twin_to_line; cat "$C/ARCHIVE_COMMIT" | tee -a "$W/v2_render.log"
for d in csi_digital_twin_v1 twin_fix1; do test -d "$T1/$d" || { log "STOP: $T1/$d missing (the train_v1 twins)"; exit 1; }; done
rm -rf "$T1/twin_fix2" && tar -xzf upload/twin_fix2.tar.gz -C "$T1"

log "== 3 python venv (usd-core, numpy, OpenCV)"
if [ ! -x "$W/venv/bin/python" ]; then
    python3 -m venv "$W/venv"
    "$W/venv/bin/pip" install -q usd-core==26.8 numpy==2.4.6 opencv-python-headless==5.0.0.93 \
        || { log "pinned versions unavailable for $(python3 -V); installing current ones (float-noise-level differences only)";
             "$W/venv/bin/pip" install -q usd-core numpy opencv-python-headless; }
fi
PY=$W/venv/bin/python
INDICES=$($PY -c "import json; print(','.join(map(str, json.load(open('$C/configs/v2_check.json'))['indices'])))")
$PY -c "from pxr import Usd; import numpy, cv2; print('usd', Usd.GetVersion(), 'numpy', numpy.__version__, 'cv2', cv2.__version__)" | tee -a "$W/v2_render.log"
(cd "$C" && $PY twin_fix2.py --check "$T1/twin_fix1" "$T1/twin_fix2") | tail -3 | tee -a "$W/v2_render.log"
(cd "$C" && $PY gen_scenes.py "$T1/twin_fix2" --selftest) | tail -1 | tee -a "$W/v2_render.log"

log "== 4 check render (indices $INDICES, at most 3 tries)"
PASS=0
for N in 1 2 3; do
    D=$W/check_try$N; rm -rf "$D"
    (cd "$C" && $PY gen_scenes.py "$T1/twin_fix2" "$D/gen" --config configs/train_v2.json --indices "$INDICES") | tail -2 | tee -a "$W/v2_render.log"
    $ISAAC "$C/render_batch.py" --twin "$T1/twin_fix2" --scenes "$D/gen/scenes" --out "$D/renders" > "$D/render.log" 2>&1
    log "try $N rendered $(ls "$D/renders"/*.png 2>/dev/null | grep -vc pinhole) scenes; cost so far \$$(cost_now)"
    set +e; (cd "$C" && $PY v2_check.py --renders "$D/renders" --scenes "$D/gen/scenes" --try $N --out "$D/v2_check.json" $CRIT \
             --update configs/train_v2.json configs/tilt_train_v2.json) | tee -a "$W/v2_render.log"; rc=${PIPESTATUS[0]}; set -e
    if [ "$rc" = 0 ]; then PASS=1; break; fi
done
cp "$C/configs/train_v2.json" "$C/configs/tilt_train_v2.json" "$W/"
if [ "$PASS" != 1 ]; then
    log "STOP: check render failed after 3 tries: no batch (D040)"
    tar -cf "$W/train_v2_check_only.tar" -C "$W" check_try1 check_try2 check_try3 v2_render.log train_v2.json tilt_train_v2.json 2>/dev/null || true
    (cd "$W" && sha256sum train_v2_check_only.tar > train_v2_check_only.tar.sha256); exit 2
fi

log "== 5 budget guard: so far \$$(cost_now), batch about $BATCH_HOURS h"
awk -v c="$(cost_now)" -v h=$BATCH_HOURS -v r=$RATE -v b=$BUDGET 'BEGIN { exit !(c + h * r <= b - 0.5) }' \
    || { log "STOP: \$$(cost_now) + batch would exceed \$$BUDGET (with a \$0.50 margin)"; exit 3; }

log "== 6 batch: generate, render, crops"
for S in train tilt; do
    CFG=$([ $S = train ] && echo train_v2.json || echo tilt_train_v2.json); D=$W/data_$S; rm -rf "$D" "$W/gen_$S"
    (cd "$C" && $PY gen_scenes.py "$T1/twin_fix2" "$W/gen_$S" --config "configs/$CFG") | tail -3 | tee -a "$W/v2_render.log"
    $ISAAC "$C/render_batch.py" --twin "$T1/twin_fix2" --scenes "$W/gen_$S/scenes" --out "$D/renders" > "$W/render_$S.log" 2>&1
    log "$S: rendered $(ls "$D/renders"/*.png | grep -vc pinhole) of $(ls "$W/gen_$S/scenes" | wc -l); not ok: $(tail -n +2 "$D/renders/render_log.csv" | grep -vc ',ok,' || true); cost so far \$$(cost_now)"
    (cd "$C" && $PY cut_train_crops.py --renders "$D/renders" --scenes "$W/gen_$S/scenes" --out "$D/crops") | tee -a "$W/v2_render.log"
    mv "$D/crops/crops.csv" "$D/crops.csv"
    tar -czf "$D/scenes.tar.gz" -C "$W/gen_$S" manifest.csv scenes.csv run.json SHA256SUMS scenes
    rm -f "$D/renders"/*_pinhole768.png
done

log "== 7 pack"
cp "$W/v2_render.log" "$W/data_train/"
(cd "$W" && find data_train data_tilt check_try* train_v2.json tilt_train_v2.json -type f | sort | xargs sha256sum > SHA256SUMS)
tar -cf "$W/train_v2_pack.tar" -C "$W" data_train data_tilt check_try* train_v2.json tilt_train_v2.json SHA256SUMS v2_render.log
(cd "$W" && sha256sum train_v2_pack.tar > train_v2_pack.tar.sha256 && cat train_v2_pack.tar.sha256)
log "PACK READY ($(du -h "$W/train_v2_pack.tar" | cut -f1)); cost so far \$$(cost_now). Copy it back, compare sha256, stop the instance."

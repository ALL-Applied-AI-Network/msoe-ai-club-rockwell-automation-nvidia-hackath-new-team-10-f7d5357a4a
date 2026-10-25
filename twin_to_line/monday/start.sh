#!/usr/bin/env bash
# Run from the USB root. All retries create a new visit; nothing is removed.
set -Eeuo pipefail
REASON='setup failed; preserve the visit and do not start live'
trap 'rc=$?; if (( rc )); then printf "\n========== STOP: %s ==========\n" "$REASON" >&2; fi' EXIT
trap 'exit 130' INT TERM
read -r -p 'Mounted USB directory: ' USB
USB=$(cd "$USB" && pwd -P)
read -r -p 'Sunday package basename, without .tar: ' PKG
[[ "$PKG" =~ ^ttl_gb10_[0-9a-f]+$ ]]
test -w "$USB"
mkdir -p "$HOME/ttl"
exec 9>"$HOME/ttl/operator.lock"
REASON="another operator command is active; finish it before setup"
flock -n 9
DAY=$(mktemp -d "$HOME/ttl/visit_$(date +%Y%m%dT%H%M%S)_XXXXXX")
STAGED="$DAY/staged"
IN="$DAY/live_in"
INSTALL="$DAY/install"
PY="$INSTALL/$PKG/venv/bin/python"
mkdir "$STAGED" "$IN" "$INSTALL" "$DAY/attempts" "$DAY/records"
printf '%s\n' 'time,action,operator,notes' > "$DAY/records/interventions.csv"
printf '%s\n' 'setup,status,source_name,copy_name,time,sha256' > "$DAY/records/staging_log.csv"
printf '%s\n' 'No live attempt yet.' > "$DAY/records/live_status.txt"
# Persist even a failed setup, so export/recovery never depends on terminal variables.
umask 077
printf 'USB=%q\nPKG=%q\nDAY=%q\nSTAGED=%q\nIN=%q\nINSTALL=%q\nPY=%q\nREADY=0\n' \
    "$USB" "$PKG" "$DAY" "$STAGED" "$IN" "$INSTALL" "$PY" > "$DAY/operator.env"
cp "$DAY/operator.env" "$HOME/ttl/operator.env"
REASON='USB script checksum failed; do not start live'
(cd "$USB" && sha256sum -c monday.sha256)
for f in start.sh stage.sh seal.sh live.sh export.sh operator.py; do
    cp "$USB/$f" "$HOME/ttl/$f"
done
REASON='package checksum or bootstrap failed; preserve this visit, repeat start for a fresh visit'
cp "$USB/$PKG.tar" "$USB/$PKG.tar.sha256" "$INSTALL/"
cd "$INSTALL"
sha256sum -c "$PKG.tar.sha256"
tar -xf "$PKG.tar"
cd "$PKG"
bash bootstrap.sh 2>&1 | tee "$DAY/records/bootstrap.log"
printf 'READY=1\n' >> "$DAY/operator.env"
cp "$DAY/operator.env" "$HOME/ttl/operator.env"
printf '\n========== SETUP OK: preflight passed; context %s ==========\n' "$DAY/operator.env"

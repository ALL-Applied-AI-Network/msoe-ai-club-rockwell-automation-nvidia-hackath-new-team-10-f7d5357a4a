#!/usr/bin/env bash
# Run from the USB root. Resume a successful visit unless --new-visit is explicit.
set -Eeuo pipefail
REASON='setup failed; preserve the visit and do not start live'
trap 'rc=$?; if (( rc )); then printf "\n========== STOP: %s ==========\n" "$REASON" >&2; fi' EXIT
trap 'exit 130' INT TERM
NEW_VISIT=0
REASON='usage: bash start.sh [--new-visit]'
if (( $# )); then
    test "$#" -eq 1
    test "$1" = --new-visit
    NEW_VISIT=1
fi
USB=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)
shopt -s nullglob
PACKAGES=("$USB"/ttl_gb10_*.tar)
printf 'USB: %s\nPackages found (%s):\n' "$USB" "${#PACKAGES[@]}"
if (( ${#PACKAGES[@]} )); then printf '  %s\n' "${PACKAGES[@]##*/}"; else printf '  (none)\n'; fi
REASON='expected exactly one ttl_gb10_*.tar beside start.sh; fix the USB package selection and retry'
test "${#PACKAGES[@]}" -eq 1
PKG=$(basename "${PACKAGES[0]}" .tar)
[[ "$PKG" =~ ^ttl_gb10_[0-9a-f]+$ ]]
test -w "$USB"
REASON='USB script or package checksum failed; preserve the current visit, fix the USB and retry'
(cd "$USB" && sha256sum -c monday.sha256 && sha256sum -c "$PKG.tar.sha256")
# Keep discovery separate from variables loaded from a previous operator.env.
SOURCE_USB="$USB"
SOURCE_PKG="$PKG"
SOURCE_SHA=$(sha256sum "$USB/$PKG.tar")
SOURCE_SHA=${SOURCE_SHA%% *}
mkdir -p "$HOME/ttl"
exec 9>"$HOME/ttl/operator.lock"
REASON="another operator command is active; finish it before setup"
flock -n 9
if (( ! NEW_VISIT )) && [[ -r "$HOME/ttl/operator.env" ]]; then
    REASON='current visit context is unreadable; preserve it and inspect operator.env'
    source "$HOME/ttl/operator.env"
    if [[ "${READY:-0}" == 1 ]]; then
        # READY=1 is the durable successful-setup record. Older kits retained the
        # verified package checksum in INSTALL rather than a PACKAGE_SHA256 field.
        CURRENT_SHA=${PACKAGE_SHA256:-}
        if [[ -z "$CURRENT_SHA" ]]; then
            REASON='current successful visit has no recorded package checksum; inspect it before --new-visit'
            read -r CURRENT_SHA _ < "$INSTALL/$PKG.tar.sha256"
        fi
        [[ "$CURRENT_SHA" =~ ^[0-9a-f]{64}$ ]]
        REASON="current $DAY used package $CURRENT_SHA; USB has $SOURCE_SHA -> bash start.sh --new-visit"
        test "$CURRENT_SHA" = "$SOURCE_SHA"
        REASON='cannot reconcile the current visit; preserve it and inspect its staging log'
        # Only reads existing copies/logs. Retakes count once per final setup.
        N=$("$PY" -I - "$HOME/ttl/monday_ops.py" <<'PYTHON'
import importlib.util, sys
spec = importlib.util.spec_from_file_location('monday_resume', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(len(module.reconcile(module.read_context())))
PYTHON
        )
        SEALED_STATE=no
        [[ -z "${SEALED:-}" ]] || SEALED_STATE=yes
        # A remounted copy of the same USB can have a different path. Append the
        # current mount without losing any previous context or resetting the visit.
        if [[ "$USB" != "$SOURCE_USB" ]]; then
            printf 'USB=%q\n' "$SOURCE_USB" >> "$DAY/operator.env"
            printf 'USB=%q\n' "$SOURCE_USB" >> "$HOME/ttl/operator.env"
        fi
        printf '\n========== SETUP OK: resumed %s (%s staged, sealed %s) ==========\n' "$DAY" "$N" "$SEALED_STATE"
        exit 0
    fi
fi
USB="$SOURCE_USB"
PKG="$SOURCE_PKG"
REASON='new visit setup failed; preserve this visit and retry'
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
printf 'USB=%q\nPKG=%q\nDAY=%q\nSTAGED=%q\nIN=%q\nINSTALL=%q\nPY=%q\nPACKAGE_SHA256=%q\nREADY=0\n' \
    "$USB" "$PKG" "$DAY" "$STAGED" "$IN" "$INSTALL" "$PY" "$SOURCE_SHA" > "$DAY/operator.env"
cp "$DAY/operator.env" "$HOME/ttl/operator.env"
REASON='USB script checksum failed; do not start live'
(cd "$USB" && sha256sum -c monday.sha256)
for f in start.sh stage.sh seal.sh live.sh export.sh monday_ops.py; do
    if ! cmp -s "$USB/$f" "$HOME/ttl/$f"; then
        cp "$USB/$f" "$HOME/ttl/$f"
    fi
done
REASON='package checksum or bootstrap failed; preserve this visit, repeat start for a fresh visit'
cp "$USB/$PKG.tar" "$USB/$PKG.tar.sha256" "$INSTALL/"
cd "$INSTALL"
sha256sum -c "$PKG.tar.sha256"
tar -xf "$PKG.tar"
cd "$PKG"
bash bootstrap.sh 2>&1 | tee "$DAY/records/bootstrap.log"
printf 'SETUP OK: package %s\n' "$SOURCE_SHA" > "$DAY/records/setup_ok.txt"
printf 'READY=1\n' >> "$DAY/operator.env"
cp "$DAY/operator.env" "$HOME/ttl/operator.env"
printf '\n========== SETUP OK: preflight passed; context %s ==========\n' "$DAY/operator.env"

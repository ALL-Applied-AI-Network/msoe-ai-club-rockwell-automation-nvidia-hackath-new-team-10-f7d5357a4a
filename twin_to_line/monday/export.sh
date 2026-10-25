#!/usr/bin/env bash
# No Python needed: even a failed first bootstrap can be backed up.
set -Eeuo pipefail
REASON='final export failed; keep GB10 files and arrange CSI-assisted recovery'
trap 'rc=$?; if (( rc )); then printf "\n========== STOP: %s ==========\n" "$REASON" >&2; fi' EXIT
trap 'exit 130' INT TERM
test -r "$HOME/ttl/operator.env"
source "$HOME/ttl/operator.env"
exec 9>"$HOME/ttl/operator.lock"
REASON='another operator command is active; stop all writers before export'
flock -n 9
REASON='final export failed; keep GB10 files and arrange CSI-assisted recovery'
shopt -s nullglob
VISITS=("$HOME"/ttl/visit_*)
test "${#VISITS[@]}" -gt 0
NAMES=()
for visit in "${VISITS[@]}"; do
    test -d "$visit"
    NAMES+=("$(basename "$visit")")
done
EXPORT=$(mktemp -d "$HOME/ttl/export_$(date +%Y%m%dT%H%M%S)_XXXXXX")
ARCHIVE="$EXPORT/evidence.tar"
tar -cf "$ARCHIVE" -C "$HOME/ttl" "${NAMES[@]}"
(cd "$EXPORT" && sha256sum evidence.tar > evidence.tar.sha256 && sha256sum -c evidence.tar.sha256)
DEST=$(mktemp -d "$USB/visit_export_$(date +%Y%m%dT%H%M%S)_XXXXXX")
cp "$ARCHIVE" "$ARCHIVE.sha256" "$DEST/"
sync
(cd "$DEST" && sha256sum -c evidence.tar.sha256)
printf '\n========== RESULTS USB OK: %s visits verified at %s; keep GB10 files until Mac verification ==========\n' "${#VISITS[@]}" "$DEST"

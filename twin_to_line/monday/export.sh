#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; if (( rc )); then printf "\n========== STOP: cannot start export; check ~/ttl/operator.env and setup ==========\n" >&2; fi' EXIT
test -r "$HOME/ttl/operator.env"
source "$HOME/ttl/operator.env"
# Export can recover a failed setup if another installed/system Python is usable.
if [[ 'export' == export && ! -x "$PY" ]]; then
    PY=$(command -v python3)
fi
test -x "$PY"
exec "$PY" "$HOME/ttl/operator.py" export

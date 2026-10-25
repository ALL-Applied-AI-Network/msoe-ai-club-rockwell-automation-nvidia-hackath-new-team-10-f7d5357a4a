#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; if (( rc )); then printf "\n========== STOP: cannot start stage; check ~/ttl/operator.env and setup ==========\n" >&2; fi' EXIT
test -r "$HOME/ttl/operator.env"
source "$HOME/ttl/operator.env"
test -x "$PY"
exec "$PY" -I "$HOME/ttl/monday_ops.py" stage

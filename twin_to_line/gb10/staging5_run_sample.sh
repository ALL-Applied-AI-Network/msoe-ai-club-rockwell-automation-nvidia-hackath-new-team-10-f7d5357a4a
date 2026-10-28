#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HOME/ttl_staging5_sample/$(date -u +%Y%m%dT%H%M%SZ)}"
exec bash "$ROOT/live_demo.sh" "replay:$ROOT/preflight/frames" "$OUT" each

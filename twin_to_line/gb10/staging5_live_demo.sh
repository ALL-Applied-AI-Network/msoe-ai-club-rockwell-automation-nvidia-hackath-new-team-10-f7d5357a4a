#!/usr/bin/env bash
# Launch the authenticated staging5 inspector. No inspector/model override is accepted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${1:-}"
OUT="${2:-$HOME/ttl_staging5_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
GATE="${3:-each}"

if [ -z "$SOURCE" ] || { [ "$GATE" != "each" ] && [ "$GATE" != "events" ]; }; then
    echo "Usage: bash $0 folder:/camera/images [OUTPUT_DIR] [each|events]" >&2
    echo "       bash $0 replay:/saved/images [OUTPUT_DIR] each" >&2
    exit 2
fi
[ -f "$ROOT/.READY" ] || { echo "Run bash $ROOT/bootstrap.sh first" >&2; exit 1; }

cd "$ROOT"
sha256sum --quiet -c SHA256SUMS || {
    echo "LAUNCH REFUSED: payload checksum mismatch" >&2
    exit 1
}

echo "======================================================================"
echo "VIAL GUARD staging5 — SHADOW / EVALUATION ONLY"
echo "Fixture G only. No PLC actuation. A PASS is not a product-release signal."
echo "damaged_vial has no training examples and is conservatively UNSURE."
echo "======================================================================"

mkdir -p "$(dirname "$OUT")"
cd app/twin_to_line
exec "$ROOT/.venv/bin/python" live_inspect.py \
    --source "$SOURCE" \
    --out "$OUT" \
    --gate "$GATE" \
    --fixture G \
    --inspector staging5 \
    --model-dir "$ROOT/model/resnet18_staging5" \
    --note "staging5 USB kit; shadow evaluation only; no release authority"


#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/job/in/team_repo}"
DATASET_ROOT="${DATASET_ROOT:-/job/in/dataset}"
MODEL_DIR="${MODEL_DIR:-/job/in/model/resnet18_staging5}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATASET_ROOT}/splits}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATASET_ROOT}/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/job/out/protected_test}"

test -d "$PROJECT_ROOT/twin_to_line"
test -d "$IMAGE_ROOT"
test -f "$SPLIT_ROOT/split_lock.json"
test -f "$SPLIT_ROOT/protected_test/test_manifest.csv"
test -f "$MODEL_DIR/model.pt"
test -f "$MODEL_DIR/run.json"
test -f "$MODEL_DIR/thresholds.json"
test ! -e "$SPLIT_ROOT/train_manifest.csv"
test ! -e "$SPLIT_ROOT/validation_manifest.csv"
mkdir -p /job/out

cd "$PROJECT_ROOT"
python -m pip install --no-cache-dir -r twin_to_line/requirements_train_resnet5.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable in the YAAD evaluation container"
print("CUDA:", torch.cuda.get_device_name(0))
PY
PROJECT_ROOT="$PROJECT_ROOT" DATASET_ROOT="$DATASET_ROOT" MODEL_DIR="$MODEL_DIR" \
  PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  twin_to_line/tests/test_yaad_test_preflight.py -q -p no:cacheprovider \
  2>&1 | tee /job/out/preflight-tests.log

cd "$PROJECT_ROOT/twin_to_line"
python evaluate_resnet5_test.py \
  --model-dir "$MODEL_DIR" \
  --test-manifest "$SPLIT_ROOT/protected_test/test_manifest.csv" \
  --split-lock "$SPLIT_ROOT/split_lock.json" \
  --image-root "$IMAGE_ROOT" \
  --out "$OUTPUT_DIR" \
  --device cuda \
  2>&1 | tee /job/out/evaluation-console.log

test -s "$OUTPUT_DIR/test_metrics.json"
test -s "$OUTPUT_DIR/test_predictions.csv"
sha256sum "$OUTPUT_DIR"/* > /job/out/artifacts.sha256
echo "PROTECTED TEST ARTIFACTS OK: $OUTPUT_DIR"

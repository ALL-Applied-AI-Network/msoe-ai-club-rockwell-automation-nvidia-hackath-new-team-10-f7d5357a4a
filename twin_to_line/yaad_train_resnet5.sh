#!/usr/bin/env bash
set -Eeuo pipefail

# Expected YAAD input bundle:
#   /job/in/team_repo/twin_to_line
#   /job/in/dataset/images
#   /job/in/dataset/splits/{train_manifest.csv,validation_manifest.csv,split_lock.json}
# The protected_test directory need not be copied into the training job.
PROJECT_ROOT="${PROJECT_ROOT:-/job/in/team_repo}"
DATASET_ROOT="${DATASET_ROOT:-/job/in/dataset}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATASET_ROOT}/images}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATASET_ROOT}/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-/job/out/resnet18_staging5}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-128}"
SEED="${SEED:-0}"

test -d "$PROJECT_ROOT/twin_to_line"
test -d "$IMAGE_ROOT"
test -f "$SPLIT_ROOT/train_manifest.csv"
test -f "$SPLIT_ROOT/validation_manifest.csv"
test -f "$SPLIT_ROOT/split_lock.json"
mkdir -p /job/out

cd "$PROJECT_ROOT"
python -m pip install --no-cache-dir -r twin_to_line/requirements_train_resnet5.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable in the YAAD job container"
print("CUDA:", torch.cuda.get_device_name(0))
PY
PROJECT_ROOT="$PROJECT_ROOT" DATASET_ROOT="$DATASET_ROOT" \
  PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  twin_to_line/tests/test_yaad_training_preflight.py -q -p no:cacheprovider \
  2>&1 | tee /job/out/preflight-tests.log

cd "$PROJECT_ROOT/twin_to_line"
python train_resnet5.py \
  --train-manifest "$SPLIT_ROOT/train_manifest.csv" \
  --validation-manifest "$SPLIT_ROOT/validation_manifest.csv" \
  --split-lock "$SPLIT_ROOT/split_lock.json" \
  --image-root "$IMAGE_ROOT" \
  --out "$OUTPUT_DIR" \
  --device cuda \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --seed "$SEED" \
  2>&1 | tee /job/out/training-console.log

test -s "$OUTPUT_DIR/model.pt"
test -s "$OUTPUT_DIR/run.json"
test -s "$OUTPUT_DIR/thresholds.json"
test -s "$OUTPUT_DIR/validation_predictions.csv"
sha256sum "$OUTPUT_DIR"/* > /job/out/artifacts.sha256
echo "TRAINING ARTIFACTS OK: $OUTPUT_DIR"

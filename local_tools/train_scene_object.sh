#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <scene_dir> <model_dir> <object_masks_dir> <num_objects> [gpu=0] [iterations=7000] [lambda_object=0.05]"
  echo
  echo "object_masks_dir must contain class-index PNGs named like frame_000001.png."
  echo "Mask label 0 is ignored/background; labels 1..num_objects-1 are supervised object ids."
  exit 1
fi

SCENE_DIR="$1"
MODEL_DIR="$2"
OBJECT_MASKS_DIR="$3"
NUM_OBJECTS="$4"
GPU_ID="${5:-0}"
ITERATIONS="${6:-7000}"
LAMBDA_OBJECT="${7:-0.05}"

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -d "$SCENE_DIR" ]; then
  echo "Scene directory not found: $SCENE_DIR" >&2
  exit 1
fi

if [ ! -d "$OBJECT_MASKS_DIR" ]; then
  echo "Object masks directory not found: $OBJECT_MASKS_DIR" >&2
  exit 1
fi

SCENE_DIR="$(cd -- "$SCENE_DIR" && pwd)"
OBJECT_MASKS_DIR="$(cd -- "$OBJECT_MASKS_DIR" && pwd)"
mkdir -p "$MODEL_DIR"
MODEL_DIR="$(cd -- "$MODEL_DIR" && pwd)"

cd "$REPO_DIR"
echo "Training object-aware 3DGS..."
echo "  scene:          $SCENE_DIR"
echo "  model_dir:      $MODEL_DIR"
echo "  object_masks:   $OBJECT_MASKS_DIR"
echo "  num_objects:    $NUM_OBJECTS"
echo "  gpu:            $GPU_ID"
echo "  iterations:     $ITERATIONS"
echo "  lambda_object:  $LAMBDA_OBJECT"

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train.py \
  -s "$SCENE_DIR" \
  -m "$MODEL_DIR" \
  --object_masks "$OBJECT_MASKS_DIR" \
  --num_objects "$NUM_OBJECTS" \
  --lambda_object "$LAMBDA_OBJECT" \
  --object_ignore_label 0 \
  --data_device cpu \
  --iterations "$ITERATIONS" \
  --disable_viewer

echo
echo "Done."

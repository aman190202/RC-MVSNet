#!/usr/bin/env bash
# run_colmap2mvsnet.sh
# Loop over every dataset folder and run colmap2mvsnet.py with matching save_folder.
# Usage:  chmod +x run_colmap2mvsnet.sh && ./run_colmap2mvsnet.sh


set -o errexit
set -o nounset
set -o pipefail


DATASET_ROOT="/home/works/coolant-dataset/dataset"
OUTPUT_ROOT="coolant-data"

# Build an array of first-level directories
mapfile -t DIRS < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

TOTAL=${#DIRS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No sub-directories found in $DATASET_ROOT – nothing to do."
  exit 0
fi

echo "Found $TOTAL dataset folders. Starting processing…"
echo

COUNT=0
for DENSE_DIR in "${DIRS[@]}"; do
  ((COUNT++))
  BASENAME=$(basename "$DENSE_DIR")
  SAVE_DIR="$OUTPUT_ROOT/$BASENAME"

  # Show progress
  PCT=$(( COUNT * 100 / TOTAL ))
  printf "[%2d/%d] %-40s  %3d%%\n" "$COUNT" "$TOTAL" "$BASENAME" "$PCT"

  # Create output directory if it doesn’t exist
  mkdir -p "$SAVE_DIR"

  # Run the conversion
  python colmap2mvsnet.py \
    --dense_folder "$DENSE_DIR" \
    --save_folder  "$SAVE_DIR" \
    --minimal_depth \
    --train
done

echo
echo "✓ All $TOTAL folders processed."

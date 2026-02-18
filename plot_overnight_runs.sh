#!/bin/bash

# Create plots directory
PLOTS_DIR="plots_17_02"
mkdir -p "$PLOTS_DIR"

# Input directory containing sweep results
INPUT_DIR="results/input_sweeps/17_02"

# Layers to exclude from plots (space-separated, e.g. "L23 L4AB")
# Valid layers: L23 L4AB L4C L5 L6
EXCLUDE_LAYERS="L23 L4AB"

echo "Plotting overnight sweep results..."
echo "Output directory: $PLOTS_DIR"
if [ -n "$EXCLUDE_LAYERS" ]; then
    echo "Excluding layers: $EXCLUDE_LAYERS"
fi
echo ""

# Build --exclude flag (empty string if no layers to exclude)
EXCLUDE_FLAG=""
if [ -n "$EXCLUDE_LAYERS" ]; then
    EXCLUDE_FLAG="--exclude $EXCLUDE_LAYERS"
fi

# Loop through all .npz files
for npz_file in "$INPUT_DIR"/*.npz; do
    # Get the filename without path and extension
    filename=$(basename "$npz_file" .npz)

    echo "Plotting: $filename"

    # Run plot_input_sweep.py with --diff and save to plots folder
    # shellcheck disable=SC2086
    python plot_input_sweep.py "$npz_file" --diff $EXCLUDE_FLAG --save "$PLOTS_DIR/${filename}.png"
done

echo ""
echo "Done! Plots saved to: $PLOTS_DIR/"

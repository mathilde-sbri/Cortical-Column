#!/bin/bash

# Create plots directory
PLOTS_DIR="plots"
mkdir -p "$PLOTS_DIR"

# Input directory containing sweep results
INPUT_DIR="results/input_sweeps/17_02"

echo "Plotting overnight sweep results..."
echo "Output directory: $PLOTS_DIR"
echo ""

# Loop through all .npz files
for npz_file in "$INPUT_DIR"/*.npz; do
    # Get the filename without path and extension
    filename=$(basename "$npz_file" .npz)

    echo "Plotting: $filename"

    # Run plot_input_sweep.py with --diff and save to plots folder
    python plot_input_sweep.py "$npz_file" --diff --save "$PLOTS_DIR/${filename}.png"
done

echo ""
echo "Done! Plots saved to: $PLOTS_DIR/"

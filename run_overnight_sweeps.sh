#!/bin/bash

# Exit on error
set -e

# Create log directory
LOG_DIR="results/sweep_logs"
mkdir -p "$LOG_DIR"

# Timestamp for this batch run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="$LOG_DIR/batch_run_${TIMESTAMP}.log"
SAVE_DIR="results/input_sweeps/18_02_L4_alpha"

echo "======================================" | tee -a "$BATCH_LOG"
echo "Starting batch run at $(date)" | tee -a "$BATCH_LOG"
echo "Save dir: $SAVE_DIR" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"

# ---------------------------------------------------------------------------
# run_sweep: single-population sweep (legacy --layer/--pop style)
#   $1 = layer   $2 = pop   $3 = input_type   $4 = weight
# ---------------------------------------------------------------------------
run_sweep() {
    local layer=$1
    local pop=$2
    local input_type=$3
    local weight=$4

    local label="${layer}_${pop}_${input_type}_w${weight}"
    echo "--------------------------------------" | tee -a "$BATCH_LOG"
    echo "Running: $label" | tee -a "$BATCH_LOG"
    echo "Started at: $(date)" | tee -a "$BATCH_LOG"

    local run_log="$LOG_DIR/${label}_${TIMESTAMP}.log"

    if python input_sweep.py \
        --layer "$layer" \
        --pop "$pop" \
        --input-type "$input_type" \
        --weight-scale "$weight" \
        --rate-min 0 \
        --rate-max 15 \
        --rate-step 1 \
        --baseline-ms 1000 \
        --stim-ms 1500 \
        --save-dir "$SAVE_DIR" \
        2>&1 | tee "$run_log"; then
        echo "✓ Completed: $label at $(date)" | tee -a "$BATCH_LOG"
    else
        echo "✗ FAILED: $label at $(date)" | tee -a "$BATCH_LOG"
        echo "  Log: $run_log" | tee -a "$BATCH_LOG"
    fi
    echo "" | tee -a "$BATCH_LOG"
}

# ---------------------------------------------------------------------------
# run_multi: multi-population sweep (new --target style)
#   Pass any number of "LAYER:POP:INPUT_TYPE:WEIGHT" strings as arguments.
#   Example: run_multi "L4C:E:AMPA:1.0" "L4C:PV:AMPA:1.0"
# ---------------------------------------------------------------------------
run_multi() {
    local targets_args=""
    local label=""
    for spec in "$@"; do
        targets_args="$targets_args --target $spec"
        # Build a short label: replace : with _ and join with +
        local part
        part=$(echo "$spec" | tr ':' '_')
        label="${label}+${part}"
    done
    label="${label:1}"  # strip leading +

    echo "--------------------------------------" | tee -a "$BATCH_LOG"
    echo "Running multi: $label" | tee -a "$BATCH_LOG"
    echo "Started at: $(date)" | tee -a "$BATCH_LOG"

    local run_log="$LOG_DIR/${label}_${TIMESTAMP}.log"

    # shellcheck disable=SC2086
    if python input_sweep.py \
        $targets_args \
        --rate-min 0 \
        --rate-max 15 \
        --rate-step 1 \
        --baseline-ms 1000 \
        --stim-ms 1500 \
        --save-dir "$SAVE_DIR" \
        2>&1 | tee "$run_log"; then
        echo "✓ Completed: $label at $(date)" | tee -a "$BATCH_LOG"
    else
        echo "✗ FAILED: $label at $(date)" | tee -a "$BATCH_LOG"
        echo "  Log: $run_log" | tee -a "$BATCH_LOG"
    fi
    echo "" | tee -a "$BATCH_LOG"
}


# =============================================================================
# L4C  —  every single-population target
# =============================================================================
echo "### L4C single-population sweeps ###" | tee -a "$BATCH_LOG"

# Excitatory drive
run_sweep "L4C" "E"   "AMPA" 1.0   # pure E drive → how fast does alpha break?
run_sweep "L4C" "E"   "AMPA" 2.0   # stronger E drive

# Inhibitory populations driven by AMPA (excitatory input TO the inhibitory cell)
run_sweep "L4C" "PV"  "AMPA" 1.0   # drive PV only → PV-mediated inhibition of E
run_sweep "L4C" "PV"  "AMPA" 2.0
run_sweep "L4C" "SOM" "AMPA" 1.0   # drive SOM → slow inhibition onto E
run_sweep "L4C" "SOM" "AMPA" 2.0
run_sweep "L4C" "VIP" "AMPA" 1.0   # drive VIP → disinhibition (VIP suppresses SOM)
run_sweep "L4C" "VIP" "AMPA" 2.0

# Inhibitory inputs delivered directly onto E cells (mimicking top-down inh)
run_sweep "L4C" "E"   "PV"  1.0    # PV-type GABA directly onto E
run_sweep "L4C" "E"   "SOM" 1.0    # SOM-type GABA directly onto E

# =============================================================================
# L4C  —  multi-population combinations
# =============================================================================
echo "" | tee -a "$BATCH_LOG"
echo "### L4C multi-population sweeps ###" | tee -a "$BATCH_LOG"

# E + PV together (canonical feedforward drive: both excited equally)
run_multi "L4C:E:AMPA:1.0" "L4C:PV:AMPA:1.0"

# E stronger than PV (E dominates → more likely gamma)
run_multi "L4C:E:AMPA:2.0" "L4C:PV:AMPA:1.0"

# PV stronger than E (inhibition-dominated → may suppress alpha differently)
run_multi "L4C:E:AMPA:1.0" "L4C:PV:AMPA:2.0"

# E + SOM (slow inh pathway recruited alongside E)
run_multi "L4C:E:AMPA:1.0" "L4C:SOM:AMPA:1.0"

# E + VIP (disinhibitory: VIP suppresses SOM, frees E from slow inh)
run_multi "L4C:E:AMPA:1.0" "L4C:VIP:AMPA:1.0"

# E + PV + SOM (full feedforward: all excitatory and slow inh recruited)
run_multi "L4C:E:AMPA:1.0" "L4C:PV:AMPA:1.0" "L4C:SOM:AMPA:1.0"

# E + PV + VIP (disinhibitory + fast inh together)
run_multi "L4C:E:AMPA:1.0" "L4C:PV:AMPA:1.0" "L4C:VIP:AMPA:1.0"

# PV + SOM only (no direct E drive: purely inhibitory push)
run_multi "L4C:PV:AMPA:1.0" "L4C:SOM:AMPA:1.0"

# VIP + SOM (VIP suppresses SOM: disinhibition of SOM targets)
run_multi "L4C:VIP:AMPA:1.0" "L4C:SOM:AMPA:1.0"


# =============================================================================
# L4AB  —  every single-population target
# =============================================================================
echo "" | tee -a "$BATCH_LOG"
echo "### L4AB single-population sweeps ###" | tee -a "$BATCH_LOG"

run_sweep "L4AB" "E"   "AMPA" 1.0
run_sweep "L4AB" "E"   "AMPA" 2.0
run_sweep "L4AB" "PV"  "AMPA" 1.0
run_sweep "L4AB" "PV"  "AMPA" 2.0
run_sweep "L4AB" "SOM" "AMPA" 1.0
run_sweep "L4AB" "SOM" "AMPA" 2.0
run_sweep "L4AB" "VIP" "AMPA" 1.0
run_sweep "L4AB" "VIP" "AMPA" 2.0

run_sweep "L4AB" "E"   "PV"  1.0   # PV-type GABA directly onto E
run_sweep "L4AB" "E"   "SOM" 1.0   # SOM-type GABA directly onto E

# =============================================================================
# L4AB  —  multi-population combinations
# =============================================================================
echo "" | tee -a "$BATCH_LOG"
echo "### L4AB multi-population sweeps ###" | tee -a "$BATCH_LOG"

run_multi "L4AB:E:AMPA:1.0" "L4AB:PV:AMPA:1.0"
run_multi "L4AB:E:AMPA:2.0" "L4AB:PV:AMPA:1.0"
run_multi "L4AB:E:AMPA:1.0" "L4AB:PV:AMPA:2.0"
run_multi "L4AB:E:AMPA:1.0" "L4AB:SOM:AMPA:1.0"
run_multi "L4AB:E:AMPA:1.0" "L4AB:VIP:AMPA:1.0"
run_multi "L4AB:E:AMPA:1.0" "L4AB:PV:AMPA:1.0" "L4AB:SOM:AMPA:1.0"
run_multi "L4AB:E:AMPA:1.0" "L4AB:PV:AMPA:1.0" "L4AB:VIP:AMPA:1.0"
run_multi "L4AB:PV:AMPA:1.0" "L4AB:SOM:AMPA:1.0"
run_multi "L4AB:VIP:AMPA:1.0" "L4AB:SOM:AMPA:1.0"

# =============================================================================
# Cross-layer combos (L4C + L4AB driven simultaneously)
# =============================================================================
echo "" | tee -a "$BATCH_LOG"
echo "### Cross-layer L4C+L4AB sweeps ###" | tee -a "$BATCH_LOG"

# Both E layers together
run_multi "L4C:E:AMPA:1.0" "L4AB:E:AMPA:1.0"

# Both E + both PV (full feedforward into both L4 sublayers)
run_multi "L4C:E:AMPA:1.0" "L4C:PV:AMPA:1.0" "L4AB:E:AMPA:1.0" "L4AB:PV:AMPA:1.0"

# L4C E only vs L4AB E only (isolate which layer is driving alpha change)
# (already done individually above; cross combo adds new info)
run_multi "L4C:E:AMPA:2.0" "L4AB:E:AMPA:1.0"
run_multi "L4C:E:AMPA:1.0" "L4AB:E:AMPA:2.0"


# =============================================================================
echo "" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "Batch run completed at $(date)" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Summary log: $BATCH_LOG"
echo "Results in:  $SAVE_DIR"
echo "Run logs in: $LOG_DIR"

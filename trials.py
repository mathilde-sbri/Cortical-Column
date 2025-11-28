import os
import numpy as np
import brian2 as b2
from brian2 import *
from config.config_test2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *

def run_single_trial(
    config,
    trial_id=0,
    base_seed=None,
    baseline_ms=1000,    
    post_ms=500,
    stim_jitter_ms=10,
    rate_variability=0.2,
    # Stimulus condition parameters
    stim_rate_multiplier=1.0,  # Scale all rates by this factor
    stim_n_multiplier=1.0,     # Scale number of target neurons by this factor
    condition_name="baseline",
    verbose=True,
):
    """
    Run a single trial and save only the raw spike/state/rate monitor data.
    All expensive LFP/CSD processing is deferred to batch post-processing.
    
    baseline_ms : float
        Mean duration of pre-stimulus simulation.
    post_ms : float
        Duration of post-stimulus simulation (after adding PoissonInput).
    stim_jitter_ms : float
        Stimulus onset jitter (uniform random ±stim_jitter_ms).
    rate_variability : float
        Fractional variability in firing rates (e.g., 0.2 = ±20%).
    stim_rate_multiplier : float
        Multiplier for stimulus rates (e.g., 2.0 = double the base rate).
    stim_n_multiplier : float
        Multiplier for number of target neurons (e.g., 0.5 = half the neurons).
    condition_name : str
        Name of the stimulus condition for tracking.
    """

    # --------- RNG / Brian2 setup ----------
    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    trial_seed = int(base_seed + trial_id)
    np.random.seed(trial_seed)
    b2.seed(trial_seed)

    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    # Add variability to stimulus onset
    actual_baseline_ms = baseline_ms + np.random.uniform(-stim_jitter_ms, stim_jitter_ms)
    actual_baseline_ms = max(100, actual_baseline_ms)  # Ensure minimum baseline

    if verbose:
        print(f"\n=== Trial {trial_id} | Condition: {condition_name} | Seed: {trial_seed} ===")
        print(f"Stimulus onset: {actual_baseline_ms:.1f} ms")
        print(f"Rate multiplier: {stim_rate_multiplier}x, N multiplier: {stim_n_multiplier}x")

    # --------- Build network ----------
    column = CorticalColumn(column_id=0, config=config)

    # add heterogeneity
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, config)

    all_monitors = column.get_all_monitors()

    # --------- Baseline run ----------
    column.network.run(actual_baseline_ms * ms)

    # --------- Stimulus setup with variability ----------
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    w_ext_NMDA = CONFIG['synapses']['Q']['EXT_NMDA']
    
    L4C = column.layers['L4C']
    cfg_L4C = CONFIG['layers']['L4C']
    L4C_E_grp = L4C.neuron_groups['E']
    
    # Apply multipliers to number of inputs and rates
    N_stim_E = int(cfg_L4C['poisson_inputs']['E']['N'] * stim_n_multiplier)
    base_rate_E = 10  # Hz
    stim_rate_E = (base_rate_E * stim_rate_multiplier * 
                   (1 + np.random.uniform(-rate_variability, rate_variability))) * Hz
    
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA', 
                             N=N_stim_E, rate=stim_rate_E, weight=w_ext_AMPA)
    L4C_E_stimNMDA = PoissonInput(L4C_E_grp, 'gE_NMDA', 
                             N=N_stim_E, rate=stim_rate_E, weight=w_ext_NMDA)
    
    L4C_PV_grp = L4C.neuron_groups['PV']
    N_stim_PV = int(cfg_L4C['poisson_inputs']['PV']['N'] * stim_n_multiplier)
    base_rate_PV = 1  # Hz
    stim_rate_PV = (base_rate_PV * stim_rate_multiplier * 
                    (1 + np.random.uniform(-rate_variability, rate_variability))) * Hz
    
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA', 
                              N=N_stim_PV, rate=stim_rate_PV, weight=w_ext_AMPA)
    
    # L6 receives weaker but concurrent thalamic input
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    L6_E_grp = L6.neuron_groups['E']
    
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'] * stim_n_multiplier)
    base_rate_L6_E = 10  # Hz
    stim_rate_L6_E = (base_rate_L6_E * stim_rate_multiplier * 
                      (1 + np.random.uniform(-rate_variability, rate_variability))) * Hz
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=N_stim_L6_E, rate=stim_rate_L6_E, weight=w_ext_AMPA)
    
    N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'] * stim_n_multiplier)
    base_rate_L6_PV = 5  # Hz
    stim_rate_L6_PV = (base_rate_L6_PV * stim_rate_multiplier * 
                       (1 + np.random.uniform(-rate_variability, rate_variability))) * Hz
    
    L6_PV_stim = PoissonInput(L6.neuron_groups['PV'], 'gE_AMPA',
                              N=N_stim_L6_PV, rate=stim_rate_L6_PV, weight=w_ext_AMPA)

    column.network.add(L4C_E_stimAMPA, L4C_E_stimNMDA, L4C_PV_stim, L6_E_stim, L6_PV_stim)

    # --------- Post-stimulus simulation ----------
    column.network.run(post_ms * ms)

    if verbose:
        print(f"Complete - L4E: {N_stim_E} inputs @ {stim_rate_E/Hz:.1f} Hz")

    # --------- Extract and save ONLY monitor data ----------
    spike_data = {}
    state_data = {}
    rate_data = {}
    neuron_group_info = {}

    for layer_name, monitors in all_monitors.items():
        # Spike monitors
        spike_data[layer_name] = {}
        for mon_name, mon in monitors.items():
            if 'spikes' in mon_name:
                spike_data[layer_name][mon_name] = {
                    'i': np.array(mon.i),
                    't': np.array(mon.t / ms)
                }
        
        # State monitors
        state_data[layer_name] = {}
        for mon_name, mon in monitors.items():
            if 'state' in mon_name:
                state_dict = {}
                for var_name in mon.record_variables:
                    state_dict[var_name] = np.array(getattr(mon, var_name))
                state_dict['t'] = np.array(mon.t / ms)
                state_data[layer_name][mon_name] = state_dict
        
        # Rate monitors
        rate_data[layer_name] = {}
        for mon_name, mon in monitors.items():
            if 'rate' in mon_name:
                if len(mon.t) > 0:
                    rate_data[layer_name][mon_name] = {
                        't_ms': np.array(mon.t / ms),
                        'rate_hz': np.array(mon.rate / Hz)
                    }
        
        # Store neuron group info needed for later LFP calculation
        layer = column.layers[layer_name]
        neuron_group_info[layer_name] = {}
        for pop_name, ng in layer.neuron_groups.items():
            neuron_group_info[layer_name][pop_name] = {
                'N': len(ng),
                'x': np.array(ng.x / meter),
                'y': np.array(ng.y / meter),
                'z': np.array(ng.z / meter)
            }

    # --------- Pack trial data ----------
    data = {
        "trial_id": trial_id,
        "seed": trial_seed,
        "condition_name": condition_name,
        "stim_rate_multiplier": stim_rate_multiplier,
        "stim_n_multiplier": stim_n_multiplier,
        "spike_data": spike_data,
        "state_data": state_data,
        "rate_data": rate_data,
        "neuron_group_info": neuron_group_info,
        "baseline_ms": baseline_ms,
        "actual_baseline_ms": actual_baseline_ms,
        "post_ms": post_ms,
        "stim_onset_ms": actual_baseline_ms,
        "stim_params": {
            "L4C_E_N": N_stim_E,
            "L4C_E_rate": stim_rate_E / Hz,
            "L4C_PV_N": N_stim_PV,
            "L4C_PV_rate": stim_rate_PV / Hz,
            "L6_E_N": N_stim_L6_E,
            "L6_E_rate": stim_rate_L6_E / Hz,
            "L6_PV_N": N_stim_L6_PV,
            "L6_PV_rate": stim_rate_L6_PV / Hz,
        }
    }

    if verbose:
        print(f"Trial {trial_id} finished.\n")

    return data


def run_multiple_conditions(
    config,
    conditions,  # List of dicts with condition parameters
    trials_per_condition=20,
    base_seed=None,
    baseline_ms=1000,
    post_ms=500,
    stim_jitter_ms=10,
    rate_variability=0.2,
    save_dir="results/lfp_trials_conditions",
    verbose=True,
):
    """
    Run multiple trials across different stimulus conditions.
    
    conditions : list of dict
        Each dict should contain:
        - 'name': str, condition identifier
        - 'rate_multiplier': float, scale factor for rates
        - 'n_multiplier': float, scale factor for number of inputs
    
    Example:
        conditions = [
            {'name': 'low_rate', 'rate_multiplier': 0.5, 'n_multiplier': 1.0},
            {'name': 'baseline', 'rate_multiplier': 1.0, 'n_multiplier': 1.0},
            {'name': 'high_rate', 'rate_multiplier': 2.0, 'n_multiplier': 1.0},
            {'name': 'low_n', 'rate_multiplier': 1.0, 'n_multiplier': 0.5},
            {'name': 'high_n', 'rate_multiplier': 1.0, 'n_multiplier': 2.0},
        ]
    """
    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    os.makedirs(save_dir, exist_ok=True)
    
    # Save condition information
    conditions_file = os.path.join(save_dir, "conditions_info.npz")
    np.savez_compressed(conditions_file, conditions=conditions)
    
    trial_counter = 0
    
    for cond_idx, condition in enumerate(conditions):
        cond_name = condition['name']
        rate_mult = condition['rate_multiplier']
        n_mult = condition['n_multiplier']
        
        print(f"\n{'='*60}")
        print(f"CONDITION {cond_idx+1}/{len(conditions)}: {cond_name}")
        print(f"Rate multiplier: {rate_mult}x, N multiplier: {n_mult}x")
        print(f"Running {trials_per_condition} trials...")
        print(f"{'='*60}")
        
        for trial_in_cond in range(trials_per_condition):
            data = run_single_trial(
                config=config,
                trial_id=trial_counter,
                base_seed=base_seed,
                baseline_ms=baseline_ms,
                post_ms=post_ms,
                stim_jitter_ms=stim_jitter_ms,
                rate_variability=rate_variability,
                stim_rate_multiplier=rate_mult,
                stim_n_multiplier=n_mult,
                condition_name=cond_name,
                verbose=verbose,
            )

            # Save with both global trial ID and condition info
            fname = os.path.join(save_dir, 
                                f"trial_{trial_counter:03d}_{cond_name}_raw.npz")
            np.savez_compressed(fname, **data)

            if verbose:
                print(f"Saved to {fname}")
            
            trial_counter += 1
    
    print(f"\n{'='*60}")
    print(f"Completed all {trial_counter} trials across {len(conditions)} conditions")
    print(f"{'='*60}\n")


def process_trials_batch(
    raw_data_dir="results/lfp_trials_conditions",
    output_dir="results/lfp_trials_processed",
    config=None,
    fs=10000,
    electrode_positions=None,
    verbose=True,
):
    """
    Batch process all raw trial data to compute LFP, CSD, bipolar, etc.
    This is done AFTER all trials are saved to maximize efficiency.
    """
    
    if electrode_positions is None:
        electrode_positions = [
            (0, 0, -0.94), (0, 0, -0.79), (0, 0, -0.64), (0, 0, -0.49),
            (0, 0, -0.34), (0, 0, -0.19), (0, 0, -0.04), (0, 0, 0.10),
            (0, 0, 0.26), (0, 0, 0.40), (0, 0, 0.56), (0, 0, 0.70),
            (0, 0, 0.86), (0, 0, 1.00), (0, 0, 1.16), (0, 0, 1.30),
        ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all raw trial files
    raw_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('_raw.npz')])
    
    if verbose:
        print(f"\n=== Batch processing {len(raw_files)} trials ===\n")
    
    for raw_file in raw_files:
        raw_path = os.path.join(raw_data_dir, raw_file)
        
        if verbose:
            print(f"Processing {raw_file}...")
        
        # Load raw data
        raw_data = np.load(raw_path, allow_pickle=True)
        
        trial_id = int(raw_data['trial_id'])
        spike_data = raw_data['spike_data'].item()
        neuron_group_info = raw_data['neuron_group_info'].item()
        actual_baseline_ms = float(raw_data['actual_baseline_ms'])
        post_ms = float(raw_data['post_ms'])
        total_sim_ms = actual_baseline_ms + post_ms
        
        # Reconstruct spike monitors format for LFP calculation
        spike_monitors = {}
        neuron_groups = {}
        
        for layer_name in spike_data.keys():
            spike_monitors[layer_name] = spike_data[layer_name]
            neuron_groups[layer_name] = neuron_group_info[layer_name]
        
        # Compute LFP
        if verbose:
            print("  Computing LFP...")
        
        lfp_signals, time_array = calculate_lfp_kernel_method(
            spike_monitors,
            neuron_groups,
            config['layers'],
            electrode_positions,
            fs=fs,
            sim_duration_ms=total_sim_ms,
        )
        
        # Compute CSD
        if verbose:
            print("  Computing CSD...")
        
        csd, csd_depths, csd_sort_idx = compute_csd_from_lfp(
            lfp_signals,
            electrode_positions,
            sigma=0.3,
            vaknin=True,
        )
        
        # Compute bipolar
        if verbose:
            print("  Computing bipolar LFP...")
        
        bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
            lfp_signals,
            electrode_positions,
        )
        
        # Pack processed data
        n_elec = len(lfp_signals)
        lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])
        bipolar_matrix = np.vstack([bipolar_signals[i] for i in range(len(bipolar_signals))])
        
        processed_data = {
            "trial_id": trial_id,
            "seed": int(raw_data['seed']),
            "condition_name": str(raw_data['condition_name']),
            "stim_rate_multiplier": float(raw_data['stim_rate_multiplier']),
            "stim_n_multiplier": float(raw_data['stim_n_multiplier']),
            "time_array_ms": np.array(time_array),
            "electrode_positions": np.array(electrode_positions),
            "lfp_matrix": lfp_matrix,
            "bipolar_matrix": bipolar_matrix,
            "csd": np.array(csd),
            "csd_depths": np.array(csd_depths),
            "csd_sort_idx": np.array(csd_sort_idx),
            "channel_labels": np.array(channel_labels, dtype=object),
            "channel_depths": np.array(channel_depths),
            "rate_data": raw_data['rate_data'].item(),
            "baseline_ms": float(raw_data['baseline_ms']),
            "actual_baseline_ms": actual_baseline_ms,
            "post_ms": post_ms,
            "stim_onset_ms": float(raw_data['stim_onset_ms']),
            "stim_params": raw_data['stim_params'].item(),
        }
        
        # Save processed data
        output_file = raw_file.replace('_raw.npz', '_processed.npz')
        output_path = os.path.join(output_dir, output_file)
        np.savez_compressed(output_path, **processed_data)
        
        if verbose:
            print(f"  Saved to {output_path}\n")
    
    if verbose:
        print("=== Batch processing complete ===")


if __name__ == "__main__":
    # Define stimulus conditions to test
    conditions = [
        # Vary firing rates
        {'name': 'rate_0.5x', 'rate_multiplier': 0.5, 'n_multiplier': 1.0},
        {'name': 'rate_0.75x', 'rate_multiplier': 0.75, 'n_multiplier': 1.0},
        {'name': 'rate_1.0x', 'rate_multiplier': 1.0, 'n_multiplier': 1.0},  # baseline
        {'name': 'rate_1.5x', 'rate_multiplier': 1.5, 'n_multiplier': 1.0},
        {'name': 'rate_2.0x', 'rate_multiplier': 2.0, 'n_multiplier': 1.0},
        
        # Vary number of target neurons
        {'name': 'n_0.5x', 'rate_multiplier': 1.0, 'n_multiplier': 0.5},
        {'name': 'n_0.75x', 'rate_multiplier': 1.0, 'n_multiplier': 0.75},
        {'name': 'n_1.5x', 'rate_multiplier': 1.0, 'n_multiplier': 1.5},
        {'name': 'n_2.0x', 'rate_multiplier': 1.0, 'n_multiplier': 2.0},
        
        # Combined variations
        {'name': 'both_low', 'rate_multiplier': 0.75, 'n_multiplier': 0.75},
        {'name': 'both_high', 'rate_multiplier': 1.5, 'n_multiplier': 1.5},
    ]
    
    # Step 1: Run all trials across conditions (fast - no LFP processing)
    print("STEP 1: Running trials across multiple conditions...")
    run_multiple_conditions(
        CONFIG,
        conditions=conditions,
        trials_per_condition=20,
        baseline_ms=1000,
        post_ms=500,
        stim_jitter_ms=10,
        rate_variability=0.2,
        save_dir="results/lfp_trials_conditions",
        verbose=True,
    )
    
    # Step 2: Batch process all trials (compute LFP, CSD, etc.)
    print("\n\nSTEP 2: Batch processing all trials...")
    process_trials_batch(
        raw_data_dir="results/lfp_trials_conditions",
        output_dir="results/lfp_trials_processed",
        config=CONFIG,
        fs=10000,
        verbose=True,
    )
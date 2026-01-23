
import os
import numpy as np
import brian2 as b2
from brian2 import *
from config.config import CONFIG
from src.column import CorticalColumn
from src.analysis import calculate_lfp_kernel_method, compute_power_spectrum
import copy


def run_single_rate(
    config,
    targets,
    stim_rate_hz,
    trial_id=0,
    base_seed=None,
    baseline_ms=1000,
    stim_ms=1500,
    fs=10000,
    verbose=True,
):
    """
    Run a single stimulation trial.

    Parameters
    ----------
    targets : dict
        Dictionary mapping (layer, pop, input_type) tuples to weight scales.
        Example: {('L4C', 'E', 'AMPA'): 1.0, ('L4C', 'PV', 'PV'): 2.0}

        input_type can be:
        - 'AMPA': excitatory input targeting gE_AMPA (default if using 2-tuple key)
        - 'PV': inhibitory input targeting gPV (PV-like GABA)
        - 'SOM': inhibitory input targeting gSOM (SOM-like GABA)
        - 'VIP': inhibitory input targeting gVIP (VIP-like GABA)

        For backwards compatibility, 2-tuple keys like ('L4C', 'E') are
        interpreted as ('L4C', 'E', 'AMPA').

        The weight scale multiplies the base external weight.
    """


    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    trial_seed = int(base_seed + trial_id)
    np.random.seed(trial_seed)
    b2.seed(trial_seed)

    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    if verbose:
        # Handle both 2-tuple and 3-tuple keys for display
        parts = []
        for key, scale in targets.items():
            if len(key) == 2:
                layer, pop = key
                input_type = 'AMPA'
            else:
                layer, pop, input_type = key
            parts.append(f"{pop}_{layer}[{input_type}](x{scale})")
        targets_str = ", ".join(parts)
        print(f"\n=== Stimulating {targets_str} at {stim_rate_hz} Hz ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=config)
    all_monitors = column.get_all_monitors()

    # Add external stimulation BEFORE running baseline - so network stabilizes with input
    w_ext_AMPA = config['synapses']['Q']['EXT_AMPA']

    # Mapping from input_type to conductance variable and base weight
    input_type_map = {
        'AMPA': ('gE_AMPA', w_ext_AMPA),
        'PV': ('gPV', w_ext_AMPA),      # Use same base weight, can be scaled
        'SOM': ('gSOM', w_ext_AMPA),
        'VIP': ('gVIP', w_ext_AMPA),
    }

    if stim_rate_hz > 0:
        for target_key, weight_scale in targets.items():
            # Handle both 2-tuple (backwards compat) and 3-tuple keys
            if len(target_key) == 2:
                target_layer, target_pop = target_key
                input_type = 'AMPA'
            else:
                target_layer, target_pop, input_type = target_key

            if input_type not in input_type_map:
                raise ValueError(f"Unknown input_type '{input_type}'. Must be one of: {list(input_type_map.keys())}")

            conductance_var, base_weight = input_type_map[input_type]

            target_layer_obj = column.layers[target_layer]
            cfg_target = config['layers'][target_layer]
            target_grp = target_layer_obj.neuron_groups[target_pop]
            N_stim = int(cfg_target['poisson_inputs'][target_pop]['N'])

            stim_input = PoissonInput(target_grp, conductance_var,
                                      N=N_stim,
                                      rate=stim_rate_hz * Hz,
                                      weight=base_weight * weight_scale)
            column.network.add(stim_input)

            if verbose:
                print(f"Added stimulation: {target_pop}_{target_layer} -> {conductance_var} at {stim_rate_hz} Hz (weight x{weight_scale})")

    if verbose:
        print(f"Running baseline ({baseline_ms} ms) for stabilization...")
    column.network.run(baseline_ms * ms)

    if verbose:
        print(f"Running analysis period ({stim_ms} ms)...")
    column.network.run(stim_ms * ms)

    if verbose:
        print("Simulation complete. Computing LFP...")

    spike_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    electrode_positions = config['electrode_positions']
    total_sim_ms = baseline_ms + stim_ms

    # Compute LFP using kernel method
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        config['layers'],
        electrode_positions,
        fs=fs,
        sim_duration_ms=total_sim_ms,
    )

    analysis_start_ms = baseline_ms + 500
    analysis_end_ms = baseline_ms + 1500

    start_idx = int(analysis_start_ms * fs / 1000)
    end_idx = int(analysis_end_ms * fs / 1000)


    window_samples = end_idx - start_idx
    nperseg = window_samples // 2

    power_spectra = {}
    freqs = None

    for elec_idx in range(len(electrode_positions)):
        lfp_window = lfp_signals[elec_idx][start_idx:end_idx]
        freq, psd = compute_power_spectrum(lfp_window, fs=fs, nperseg=nperseg)
        power_spectra[elec_idx] = psd
        if freqs is None:
            freqs = freq

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])

    psd_matrix = np.vstack([power_spectra[i] for i in range(n_elec)])

    data = {
        "targets": targets,
        "stim_rate_hz": stim_rate_hz,
        "trial_id": trial_id,
        "seed": trial_seed,
        "time_array_ms": np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "lfp_matrix": lfp_matrix,
        "frequencies": freqs,
        "psd_matrix": psd_matrix,
        "baseline_ms": baseline_ms,
        "stim_ms": stim_ms,
        "analysis_start_ms": analysis_start_ms,
        "analysis_end_ms": analysis_end_ms,
        "fs": fs,
    }

    if verbose:
        print(f"Done. PSD matrix shape: {psd_matrix.shape}")

    return data


def run_rate_sweep(
    config,
    targets,
    rate_values,
    base_seed=None,
    baseline_ms=1000,
    stim_ms=1500,
    fs=10000,
    save_dir="results/input_sweeps",
    verbose=True,
):
    """
    Run a sweep over different stimulation rates.

    Parameters
    ----------
    targets : dict
        Dictionary mapping (layer, pop, input_type) tuples to weight scales.
        Example: {('L4C', 'E', 'AMPA'): 1.0, ('L4C', 'PV', 'PV'): 2.0}

        input_type can be 'AMPA', 'PV', 'SOM', or 'VIP'.
        For backwards compatibility, 2-tuple keys are interpreted as AMPA.
    """

    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    os.makedirs(save_dir, exist_ok=True)

    all_psd_matrices = []
    all_rates = []
    freqs = None
    electrode_positions = None

    for i, rate in enumerate(rate_values):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Rate sweep: {i+1}/{len(rate_values)} - {rate} Hz")
            print(f"{'='*60}")

        data = run_single_rate(
            config=config,
            targets=targets,
            stim_rate_hz=rate,
            trial_id=i,
            base_seed=base_seed,
            baseline_ms=baseline_ms,
            stim_ms=stim_ms,
            fs=fs,
            verbose=verbose,
        )

        all_psd_matrices.append(data['psd_matrix'])
        all_rates.append(rate)

        if freqs is None:
            freqs = data['frequencies']
            electrode_positions = data['electrode_positions']

    psd_stack = np.stack(all_psd_matrices, axis=0)

    rate_min = int(min(rate_values))
    rate_max = int(max(rate_values))
    # Build filename from targets (handle both 2-tuple and 3-tuple keys)
    targets_str_parts = []
    target_layers = []
    target_pops = []
    input_types = []
    weight_scales = []

    for target_key, scale in targets.items():
        if len(target_key) == 2:
            layer, pop = target_key
            input_type = 'AMPA'
        else:
            layer, pop, input_type = target_key

        targets_str_parts.append(f"{pop}{layer}_{input_type}")
        target_layers.append(layer)
        target_pops.append(pop)
        input_types.append(input_type)
        weight_scales.append(scale)

    targets_str = "_".join(targets_str_parts)
    fname = f"sweep_{targets_str}_{rate_min}-{rate_max}Hz.npz"
    fpath = os.path.join(save_dir, fname)

    # Convert to numpy arrays for saving
    target_layers = np.array(target_layers)
    target_pops = np.array(target_pops)
    input_types = np.array(input_types)
    weight_scales = np.array(weight_scales)

    np.savez_compressed(
        fpath,
        target_layers=target_layers,
        target_pops=target_pops,
        input_types=input_types,
        weight_scales=weight_scales,
        rate_values=np.array(rate_values),
        frequencies=freqs,
        psd_stack=psd_stack,
        electrode_positions=electrode_positions,
        baseline_ms=baseline_ms,
        stim_ms=stim_ms,
        fs=fs,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sweep complete! Saved to: {fpath}")
        print(f"PSD stack shape: {psd_stack.shape}")
        print(f"  - {len(rate_values)} rates")
        print(f"  - {psd_stack.shape[1]} electrodes")
        print(f"  - {psd_stack.shape[2]} frequency bins")
        print(f"{'='*60}")

    return fpath


if __name__ == "__main__":
    rate_values = np.arange(0, 16, 1)

    # Define targets as a dict: {(layer, pop, input_type): weight_scale}
    # input_type can be: 'AMPA', 'PV', 'SOM', 'VIP'
    #
    # Examples:
    #   AMPA input to L4C E cells:     ('L4C', 'E', 'AMPA'): 1.0
    #   PV-like GABA to L4C E cells:   ('L4C', 'E', 'PV'): 1.0
    #   SOM-like GABA to L23 E cells:  ('L23', 'E', 'SOM'): 1.0
    #
    # For backwards compatibility, 2-tuple keys default to AMPA:
    #   ('L1', 'VIP'): 1.0  is equivalent to  ('L1', 'VIP', 'AMPA'): 1.0
    targets = {
        ('L4C', 'E', 'AMPA'): 1.0,
    }

    result_path = run_rate_sweep(
        config=CONFIG,
        targets=targets,
        rate_values=rate_values,
        baseline_ms=1000,
        stim_ms=1500,
        fs=10000,
        save_dir="results/input_sweeps",
        verbose=True,
    )

    print(f"\nResults saved to: {result_path}")

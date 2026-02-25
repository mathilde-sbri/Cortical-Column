import os
import numpy as np
import brian2 as b2
from brian2 import *
import copy as _copy
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.analysis import compute_power_spectrum, calculate_lfp_kernel_method
import argparse


def run_single_p(
    config,
    p,
    trial_id=0,
    base_seed=None,
    sim_ms=2500,
    analysis_start_ms=1000,
    fs=10000,
    verbose=True,
):
    """
    Run a single simulation with inter-layer connection probabilities scaled by p.

    Parameters
    ----------
    config : dict
        CONFIG dict (will be deep-copied and modified internally)
    p : float
        Inter-layer scaling factor. 0 = isolated layers, 1 = original, 2 = double.
    sim_ms : float
        Total simulation duration in ms.
    analysis_start_ms : float
        Time (ms from start) at which to begin the PSD analysis window.
    """
    cfg = _copy.deepcopy(config)

    # Scale all inter-layer connection probabilities by p
    for layer_pair in cfg['inter_layer_connections']:
        for conn_key in cfg['inter_layer_connections'][layer_pair]:
            cfg['inter_layer_connections'][layer_pair][conn_key] *= p

    cfg['inter_layer_scaling'] = p

    if base_seed is None:
        base_seed = cfg['simulation']['RANDOM_SEED']

    trial_seed = int(base_seed + trial_id)
    np.random.seed(trial_seed)
    b2.seed(trial_seed)

    b2.start_scope()
    b2.defaultclock.dt = cfg['simulation']['DT']

    if verbose:
        print(f"\n=== p = {p:.4f} (trial {trial_id}) ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=cfg)
    all_monitors = column.get_all_monitors()

    if verbose:
        print(f"Running simulation ({sim_ms} ms, no external stimulus)...")

    column.network.run(sim_ms * ms)

    if verbose:
        print("Simulation complete. Computing LFP...")

    spike_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    electrode_positions = cfg['electrode_positions']

    print("Computing LFP using kernel method...")
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        fs=10000,
        sim_duration_ms=sim_ms
    )

    start_idx = int(analysis_start_ms * fs / 1000)
    end_idx = len(time_array)

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
    psd_matrix = np.vstack([power_spectra[i] for i in range(n_elec)])

    if verbose:
        print(f"Done. PSD matrix shape: {psd_matrix.shape}")

    return {
        "p": p,
        "trial_id": trial_id,
        "seed": trial_seed,
        "electrode_positions": np.array(electrode_positions),
        "frequencies": freqs,
        "psd_matrix": psd_matrix,
        "sim_ms": sim_ms,
        "analysis_start_ms": analysis_start_ms,
        "fs": fs,
    }


def run_p_sweep(
    config,
    p_values,
    base_seed=None,
    sim_ms=2500,
    analysis_start_ms=1000,
    fs=10000,
    save_dir="results/p_sweeps",
    verbose=True,
):
    """
    Run a sweep over inter-layer scaling values.
    Each result is saved individually as it completes (like trials.py),
    then all are stacked into one final .npz at the end.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, p in enumerate(p_values):
        if verbose:
            print(f"\n{'='*60}")
            print(f"P sweep: {i+1}/{len(p_values)} — p = {p:.4f}")
            print(f"{'='*60}")

        data = run_single_p(
            config=config,
            p=p,
            trial_id=i,
            base_seed=base_seed,
            sim_ms=sim_ms,
            analysis_start_ms=analysis_start_ms,
            fs=fs,
            verbose=verbose,
        )

        # Save immediately after each run (same pattern as trials.py)
        fname = os.path.join(save_dir, f"p_{p:.6f}_trial{i:03d}.npz")
        np.savez_compressed(
            fname,
            p=data['p'],
            trial_id=data['trial_id'],
            seed=data['seed'],
            electrode_positions=data['electrode_positions'],
            frequencies=data['frequencies'],
            psd_matrix=data['psd_matrix'],
            sim_ms=data['sim_ms'],
            analysis_start_ms=data['analysis_start_ms'],
            fs=data['fs'],
        )
        if verbose:
            print(f"Saved p={p:.4f} -> {fname}")

    # Combine all individual files into one stacked .npz
    import glob
    files = sorted(glob.glob(os.path.join(save_dir, 'p_*.npz')))

    all_p, all_psd = [], []
    freqs = electrode_positions = None

    for f in files:
        d = np.load(f, allow_pickle=True)
        all_p.append(float(d['p']))
        all_psd.append(d['psd_matrix'])
        if freqs is None:
            freqs = d['frequencies']
            electrode_positions = d['electrode_positions']

    order = np.argsort(all_p)
    p_arr = np.array(all_p)[order]
    psd_stack = np.stack([all_psd[i] for i in order], axis=0)

    p_min, p_max, n = float(p_arr[0]), float(p_arr[-1]), len(p_arr)
    fpath = os.path.join(save_dir, f"sweep_p_{p_min:.2f}-{p_max:.2f}_n{n}.npz")

    np.savez_compressed(
        fpath,
        p_values=p_arr,
        frequencies=freqs,
        psd_stack=psd_stack,
        electrode_positions=electrode_positions,
        sim_ms=sim_ms,
        analysis_start_ms=analysis_start_ms,
        fs=fs,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sweep complete! Saved to: {fpath}")
        print(f"PSD stack shape: {psd_stack.shape}")
        print(f"  - {n} p values")
        print(f"  - {psd_stack.shape[1]} electrodes")
        print(f"  - {psd_stack.shape[2]} frequency bins")
        print(f"{'='*60}")

    return fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sweep inter-layer connection probability scaling factor p',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_p.py --p-min 0.0 --p-max 2.0 --n-p 40 --save-dir results/p_sweeps
        """
    )

    parser.add_argument('--p-min', type=float, default=0.0,
                        help='Minimum p value (default: 0.0)')
    parser.add_argument('--p-max', type=float, default=2.0,
                        help='Maximum p value (default: 2.0)')
    parser.add_argument('--n-p', type=int, default=40,
                        help='Number of p values (default: 40)')
    parser.add_argument('--sim-ms', type=float, default=2500,
                        help='Total simulation duration in ms (default: 2500)')
    parser.add_argument('--analysis-start-ms', type=float, default=1000,
                        help='Start of PSD analysis window in ms (default: 1000)')
    parser.add_argument('--save-dir', type=str, default='results/p_sweeps',
                        help='Directory to save results (default: results/p_sweeps)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    p_values = np.linspace(args.p_min, args.p_max, args.n_p)

    print(f"\n{'='*70}")
    print(f"Inter-layer scaling sweep")
    print(f"p values: {p_values[0]:.4f} to {p_values[-1]:.4f} ({len(p_values)} values)")
    print(f"Simulation: {args.sim_ms} ms, analysis from {args.analysis_start_ms} ms")
    print(f"Save directory: {args.save_dir}")
    print(f"{'='*70}\n")

    result_path = run_p_sweep(
        config=CONFIG,
        p_values=p_values,
        sim_ms=args.sim_ms,
        analysis_start_ms=args.analysis_start_ms,
        fs=10000,
        save_dir=args.save_dir,
        verbose=not args.quiet,
    )

    print(f"\n{'='*70}")
    print(f"Results saved to: {result_path}")
    print(f"{'='*70}\n")

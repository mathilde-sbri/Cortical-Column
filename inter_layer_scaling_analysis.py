

import numpy as np
import brian2 as b2
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import welch
import copy
from tqdm import tqdm
import pickle

from config.config import CONFIG
from src.column import CorticalColumn
from src.analysis import calculate_lfp_mazzoni, calculate_lfp_kernel_method, compute_power_spectrum


def scale_inter_layer_connections_for_L4C(config, p):

    config_scaled = copy.deepcopy(config)

    # Scale connection probabilities
    for (source_layer, target_layer), conns in config_scaled['inter_layer_connections'].items():
        # Only scale if L4C is involved
        if source_layer == 'L4C' or target_layer == 'L4C':
            for conn_key in conns.keys():
                config_scaled['inter_layer_connections'][(source_layer, target_layer)][conn_key] *= p

    # Scale conductances
    for (source_layer, target_layer), conds in config_scaled['inter_layer_conductances'].items():
        # Only scale if L4C is involved
        if source_layer == 'L4C' or target_layer == 'L4C':
            for cond_key in conds.keys():
                config_scaled['inter_layer_conductances'][(source_layer, target_layer)][cond_key] *= p

    return config_scaled


def run_simulation_for_p(p, baseline_time=800, sim_time=1200, random_seed=None, l4c_electrode_idx=8):
 
    if random_seed is not None:
        np.random.seed(random_seed)

    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    config_scaled = scale_inter_layer_connections_for_L4C(CONFIG, p)

    column = CorticalColumn(column_id=0, config=config_scaled)

    L4C = column.layers['L4C']
    L4C_E_grp = L4C.neuron_groups['E']

    # Run the full simulation with only baseline Poisson inputs (no extra stimulation)
    total_time = baseline_time + sim_time
    column.network.run(total_time * ms)

    all_monitors = column.get_all_monitors()
    spike_monitors = {}
    state_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'state' in k
        }
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    L4C_state_monitor = column.layers['L4C'].monitors['E_state']

    neuron_params = {'E_E': 0.0, 'E_I': -80.0}
    lfp_signal_mazzoni, time_array_mazzoni = calculate_lfp_mazzoni(
        L4C_state_monitor, neuron_params, method='weighted'
    )

    stim_start_idx = np.argmax(time_array_mazzoni >= baseline_time)
    lfp_stim_mazzoni = lfp_signal_mazzoni[stim_start_idx:]

    fs_mazzoni = 1000.0 / (time_array_mazzoni[1] - time_array_mazzoni[0])
    freq_mazzoni, psd_mazzoni = compute_power_spectrum(
        lfp_stim_mazzoni, fs=fs_mazzoni, nperseg=min(2048, len(lfp_stim_mazzoni)//4)
    )

    total_time = baseline_time + sim_time
    electrode_positions = config_scaled['electrode_positions']
    lfp_signals_kernel, time_array_kernel = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        config_scaled['layers'],
        electrode_positions,
        fs=10000,
        sim_duration_ms=total_time
    )

    lfp_signal_kernel = lfp_signals_kernel[l4c_electrode_idx]

    stim_start_idx_kernel = np.argmax(time_array_kernel >= baseline_time)
    lfp_stim_kernel = lfp_signal_kernel[stim_start_idx_kernel:]

    fs_kernel = 1000.0 / (time_array_kernel[1] - time_array_kernel[0])
    freq_kernel, psd_kernel = compute_power_spectrum(
        lfp_stim_kernel, fs=fs_kernel, nperseg=min(2048, len(lfp_stim_kernel)//4)
    )

    del column

    return {
        'freq_mazzoni': freq_mazzoni,
        'psd_mazzoni': psd_mazzoni,
        'freq_kernel': freq_kernel,
        'psd_kernel': psd_kernel
    }


def load_partial_results(output_file):
    """Load existing partial results if available."""
    import os
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                results = pickle.load(f)
            print(f"Loaded existing results from {output_file}")
            return results
        except Exception as e:
            print(f"Could not load existing results: {e}")
    return None


def run_parameter_sweep(p_values, baseline_time=800, sim_time=1200,
                       save_results=True, output_file='inter_layer_sweep_results.pkl',
                       resume=True):

    n_p = len(p_values)
    all_psds_mazzoni = []
    all_psds_kernel = []
    freq_array_mazzoni = None
    freq_array_kernel = None
    start_idx = 0

    # Try to resume from existing results
    if resume:
        existing_results = load_partial_results(output_file)
        if existing_results is not None:
            # Check if p_values match
            if np.array_equal(existing_results.get('p_values'), p_values):
                completed = existing_results.get('completed_indices', [])
                if len(completed) > 0:
                    start_idx = max(completed) + 1
                    freq_array_mazzoni = existing_results.get('frequencies_mazzoni')
                    freq_array_kernel = existing_results.get('frequencies_kernel')
                    # Load existing PSDs as lists
                    all_psds_mazzoni = list(existing_results.get('power_spectra_mazzoni', []))
                    all_psds_kernel = list(existing_results.get('power_spectra_kernel', []))
                    print(f"Resuming from simulation {start_idx}/{n_p} ({len(completed)} already completed)")
            else:
                print("p_values don't match existing results, starting fresh")

    print(f"Running parameter sweep for {n_p} values of p...")
    print(f"Baseline time: {baseline_time} ms, Simulation time: {sim_time} ms")
    print("Computing LFP using both Mazzoni and Kernel methods")

    if start_idx >= n_p:
        print("All simulations already completed!")
        return existing_results

    for i, p in enumerate(tqdm(p_values[start_idx:], desc="Simulating", initial=start_idx, total=n_p)):
        actual_idx = start_idx + i
        result = run_simulation_for_p(p, baseline_time=baseline_time,
                                      sim_time=sim_time, random_seed=42 + actual_idx)

        if freq_array_mazzoni is None:
            freq_array_mazzoni = result['freq_mazzoni']
        if freq_array_kernel is None:
            freq_array_kernel = result['freq_kernel']

        all_psds_mazzoni.append(result['psd_mazzoni'])
        all_psds_kernel.append(result['psd_kernel'])

        # Save results after each simulation
        if save_results:
            results = {
                'p_values': p_values,
                'frequencies_mazzoni': freq_array_mazzoni,
                'frequencies_kernel': freq_array_kernel,
                'power_spectra_mazzoni': np.array(all_psds_mazzoni),
                'power_spectra_kernel': np.array(all_psds_kernel),
                'baseline_time': baseline_time,
                'sim_time': sim_time,
                'completed_indices': list(range(actual_idx + 1))
            }
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)

        # Clear memory after saving
        del result
        import gc
        gc.collect()

    all_psds_mazzoni = np.array(all_psds_mazzoni)
    all_psds_kernel = np.array(all_psds_kernel)

    results = {
        'p_values': p_values,
        'frequencies_mazzoni': freq_array_mazzoni,
        'frequencies_kernel': freq_array_kernel,
        'power_spectra_mazzoni': all_psds_mazzoni,
        'power_spectra_kernel': all_psds_kernel,
        'baseline_time': baseline_time,
        'sim_time': sim_time,
        'completed_indices': list(range(n_p))
    }

    if save_results:
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {output_file}")

    return results


def plot_power_spectrum_heatmap(results, method='mazzoni', freq_max=100, vmin=None, vmax=None,
                               use_log_scale=True, save_fig=True,
                               output_file='inter_layer_influence_heatmap.png'):

    p_values = results['p_values']

    if method == 'mazzoni':
        frequencies = results['frequencies_mazzoni']
        power_spectra = results['power_spectra_mazzoni']
        method_label = 'Mazzoni'
    elif method == 'kernel':
        frequencies = results['frequencies_kernel']
        power_spectra = results['power_spectra_kernel']
        method_label = 'Kernel'
    else:
        raise ValueError("method must be 'mazzoni' or 'kernel'")

    freq_mask = frequencies <= freq_max
    freq_plot = frequencies[freq_mask]
    psd_plot = power_spectra[:, freq_mask]

    fig, ax = plt.subplots(figsize=(12, 8))

    if use_log_scale:
        psd_plot_safe = psd_plot + 1e-10
        im = ax.imshow(psd_plot_safe.T, aspect='auto', origin='lower',
                      extent=[p_values[0], p_values[-1], freq_plot[0], freq_plot[-1]],
                      cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
                      interpolation='bilinear')
    else:
        im = ax.imshow(psd_plot.T, aspect='auto', origin='lower',
                      extent=[p_values[0], p_values[-1], freq_plot[0], freq_plot[-1]],
                      cmap='viridis', vmin=vmin, vmax=vmax,
                      interpolation='bilinear')

    ax.set_xlabel('Inter-layer connection strength (p)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_title(f'Influence of Inter-layer Connectivity on L4C Power Spectrum ({method_label} Method)',
                fontsize=16, fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power Spectral Density (a.u.)', fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax.axvline(x=1.0, color='white', linestyle='--', linewidth=2, alpha=0.7,
              label='Original model (p=1)')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_fig:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig, ax


def plot_power_spectra_comparison(results, method='mazzoni', p_values_to_plot=None, freq_max=100,
                                  save_fig=True, output_file='power_spectra_comparison.png'):
  
    if p_values_to_plot is None:
        p_values_to_plot = [0.0, 0.25, 0.5, 0.75, 1.0]

    p_values = results['p_values']

    if method == 'mazzoni':
        frequencies = results['frequencies_mazzoni']
        power_spectra = results['power_spectra_mazzoni']
        method_label = 'Mazzoni'
    elif method == 'kernel':
        frequencies = results['frequencies_kernel']
        power_spectra = results['power_spectra_kernel']
        method_label = 'Kernel'
    else:
        raise ValueError("method must be 'mazzoni' or 'kernel'")

    freq_mask = frequencies <= freq_max
    freq_plot = frequencies[freq_mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.plasma(np.linspace(0, 1, len(p_values_to_plot)))

    for p_target, color in zip(p_values_to_plot, colors):
        idx = np.argmin(np.abs(p_values - p_target))
        p_actual = p_values[idx]
        psd = power_spectra[idx, freq_mask]

        ax.plot(freq_plot, psd, linewidth=2, alpha=0.8,
               label=f'p = {p_actual:.2f}', color=color)

    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Spectral Density (a.u.)', fontsize=12, fontweight='bold')
    ax.set_title(f'L4C Power Spectra for Different Inter-layer Connection Strengths ({method_label} Method)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig, ax


def plot_method_comparison(results, p_value=1.0, freq_max=100,
                          save_fig=True, output_file='method_comparison.png'):

    p_values = results['p_values']

    idx = np.argmin(np.abs(p_values - p_value))
    p_actual = p_values[idx]

    freq_mazzoni = results['frequencies_mazzoni']
    freq_kernel = results['frequencies_kernel']
    psd_mazzoni = results['power_spectra_mazzoni'][idx]
    psd_kernel = results['power_spectra_kernel'][idx]

    mask_mazzoni = freq_mazzoni <= freq_max
    mask_kernel = freq_kernel <= freq_max

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(freq_mazzoni[mask_mazzoni], psd_mazzoni[mask_mazzoni],
           linewidth=2, alpha=0.8, label='Mazzoni Method', color='#1f77b4')
    ax.plot(freq_kernel[mask_kernel], psd_kernel[mask_kernel],
           linewidth=2, alpha=0.8, label='Kernel Method', color='#ff7f0e', linestyle='--')

    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Spectral Density (a.u.)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparison of LFP Methods (p = {p_actual:.2f})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig, ax


def main():

    n_points = 20
    p_values = np.linspace(0, 1, n_points)

    print("="*70)
    print("Inter-layer Connectivity Influence on L4C Dynamics")
    print("="*70)
    print(f"Number of simulations: {n_points}")
    print(f"p range: {p_values[0]:.2f} to {p_values[-1]:.2f}")
    print("="*70)

    results = run_parameter_sweep(
        p_values,
        baseline_time=800,  
        sim_time=3000,     
        save_results=True,
        output_file='inter_layer_sweep_results_longer.pkl'
    )

    print("\nGenerating visualizations...")

    print("  Creating Mazzoni method heatmap...")
    plot_power_spectrum_heatmap(
        results,
        method='mazzoni',
        freq_max=100,
        use_log_scale=True,
        save_fig=True,
        output_file='inter_layer_influence_heatmap_mazzoni.png'
    )

    print("  Creating Kernel method heatmap...")
    plot_power_spectrum_heatmap(
        results,
        method='kernel',
        freq_max=100,
        use_log_scale=True,
        save_fig=True,
        output_file='inter_layer_influence_heatmap_kernel.png'
    )

    print("  Creating Mazzoni method comparison plot...")
    plot_power_spectra_comparison(
        results,
        method='mazzoni',
        p_values_to_plot=[0.0, 0.25, 0.5, 0.75, 1.0],
        freq_max=100,
        save_fig=True,
        output_file='power_spectra_comparison_mazzoni.png'
    )

    print("  Creating Kernel method comparison plot...")
    plot_power_spectra_comparison(
        results,
        method='kernel',
        p_values_to_plot=[0.0, 0.25, 0.5, 0.75, 1.0],
        freq_max=100,
        save_fig=True,
        output_file='power_spectra_comparison_kernel.png'
    )

    print("  Creating method comparison plot...")
    plot_method_comparison(
        results,
        p_value=1.0,
        freq_max=100,
        save_fig=True,
        output_file='method_comparison_p1.0.png'
    )

    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - inter_layer_sweep_results.pkl (raw data)")
    print("\nMazzoni Method:")
    print("  - inter_layer_influence_heatmap_mazzoni.png (2D heatmap)")
    print("  - power_spectra_comparison_mazzoni.png (line plots)")
    print("\nKernel Method:")
    print("  - inter_layer_influence_heatmap_kernel.png (2D heatmap)")
    print("  - power_spectra_comparison_kernel.png (line plots)")
    print("\nMethod Comparison:")
    print("  - method_comparison_p1.0.png (comparison at p=1.0)")

    plt.show()


def plot_partial_results(results_file='inter_layer_sweep_results.pkl', freq_max=100):
    """Plot results from a partial or complete sweep."""
    import os

    if not os.path.exists(results_file):
        print(f"No results file found at {results_file}")
        return

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    completed = results.get('completed_indices', [])
    n_completed = len(completed)
    n_total = len(results['p_values'])

    print(f"Loaded results: {n_completed}/{n_total} simulations completed")

    if n_completed == 0:
        print("No completed simulations to plot.")
        return

    # Get only the completed p_values
    p_values = results['p_values'][:n_completed]

    # Plot for both methods
    for method in ['mazzoni', 'kernel']:
        if method == 'mazzoni':
            frequencies = results['frequencies_mazzoni']
            power_spectra = results['power_spectra_mazzoni']
            method_label = 'Mazzoni'
        else:
            frequencies = results['frequencies_kernel']
            power_spectra = results['power_spectra_kernel']
            method_label = 'Kernel'

        if frequencies is None or len(power_spectra) == 0:
            continue

        freq_mask = frequencies <= freq_max
        freq_plot = frequencies[freq_mask]
        psd_plot = power_spectra[:, freq_mask]

        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        psd_plot_safe = psd_plot + 1e-10
        im = ax.imshow(psd_plot_safe.T, aspect='auto', origin='lower',
                      extent=[p_values[0], p_values[-1], freq_plot[0], freq_plot[-1]],
                      cmap='viridis', norm=LogNorm(),
                      interpolation='bilinear')

        ax.set_xlabel('Inter-layer connection strength (p)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        ax.set_title(f'L4C Power Spectrum ({method_label}) - {n_completed}/{n_total} completed',
                    fontsize=16, fontweight='bold', pad=20)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power Spectral Density (a.u.)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        if 1.0 <= p_values[-1]:
            ax.axvline(x=1.0, color='white', linestyle='--', linewidth=2, alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'partial_heatmap_{method}.png', dpi=300, bbox_inches='tight')
        print(f"Saved partial_heatmap_{method}.png")

    plt.show()
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--plot':
        plot_partial_results()
    else:
        main()

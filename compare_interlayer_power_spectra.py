"""
Compare power spectra with and without interlayer connections.

This script runs two simulations:
1. With interlayer connections (p=1.0, normal model)
2. Without interlayer connections (p=0.0, isolated layers)

For each electrode position (corresponding to different layers), it computes
and plots the power spectrum to show the effect of interlayer connectivity.
"""

import numpy as np
import brian2 as b2
from brian2 import *
import matplotlib.pyplot as plt
from scipy.signal import welch
import copy

from config.config import CONFIG
from src.column import CorticalColumn
from src.analysis import calculate_lfp_kernel_method, compute_power_spectrum


def disable_inter_layer_connections(config):
    """Create a config with all interlayer connections set to zero."""
    config_no_inter = copy.deepcopy(config)

    # Set all inter-layer connection probabilities to 0
    for (source_layer, target_layer), conns in config_no_inter['inter_layer_connections'].items():
        for conn_key in conns.keys():
            config_no_inter['inter_layer_connections'][(source_layer, target_layer)][conn_key] = 0.0

    # Set all inter-layer conductances to 0
    for (source_layer, target_layer), conds in config_no_inter['inter_layer_conductances'].items():
        for cond_key in conds.keys():
            config_no_inter['inter_layer_conductances'][(source_layer, target_layer)][cond_key] = 0.0

    return config_no_inter


def run_simulation(config, baseline_time=800, sim_time=1200, random_seed=42):
    """Run a single simulation and return LFP signals for all electrodes."""

    np.random.seed(random_seed)
    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    # Create cortical column
    column = CorticalColumn(column_id=0, config=config)

    # Run simulation
    total_time = baseline_time + sim_time
    column.network.run(total_time * ms)

    # Get monitors
    all_monitors = column.get_all_monitors()
    spike_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    # Calculate LFP using kernel method for all electrodes
    electrode_positions = config['electrode_positions']
    lfp_signals_dict, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        config['layers'],
        electrode_positions,
        fs=10000,
        sim_duration_ms=total_time
    )

    # Convert dictionary to array (electrodes x time)
    n_electrodes = len(electrode_positions)
    lfp_signals = np.array([lfp_signals_dict[i] for i in range(n_electrodes)])

    # Extract only the stimulus period
    stim_start_idx = np.argmax(time_array >= baseline_time)
    lfp_signals_stim = lfp_signals[:, stim_start_idx:]
    time_array_stim = time_array[stim_start_idx:]

    # Clean up
    del column

    return lfp_signals_stim, time_array_stim, electrode_positions


def compute_power_spectra_all_electrodes(lfp_signals, time_array, nperseg=2048):
    """Compute power spectra for all electrodes."""

    n_electrodes = lfp_signals.shape[0]
    fs = 1000.0 / (time_array[1] - time_array[0])

    freqs = None
    psds = []

    for i in range(n_electrodes):
        freq, psd = compute_power_spectrum(
            lfp_signals[i],
            fs=fs,
            nperseg=min(nperseg, len(lfp_signals[i])//4)
        )

        if freqs is None:
            freqs = freq

        psds.append(psd)

    return freqs, np.array(psds)


def plot_comparison(freqs, psds_with, psds_without, electrode_positions,
                   freq_max=100, save_fig=True):
    """
    Plot comparison of power spectra with and without interlayer connections.

    Creates two types of plots:
    1. Grid of individual electrode comparisons
    2. Heatmap comparison showing all electrodes
    """

    n_electrodes = len(electrode_positions)
    freq_mask = freqs <= freq_max
    freqs_plot = freqs[freq_mask]

    # Map electrode positions to layer names based on z-coordinates
    # Layer ranges: L6: [-0.62, -0.34], L5: [-0.34, -0.14], L4C: [-0.14, 0.14],
    #               L4AB: [0.14, 0.45], L23: [0.45, 1.1]
    layer_map = {
        0: 'Below L6',      # z=-0.94
        1: 'Below L6',      # z=-0.79
        2: 'Below L6',      # z=-0.64
        3: 'L6',            # z=-0.49
        4: 'L6/L5 boundary',# z=-0.34
        5: 'L5',            # z=-0.19
        6: 'L4C',           # z=-0.04
        7: 'L4C',           # z=0.10
        8: 'L4AB',          # z=0.26
        9: 'L4AB',          # z=0.40
        10: 'L23',          # z=0.56
        11: 'L23',          # z=0.70
        12: 'L23',          # z=0.86
        13: 'L23',          # z=1.00
        14: 'Above L23',    # z=1.16
        15: 'Above L23'     # z=1.30
    }

    # --- Plot 1: Grid of individual comparisons ---
    n_cols = 4
    n_rows = (n_electrodes + n_cols - 1) // n_cols

    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten()

    for i in range(n_electrodes):
        ax = axes[i]

        # Plot both conditions
        ax.plot(freqs_plot, psds_with[i, freq_mask],
               linewidth=2, alpha=0.8, label='With inter-layer', color='#1f77b4')
        ax.plot(freqs_plot, psds_without[i, freq_mask],
               linewidth=2, alpha=0.8, label='Without inter-layer', color='#ff7f0e', linestyle='--')

        # Labels and formatting
        electrode_label = layer_map.get(i, f'Electrode {i}')
        z_pos = electrode_positions[i][2]
        ax.set_title(f'{electrode_label}\nz={z_pos:.3f}mm', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('PSD (a.u.)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    # Hide extra subplots
    for i in range(n_electrodes, len(axes)):
        axes[i].axis('off')

    fig1.suptitle('Power Spectrum Comparison: With vs Without Inter-layer Connections\n(Per Electrode/Layer)',
                  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_fig:
        plt.savefig('interlayer_comparison_grid.png', dpi=300, bbox_inches='tight')
        print("Saved: interlayer_comparison_grid.png")

    # Filter to only include electrodes within layers (3-13: L6 to L23)
    valid_electrodes = slice(3, 14)  # Electrodes 3-13 inclusive
    n_valid = 11  # Number of valid electrodes

    # --- Plot 2: Heatmap comparison WITH/WITHOUT ---
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))

    # With inter-layer connections
    im1 = ax1.imshow(psds_with[valid_electrodes, :][:, freq_mask].T, aspect='auto', origin='lower',
                     extent=[3, 13, freqs_plot[0], freqs_plot[-1]],
                     cmap='viridis', interpolation='bilinear')
    ax1.set_xlabel('Electrode Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('WITH Inter-layer Connections', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='PSD (a.u.)')
    ax1.grid(True, alpha=0.3, axis='x')

    # Without inter-layer connections
    im2 = ax2.imshow(psds_without[valid_electrodes, :][:, freq_mask].T, aspect='auto', origin='lower',
                     extent=[3, 13, freqs_plot[0], freqs_plot[-1]],
                     cmap='viridis', interpolation='bilinear')
    ax2.set_xlabel('Electrode Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_title('WITHOUT Inter-layer Connections', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='PSD (a.u.)')
    ax2.grid(True, alpha=0.3, axis='x')

    # Difference (WITH - WITHOUT): positive = inter-layer connectivity increases power
    diff = psds_with[valid_electrodes, :][:, freq_mask] - psds_without[valid_electrodes, :][:, freq_mask]
    vmax_abs = np.max(np.abs(diff))
    im3 = ax3.imshow(diff.T, aspect='auto', origin='lower',
                     extent=[3, 13, freqs_plot[0], freqs_plot[-1]],
                     cmap='RdBu_r', interpolation='bilinear',
                     vmin=-vmax_abs, vmax=vmax_abs)
    ax3.set_xlabel('Electrode Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax3.set_title('DIFFERENCE (With - Without)', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='PSD Difference (a.u.)')
    ax3.grid(True, alpha=0.3, axis='x')

    fig2.suptitle('Laminar Power Spectrum Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_fig:
        plt.savefig('interlayer_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: interlayer_comparison_heatmap.png")

    # --- Plot 2b: Heatmap comparison PERCENT CHANGE ---
    fig2b, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))

    # With inter-layer connections
    im1 = ax1.imshow(psds_with[valid_electrodes, :][:, freq_mask].T, aspect='auto', origin='lower',
                     extent=[3, 13, freqs_plot[0], freqs_plot[-1]],
                     cmap='viridis', interpolation='bilinear')
    ax1.set_xlabel('Electrode Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('WITH Inter-layer Connections', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='PSD (a.u.)')
    ax1.grid(True, alpha=0.3, axis='x')

    # Without inter-layer connections
    im2 = ax2.imshow(psds_without[valid_electrodes, :][:, freq_mask].T, aspect='auto', origin='lower',
                     extent=[3, 13, freqs_plot[0], freqs_plot[-1]],
                     cmap='viridis', interpolation='bilinear')
    ax2.set_xlabel('Electrode Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_title('WITHOUT Inter-layer Connections', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='PSD (a.u.)')
    ax2.grid(True, alpha=0.3, axis='x')

    # Percent change: ((WITH - WITHOUT) / WITHOUT) * 100
    epsilon = 1e-10
    percent_change = ((psds_with[valid_electrodes, :][:, freq_mask] - psds_without[valid_electrodes, :][:, freq_mask]) /
                      (psds_without[valid_electrodes, :][:, freq_mask] + epsilon)) * 100

    # Set symmetric color scale
    vmax_pct = np.percentile(np.abs(percent_change), 95)  # Use 95th percentile to avoid outliers
    im3 = ax3.imshow(percent_change.T, aspect='auto', origin='lower',
                     extent=[3, 13, freqs_plot[0], freqs_plot[-1]],
                     cmap='RdBu_r', interpolation='bilinear',
                     vmin=-vmax_pct, vmax=vmax_pct)
    ax3.set_xlabel('Electrode Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax3.set_title('PERCENT CHANGE (With vs Without)', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='% Change in PSD')
    ax3.grid(True, alpha=0.3, axis='x')

    fig2b.suptitle('Laminar Power Spectrum Comparison - Percent Change', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_fig:
        plt.savefig('interlayer_comparison_percent.png', dpi=300, bbox_inches='tight')
        print("Saved: interlayer_comparison_percent.png")

    # --- Plot 3: Selected layers overlay ---
    fig3, ax = plt.subplots(figsize=(10, 6))

    # Select representative electrodes from each layer (updated to match correct depths)
    selected_electrodes = [3, 5, 7, 8, 11]  # L6, L5, L4C, L4AB, L23
    colors = plt.cm.tab10(np.arange(len(selected_electrodes)))

    for idx, elec_idx in enumerate(selected_electrodes):
        layer_label = layer_map.get(elec_idx, f'E{elec_idx}')

        # With inter-layer
        ax.plot(freqs_plot, psds_with[elec_idx, freq_mask],
               linewidth=2.5, alpha=0.7, color=colors[idx],
               label=f'{layer_label} - WITH')

        # Without inter-layer
        ax.plot(freqs_plot, psds_without[elec_idx, freq_mask],
               linewidth=2.5, alpha=0.7, color=colors[idx],
               linestyle='--', label=f'{layer_label} - WITHOUT')

    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Spectral Density (a.u.)', fontsize=12, fontweight='bold')
    ax.set_title('Selected Layer Power Spectra: With vs Without Inter-layer Connections',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig('interlayer_comparison_selected_layers.png', dpi=300, bbox_inches='tight')
        print("Saved: interlayer_comparison_selected_layers.png")

    return fig1, fig2, fig3


def main():
    """Main execution function."""

    print("="*70)
    print("Inter-layer Connection Effect on Power Spectra")
    print("="*70)
    print()

    # Run simulation WITH inter-layer connections
    print("Running simulation WITH inter-layer connections...")
    config_with = CONFIG
    lfp_with, time_array, electrode_positions = run_simulation(
        config_with,
        baseline_time=800,
        sim_time=1200,
        random_seed=42
    )
    print(f"  Completed. LFP shape: {lfp_with.shape}")
    print(f"  Number of electrodes: {len(electrode_positions)}")

    # Run simulation WITHOUT inter-layer connections
    print("\nRunning simulation WITHOUT inter-layer connections...")
    config_without = disable_inter_layer_connections(CONFIG)
    lfp_without, _, _ = run_simulation(
        config_without,
        baseline_time=800,
        sim_time=1200,
        random_seed=42
    )
    print(f"  Completed. LFP shape: {lfp_without.shape}")

    # Compute power spectra
    print("\nComputing power spectra for all electrodes...")
    freqs, psds_with = compute_power_spectra_all_electrodes(lfp_with, time_array)
    _, psds_without = compute_power_spectra_all_electrodes(lfp_without, time_array)
    print(f"  Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"  Number of frequency bins: {len(freqs)}")

    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(freqs, psds_with, psds_without, electrode_positions,
                   freq_max=100, save_fig=True)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("\nGenerated files:")
    print("  1. interlayer_comparison_grid.png")
    print("     - Individual power spectrum plots for each electrode")
    print("  2. interlayer_comparison_heatmap.png")
    print("     - Laminar heatmaps showing WITH, WITHOUT, and DIFFERENCE")
    print("  3. interlayer_comparison_percent.png")
    print("     - Laminar heatmaps showing WITH, WITHOUT, and PERCENT CHANGE")
    print("  4. interlayer_comparison_selected_layers.png")
    print("     - Overlay plot of representative layers from each region")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()

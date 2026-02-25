import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse


def load_p_sweep(path):
    """
    Load p-sweep data from either:
    - a single combined .npz file (has 'psd_stack' and 'p_values')
    - a directory of individual p_*.npz files
    Returns dict with keys: p_values, psd_stack, frequencies, electrode_positions
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'p_*.npz')))
        if not files:
            raise FileNotFoundError(f"No p_*.npz files found in {path}")
        all_p, all_psd, freqs, elec = [], [], None, None
        for f in files:
            d = np.load(f, allow_pickle=True)
            all_p.append(float(d['p']))
            all_psd.append(d['psd_matrix'])
            if freqs is None:
                freqs = d['frequencies']
                elec = d['electrode_positions']
        order = np.argsort(all_p)
        return {
            'p_values': np.array(all_p)[order],
            'psd_stack': np.stack([all_psd[i] for i in order], axis=0),
            'frequencies': freqs,
            'electrode_positions': elec,
        }
    else:
        d = np.load(path, allow_pickle=True)
        return {
            'p_values': d['p_values'],
            'psd_stack': d['psd_stack'],
            'frequencies': d['frequencies'],
            'electrode_positions': d['electrode_positions'],
        }

LAYER_DEPTH_RANGES = {
    'L23':  (0.45, 1.30),
    'L4AB': (0.14, 0.45),
    'L4C':  (-0.14, 0.14),
    'L5':   (-0.49, -0.14),
    'L6':   (-0.94, -0.49),
}

FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
}

BAND_COLORS = {
    'delta': '#e41a1c',
    'theta': '#377eb8',
    'alpha': '#4daf4a',
    'beta': '#984ea3',
    'gamma': '#ff7f00',
}


def add_frequency_band_lines(ax, fmax, show_labels=True):
    for band_name, (fmin_band, fmax_band) in FREQUENCY_BANDS.items():
        if fmin_band > fmax:
            continue
        color = BAND_COLORS[band_name]
        if fmin_band > 0 and fmin_band <= fmax:
            ax.axhline(y=fmin_band, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
        if show_labels:
            visible_fmax = min(fmax_band, fmax)
            if visible_fmax > fmin_band:
                label_y = (fmin_band + visible_fmax) / 2
                ax.text(
                    1.02, label_y, band_name.capitalize(),
                    transform=ax.get_yaxis_transform(),
                    fontsize=6, color=color, fontweight='bold',
                    va='center', ha='left'
                )


def get_excluded_electrode_indices(electrode_positions, exclude_layers):
    excluded = set()
    for i, pos in enumerate(electrode_positions):
        z = pos[2]
        for layer in exclude_layers:
            layer = layer.upper()
            if layer not in LAYER_DEPTH_RANGES:
                raise ValueError(f"Unknown layer '{layer}'. Valid layers: {list(LAYER_DEPTH_RANGES.keys())}")
            zmin, zmax = LAYER_DEPTH_RANGES[layer]
            if zmin <= z <= zmax:
                excluded.add(i)
                break
    return excluded


def get_electrode_labels(electrode_positions):
    depths = [pos[2] for pos in electrode_positions]
    labels = [f"Elec {i+1}\n(z={z:.2f}mm)" for i, z in enumerate(depths)]
    return labels, depths


def plot_p_sweep_heatmaps(
    filepath,
    fmax=100,
    fmin=None,
    log_scale=False,
    cmap='viridis',
    save_path=None,
    figsize=(16, 14),
    show_bands=True,
    exclude_layers=None,
):
    data = load_p_sweep(filepath)
    p_values = data['p_values']
    frequencies = data['frequencies']
    psd_stack = data['psd_stack']
    electrode_positions = data['electrode_positions']

    freq_mask = frequencies <= fmax
    if fmin is not None:
        freq_mask &= frequencies >= fmin
    frequencies = frequencies[freq_mask]
    psd_stack = psd_stack[:, :, freq_mask]

    electrode_labels, depths = get_electrode_labels(electrode_positions)
    excluded = get_excluded_electrode_indices(electrode_positions, exclude_layers or [])
    depth_order = [i for i in np.argsort(depths) if i not in excluded]
    n_electrodes = len(depth_order)

    n_cols = 4
    n_rows = max(1, (n_electrodes + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 4))
    axes = np.array(axes).flatten()

    psd_included = psd_stack[:, list(depth_order), :]
    vmin = psd_included.min()
    vmax = psd_included.max()

    if log_scale:
        vmin = max(vmin, 1e-10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for plot_idx, elec_idx in enumerate(depth_order):
        ax = axes[plot_idx]
        psd_elec = psd_stack[:, elec_idx, :].T

        if log_scale:
            psd_elec = np.maximum(psd_elec, 1e-10)

        im = ax.imshow(
            psd_elec,
            aspect='auto',
            origin='lower',
            extent=[p_values[0], p_values[-1], frequencies[0], frequencies[-1]],
            cmap=cmap,
            norm=norm if log_scale else None,
            vmin=None if log_scale else vmin,
            vmax=None if log_scale else vmax,
        )

        z = depths[elec_idx]
        ax.set_title(f"Electrode {elec_idx+1} (z={z:.2f}mm)", fontsize=10)

        last_row_start = (n_electrodes - 1) // n_cols * n_cols
        if plot_idx >= last_row_start:
            ax.set_xlabel("Inter-layer scaling p", fontsize=9)
        if plot_idx % n_cols == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=9)

        ax.tick_params(labelsize=8)

        if show_bands:
            show_labels = (plot_idx % n_cols == n_cols - 1) or (plot_idx == n_electrodes - 1)
            add_frequency_band_lines(ax, fmax, show_labels=show_labels)

    for idx in range(n_electrodes, len(axes)):
        axes[idx].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power (a.u.)', fontsize=11)

    fig.suptitle(
        f"Power spectra vs inter-layer scaling p\n"
        f"(p=0: isolated layers, p=1: original, p=2: doubled)",
        fontsize=13,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    return fig


def plot_p_sweep_difference(
    filepath,
    fmax=100,
    fmin=None,
    baseline_idx=None,
    cmap='RdBu_r',
    save_path=None,
    figsize=(16, 14),
    show_bands=True,
    exclude_layers=None,
):
    """
    Plot difference from baseline (default: p=1 index, i.e. original connectivity).
    """
    data = load_p_sweep(filepath)
    p_values = data['p_values']
    frequencies = data['frequencies']
    psd_stack = data['psd_stack']
    electrode_positions = data['electrode_positions']

    # Default baseline: p closest to 0.0
    if baseline_idx is None:
        baseline_idx = int(np.argmin(np.abs(p_values - 0.0)))

    freq_mask = frequencies <= fmax
    if fmin is not None:
        freq_mask &= frequencies >= fmin
    frequencies = frequencies[freq_mask]
    psd_stack = psd_stack[:, :, freq_mask]

    baseline_psd = psd_stack[baseline_idx, :, :]
    psd_diff = psd_stack - baseline_psd[np.newaxis, :, :]

    electrode_labels, depths = get_electrode_labels(electrode_positions)
    excluded = get_excluded_electrode_indices(electrode_positions, exclude_layers or [])
    depth_order = [i for i in np.argsort(depths) if i not in excluded]
    n_electrodes = len(depth_order)

    n_cols = 4
    n_rows = max(1, (n_electrodes + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 4))
    axes = np.array(axes).flatten()

    psd_diff_included = psd_diff[:, list(depth_order), :]
    max_abs = np.abs(psd_diff_included).max()
    vmin, vmax = -max_abs, max_abs

    for plot_idx, elec_idx in enumerate(depth_order):
        ax = axes[plot_idx]
        psd_diff_elec = psd_diff[:, elec_idx, :].T

        im = ax.imshow(
            psd_diff_elec,
            aspect='auto',
            origin='lower',
            extent=[p_values[0], p_values[-1], frequencies[0], frequencies[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        z = depths[elec_idx]
        ax.set_title(f"Electrode {elec_idx+1} (z={z:.2f}mm)", fontsize=10)

        # Mark p=1
        ax.axvline(x=p_values[baseline_idx], color='white', linestyle='--', linewidth=0.8, alpha=0.8)

        last_row_start = (n_electrodes - 1) // n_cols * n_cols
        if plot_idx >= last_row_start:
            ax.set_xlabel("Inter-layer scaling p", fontsize=9)
        if plot_idx % n_cols == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=9)

        ax.tick_params(labelsize=8)

        if show_bands:
            show_labels = (plot_idx % n_cols == n_cols - 1) or (plot_idx == n_electrodes - 1)
            add_frequency_band_lines(ax, fmax, show_labels=show_labels)

    for idx in range(n_electrodes, len(axes)):
        axes[idx].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power difference (a.u.)', fontsize=11)

    fig.suptitle(
        f"Power spectra difference from p={p_values[baseline_idx]:.2f} (isolated layers)",
        fontsize=13,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot inter-layer scaling sweep results")
    parser.add_argument("filepath", type=str, help="Path to p-sweep results .npz file")
    parser.add_argument("--fmax", type=float, default=100, help="Max frequency to display (default: 100)")
    parser.add_argument("--high-pass", type=float, default=None, metavar="FREQ",
                        help="Only plot frequencies >= FREQ Hz")
    parser.add_argument("--log", action="store_true", help="Use logarithmic color scale")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap (default: viridis)")
    parser.add_argument("--save", type=str, default=None, help="Save path for figure")
    parser.add_argument("--diff", action="store_true",
                        help="Also plot difference from p=0 (isolated layers)")
    parser.add_argument("--baseline-idx", type=int, default=None,
                        help="Index in p_values to use as baseline for diff (default: closest to p=1)")
    parser.add_argument("--no-bands", action="store_true", help="Hide frequency band annotations")
    parser.add_argument(
        "--exclude", type=str, nargs="+", default=[], metavar="LAYER",
        help=f"Exclude electrodes from these layers. Valid: {list(LAYER_DEPTH_RANGES.keys())}"
    )

    args = parser.parse_args()

    plot_p_sweep_heatmaps(
        args.filepath,
        fmax=args.fmax,
        fmin=args.high_pass,
        log_scale=args.log,
        cmap=args.cmap,
        save_path=args.save,
        show_bands=not args.no_bands,
        exclude_layers=args.exclude,
    )

    if args.diff:
        diff_save = None
        if args.save:
            base, ext = os.path.splitext(args.save)
            diff_save = f"{base}_diff{ext}"

        plot_p_sweep_difference(
            args.filepath,
            fmax=args.fmax,
            fmin=args.high_pass,
            baseline_idx=args.baseline_idx,
            save_path=diff_save,
            show_bands=not args.no_bands,
            exclude_layers=args.exclude,
        )

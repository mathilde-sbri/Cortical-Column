import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config.config_test import CONFIG
from src.analysis import *
from src.visualization import *
from scipy.signal import spectrogram  # at top of file


def compute_spectrogram_lfp(lfp, time_array_ms,
                            fmin=1.0, fmax=100.0,
                            nperseg_ms=50.0,
                            noverlap_ms=45.0,
                            nfft=None):
    """
    Compute a spectrogram for a single LFP trace using STFT.
    """
    from scipy.signal.windows import hamming  # Correct import location
    
    time_array_ms = np.asarray(time_array_ms)
    lfp = np.asarray(lfp)

    # Sampling interval and freq
    dt_ms = np.diff(time_array_ms).mean()
    fs = 1000.0 / dt_ms  # Hz

    # Window length / overlap in samples
    nperseg = int(round(nperseg_ms / dt_ms))
    noverlap = int(round(noverlap_ms / dt_ms))
    
    if nperseg <= 1:
        raise ValueError("nperseg_ms too small for the time resolution of the data.")
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    # Create Hamming window to match MATLAB
    window = hamming(nperseg)
    
    freqs, t_spec_s, Sxx = spectrogram(
        lfp,
        fs=fs,
        window=window,  # Use Hamming window
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft if nfft is not None else nperseg,
        scaling='density',
        mode='psd',
    )

    # Convert to ms, shift to match original time start
    t_spec_ms = t_spec_s * 1000.0 + time_array_ms[0]

    # Keep only desired frequency range
    fmask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[fmask]
    Sxx = Sxx[fmask, :]

    # Convert to dB
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

    return t_spec_ms, freqs, Sxx_db

def plot_spectrogram_per_channel(bipolar_signals, channel_labels, channel_depths, time_array_ms,
                                 time_range=(200, 1000),
                                 fmin=1.0, fmax=100.0,
                                 nperseg_ms=50.0, noverlap_ms=45.0,
                                 figsize=(14, 20)):
    """
    Plot spectrogram (STFT) for each bipolar channel in a vertical stack.
    """
    time_array_ms = np.asarray(time_array_ms)

    if time_range is None:
        time_range = (time_array_ms[0], time_array_ms[-1])

    # optional: just use time_range for x-limits; spectrogram itself is computed on full signal
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    pcm = None

    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        t_spec_ms, freqs, Sxx_db = compute_spectrogram_lfp(
            lfp, time_array_ms,
            fmin=fmin, fmax=fmax,
            nperseg_ms=80.0,
            noverlap_ms=70.0,
            nfft=4096    
        )

        # Calculate bin edges instead of centers for accurate extent
        dt = np.mean(np.diff(t_spec_ms)) if len(t_spec_ms) > 1 else 1.0
        df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0

        extent = [
            t_spec_ms[0] - dt/2,   # Left edge
            t_spec_ms[-1] + dt/2,  # Right edge
            freqs[0] - df/2,       # Bottom edge
            freqs[-1] + df/2       # Top edge
        ]

        ax = axes[i]
        pcm = ax.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='jet',
            interpolation='bilinear'
        )

        ax.set_ylim(fmin, fmax)
        ax.set_ylabel("Freq (Hz)", fontsize=9)

        # Add label with channel name and depth
        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')

        ax.set_xlim(time_range[0], time_range[1])

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)
    plt.tight_layout()

    return fig

def plot_spectrogram_by_layer(bipolar_signals, channel_depths, time_array_ms,
                              time_range=(200, 1000),
                              fmin=1.0, fmax=100.0,
                              nperseg_ms=50.0, noverlap_ms=45.0,
                              figsize=(12, 10)):
    """
    Plot layer-averaged spectrogram (anatomical layers: L1, L23, L4AB, L4C, L5, L6).
    """
    time_array_ms = np.asarray(time_array_ms)
    if time_range is None:
        time_range = (time_array_ms[0], time_array_ms[-1])

    # Assign channels to layers
    layer_masks = assign_channels_to_layers(channel_depths)
    

    layer_names = ['SG', 'G', 'IG']

    fig, axes = plt.subplots(len(layer_names), 1, figsize=figsize, sharex=True)
    pcm = None

    for idx, layer_name in enumerate(layer_names):
        ax = axes[idx]
        layer_mask = layer_masks[layer_name]

        # channels in this layer
        layer_channels = [
            ch for ch in bipolar_signals.keys()
            if ch < len(layer_mask) and layer_mask[ch]
        ]

        if len(layer_channels) == 0:
            ax.text(
                0.5, 0.5, f'{layer_name}\n(no channels)',
                ha='center', va='center', transform=ax.transAxes
            )
            ax.set_ylabel(layer_name, fontsize=11)
            continue

        # Average LFP of all channels in this layer
        lfp_layer = np.mean([bipolar_signals[ch] for ch in layer_channels], axis=0)

        t_spec_ms, freqs, Sxx_db = compute_spectrogram_lfp(
            lfp_layer, time_array_ms,
            fmin=fmin, fmax=fmax,
            nperseg_ms=80.0,
            noverlap_ms=70.0,
            nfft=4096    
        )

        extent = [t_spec_ms[0], t_spec_ms[-1], freqs[0], freqs[-1]]

        pcm = ax.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='jet',
            interpolation='bilinear'  
        )

        ax.set_ylim(fmin, fmax)
        ax.set_ylabel(f'{layer_name}\n(n={len(layer_channels)} ch)', fontsize=9)
        ax.set_xlim(time_range[0], time_range[1])

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)
    plt.tight_layout()

    return fig

def compute_psth(t_ms, rate_hz, bin_width_ms=10.0):
    """
    Bin a continuous rate trace into a PSTH-style binned rate.

    Args:
        t_ms : 1D array of time points in ms
        rate_hz : 1D array of rates (Hz) at each time point
        bin_width_ms : width of time bins (in ms)

    Returns:
        bin_centers_ms : 1D array of bin center times (ms)
        psth_rate_hz   : 1D array of mean rate (Hz) in each bin
    """
    t_ms = np.asarray(t_ms).ravel()
    rate_hz = np.asarray(rate_hz).ravel()

    if t_ms.size == 0:
        return np.array([]), np.array([])

    t_start = t_ms[0]
    t_end = t_ms[-1]

    # Bin edges and centers
    bin_edges = np.arange(t_start, t_end + bin_width_ms, bin_width_ms)
    if bin_edges.size < 2:
        return t_ms, rate_hz

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    psth_rate = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        mask = (t_ms >= bin_edges[i]) & (t_ms < bin_edges[i+1])
        if np.any(mask):
            psth_rate[i] = rate_hz[mask].mean()
        else:
            psth_rate[i] = 0.0

    return bin_centers, psth_rate

def plot_psth_from_rate_data(rate_data, stim_onset_ms, 
                             bin_width_ms=10.0, time_range=None,
                             figsize=(10, 8)):
    """
    Plot PSTH (binned rate) by layer and monitor.

    Args:
        rate_data: avg_data["rate_data"] dict:
            {layer_name: {mon_name: {"t_ms": ..., "rate_hz": ...}, ...}, ...}
        stim_onset_ms: stimulus onset time in ms (vertical line)
        bin_width_ms: bin width for PSTH in ms
        time_range: (t_min, t_max) in ms for plotting
        figsize: matplotlib figure size
    """
    layer_names = list(rate_data.keys())
    n_layers = len(layer_names)

    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, layer_names):
        layer_rates = rate_data[layer_name]

        for mon_name, mon_data in layer_rates.items():
            t_ms = mon_data["t_ms"]
            rate_hz = mon_data["rate_hz"]

            bin_t, psth_rate = compute_psth(t_ms, rate_hz, bin_width_ms=bin_width_ms)
            if bin_t.size == 0:
                continue

            if time_range is not None:
                mask = (bin_t >= time_range[0]) & (bin_t <= time_range[1])
                bin_t_plot = bin_t[mask]
                psth_plot = psth_rate[mask]
            else:
                bin_t_plot = bin_t
                psth_plot = psth_rate

            ax.plot(bin_t_plot, psth_plot, label=mon_name, linewidth=1.5)

        ax.axvline(stim_onset_ms, color='r', linestyle='--', linewidth=1)
        ax.set_ylabel(f'{layer_name}\nRate (Hz)', fontsize=10)
        ax.set_xlim(300, 800)
        ax.set_ylim(0, 80)
        ax.grid(True, alpha=0.3)
        if len(layer_rates) > 1:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (ms)", fontsize=11)
    fig.tight_layout()

    return fig


def load_all_trials(trial_dir="results/lfp_trials", n_trials=15):
    """
    Load all trial data from saved .npz files.
    
    Returns:
        trials_data: list of dictionaries containing trial data
    """
    trials_data = []
    
    for trial_id in range(n_trials):
        fname = os.path.join(trial_dir, f"trial_{trial_id:03d}.npz")
        
        if not os.path.exists(fname):
            print(f"Warning: {fname} not found, skipping...")
            continue
        
        data = np.load(fname, allow_pickle=True)
        
        trial_dict = {
            "trial_id": int(data["trial_id"]),
            "seed": int(data["seed"]),
            "time_array_ms": data["time_array_ms"],
            "electrode_positions": data["electrode_positions"],
            "lfp_matrix": data["lfp_matrix"],
            "bipolar_matrix": data["bipolar_matrix"],
            "csd": data["csd"],
            "csd_depths": data["csd_depths"],
            "csd_sort_idx": data["csd_sort_idx"],
            "channel_labels": data["channel_labels"],
            "channel_depths": data["channel_depths"],
            "rate_data": data["rate_data"].item(),
            "baseline_ms": float(data["baseline_ms"]),
            "post_ms": float(data["post_ms"]),
            "stim_onset_ms": float(data["stim_onset_ms"]),
        }
        
        trials_data.append(trial_dict)
        print(f"Loaded trial {trial_id}")
    
    print(f"\nSuccessfully loaded {len(trials_data)}/{n_trials} trials")
    return trials_data


def average_trials(trials_data):
    """
    Average LFP, bipolar, CSD, and rate data across all trials.
    
    Returns:
        avg_data: dictionary containing trial-averaged data
    """
    if len(trials_data) == 0:
        raise ValueError("No trials data to average!")
    
    n_trials = len(trials_data)
    
    # Get reference structures from first trial
    ref_trial = trials_data[0]
    time_array_ms = ref_trial["time_array_ms"]
    electrode_positions = ref_trial["electrode_positions"]
    channel_labels = ref_trial["channel_labels"]
    channel_depths = ref_trial["channel_depths"]
    csd_depths = ref_trial["csd_depths"]
    baseline_ms = ref_trial["baseline_ms"]
    stim_onset_ms = ref_trial["stim_onset_ms"]
    
    # Initialize arrays for averaging
    n_electrodes = ref_trial["lfp_matrix"].shape[0]
    n_timepoints = ref_trial["lfp_matrix"].shape[1]
    n_bipolar = ref_trial["bipolar_matrix"].shape[0]
    n_csd = ref_trial["csd"].shape[0]
    
    lfp_sum = np.zeros((n_electrodes, n_timepoints))
    bipolar_sum = np.zeros((n_bipolar, n_timepoints))
    csd_sum = np.zeros((n_csd, n_timepoints))
    
    # Accumulate data across trials
    for trial in trials_data:
        lfp_sum += trial["lfp_matrix"]
        bipolar_sum += trial["bipolar_matrix"]
        csd_sum += trial["csd"]
    
    # Compute averages
    lfp_avg = lfp_sum / n_trials
    bipolar_avg = bipolar_sum / n_trials
    csd_avg = csd_sum / n_trials
    
    # Average rate data
    rate_data_avg = {}
    for layer_name in ref_trial["rate_data"].keys():
        rate_data_avg[layer_name] = {}
        
        for mon_name in ref_trial["rate_data"][layer_name].keys():
            t_ms = ref_trial["rate_data"][layer_name][mon_name]["t_ms"]
            n_rate_points = len(t_ms)
            
            rate_sum = np.zeros(n_rate_points)
            for trial in trials_data:
                rate_sum += trial["rate_data"][layer_name][mon_name]["rate_hz"]
            
            rate_data_avg[layer_name][mon_name] = {
                "t_ms": t_ms,
                "rate_hz": rate_sum / n_trials
            }
    
    # Convert averaged matrices back to dictionaries
    lfp_signals = {i: lfp_avg[i, :] for i in range(n_electrodes)}
    bipolar_signals = {i: bipolar_avg[i, :] for i in range(n_bipolar)}
    
    avg_data = {
        "n_trials_averaged": n_trials,
        "time_array_ms": time_array_ms,
        "electrode_positions": electrode_positions,
        "lfp_signals": lfp_signals,
        "lfp_matrix": lfp_avg,
        "bipolar_signals": bipolar_signals,
        "bipolar_matrix": bipolar_avg,
        "csd": csd_avg,
        "csd_depths": csd_depths,
        "channel_labels": channel_labels,
        "channel_depths": channel_depths,
        "rate_data": rate_data_avg,
        "baseline_ms": baseline_ms,
        "stim_onset_ms": stim_onset_ms,
    }
    
    print(f"\nAveraged {n_trials} trials successfully")
    return avg_data


def compute_bipolar_power_spectrum_prestim_poststim(bipolar_signals, time_array, 
                                                     stim_onset_ms=500,
                                                     prestim_window=(200, 500),
                                                     poststim_window=(500, 1000),
                                                     fs=10000, fmax=100, method='welch'):
    """
    Compute power spectra for pre-stimulus and post-stimulus periods separately.
    
    Returns:
        freq: frequency array
        psds_pre: dict of pre-stimulus PSDs per channel
        psds_post: dict of post-stimulus PSDs per channel
    """
    # Find time indices
    pre_mask = (time_array >= prestim_window[0]) & (time_array < prestim_window[1])
    post_mask = (time_array >= poststim_window[0]) & (time_array < poststim_window[1])
    
    psds_pre = {}
    psds_post = {}
    
    for ch_idx, lfp in bipolar_signals.items():
        # Pre-stimulus period
        lfp_pre = lfp[pre_mask]
        freq_pre, psd_pre = compute_power_spectrum(lfp_pre, fs=fs, method=method)
        
        # Post-stimulus period
        lfp_post = lfp[post_mask]
        freq_post, psd_post = compute_power_spectrum(lfp_post, fs=fs, method=method)
        
        # Keep only frequencies up to fmax
        freq_mask = freq_pre <= fmax
        psds_pre[ch_idx] = psd_pre[freq_mask]
        psds_post[ch_idx] = psd_post[freq_mask]
    
    return freq_pre[freq_mask], psds_pre, psds_post


def plot_bipolar_power_spectra_comparison(bipolar_signals, channel_labels, channel_depths, 
                                         time_array, stim_onset_ms=500,
                                         prestim_window=(200, 500),
                                         poststim_window=(500, 1000),
                                         fs=10000, fmax=100, figsize=(16, 20)):
    """
    Plot power spectra comparing pre-stimulus (blue) vs post-stimulus (red) periods.
    """
    freq, psds_pre, psds_post = compute_bipolar_power_spectrum_prestim_poststim(
        bipolar_signals, time_array, stim_onset_ms,
        prestim_window, poststim_window, fs, fmax
    )
    
    n_channels = len(psds_pre)
    
    fig, axes = plt.subplots(n_channels, 1, sharex=True, sharey=True, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    
    for (ch_idx, psd_pre), (_, psd_post), ax in zip(psds_pre.items(), psds_post.items(), axes[:n_channels]):
        # Plot pre-stimulus in blue
        ax.plot(freq, psd_pre, linewidth=1.5, alpha=0.8, color='blue', 
                label=f'Pre-stim ({prestim_window[0]}-{prestim_window[1]} ms)')
        
        # Plot post-stimulus in red
        ax.plot(freq, psd_post, linewidth=1.5, alpha=0.8, color='red',
                label=f'Post-stim ({poststim_window[0]}-{poststim_window[1]} ms)')
        
        ax.set_ylabel(
            f"{channel_labels[ch_idx]}\n(z={channel_depths[ch_idx]:.3f} mm)",
            fontsize=9
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, fmax)
        
        if ch_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Power Spectral Density (a.u.)',
             va='center', rotation='vertical', fontsize=12)
    
    fig.suptitle('Bipolar LFP Power Spectra: Pre vs Post Stimulus', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.06, 0.06, 1.0, 0.96])
    
    return fig


def assign_channels_to_layers(channel_depths):
    """
    Assign channels to cortical layers based on depth.
    Adjust these boundaries based on your model's anatomy.
    
    Returns:
        layer_masks: dict with 'SG', 'G', 'IG' keys containing boolean masks
    """
    # Define layer boundaries (in mm, adjust to match your model)
    # Positive = superficial, Negative = deep
    layer_masks = {
        'SG': (channel_depths > 0.5) & (channel_depths <= 1.3),   # Superficial (L2/3)
        'G': (channel_depths > -0.2) & (channel_depths <= 0.5),    # Granular (L4)
        'IG': (channel_depths >= -1.0) & (channel_depths <= -0.2)  # Infragranular (L5/6)
    }
    
    return layer_masks

def assign_channels_to_cortical_layers(channel_depths):
    """
    Assign channels to anatomical layers using depth (z, in mm).

    Uses the same z ranges as in _LAYER_CONFIGS['...']['coordinates']['z']:

        L1   :  1.10 to  1.19
        L23  :  0.45 to  1.10
        L4AB :  0.14 to  0.45
        L4C  : -0.14 to  0.14
        L5   : -0.34 to -0.14
        L6   : -0.62 to -0.34
    """
    z = np.asarray(channel_depths)

    layer_masks = {
        'L1':   (z >=  1.10) & (z <=  1.19),
        'L23':  (z >=  0.45) & (z <   1.10),
        'L4AB': (z >=  0.14) & (z <   0.45),
        'L4C':  (z >= -0.14) & (z <   0.14),
        'L5':   (z >= -0.34) & (z <  -0.14),
        'L6':   (z >= -0.62) & (z <  -0.34),
    }

    return layer_masks

def plot_wavelet_by_layer(bipolar_signals, channel_labels, channel_depths, time_array,
                          figsize=(14, 12), time_range=(200, 1000),
                          freq_min=1.0, freq_max=100.0, n_freqs=100, wavelet_cycles=7.0):
    """
    Plot wavelet analysis averaged by cortical layer (SG, G, IG).
    Includes edge tapering to reduce border effects.
    """
    time_array = np.asarray(time_array)
    
    if time_range is None:
        time_range = (time_array[0], time_array[-1])
    
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    time_plot = time_array[time_mask]
    
    dt_ms = np.diff(time_array).mean()
    dt = dt_ms / 1000.0
    
    wavelet_freqs = np.linspace(freq_min, freq_max, n_freqs)
    
    # Assign channels to layers
    layer_masks = assign_channels_to_layers(channel_depths)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    layer_names = ['SG', 'G', 'IG']
    
    for ax, layer_name in zip(axes, layer_names):
        layer_mask = layer_masks[layer_name]
        layer_channels = [ch_idx for ch_idx in bipolar_signals.keys() 
                         if ch_idx < len(layer_mask) and layer_mask[ch_idx]]
        
        if len(layer_channels) == 0:
            ax.text(0.5, 0.5, f'{layer_name}\n(no channels)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylabel(f'{layer_name}', fontsize=12)
            continue
        
        # Average LFP across channels in this layer
        lfp_layer = np.mean([bipolar_signals[ch] for ch in layer_channels], axis=0)
        lfp_layer = np.asarray(lfp_layer)
        
        # Apply time window
        lfp_plot = lfp_layer[time_mask]
        L = len(lfp_plot)
        
        # EDGE TAPERING: Apply a Tukey (tapered cosine) window to reduce edge effects
        # This is what MATLAB likely does by default or through preprocessing
        from scipy.signal import windows
        taper_fraction = 0.1  # Taper 10% on each end
        window = windows.tukey(L, alpha=taper_fraction)
        lfp_plot_tapered = lfp_plot * window
        
        # Compute wavelet transform
        wavelet_power = np.zeros((len(wavelet_freqs), L), dtype=float)
        
        for fi, freq in enumerate(wavelet_freqs):
            s = wavelet_cycles / (2.0 * np.pi * freq)
            t_wave = np.arange(-3.0 * s, 3.0 * s + dt, dt)
            
            # Morlet wavelet
            wavelet = np.exp(2j * np.pi * freq * t_wave) * \
                      np.exp(-t_wave**2 / (2.0 * s**2))
            
            # Convolve with tapered signal
            conv_full = np.convolve(lfp_plot_tapered, wavelet, mode='full')
            total_len = conv_full.size
            start = (total_len - L) // 2
            end = start + L
            conv_res = conv_full[start:end]
            
            wavelet_power[fi, :] = np.abs(conv_res)**2
        
        # Convert to dB
        wavelet_power_db = 10.0 * np.log10(wavelet_power + 1e-12)
        
        # Plot
        pcm = ax.pcolormesh(
            time_plot,
            wavelet_freqs,
            wavelet_power_db,
            cmap='jet',
            shading="auto", vmax=80
        )
        
        ax.set_ylim(freq_min, freq_max)
        ax.set_ylabel(f'{layer_name}\n({len(layer_channels)} ch)\nFreq (Hz)', fontsize=10)
        ax.axvline(500, color='white', linestyle='--', linewidth=1.5, alpha=0.7)  # Stimulus onset
        
    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    axes[-1].set_xlim(350, 800)
    
    # Add colorbar
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Power (dB)", fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_all_analyses(avg_data, save_dir="results/averaged_analysis"):
    """
    Generate all plots using averaged data (mimicking MATLAB analysis).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n=== Generating all averaged plots ===\n")
    
    # Extract data
    lfp_signals = avg_data["lfp_signals"]
    bipolar_signals = avg_data["bipolar_signals"]
    time_array = avg_data["time_array_ms"]
    electrode_positions = avg_data["electrode_positions"]
    channel_labels = avg_data["channel_labels"]
    channel_depths = avg_data["channel_depths"]
    csd = avg_data["csd"]
    csd_depths = avg_data["csd_depths"]
    rate_data = avg_data["rate_data"]
    n_trials = avg_data["n_trials_averaged"]
    stim_onset_ms = avg_data["stim_onset_ms"]
    
    # 1. Monopolar LFP (kernel method) - per channel
    print("1. Plotting monopolar LFP per channel...")
    fig_lfp_kernel = plot_lfp_kernel(lfp_signals, time_array, electrode_positions)
    fig_lfp_kernel.suptitle(f'Monopolar LFP per channel (avg of {n_trials} trials)', 
                             fontsize=14, fontweight='bold')
    fig_lfp_kernel.savefig(os.path.join(save_dir, "01_monopolar_lfp_per_channel.png"), 
                           dpi=300, bbox_inches='tight')
    plt.close(fig_lfp_kernel)
    
    # 2. Bipolar LFP - per channel
    print("2. Plotting bipolar LFP per channel...")
    fig_bipolar = plot_bipolar_lfp(
        bipolar_signals, channel_labels, channel_depths, 
        time_array, time_range=(-200, 500)
    )
    fig_bipolar.suptitle(f'Bipolar LFP per channel (avg of {n_trials} trials)', 
                         fontsize=14, fontweight='bold', y=0.995)
    fig_bipolar.savefig(os.path.join(save_dir, "02_bipolar_lfp_per_channel.png"), 
                        dpi=300, bbox_inches='tight')
    plt.close(fig_bipolar)
    
    # 3. Monopolar vs Bipolar comparison
    print("3. Plotting monopolar vs bipolar comparison...")
    fig_comparison = plot_lfp_comparison(
        lfp_signals, bipolar_signals, time_array,
        electrode_positions, channel_labels, 
        channel_depths, time_range=(400, 600)
    )
    fig_comparison.suptitle(f'LFP Comparison (avg of {n_trials} trials)', 
                            fontsize=14, fontweight='bold', y=0.995)
    fig_comparison.savefig(os.path.join(save_dir, "03_lfp_comparison.png"), 
                           dpi=300, bbox_inches='tight')
    plt.close(fig_comparison)
    
    # 4. NEW: Bipolar LFP Power Spectra with Pre vs Post comparison
    print("4. Plotting bipolar power spectra (pre vs post stimulus)...")
    fig_bipolar_psd = plot_bipolar_power_spectra_comparison(
        bipolar_signals, channel_labels, channel_depths,
        time_array, stim_onset_ms=stim_onset_ms,
        prestim_window=(200, 500), poststim_window=(500, 800),
        fmax=100
    )
    fig_bipolar_psd.suptitle(f'Bipolar LFP Power Spectra: Pre vs Post (avg of {n_trials} trials)', 
                             fontsize=14, fontweight='bold', y=0.98)
    fig_bipolar_psd.savefig(os.path.join(save_dir, "04_bipolar_power_spectra_comparison.png"), 
                            dpi=300, bbox_inches='tight')
    plt.close(fig_bipolar_psd)
    
    # 5. Current Source Density (CSD)
    print("5. Plotting CSD...")
    fig_csd = plot_csd(
        csd,
        time_array,
        csd_depths,
        time_range=(400, 700),
        figsize=(8, 10),
        cmap='seismic'
    )
    fig_csd.suptitle(f'Laminar Current Source Density (avg of {n_trials} trials)', 
                     fontsize=14, fontweight='bold', y=0.995)
    fig_csd.savefig(os.path.join(save_dir, "05_csd.png"), 
                    dpi=300, bbox_inches='tight')
    plt.close(fig_csd)
    
    # 6. NEW: Wavelet Transform by Layer (SG/G/IG)
    print("6. Plotting wavelet transform by layer (SG/G/IG)...")
    fig_wavelet_layer = plot_wavelet_by_layer(
        bipolar_signals, channel_labels, channel_depths, 
        time_array, time_range=(200, 1000),
        freq_min=1.0, freq_max=100.0, n_freqs=100, wavelet_cycles=3.0
    )
    fig_wavelet_layer.suptitle(f'Wavelet Transform by Layer (avg of {n_trials} trials)', 
                               fontsize=14, fontweight='bold', y=0.98)
    fig_wavelet_layer.savefig(os.path.join(save_dir, "06_wavelet_by_layer.png"), 
                              dpi=300, bbox_inches='tight')
    plt.close(fig_wavelet_layer)
    
    # 7. Wavelet Transform - per channel (original, with edge tapering)
    print("7. Plotting wavelet transform per channel...")
    fig_wavelet = plot_wavelet_transform_improved(
        bipolar_signals, channel_labels, channel_depths, 
        time_array, time_range=(100, 1000),
        freq_min=1.0, freq_max=100.0, n_freqs=100, wavelet_cycles=3.0
    )
    fig_wavelet.suptitle(f'Wavelet Transform per channel (avg of {n_trials} trials)', 
                         fontsize=14, fontweight='bold', y=0.995)
    fig_wavelet.savefig(os.path.join(save_dir, "07_wavelet_per_channel.png"), 
                        dpi=300, bbox_inches='tight')
    plt.close(fig_wavelet)
    
    # 8. PSTH + Population rate FFT by layer (using binned rate)
    print("8. Plotting PSTH and population rate FFT by layer (binned)...")

    # 8a. PSTH plot (time-domain, binned)
    fig_psth = plot_psth_from_rate_data(
        rate_data,
        stim_onset_ms=stim_onset_ms,
        bin_width_ms=1.0,       
        time_range=(0, 1000)
    )
    fig_psth.suptitle(
        f'PSTH by layer (bin = 10 ms, avg of {n_trials} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_psth.savefig(os.path.join(save_dir, "08_psth_by_layer.png"),
                     dpi=300, bbox_inches='tight')
    plt.close(fig_psth)

    # 8b. Build Brian2-like monitors from PSTH (binned) for FFT
    rate_monitors_dict = {}
    for layer_name, layer_rates in rate_data.items():
        rate_monitors_dict[layer_name] = {}

        for mon_name, mon_data in layer_rates.items():
            # Compute PSTH for this monitor
            psth_t_ms, psth_rate_hz = compute_psth(
                mon_data["t_ms"], mon_data["rate_hz"],
                bin_width_ms=10.0   # same bin width as above
            )

            class MockMonitor:
                def __init__(self, t_ms, rate_hz):
                    from brian2 import ms, Hz
                    self.t = t_ms * ms
                    self.rate = rate_hz * Hz

            rate_monitors_dict[layer_name][mon_name] = MockMonitor(
                psth_t_ms,
                psth_rate_hz
            )

    fig_rate_fft = plot_rate_fft(rate_monitors_dict, fmax=100)
    fig_rate_fft.suptitle(
        f'Population Rate FFT by layer (PSTH-binned, avg of {n_trials} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_rate_fft.savefig(os.path.join(save_dir, "08_rate_fft_by_layer.png"),
                         dpi=300, bbox_inches='tight')
    plt.close(fig_rate_fft)
    # 9. Spectrogram per channel
    print("9. Plotting spectrogram per channel...")
    fig_spec_ch = plot_spectrogram_per_channel(
        bipolar_signals, channel_labels, channel_depths,
        time_array, time_range=(200, 1000),
        fmin=1.0, fmax=100.0,
        nperseg_ms=40.0, noverlap_ms=36.0  # very high temporal resolution
    )
    fig_spec_ch.suptitle(
        f'Spectrogram (STFT) per channel (avg of {n_trials} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_spec_ch.savefig(os.path.join(save_dir, "09_spectrogram_per_channel.png"),
                        dpi=300, bbox_inches='tight')
    plt.close(fig_spec_ch)

    # 10. Spectrogram by layer (anatomical)
    print("10. Plotting spectrogram by anatomical layer...")
    fig_spec_layer = plot_spectrogram_by_layer(
        bipolar_signals, channel_depths,
        time_array, time_range=(200, 1000),
        fmin=1.0, fmax=100.0,
        nperseg_ms=40.0, noverlap_ms=36.0
    )
    fig_spec_layer.suptitle(
        f'Spectrogram by anatomical layer (avg of {n_trials} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_spec_layer.savefig(os.path.join(save_dir, "10_spectrogram_by_layer.png"),
                           dpi=300, bbox_inches='tight')
    plt.close(fig_spec_layer)


    print(f"\n=== All plots saved to {save_dir} ===\n")


def plot_wavelet_transform_improved(bipolar_signals, channel_labels, channel_depths, time_array,
                                   figsize=(14, 20), time_range=(300, 800),
                                   freq_min=1.0, freq_max=100.0, n_freqs=100, wavelet_cycles=7.0):
    """
    Improved wavelet transform with edge tapering to reduce border effects.
    """
    time_array = np.asarray(time_array)
    
    if time_range is None:
        time_range = (time_array[0], time_array[-1])
    
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    time_plot = time_array[time_mask]
    
    dt_ms = np.diff(time_array).mean()
    dt = dt_ms / 1000.0
    
    wavelet_freqs = np.linspace(freq_min, freq_max, n_freqs)
    
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    pcm = None
    
    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        lfp = np.asarray(lfp)
        lfp_plot = lfp[time_mask]
        L = len(lfp_plot)
        
        # Apply Tukey window for edge tapering
        from scipy.signal import windows
        taper_fraction = 0.1
        window = windows.tukey(L, alpha=taper_fraction)
        lfp_plot_tapered = lfp_plot * window
        
        wavelet_power = np.zeros((len(wavelet_freqs), L), dtype=float)
        
        for fi, freq in enumerate(wavelet_freqs):
            s = wavelet_cycles / (2.0 * np.pi * freq)
            t_wave = np.arange(-3.0 * s, 3.0 * s + dt, dt)
            
            wavelet = np.exp(2j * np.pi * freq * t_wave) * \
                      np.exp(-t_wave**2 / (2.0 * s**2))
            
            conv_full = np.convolve(lfp_plot_tapered, wavelet, mode='full')
            total_len = conv_full.size
            start = (total_len - L) // 2
            end = start + L
            conv_res = conv_full[start:end]
            
            wavelet_power[fi, :] = np.abs(conv_res)**2
        
        wavelet_power_db = 10.0 * np.log10(wavelet_power + 1e-12)
        
        ax = axes[i]
        pcm = ax.pcolormesh(
            time_plot,
            wavelet_freqs,
            wavelet_power_db,
            cmap='jet',
            shading="auto"
        )
        
        ax.set_ylim(freq_min, freq_max)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        if channel_depths is not None and i < len(channel_depths):
            ax.text(
                1.01,
                0.5,
                f"{channel_depths[i]:.2f} mm",
                transform=ax.transAxes,
                va="center",
                fontsize=8,
            )
    
    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    axes[-1].set_xlim(350, 800)
    
    cbar = fig.colorbar(pcm, ax=axes)
    cbar.set_label("Power (dB)", fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_layer_averaged_analysis(avg_data, save_dir="results/averaged_analysis"):
    """
    Create layer-averaged LFP plots for anatomical layers:
    L1, L23, L4AB, L4C, L5, L6.
    """
    print("\n=== Generating layer-averaged LFP plots (anatomical layers) ===\n")
    
    channel_depths = avg_data["channel_depths"]
    bipolar_signals = avg_data["bipolar_signals"]
    time_array = avg_data["time_array_ms"]
    stim_onset_ms = avg_data["stim_onset_ms"]

    # Use the NEW function instead of SG/G/IG grouping
    layer_masks = assign_channels_to_cortical_layers(channel_depths)

    # Order of layers for plotting
    layer_names = ['L1', 'L23', 'L4AB', 'L4C', 'L5', 'L6']
    
    fig, axes = plt.subplots(len(layer_names), 1, figsize=(10, 12), sharex=True)
    
    for idx, layer_name in enumerate(layer_names):
        ax = axes[idx]
        layer_mask = layer_masks[layer_name]
        
        # channels whose index is within bounds & belong to this layer
        layer_channels = [
            ch for ch in bipolar_signals.keys()
            if ch < len(layer_mask) and layer_mask[ch]
        ]
        
        if len(layer_channels) == 0:
            ax.text(
                0.5, 0.5, f'{layer_name}\n(no channels)',
                ha='center', va='center', transform=ax.transAxes
            )
            ax.set_ylabel(f'{layer_name}', fontsize=11)
            continue
        
        # Average LFP across channels in this layer
        layer_lfp = np.mean([bipolar_signals[ch] for ch in layer_channels], axis=0)
        
        ax.plot(time_array, layer_lfp, 'k-', linewidth=1.5)
        ax.axvline(stim_onset_ms, color='r', linestyle='--', linewidth=1)
        ax.set_ylabel(f'{layer_name}\n(n={len(layer_channels)} ch)', fontsize=11)
        ax.set_xlim(300, 800)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    fig.suptitle(
        f'Layer-averaged Bipolar LFP (anatomical layers, avg of {avg_data["n_trials_averaged"]} trials)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "09_layer_averaged_lfp_anatomical.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Layer-averaged LFP plots (anatomical layers) complete")

def plot_spectrogram_per_channel_peristimulus(bipolar_signals, channel_labels, channel_depths, time_array_ms,
                                              stim_onset_ms=500.0,
                                              time_window_ms=50.0,  # ±75ms around stimulus
                                              fmin=1.0, fmax=100.0,
                                              nperseg_ms=50.0, noverlap_ms=45.0,
                                              figsize=(14, 20)):
    """
    Plot spectrogram (STFT) for each bipolar channel in a vertical stack,
    zoomed to a short window around stimulus onset.
    
    Args:
        time_window_ms: half-width of time window (e.g., 75 means ±75ms around stim)
    """
    time_array_ms = np.asarray(time_array_ms)
    
    time_range = (stim_onset_ms - time_window_ms, stim_onset_ms + time_window_ms)
    
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    pcm = None

    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        t_spec_ms, freqs, Sxx_db = compute_spectrogram_lfp(
            lfp, time_array_ms,
            fmin=fmin, fmax=fmax,
            nperseg_ms=nperseg_ms,
            noverlap_ms=noverlap_ms,
            nfft=4096    
        )

        dt = np.mean(np.diff(t_spec_ms)) if len(t_spec_ms) > 1 else 1.0
        df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0

        extent = [
            t_spec_ms[0] - dt/2,
            t_spec_ms[-1] + dt/2,
            freqs[0] - df/2,
            freqs[-1] + df/2
        ]

        ax = axes[i]
        pcm = ax.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='jet',
            interpolation='bilinear'
        )

        ax.set_ylim(fmin, fmax)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        # Add stimulus onset line
        ax.axvline(stim_onset_ms, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')

        ax.set_xlim(time_range[0], time_range[1])

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)
    plt.tight_layout()

    return fig


def plot_wavelet_per_channel_peristimulus(bipolar_signals, channel_labels, channel_depths, time_array,
                                          stim_onset_ms=500.0,
                                          time_window_ms=50.0,  # ±75ms around stimulus
                                          figsize=(14, 20),
                                          freq_min=1.0, freq_max=100.0, 
                                          n_freqs=100, wavelet_cycles=7.0):
    """
    Plot wavelet transform for each channel, zoomed to short window around stimulus.
    """
    time_array = np.asarray(time_array)
    
    time_range = (stim_onset_ms - time_window_ms, stim_onset_ms + time_window_ms)
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    time_plot = time_array[time_mask]
    
    dt_ms = np.diff(time_array).mean()
    dt = dt_ms / 1000.0
    
    wavelet_freqs = np.linspace(freq_min, freq_max, n_freqs)
    
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    pcm = None
    
    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        lfp = np.asarray(lfp)
        lfp_plot = lfp[time_mask]
        L = len(lfp_plot)
        
        # Apply Tukey window for edge tapering
        from scipy.signal import windows
        taper_fraction = 0.1
        window = windows.tukey(L, alpha=taper_fraction)
        lfp_plot_tapered = lfp_plot * window
        
        wavelet_power = np.zeros((len(wavelet_freqs), L), dtype=float)
        
        for fi, freq in enumerate(wavelet_freqs):
            s = wavelet_cycles / (2.0 * np.pi * freq)
            t_wave = np.arange(-3.0 * s, 3.0 * s + dt, dt)
            
            wavelet = np.exp(2j * np.pi * freq * t_wave) * \
                      np.exp(-t_wave**2 / (2.0 * s**2))
            
            conv_full = np.convolve(lfp_plot_tapered, wavelet, mode='full')
            total_len = conv_full.size
            start = (total_len - L) // 2
            end = start + L
            conv_res = conv_full[start:end]
            
            wavelet_power[fi, :] = np.abs(conv_res)**2
        
        wavelet_power_db = 10.0 * np.log10(wavelet_power + 1e-12)
        
        ax = axes[i]
        pcm = ax.pcolormesh(
            time_plot,
            wavelet_freqs,
            wavelet_power_db,
            cmap='jet',
            shading="auto"
        )
        
        ax.set_ylim(freq_min, freq_max)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        # Add stimulus onset line
        ax.axvline(stim_onset_ms, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        
        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')
    
    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    
    cbar = fig.colorbar(pcm, ax=axes)
    cbar.set_label("Power (dB)", fontsize=10)
    
    plt.tight_layout()
    return fig


def compute_trial_average_spectrogram(trials_data, 
                                     time_range=(200, 1000),
                                     fmin=1.0, fmax=100.0,
                                     nperseg_ms=50.0, noverlap_ms=45.0):
    """
    Compute spectrogram for each trial, then average across trials.
    Returns average spectrogram for each channel.
    
    Returns:
        t_spec_ms: time array for spectrogram
        freqs: frequency array
        avg_spectrograms: dict {ch_idx: averaged Sxx_db across trials}
    """
    if len(trials_data) == 0:
        raise ValueError("No trials data!")
    
    ref_trial = trials_data[0]
    time_array_ms = ref_trial["time_array_ms"]
    n_channels = ref_trial["bipolar_matrix"].shape[0]
    
    # First trial to get dimensions
    first_bipolar = {i: ref_trial["bipolar_matrix"][i, :] for i in range(n_channels)}
    t_spec_ms, freqs, _ = compute_spectrogram_lfp(
        first_bipolar[0], time_array_ms,
        fmin=fmin, fmax=fmax,
        nperseg_ms=nperseg_ms,
        noverlap_ms=noverlap_ms,
        nfft=4096
    )
    
    # Initialize accumulator
    n_freqs = len(freqs)
    n_times = len(t_spec_ms)
    spectrogram_sum = {ch: np.zeros((n_freqs, n_times)) for ch in range(n_channels)}
    
    # Accumulate spectrograms from all trials
    for trial in trials_data:
        bipolar_signals = {i: trial["bipolar_matrix"][i, :] for i in range(n_channels)}
        
        for ch_idx in range(n_channels):
            _, _, Sxx_db = compute_spectrogram_lfp(
                bipolar_signals[ch_idx], time_array_ms,
                fmin=fmin, fmax=fmax,
                nperseg_ms=nperseg_ms,
                noverlap_ms=noverlap_ms,
                nfft=4096
            )
            spectrogram_sum[ch_idx] += Sxx_db
    
    # Average
    n_trials = len(trials_data)
    avg_spectrograms = {ch: spectrogram_sum[ch] / n_trials for ch in range(n_channels)}
    
    return t_spec_ms, freqs, avg_spectrograms


def plot_averaged_spectrogram_per_channel(trials_data, channel_labels, channel_depths,
                                         time_range=(200, 1000),
                                         fmin=1.0, fmax=100.0,
                                         nperseg_ms=50.0, noverlap_ms=45.0,
                                         figsize=(14, 20)):
    """
    Plot average of trial spectrograms (not spectrogram of average).
    """
    print("Computing trial-averaged spectrograms...")
    t_spec_ms, freqs, avg_spectrograms = compute_trial_average_spectrogram(
        trials_data, time_range, fmin, fmax, nperseg_ms, noverlap_ms
    )
    
    n_channels = len(avg_spectrograms)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    pcm = None
    
    for i, (ch_idx, Sxx_db) in enumerate(avg_spectrograms.items()):
        dt = np.mean(np.diff(t_spec_ms)) if len(t_spec_ms) > 1 else 1.0
        df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0
        
        extent = [
            t_spec_ms[0] - dt/2,
            t_spec_ms[-1] + dt/2,
            freqs[0] - df/2,
            freqs[-1] + df/2
        ]
        
        ax = axes[i]
        pcm = ax.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='jet',
            interpolation='bilinear'
        )
        
        ax.set_ylim(fmin, fmax)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')
        
        ax.set_xlim(time_range[0], time_range[1])
    
    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)
    plt.tight_layout()
    
    return fig


def compute_trial_average_wavelet(trials_data,
                                  time_range=(200, 1000),
                                  freq_min=1.0, freq_max=100.0,
                                  n_freqs=100, wavelet_cycles=7.0):
    """
    Compute wavelet transform for each trial, then average across trials.
    
    Returns:
        time_plot: time array
        wavelet_freqs: frequency array  
        avg_wavelets: dict {ch_idx: averaged wavelet power (dB) across trials}
    """
    if len(trials_data) == 0:
        raise ValueError("No trials data!")
    
    ref_trial = trials_data[0]
    time_array = ref_trial["time_array_ms"]
    n_channels = ref_trial["bipolar_matrix"].shape[0]
    
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    time_plot = time_array[time_mask]
    L = len(time_plot)
    
    dt_ms = np.diff(time_array).mean()
    dt = dt_ms / 1000.0
    
    wavelet_freqs = np.linspace(freq_min, freq_max, n_freqs)
    
    # Initialize accumulator
    wavelet_sum = {ch: np.zeros((n_freqs, L)) for ch in range(n_channels)}
    
    # Accumulate wavelets from all trials
    for trial in trials_data:
        bipolar_matrix = trial["bipolar_matrix"]
        
        for ch_idx in range(n_channels):
            lfp = bipolar_matrix[ch_idx, :]
            lfp_plot = lfp[time_mask]
            
            # Apply tapering
            from scipy.signal import windows
            taper_fraction = 0.1
            window = windows.tukey(L, alpha=taper_fraction)
            lfp_plot_tapered = lfp_plot * window
            
            wavelet_power = np.zeros((len(wavelet_freqs), L), dtype=float)
            
            for fi, freq in enumerate(wavelet_freqs):
                s = wavelet_cycles / (2.0 * np.pi * freq)
                t_wave = np.arange(-3.0 * s, 3.0 * s + dt, dt)
                
                wavelet = np.exp(2j * np.pi * freq * t_wave) * \
                          np.exp(-t_wave**2 / (2.0 * s**2))
                
                conv_full = np.convolve(lfp_plot_tapered, wavelet, mode='full')
                total_len = conv_full.size
                start = (total_len - L) // 2
                end = start + L
                conv_res = conv_full[start:end]
                
                wavelet_power[fi, :] = np.abs(conv_res)**2
            
            wavelet_power_db = 10.0 * np.log10(wavelet_power + 1e-12)
            wavelet_sum[ch_idx] += wavelet_power_db
    
    # Average
    n_trials = len(trials_data)
    avg_wavelets = {ch: wavelet_sum[ch] / n_trials for ch in range(n_channels)}
    
    return time_plot, wavelet_freqs, avg_wavelets

def plot_spectrogram_per_channel_peristimulus_improved(bipolar_signals, channel_labels, channel_depths, time_array_ms,
                                                       stim_onset_ms=500.0,
                                                       time_window_ms=50.0,
                                                       fmin=10.0, fmax=100.0,  # Start at 10Hz to reduce low-freq dominance
                                                       nperseg_ms=20.0,  # Shorter window for better time resolution
                                                       noverlap_ms=18.0,  # High overlap
                                                       figsize=(14, 20)):
    """
    Improved peristimulus spectrogram with better parameters for short time windows.
    """
    time_array_ms = np.asarray(time_array_ms)
    
    time_range = (stim_onset_ms - time_window_ms, stim_onset_ms + time_window_ms)
    
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    pcm = None

    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        t_spec_ms, freqs, Sxx_db = compute_spectrogram_lfp(
            lfp, time_array_ms,
            fmin=fmin, fmax=fmax,
            nperseg_ms=nperseg_ms,
            noverlap_ms=noverlap_ms,
            nfft=2048    
        )

        dt = np.mean(np.diff(t_spec_ms)) if len(t_spec_ms) > 1 else 1.0
        df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0

        extent = [
            t_spec_ms[0] - dt/2,
            t_spec_ms[-1] + dt/2,
            freqs[0] - df/2,
            freqs[-1] + df/2
        ]

        ax = axes[i]
        
        # Use symmetric color limits around median to see changes better
        vmin, vmax = np.percentile(Sxx_db, [5, 95])
        
        pcm = ax.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='jet',
            interpolation='bilinear',
            vmin=vmin, vmax=vmax  # Better contrast
        )

        ax.set_ylim(fmin, fmax)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        # Add stimulus onset line
        ax.axvline(stim_onset_ms, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')

        ax.set_xlim(time_range[0], time_range[1])

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)
    plt.tight_layout()

    return fig


def plot_baseline_normalized_wavelet_peristimulus(bipolar_signals, channel_labels, channel_depths, time_array,
                                                  stim_onset_ms=500.0,
                                                  time_window_ms=50.0,
                                                  baseline_window=(-200, -50),  # ms before stimulus
                                                  figsize=(14, 20),
                                                  freq_min=10.0, freq_max=100.0,  # Skip very low frequencies
                                                  n_freqs=50, wavelet_cycles=6.0):
    """
    Plot baseline-normalized wavelet to see stimulus-evoked changes clearly.
    Shows percentage change from baseline.
    """
    time_array = np.asarray(time_array)
    
    time_range = (stim_onset_ms - time_window_ms, stim_onset_ms + time_window_ms)
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    time_plot = time_array[time_mask]
    
    # Baseline mask
    baseline_mask = (time_array >= baseline_window[0]) & (time_array < baseline_window[1])
    
    dt_ms = np.diff(time_array).mean()
    dt = dt_ms / 1000.0
    
    wavelet_freqs = np.linspace(freq_min, freq_max, n_freqs)
    
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    pcm = None
    
    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        lfp = np.asarray(lfp)
        
        # Compute wavelet on full signal
        L_full = len(lfp)
        wavelet_power_full = np.zeros((len(wavelet_freqs), L_full), dtype=float)
        
        for fi, freq in enumerate(wavelet_freqs):
            s = wavelet_cycles / (2.0 * np.pi * freq)
            t_wave = np.arange(-3.0 * s, 3.0 * s + dt, dt)
            
            wavelet = np.exp(2j * np.pi * freq * t_wave) * \
                      np.exp(-t_wave**2 / (2.0 * s**2))
            
            conv_full = np.convolve(lfp, wavelet, mode='same')
            wavelet_power_full[fi, :] = np.abs(conv_full)**2
        
        # Extract baseline and compute mean
        baseline_power = wavelet_power_full[:, baseline_mask]
        baseline_mean = np.mean(baseline_power, axis=1, keepdims=True)
        
        # Normalize: percent change from baseline
        wavelet_power_normalized = ((wavelet_power_full - baseline_mean) / (baseline_mean + 1e-12)) * 100
        
        # Extract peristimulus window
        wavelet_plot = wavelet_power_full[:, time_mask]
        
        ax = axes[i]
        
        # Use symmetric colormap centered at 0
        vmax = np.percentile(np.abs(wavelet_plot), 95)
        
        pcm = ax.pcolormesh(
            time_plot,
            wavelet_freqs,
            wavelet_plot,
            cmap='jet',  # Red=increase, Blue=decrease
            shading="auto",
            vmin=-vmax, vmax=vmax
        )
        
        ax.set_ylim(freq_min, freq_max)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        # Add stimulus onset line
        ax.axvline(stim_onset_ms, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        
        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')
    
    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    
    cbar = fig.colorbar(pcm, ax=axes)
    cbar.set_label("% Change from baseline", fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_baseline_normalized_spectrogram_peristimulus(bipolar_signals, channel_labels, channel_depths, time_array_ms,
                                                      stim_onset_ms=500.0,
                                                      time_window_ms=50.0,
                                                      baseline_window=(-200, -50),
                                                      fmin=10.0, fmax=100.0,
                                                      nperseg_ms=20.0, noverlap_ms=18.0,
                                                      figsize=(14, 20)):
    """
    Baseline-normalized spectrogram showing percentage change from baseline.
    """
    time_array_ms = np.asarray(time_array_ms)
    
    time_range = (stim_onset_ms - time_window_ms, stim_onset_ms + time_window_ms)
    
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    pcm = None

    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        # Compute full spectrogram
        t_spec_ms, freqs, Sxx_db = compute_spectrogram_lfp(
            lfp, time_array_ms,
            fmin=fmin, fmax=fmax,
            nperseg_ms=nperseg_ms,
            noverlap_ms=noverlap_ms,
            nfft=2048
        )
        
        # Find baseline period in spectrogram
        baseline_mask = (t_spec_ms >= baseline_window[0]) & (t_spec_ms < baseline_window[1])
        baseline_power = Sxx_db[:, baseline_mask]
        baseline_mean = np.mean(baseline_power, axis=1, keepdims=True)
        
        # Normalize: difference from baseline (dB difference = power ratio)
        Sxx_normalized = Sxx_db #- baseline_mean
        
        dt = np.mean(np.diff(t_spec_ms)) if len(t_spec_ms) > 1 else 1.0
        df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0

        extent = [
            t_spec_ms[0] - dt/2,
            t_spec_ms[-1] + dt/2,
            freqs[0] - df/2,
            freqs[-1] + df/2
        ]

        ax = axes[i]
        
        # Symmetric colormap
        vmax = np.percentile(np.abs(Sxx_normalized), 95)
        
        pcm = ax.imshow(
            Sxx_normalized,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='RdBu_r',
            interpolation='bilinear',
            vmin=-vmax, vmax=vmax
        )

        ax.set_ylim(fmin, fmax)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        ax.axvline(stim_onset_ms, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')

        ax.set_xlim(time_range[0], time_range[1])

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    cbar = fig.colorbar(pcm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Power change (dB)", fontsize=10)
    plt.tight_layout()

    return fig
def plot_averaged_wavelet_per_channel(trials_data, channel_labels, channel_depths,
                                      time_range=(200, 1000),
                                      freq_min=1.0, freq_max=100.0,
                                      n_freqs=100, wavelet_cycles=7.0,
                                      figsize=(14, 20)):
    """
    Plot average of trial wavelets (not wavelet of average).
    """
    print("Computing trial-averaged wavelets...")
    time_plot, wavelet_freqs, avg_wavelets = compute_trial_average_wavelet(
        trials_data, time_range, freq_min, freq_max, n_freqs, wavelet_cycles
    )
    
    n_channels = len(avg_wavelets)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    pcm = None
    
    for i, (ch_idx, wavelet_power_db) in enumerate(avg_wavelets.items()):
        ax = axes[i]
        pcm = ax.pcolormesh(
            time_plot,
            wavelet_freqs,
            wavelet_power_db,
            cmap='jet',
            shading="auto"
        )
        
        ax.set_ylim(freq_min, freq_max)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        
        depth_txt = f"{channel_depths[ch_idx]:.2f} mm" if channel_depths is not None else ""
        ax.set_title(f"{channel_labels[ch_idx]}  ({depth_txt})", fontsize=9, loc='left')
    
    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    
    cbar = fig.colorbar(pcm, ax=axes)
    cbar.set_label("Power (dB)", fontsize=10)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Configuration
    TRIAL_DIR = "results/lfp_trials"
    N_TRIALS = 15
    SAVE_DIR = "results/averaged_analysis"
    
    print("="*60)
    print("TRIAL-AVERAGED LFP ANALYSIS")
    print("="*60)
    
    # Step 1: Load all trials
    print("\n### Step 1: Loading all trial data ###")
    trials_data = load_all_trials(trial_dir=TRIAL_DIR, n_trials=N_TRIALS)
    
    # Step 2: Average across trials
    print("\n### Step 2: Averaging across trials ###")
    avg_data = average_trials(trials_data)
    
    # Step 3: Generate all plots
    print("\n### Step 3: Generating plots ###")
    plot_all_analyses(avg_data, save_dir=SAVE_DIR)
    
    # Step 4: Generate layer-averaged plots
    print("\n### Step 4: Generating layer-averaged plots ###")
    plot_layer_averaged_analysis(avg_data, save_dir=SAVE_DIR)
    # After Step 4, add:
    
    # Step 5: Peristimulus time-frequency analysis (zoomed around stimulus)
    print("\n### Step 5: Generating peristimulus time-frequency plots ###")
    
    stim_onset_ms = avg_data["stim_onset_ms"]
    
    # Spectrogram - peristimulus (±75ms)
    fig_spec_peri = plot_spectrogram_per_channel_peristimulus(
        avg_data["bipolar_signals"], 
        avg_data["channel_labels"],
        avg_data["channel_depths"],
        avg_data["time_array_ms"],
        stim_onset_ms=stim_onset_ms,
        time_window_ms=50.0,
        fmin=1.0, fmax=100.0
    )
    fig_spec_peri.suptitle(
        f'Spectrogram Peristimulus (±75ms, avg of {N_TRIALS} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_spec_peri.savefig(os.path.join(SAVE_DIR, "11_spectrogram_peristimulus.png"),
                          dpi=300, bbox_inches='tight')
    plt.close(fig_spec_peri)
    
    # Wavelet - peristimulus (±75ms)
    fig_wav_peri = plot_wavelet_per_channel_peristimulus(
        avg_data["bipolar_signals"],
        avg_data["channel_labels"],
        avg_data["channel_depths"],
        avg_data["time_array_ms"],
        stim_onset_ms=stim_onset_ms,
        time_window_ms=50.0,
        freq_min=1.0, freq_max=100.0,
        n_freqs=100, wavelet_cycles=3.0
    )
    fig_wav_peri.suptitle(
        f'Wavelet Peristimulus (±75ms, avg of {N_TRIALS} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_wav_peri.savefig(os.path.join(SAVE_DIR, "12_wavelet_peristimulus.png"),
                         dpi=300, bbox_inches='tight')
    plt.close(fig_wav_peri)
    
    # Step 6: Average of trial time-frequency (not time-frequency of average)
    print("\n### Step 6: Generating average-of-trials time-frequency plots ###")
    
    # Spectrogram - average of trials
    fig_spec_avg_trials = plot_averaged_spectrogram_per_channel(
        trials_data,
        avg_data["channel_labels"],
        avg_data["channel_depths"],
        time_range=(200, 1000),
        fmin=1.0, fmax=100.0,
        nperseg_ms=80.0, noverlap_ms=70.0
    )
    fig_spec_avg_trials.suptitle(
        f'Spectrogram (Average of {N_TRIALS} Trial Spectrograms)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_spec_avg_trials.savefig(os.path.join(SAVE_DIR, "13_spectrogram_avg_of_trials.png"),
                                dpi=300, bbox_inches='tight')
    plt.close(fig_spec_avg_trials)
    
    # Wavelet - average of trials
    fig_wav_avg_trials = plot_averaged_wavelet_per_channel(
        trials_data,
        avg_data["channel_labels"],
        avg_data["channel_depths"],
        time_range=(200, 1000),
        freq_min=1.0, freq_max=100.0,
        n_freqs=100, wavelet_cycles=3.0
    )
    fig_wav_avg_trials.suptitle(
        f'Wavelet (Average of {N_TRIALS} Trial Wavelets)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_wav_avg_trials.savefig(os.path.join(SAVE_DIR, "14_wavelet_avg_of_trials.png"),
                                dpi=300, bbox_inches='tight')
    plt.close(fig_wav_avg_trials)
    # Step 7: Improved peristimulus plots with baseline normalization
    print("\n### Step 7: Baseline-normalized peristimulus plots ###")
    
    # Baseline-normalized wavelet
    fig_wav_norm = plot_baseline_normalized_wavelet_peristimulus(
        avg_data["bipolar_signals"],
        avg_data["channel_labels"],
        avg_data["channel_depths"],
        avg_data["time_array_ms"],
        stim_onset_ms=stim_onset_ms,
        time_window_ms=50.0,
        baseline_window=(-200, -50),
        freq_min=10.0, freq_max=100.0
    )
    fig_wav_norm.suptitle(
        f'Wavelet Peristimulus - Baseline Normalized (avg of {N_TRIALS} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_wav_norm.savefig(os.path.join(SAVE_DIR, "15_wavelet_peristimulus_normalized.png"),
                         dpi=300, bbox_inches='tight')
    plt.close(fig_wav_norm)
    
    # Baseline-normalized spectrogram  
    fig_spec_norm = plot_baseline_normalized_spectrogram_peristimulus(
        avg_data["bipolar_signals"],
        avg_data["channel_labels"],
        avg_data["channel_depths"],
        avg_data["time_array_ms"],
        stim_onset_ms=stim_onset_ms,
        time_window_ms=50.0,
        baseline_window=(-200, -50),
        fmin=10.0, fmax=100.0,
        nperseg_ms=20.0, noverlap_ms=18.0
    )
    fig_spec_norm.suptitle(
        f'Spectrogram Peristimulus - Baseline Normalized (avg of {N_TRIALS} trials)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig_spec_norm.savefig(os.path.join(SAVE_DIR, "16_spectrogram_peristimulus_normalized.png"),
                          dpi=300, bbox_inches='tight')
    plt.close(fig_spec_norm)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved to: {SAVE_DIR}")
    print("="*60)
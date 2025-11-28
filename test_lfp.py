import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict

# Import your analysis functions
from src.analysis import calculate_lfp_kernel_method, compute_csd_from_lfp, compute_bipolar_lfp
from config.config_test2 import CONFIG

# Superlets (optional)
try:
    from superlets import superlets
except ImportError:
    print("WARNING: superlets not installed. Install with: pip install superlets")
    superlets = None


def process_raw_trials_batch(
    raw_data_dir="results/lfp_trials_conditions",
    output_dir="results/lfp_trials_processed",
    config=None,
    fs=10000,
    electrode_positions=None,
    verbose=True,
):
    """
    Batch process all raw trial data to compute LFP, CSD, bipolar, etc.
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
    
    for idx, raw_file in enumerate(raw_files):
        raw_path = os.path.join(raw_data_dir, raw_file)
        
        if verbose:
            print(f"[{idx+1}/{len(raw_files)}] Processing {raw_file}...")
        
        # Load raw data
        raw_data = np.load(raw_path, allow_pickle=True)
        
        trial_id = int(raw_data['trial_id'])
        spike_data = raw_data['spike_data'].item()
        neuron_group_info = raw_data['neuron_group_info'].item()
        actual_baseline_ms = float(raw_data['actual_baseline_ms'])
        post_ms = float(raw_data['post_ms'])
        total_sim_ms = actual_baseline_ms + post_ms
        
        # Reconstruct spike monitors format for LFP calculation
        # Create mock spike monitor objects that mimic Brian2 SpikeMonitor interface
        class MockSpikeMonitor:
            """Mock Brian2 SpikeMonitor for LFP calculation"""
            def __init__(self, spike_indices, spike_times_ms):
                self.i = spike_indices
                # Store times in ms as array, calculate_lfp_kernel_method will convert
                self._t_ms = spike_times_ms
                
            @property
            def t(self):
                # Return a mock object that supports division
                class MockTime:
                    def __init__(self, t_ms):
                        self._t_ms = t_ms
                    def __truediv__(self, unit):
                        # When divided by ms, return the time in ms
                        return self._t_ms
                return MockTime(self._t_ms)
        
        class MockNeuronGroup:
            """Mock Brian2 NeuronGroup for LFP calculation"""
            def __init__(self, N, x_vals, y_vals, z_vals):
                self.N = N
                self._x = x_vals
                self._y = y_vals
                self._z = z_vals
            
            @property
            def x(self):
                class MockPosition:
                    def __init__(self, vals):
                        self._vals = vals
                    def __truediv__(self, unit):
                        # When divided by mm or meter, return the values
                        return self._vals
                    def __len__(self):
                        return len(self._vals)
                return MockPosition(self._x)
            
            @property
            def y(self):
                class MockPosition:
                    def __init__(self, vals):
                        self._vals = vals
                    def __truediv__(self, unit):
                        return self._vals
                    def __len__(self):
                        return len(self._vals)
                return MockPosition(self._y)
            
            @property
            def z(self):
                class MockPosition:
                    def __init__(self, vals):
                        self._vals = vals
                    def __truediv__(self, unit):
                        return self._vals
                    def __len__(self):
                        return len(self._vals)
                return MockPosition(self._z)
        
        spike_monitors = {}
        neuron_groups = {}
        
        for layer_name in spike_data.keys():
            spike_monitors[layer_name] = {}
            for mon_name, mon_data in spike_data[layer_name].items():
                spike_monitors[layer_name][mon_name] = MockSpikeMonitor(
                    mon_data['i'], 
                    mon_data['t']
                )
            
            # Create mock neuron groups
            neuron_groups[layer_name] = {}
            for pop_name, ng_data in neuron_group_info[layer_name].items():
                neuron_groups[layer_name][pop_name] = MockNeuronGroup(
                    ng_data['N'],
                    ng_data['x'],
                    ng_data['y'],
                    ng_data['z']
                )
        
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
        print("=== Batch processing complete ===\n")


def load_trials_by_condition(processed_dir="results/lfp_trials_processed"):
    """
    Load all processed trial files and group them by condition.
    
    Returns:
        dict: {condition_name: list of trial data dicts}
    """
    trials_by_condition = defaultdict(list)
    
    processed_files = sorted([f for f in os.listdir(processed_dir) 
                             if f.endswith('_processed.npz')])
    
    print(f"Loading {len(processed_files)} processed trials...")
    
    for fname in processed_files:
        fpath = os.path.join(processed_dir, fname)
        data = np.load(fpath, allow_pickle=True)
        
        condition = str(data['condition_name'])
        trials_by_condition[condition].append({
            'trial_id': int(data['trial_id']),
            'condition_name': condition,
            'stim_rate_multiplier': float(data['stim_rate_multiplier']),
            'stim_n_multiplier': float(data['stim_n_multiplier']),
            'time_array_ms': data['time_array_ms'],
            'lfp_matrix': data['lfp_matrix'],
            'bipolar_matrix': data['bipolar_matrix'],
            'csd': data['csd'],
            'stim_onset_ms': float(data['stim_onset_ms']),
            'stim_params': data['stim_params'].item(),
        })
    
    print(f"Found {len(trials_by_condition)} conditions:")
    for cond, trials in trials_by_condition.items():
        print(f"  {cond}: {len(trials)} trials")
    
    return trials_by_condition


def compute_superlet_analysis(
    bipolar_lfps,
    time_s,
    foi=None,
    baseline_window=(0.85, 0.99),  # Pre-stimulus baseline
    analysis_window=(0.95, 1.3),   # Around stimulus
    c1=2,
    ord=(1, 5),
):
    """
    Compute superlet time-frequency analysis averaged across trials.
    
    Parameters:
        bipolar_lfps : np.array, shape (n_trials, n_channels, n_time)
        time_s : np.array, time in seconds
        foi : frequencies of interest
        baseline_window : tuple, (start, end) in seconds for baseline
        analysis_window : tuple, (start, end) in seconds for analysis
        c1 : superlet parameter
        ord : superlet order range
    
    Returns:
        S_mean : np.array, shape (n_channels, n_freqs, n_twin)
        t_win : np.array, time points in analysis window
        foi : np.array, frequencies
    """
    if superlets is None:
        raise ImportError("superlets package not installed")
    
    if foi is None:
        foi = np.arange(1, 120, 1)
    
    n_trials, n_channels, n_time = bipolar_lfps.shape
    
    dt = np.diff(time_s).mean()
    fs = 1.0 / dt
    
    # Define windows
    baseline_idx = (time_s >= baseline_window[0]) & (time_s <= baseline_window[1])
    analysis_idx = (time_s >= analysis_window[0]) & (time_s <= analysis_window[1])
    t_win = time_s[analysis_idx]
    n_twin = analysis_idx.sum()
    
    # Accumulator for mean TF maps
    S_mean = np.zeros((n_channels, len(foi), n_twin))
    
    print(f"Computing superlets for {n_channels} channels, {n_trials} trials...")
    
    for ch in range(n_channels):
        acc = np.zeros((len(foi), n_twin))
        
        for tr in range(n_trials):
            if tr % 5 == 0:
                print(f"  Channel {ch+1}/{n_channels}, Trial {tr+1}/{n_trials}")
            
            signal_data = bipolar_lfps[tr, ch, :]
            
            # Compute superlet on full signal
            S_full = superlets(signal_data, fs, foi, c1, ord)
            
            # Baseline normalization (z-score)
            baseline_power = S_full[:, baseline_idx]
            baseline_mean = baseline_power.mean(axis=1, keepdims=True)
            baseline_std = baseline_power.std(axis=1, keepdims=True)
            baseline_std[baseline_std == 0] = np.finfo(float).eps
            
            S_full_z = (S_full - baseline_mean) / baseline_std
            
            # Crop to analysis window
            S_win = S_full_z[:, analysis_idx]
            acc += S_win
        
        # Average over trials
        S_mean[ch] = acc / n_trials
    
    return S_mean, t_win, foi


def compute_power_spectra(
    bipolar_lfps,
    time_s,
    stim_onset_s,
    pre_window=(-0.2, -0.01),   # Relative to stimulus
    post_window=(0.01, 0.2),     # Relative to stimulus
    nperseg=256,
):
    """
    Compute power spectra before and after stimulus for each channel.
    """
    n_trials, n_channels, n_time = bipolar_lfps.shape
    
    dt = np.diff(time_s).mean()
    fs = 1.0 / dt
    
    # Define time windows
    pre_idx = ((time_s >= (stim_onset_s + pre_window[0])) & 
               (time_s <= (stim_onset_s + pre_window[1])))
    post_idx = ((time_s >= (stim_onset_s + post_window[0])) & 
                (time_s <= (stim_onset_s + post_window[1])))
    
    # Storage for all trials
    pre_psds = []
    post_psds = []
    
    print(f"Computing power spectra for {n_channels} channels, {n_trials} trials...")
    
    for tr in range(n_trials):
        if tr % 5 == 0:
            print(f"  Trial {tr+1}/{n_trials}")
        
        pre_psd_trial = []
        post_psd_trial = []
        
        for ch in range(n_channels):
            signal_data = bipolar_lfps[tr, ch, :]
            
            # Pre-stimulus PSD
            pre_signal = signal_data[pre_idx]
            f_pre, psd_pre = signal.welch(pre_signal, fs=fs, nperseg=nperseg)
            pre_psd_trial.append(psd_pre)
            
            # Post-stimulus PSD
            post_signal = signal_data[post_idx]
            f_post, psd_post = signal.welch(post_signal, fs=fs, nperseg=nperseg)
            post_psd_trial.append(psd_post)
        
        pre_psds.append(np.array(pre_psd_trial))
        post_psds.append(np.array(post_psd_trial))
    
    # Convert to arrays: (n_trials, n_channels, n_freqs)
    pre_psds = np.array(pre_psds)
    post_psds = np.array(post_psds)
    
    # Compute mean and SEM across trials
    pre_power_mean = pre_psds.mean(axis=0)
    post_power_mean = post_psds.mean(axis=0)
    
    pre_power_sem = pre_psds.std(axis=0) / np.sqrt(n_trials)
    post_power_sem = post_psds.std(axis=0) / np.sqrt(n_trials)
    
    return {
        'freqs': f_pre,
        'pre_power': pre_power_mean,
        'post_power': post_power_mean,
        'pre_sem': pre_power_sem,
        'post_sem': post_power_sem,
    }


def plot_superlet_results(S_mean, t_win, foi, condition_name, save_path=None):
    """
    Plot superlet time-frequency maps for all channels.
    """
    n_channels = S_mean.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_channels)))
    n_cols = int(np.ceil(n_channels / n_rows))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 10), dpi=200)
    axs = axs.ravel()
    
    vmin = S_mean.min()
    vmax = S_mean.max()
    
    for ch in range(n_channels):
        pcm = axs[ch].pcolormesh(t_win, foi, S_mean[ch], 
                                  cmap='jet', vmin=vmin, vmax=vmax)
        axs[ch].set_title(f"Channel {ch}", fontsize=10)
        axs[ch].set_xlabel("Time (s)", fontsize=8)
        axs[ch].set_ylabel("Frequency (Hz)", fontsize=8)
        axs[ch].tick_params(labelsize=7)
        axs[ch].axvline(1.0, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    # Hide unused subplots
    for ch in range(n_channels, len(axs)):
        axs[ch].axis('off')
    
    plt.suptitle(f"Superlet Analysis - Condition: {condition_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label='Z-score')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved superlet plot to {save_path}")
    
    plt.close()


def plot_power_spectra(psd_results, condition_name, save_path=None, freq_max=100):
    """
    Plot pre/post stimulus power spectra for all channels.
    """
    freqs = psd_results['freqs']
    pre_power = psd_results['pre_power']
    post_power = psd_results['post_power']
    pre_sem = psd_results['pre_sem']
    post_sem = psd_results['post_sem']
    
    n_channels = pre_power.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_channels)))
    n_cols = int(np.ceil(n_channels / n_rows))
    
    # Limit frequency range
    freq_mask = freqs <= freq_max
    freqs_plot = freqs[freq_mask]
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 10), dpi=200)
    axs = axs.ravel()
    
    for ch in range(n_channels):
        pre_p = pre_power[ch, freq_mask]
        post_p = post_power[ch, freq_mask]
        pre_s = pre_sem[ch, freq_mask]
        post_s = post_sem[ch, freq_mask]
        
        # Plot with error bands
        axs[ch].plot(freqs_plot, pre_p, 'b-', label='Pre-stim', linewidth=1.5)
        axs[ch].fill_between(freqs_plot, pre_p - pre_s, pre_p + pre_s, 
                             color='b', alpha=0.2)
        
        axs[ch].plot(freqs_plot, post_p, 'r-', label='Post-stim', linewidth=1.5)
        axs[ch].fill_between(freqs_plot, post_p - post_s, post_p + post_s, 
                             color='r', alpha=0.2)
        
        axs[ch].set_title(f"Channel {ch}", fontsize=10)
        axs[ch].set_xlabel("Frequency (Hz)", fontsize=8)
        axs[ch].set_ylabel("Power (V²/Hz)", fontsize=8)
        axs[ch].set_yscale('log')
        axs[ch].tick_params(labelsize=7)
        axs[ch].legend(fontsize=7)
        axs[ch].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for ch in range(n_channels, len(axs)):
        axs[ch].axis('off')
    
    plt.suptitle(f"Power Spectra - Condition: {condition_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved power spectra plot to {save_path}")
    
    plt.close()


def analyze_all_conditions(
    processed_dir="results/lfp_trials_processed",
    output_dir="results/analysis_by_condition",
    foi=None,
    baseline_window=(0.85, 0.99),
    analysis_window=(0.95, 1.3),
    pre_window=(-0.2, -0.01),
    post_window=(0.01, 0.2),
    verbose=True,
):
    """
    Main analysis function: process all conditions separately.
    """
    if foi is None:
        foi = np.arange(1, 120, 1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trials grouped by condition
    trials_by_condition = load_trials_by_condition(processed_dir)
    
    # Process each condition
    for condition_name, trials in trials_by_condition.items():
        print(f"\n{'='*60}")
        print(f"Analyzing condition: {condition_name} ({len(trials)} trials)")
        print(f"{'='*60}")
        
        # Create condition-specific output directory
        cond_dir = os.path.join(output_dir, condition_name)
        os.makedirs(cond_dir, exist_ok=True)
        
        # Stack bipolar LFPs from all trials
        bipolar_lfps = []
        time_arrays = []
        stim_onsets = []
        
        for trial in trials:
            bipolar_lfps.append(trial['bipolar_matrix'])
            time_arrays.append(trial['time_array_ms'])
            stim_onsets.append(trial['stim_onset_ms'])
        
        # Use first trial's time array
        time_ms = time_arrays[0]
        time_s = time_ms / 1000.0
        
        # Average stimulus onset
        stim_onset_s = np.mean(stim_onsets) / 1000.0
        
        # Stack: (n_trials, n_channels, n_time)
        bipolar_lfps = np.stack(bipolar_lfps, axis=0)
        
        if verbose:
            print(f"Data shape: {bipolar_lfps.shape}")
            print(f"Mean stimulus onset: {stim_onset_s:.3f} s")
        
        # 1. SUPERLET ANALYSIS
        if superlets is not None:
            print("\n1. Computing superlet analysis...")
            try:
                S_mean, t_win, foi_used = compute_superlet_analysis(
                    bipolar_lfps, time_s, foi=foi,
                    baseline_window=baseline_window,
                    analysis_window=analysis_window,
                )
                
                # Save superlet results
                superlet_path = os.path.join(cond_dir, "superlet_results.npz")
                np.savez_compressed(
                    superlet_path,
                    S_mean=S_mean,
                    t_win=t_win,
                    foi=foi_used,
                    condition_name=condition_name,
                )
                print(f"Saved superlet results to {superlet_path}")
                
                # Plot superlet results
                plot_path = os.path.join(cond_dir, "superlet_plot.png")
                plot_superlet_results(S_mean, t_win, foi_used, condition_name, plot_path)
                
            except Exception as e:
                print(f"Error in superlet analysis: {e}")
        else:
            print("\n1. Skipping superlet analysis (package not installed)")
        
        # 2. POWER SPECTRA ANALYSIS
        print("\n2. Computing power spectra...")
        try:
            psd_results = compute_power_spectra(
                bipolar_lfps, time_s, stim_onset_s,
                pre_window=pre_window,
                post_window=post_window,
            )
            
            # Save power spectra results
            psd_path = os.path.join(cond_dir, "power_spectra_results.npz")
            np.savez_compressed(psd_path, **psd_results, condition_name=condition_name)
            print(f"Saved power spectra to {psd_path}")
            
            # Plot power spectra
            plot_path = os.path.join(cond_dir, "power_spectra_plot.png")
            plot_power_spectra(psd_results, condition_name, plot_path)
            
        except Exception as e:
            print(f"Error in power spectra analysis: {e}")
        
        # 3. Save summary info
        summary = {
            'condition_name': condition_name,
            'n_trials': len(trials),
            'stim_rate_multiplier': trials[0]['stim_rate_multiplier'],
            'stim_n_multiplier': trials[0]['stim_n_multiplier'],
            'stim_params': trials[0]['stim_params'],
        }
        summary_path = os.path.join(cond_dir, "summary.npz")
        np.savez_compressed(summary_path, **summary)
        
        print(f"\nCompleted analysis for {condition_name}")
    
    print(f"\n{'='*60}")
    print("All conditions analyzed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("STEP 1: Processing raw trials to compute LFP/bipolar")
    print("="*60)
    
    process_raw_trials_batch(
        raw_data_dir="results/lfp_trials_conditions",
        output_dir="results/lfp_trials_processed",
        config=CONFIG,
        fs=10000,
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("STEP 2: Analyzing all conditions")
    print("="*60)
    
    analyze_all_conditions(
        processed_dir="results/lfp_trials_processed",
        output_dir="results/analysis_by_condition",
        foi=np.arange(1, 120, 1),
        baseline_window=(0.85, 0.99),
        analysis_window=(0.95, 1.3),
        pre_window=(-0.2, -0.01),
        post_window=(0.01, 0.2),
        verbose=True,
    )
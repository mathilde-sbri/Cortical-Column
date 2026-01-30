"""
Alternative LFP calculation methods for debugging oscillation detection.

The kernel method may be filtering out oscillations due to:
1. Gaussian kernel bandwidth limitations
2. Aggressive depth scaling
3. Parameters tuned for human cortex

This file provides simpler alternatives to help diagnose the issue.
"""

import numpy as np
from scipy import signal


def calculate_lfp_simple_rate(spike_monitors, neuron_groups, layer_configs,
                               electrode_positions, fs=10000, sim_duration_ms=1000,
                               tau_ms=3.0):
    """
    Simplest LFP proxy: exponentially-filtered spike rate.
    
    This is NOT physically accurate but preserves oscillations well.
    Good for debugging whether oscillations exist in the spike data.
    
    Parameters
    ----------
    tau_ms : float
        Time constant for exponential filter (ms). Smaller = more high-freq content.
        Default 3ms allows frequencies up to ~50 Hz.
    """
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
    lfp_signals = {i: np.zeros(n_samples) for i in range(len(electrode_positions))}
    
    # Create exponential kernel
    kernel_length = int(10 * tau_ms / dt_ms)  # 10 time constants
    kernel_t = np.arange(kernel_length) * dt_ms
    kernel = np.exp(-kernel_t / tau_ms)
    kernel = kernel / kernel.sum()  # Normalize
    
    for layer_name, layer_spike_mons in spike_monitors.items():
        layer_config = layer_configs[layer_name]
        z_range = layer_config['coordinates']['z']
        layer_center_z = np.mean(z_range)
        
        for pop_name in ['E', 'PV', 'SOM', 'VIP']:
            spike_key = f'{pop_name}_spikes'
            if spike_key not in layer_spike_mons:
                continue
            
            spike_mon = layer_spike_mons[spike_key]
            neuron_grp = neuron_groups[layer_name][pop_name]
            
            # Sign: E positive, I negative (simplified)
            sign = 1.0 if pop_name == 'E' else -1.0
            
            spike_times_ms = np.array(spike_mon.t / ms) if hasattr(spike_mon.t, 'dimensions') else np.array(spike_mon.t) * 1000
            
            # Bin spikes
            spike_hist, _ = np.histogram(spike_times_ms, bins=n_samples, range=(0, sim_duration_ms))
            
            # Convolve with exponential kernel
            filtered = np.convolve(spike_hist.astype(float), kernel, mode='same')
            
            # Add to electrodes based on distance
            for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
                # Simple distance weighting based on layer center
                dist = abs(ez - layer_center_z)
                weight = np.exp(-dist / 0.3)  # 300 um space constant
                
                lfp_signals[elec_idx] += sign * weight * filtered
    
    return lfp_signals, time_array


def calculate_lfp_synaptic_current(spike_monitors, neuron_groups, layer_configs,
                                    electrode_positions, fs=10000, sim_duration_ms=1000,
                                    tau_ampa=5.0, tau_gaba=10.0):
    """
    LFP as sum of synaptic currents (more biophysical).
    
    Models LFP as proportional to total synaptic input current,
    which is the main contributor to extracellular fields.
    
    Uses double-exponential kernels that better preserve oscillations.
    """
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
    lfp_signals = {i: np.zeros(n_samples) for i in range(len(electrode_positions))}
    
    def alpha_kernel(t, tau):
        """Alpha function kernel - rises then decays, preserves oscillations better"""
        k = (t / tau) * np.exp(1 - t / tau)
        k[t < 0] = 0
        return k / k.max()
    
    # Create kernels
    kernel_length = int(50 / dt_ms)  # 50 ms
    kernel_t = np.arange(kernel_length) * dt_ms
    
    kernel_e = alpha_kernel(kernel_t, tau_ampa)
    kernel_i = alpha_kernel(kernel_t, tau_gaba)
    
    for layer_name, layer_spike_mons in spike_monitors.items():
        layer_config = layer_configs[layer_name]
        z_range = layer_config['coordinates']['z']
        layer_center_z = np.mean(z_range)
        layer_thickness = z_range[1] - z_range[0]
        
        for pop_name in ['E', 'PV', 'SOM', 'VIP']:
            spike_key = f'{pop_name}_spikes'
            if spike_key not in layer_spike_mons:
                continue
            
            spike_mon = layer_spike_mons[spike_key]
            
            is_excitatory = (pop_name == 'E')
            
            # Get spike times - handle Brian2 units
            try:
                spike_times_ms = np.array(spike_mon.t / ms)
            except:
                spike_times_ms = np.array(spike_mon.t) * 1000
            
            if len(spike_times_ms) == 0:
                continue
            
            # Bin spikes
            spike_hist, _ = np.histogram(spike_times_ms, bins=n_samples, range=(0, sim_duration_ms))
            
            # Convolve with appropriate kernel
            if is_excitatory:
                # E spikes cause EPSC in targets -> sink at dendrites
                filtered = np.convolve(spike_hist.astype(float), kernel_e, mode='same')
                sign = 1.0  # Simplified: E activity = positive LFP deflection
            else:
                # I spikes cause IPSC -> source at soma
                filtered = np.convolve(spike_hist.astype(float), kernel_i, mode='same')
                sign = -0.5  # Inhibitory contribution (often weaker in LFP)
            
            # Add to electrodes with spatial weighting
            for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
                # Distance from layer center
                rel_depth = ez - layer_center_z
                
                # Within layer: strongest signal
                # Above/below: decaying
                if abs(rel_depth) < layer_thickness / 2:
                    weight = 1.0
                else:
                    dist = abs(rel_depth) - layer_thickness / 2
                    weight = np.exp(-dist / 0.2)  # 200 um decay
                
                lfp_signals[elec_idx] += sign * weight * filtered
    
    return lfp_signals, time_array


def calculate_population_rate_spectrum(spike_monitors, layer_name, pop_name,
                                        fs=10000, sim_duration_ms=1000,
                                        time_range=None):
    """
    Direct spectrum of population spike rate.
    
    Bypasses LFP calculation entirely - if oscillations exist in spikes,
    they WILL show up here. Good for debugging.
    
    Parameters
    ----------
    time_range : tuple, optional
        (start_ms, end_ms) to analyze. If None, uses full duration.
    
    Returns
    -------
    freqs : array
        Frequency axis
    psd : array
        Power spectral density
    rate : array
        The population rate time series
    time_array : array
        Time axis for rate
    """
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
    spike_key = f'{pop_name}_spikes'
    spike_mon = spike_monitors[layer_name][spike_key]
    
    try:
        spike_times_ms = np.array(spike_mon.t / ms)
    except:
        spike_times_ms = np.array(spike_mon.t) * 1000
    
    # Bin spikes
    spike_hist, _ = np.histogram(spike_times_ms, bins=n_samples, range=(0, sim_duration_ms))
    
    # Smooth with small Gaussian to get rate
    from scipy.ndimage import gaussian_filter1d
    sigma_samples = int(2.0 / dt_ms)  # 2 ms smoothing
    rate = gaussian_filter1d(spike_hist.astype(float), sigma_samples) * (fs / 1000)
    
    # Select time range
    if time_range is not None:
        start_idx = int(time_range[0] * fs / 1000)
        end_idx = int(time_range[1] * fs / 1000)
        rate_segment = rate[start_idx:end_idx]
        time_segment = time_array[start_idx:end_idx]
    else:
        rate_segment = rate
        time_segment = time_array
    
    # Compute spectrum
    freqs, psd = signal.welch(rate_segment, fs=fs, nperseg=min(len(rate_segment)//2, fs))
    
    return freqs, psd, rate, time_array


def compare_lfp_methods(spike_monitors, neuron_groups, layer_configs,
                        electrode_positions, fs=10000, sim_duration_ms=1000,
                        electrode_idx=7, time_range=None):
    """
    Compare different LFP methods side by side.
    
    Returns spectra from each method for comparison.
    """
    results = {}
    
    # Method 1: Simple rate-based
    lfp1, time1 = calculate_lfp_simple_rate(
        spike_monitors, neuron_groups, layer_configs,
        electrode_positions, fs, sim_duration_ms, tau_ms=2.0
    )
    
    # Method 2: Synaptic current model
    lfp2, time2 = calculate_lfp_synaptic_current(
        spike_monitors, neuron_groups, layer_configs,
        electrode_positions, fs, sim_duration_ms
    )
    
    # Select time range
    if time_range is not None:
        start_idx = int(time_range[0] * fs / 1000)
        end_idx = int(time_range[1] * fs / 1000)
    else:
        start_idx = 0
        end_idx = len(time1)
    
    # Compute spectra
    nperseg = min((end_idx - start_idx) // 2, fs)
    
    freqs1, psd1 = signal.welch(lfp1[electrode_idx][start_idx:end_idx], fs=fs, nperseg=nperseg)
    freqs2, psd2 = signal.welch(lfp2[electrode_idx][start_idx:end_idx], fs=fs, nperseg=nperseg)
    
    results['simple_rate'] = {'freqs': freqs1, 'psd': psd1, 'lfp': lfp1[electrode_idx]}
    results['synaptic_current'] = {'freqs': freqs2, 'psd': psd2, 'lfp': lfp2[electrode_idx]}
    
    return results, time1


def plot_method_comparison(results, time_array, electrode_idx=7, 
                           time_range=None, figsize=(14, 10)):
    """
    Plot comparison of LFP methods.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    methods = list(results.keys())
    colors = ['blue', 'red', 'green']
    
    # Time range for plotting
    if time_range is not None:
        t_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    else:
        t_mask = np.ones(len(time_array), dtype=bool)
    
    # Plot time series
    for i, method in enumerate(methods):
        ax = axes[i, 0]
        lfp = results[method]['lfp']
        ax.plot(time_array[t_mask], lfp[t_mask], color=colors[i], linewidth=0.5)
        ax.set_title(f'{method} - Time Series')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('LFP (a.u.)')
    
    # Plot spectra
    for i, method in enumerate(methods):
        ax = axes[i, 1]
        freqs = results[method]['freqs']
        psd = results[method]['psd']
        ax.semilogy(freqs, psd, color=colors[i], linewidth=1.5)
        ax.set_xlim(0, 100)
        ax.set_title(f'{method} - Power Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        
        # Mark peaks
        peak_idx = np.argmax(psd[(freqs > 5) & (freqs < 100)])
        peak_freq = freqs[(freqs > 5) & (freqs < 100)][peak_idx]
        ax.axvline(peak_freq, color='gray', linestyle='--', alpha=0.5)
        ax.text(peak_freq + 2, psd.max() * 0.5, f'{peak_freq:.1f} Hz')
    
    plt.tight_layout()
    return fig


# Quick diagnostic function
def diagnose_oscillations(spike_monitors, layer_name='L23', pop_name='E',
                          fs=10000, sim_duration_ms=1000, baseline_end_ms=None):
    """
    Quick check for oscillations directly in spike data.
    
    Prints summary and returns spectrum.
    """
    freqs, psd, rate, time_array = calculate_population_rate_spectrum(
        spike_monitors, layer_name, pop_name, fs, sim_duration_ms
    )
    
    # Find peaks
    from scipy.signal import find_peaks
    
    # Only look at 4-100 Hz range
    freq_mask = (freqs >= 4) & (freqs <= 100)
    psd_masked = psd[freq_mask]
    freqs_masked = freqs[freq_mask]
    
    # Find peaks in spectrum
    peaks, properties = find_peaks(psd_masked, height=np.median(psd_masked) * 2, prominence=np.std(psd_masked))
    
    print(f"\n=== Oscillation Diagnosis: {layer_name} {pop_name} ===")
    print(f"Mean firing rate: {rate.mean():.2f} Hz")
    print(f"Rate std: {rate.std():.2f} Hz")
    
    if len(peaks) > 0:
        print(f"\nSpectral peaks found:")
        for peak_idx in peaks:
            print(f"  {freqs_masked[peak_idx]:.1f} Hz (power: {psd_masked[peak_idx]:.2e})")
    else:
        print("\nNo clear spectral peaks found (flat/broadband spectrum)")
    
    # Check for rhythmicity
    peak_to_mean = psd_masked.max() / psd_masked.mean()
    print(f"\nPeak-to-mean ratio: {peak_to_mean:.2f}")
    if peak_to_mean > 5:
        print("  -> Strong rhythmicity")
    elif peak_to_mean > 2:
        print("  -> Moderate rhythmicity")
    else:
        print("  -> Weak/no rhythmicity (broadband)")
    
    return freqs, psd, rate, time_array


# Unit handling helper
try:
    from brian2 import ms
except ImportError:
    ms = 0.001  # Fallback if brian2 not available

"""
LFP Calculation using the Mazzoni et al. 2015 Method
=====================================================

This implements the "Reference Weighted Sum" (RWS) proxy for LFP from point-neuron 
networks, which was validated against biophysically realistic multicompartmental 
simulations and shown to explain >90% of LFP variance.

Reference:
    Mazzoni A, Lindén H, Cuntz H, Lansner A, Panzeri S, Einevoll GT (2015) 
    Computing the Local Field Potential (LFP) from Integrate-and-Fire Network Models. 
    PLoS Comput Biol 11(12): e1004584. https://doi.org/10.1371/journal.pcbi.1004584

Key findings from the paper:
1. LFP is dominated by pyramidal (excitatory) neuron contributions
2. Inhibitory interneurons contribute negligibly due to their symmetric morphology
3. The best proxy is a weighted sum of AMPA and GABA currents with specific delays
4. Formula: LFP ∝ 1.65 * AMPA(t - 6ms) - GABA(t)

The method works by:
1. Computing summed synaptic currents from spikes (convolved with synaptic kernels)
2. Applying the RWS formula with optimal weights and delays
3. Adding spatial weighting based on electrode position relative to neural populations
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def calculate_lfp_mazzoni(spike_monitors, neuron_groups, layer_configs,
                          electrode_positions, fs=10000, sim_duration_ms=1000,
                          tau_ampa_rise=0.4, tau_ampa_decay=2.0,
                          tau_gaba_rise=0.25, tau_gaba_decay=5.0,
                          alpha=1.65, delay_ampa_ms=6.0, delay_gaba_ms=0.0,
                          spatial_decay_mm=0.3):
    """
    Calculate LFP using the Mazzoni et al. 2015 Reference Weighted Sum (RWS) method
    with proper laminar electrode support.
    
    This is the gold-standard method for computing LFP from point-neuron networks,
    validated against biophysically realistic multicompartmental simulations.
    
    Parameters
    ----------
    spike_monitors : dict
        Dictionary of spike monitors organized by layer.
        Structure: {layer_name: {'E_spikes': monitor, 'PV_spikes': monitor, ...}}
    neuron_groups : dict
        Dictionary of neuron groups organized by layer.
        Structure: {layer_name: {'E': group, 'PV': group, ...}}
    layer_configs : dict
        Configuration for each layer including coordinates.
    electrode_positions : list
        List of (x, y, z) electrode positions in mm.
    fs : float
        Sampling frequency in Hz. Default 10000.
    sim_duration_ms : float
        Total simulation duration in ms.
    tau_ampa_rise : float
        AMPA rise time constant in ms. Default 0.4.
    tau_ampa_decay : float
        AMPA decay time constant in ms. Default 2.0.
    tau_gaba_rise : float
        GABA rise time constant in ms. Default 0.25.
    tau_gaba_decay : float
        GABA decay time constant in ms. Default 5.0.
    alpha : float
        Relative weight of AMPA vs GABA. Default 1.65 (from Mazzoni et al.).
    delay_ampa_ms : float
        Delay for AMPA contribution in ms. Default 6.0 (from Mazzoni et al.).
    delay_gaba_ms : float
        Delay for GABA contribution in ms. Default 0.0.
    spatial_decay_mm : float
        Spatial decay constant in mm. Default 0.3.
    
    Returns
    -------
    lfp_signals : dict
        Dictionary mapping electrode index to LFP time series.
    time_array : array
        Time points in ms.
    
    Notes
    -----
    The RWS proxy formula from Mazzoni et al. 2015:
        LFP(t) ∝ α * I_AMPA(t - τ_AMPA) - I_GABA(t - τ_GABA)
    
    where α = 1.65, τ_AMPA = 6 ms, τ_GABA = 0 ms
    
    Spatial model: Each layer contributes a dipole-like field that:
    - Is strongest within the layer
    - Decays exponentially outside the layer
    - Inverts sign across the layer (sink/source pattern)
    """
    
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
    # Create synaptic kernels (difference of exponentials)
    kernel_length_ms = 50
    kernel_samples = int(kernel_length_ms / dt_ms)
    kernel_t = np.arange(kernel_samples) * dt_ms
    
    def diff_exp_kernel(t, tau_rise, tau_decay):
        """Double exponential (difference of exponentials) synaptic kernel."""
        if tau_rise == tau_decay:
            kernel = (t / tau_decay) * np.exp(1 - t / tau_decay)
        else:
            kernel = (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))
            t_peak = (tau_decay * tau_rise / (tau_decay - tau_rise)) * np.log(tau_decay / tau_rise)
            peak_val = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
            kernel = kernel / peak_val if peak_val > 0 else kernel
        kernel[t < 0] = 0
        return kernel
    
    kernel_ampa = diff_exp_kernel(kernel_t, tau_ampa_rise, tau_ampa_decay)
    kernel_gaba = diff_exp_kernel(kernel_t, tau_gaba_rise, tau_gaba_decay)
    
    delay_ampa_samples = int(delay_ampa_ms / dt_ms)
    delay_gaba_samples = int(delay_gaba_ms / dt_ms)
    
    # Compute RWS current for each layer (this is the key temporal signal)
    layer_rws_signals = {}
    
    for layer_name, layer_spike_mons in spike_monitors.items():
        ampa_current = np.zeros(n_samples)
        gaba_current = np.zeros(n_samples)
        
        for pop_name in ['E', 'PV', 'SOM', 'VIP']:
            spike_key = f'{pop_name}_spikes'
            if spike_key not in layer_spike_mons:
                continue
            
            spike_mon = layer_spike_mons[spike_key]
            
            try:
                spike_times_ms = np.array(spike_mon.t / ms)
            except:
                try:
                    spike_times_ms = np.array(spike_mon.t) * 1000
                except:
                    spike_times_ms = np.array(spike_mon.t)
            
            if len(spike_times_ms) == 0:
                continue
            
            spike_hist, _ = np.histogram(spike_times_ms, bins=n_samples, 
                                         range=(0, sim_duration_ms))
            
            if pop_name == 'E':
                filtered = np.convolve(spike_hist.astype(float), kernel_ampa, mode='same')
                ampa_current += filtered
            else:
                filtered = np.convolve(spike_hist.astype(float), kernel_gaba, mode='same')
                gaba_current += filtered
        
        # Apply delays
        if delay_ampa_samples > 0:
            ampa_delayed = np.zeros_like(ampa_current)
            ampa_delayed[delay_ampa_samples:] = ampa_current[:-delay_ampa_samples]
        else:
            ampa_delayed = ampa_current
        
        if delay_gaba_samples > 0:
            gaba_delayed = np.zeros_like(gaba_current)
            gaba_delayed[delay_gaba_samples:] = gaba_current[:-delay_gaba_samples]
        else:
            gaba_delayed = gaba_current
        
        # RWS formula for this layer
        layer_rws_signals[layer_name] = alpha * ampa_delayed - gaba_delayed
    
    # Now compute LFP at each electrode with proper spatial weighting
    lfp_signals = {i: np.zeros(n_samples) for i in range(len(electrode_positions))}
    
    for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
        lfp = np.zeros(n_samples)
        
        for layer_name, layer_rws in layer_rws_signals.items():
            layer_config = layer_configs[layer_name]
            z_range = layer_config['coordinates']['z']
            z_min, z_max = z_range
            layer_center_z = (z_min + z_max) / 2
            layer_thickness = z_max - z_min
            
            # Compute spatial weight using dipole model
            # The LFP from a layer forms a dipole pattern:
            # - Positive (source) at soma level (lower part)
            # - Negative (sink) at dendrite level (upper part for pyramidal)
            
            rel_depth = ez - layer_center_z
            
            # Distance-based amplitude decay
            if z_min <= ez <= z_max:
                # Electrode is within this layer
                dist_to_boundary = min(abs(ez - z_min), abs(ez - z_max))
                amplitude = 1.0
            else:
                # Electrode is outside this layer
                if ez > z_max:
                    dist_to_boundary = ez - z_max
                else:
                    dist_to_boundary = z_min - ez
                amplitude = np.exp(-dist_to_boundary / spatial_decay_mm)
            
            # Dipole sign: positive above layer center, negative below
            # This creates the characteristic LFP inversion pattern
            dipole_sign = np.sign(rel_depth) if rel_depth != 0 else 1
            
            # Weight combines amplitude and sign
            weight = dipole_sign * amplitude
            
            lfp += weight * layer_rws
        
        lfp_signals[elec_idx] = lfp
    
    return lfp_signals, time_array


def calculate_lfp_mazzoni_simple(spike_monitors, neuron_groups, layer_configs,
                                  electrode_positions, fs=10000, sim_duration_ms=1000):
    """
    Simplified version that computes LFP without layer-specific spatial weighting.
    
    Good for getting a single "global" LFP signal that represents overall network activity.
    Uses the same RWS formula but sums all contributions equally.
    
    Returns
    -------
    lfp_global : array
        Single LFP time series (not electrode-specific)
    time_array : array
        Time points in ms
    """
    
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
    # Synaptic time constants (from Mazzoni et al.)
    tau_ampa_rise, tau_ampa_decay = 0.4, 2.0
    tau_gaba_rise, tau_gaba_decay = 0.25, 5.0
    
    # RWS parameters
    alpha = 1.65
    delay_ampa_ms = 6.0
    
    # Create kernels
    kernel_length_ms = 50
    kernel_samples = int(kernel_length_ms / dt_ms)
    kernel_t = np.arange(kernel_samples) * dt_ms
    
    def diff_exp_kernel(t, tau_rise, tau_decay):
        if tau_rise == tau_decay:
            kernel = (t / tau_decay) * np.exp(1 - t / tau_decay)
        else:
            kernel = (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))
            t_peak = (tau_decay * tau_rise / (tau_decay - tau_rise)) * np.log(tau_decay / tau_rise)
            peak_val = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
            kernel = kernel / peak_val if peak_val > 0 else kernel
        kernel[t < 0] = 0
        return kernel
    
    kernel_ampa = diff_exp_kernel(kernel_t, tau_ampa_rise, tau_ampa_decay)
    kernel_gaba = diff_exp_kernel(kernel_t, tau_gaba_rise, tau_gaba_decay)
    
    delay_samples = int(delay_ampa_ms / dt_ms)
    
    # Sum all currents across layers
    total_ampa = np.zeros(n_samples)
    total_gaba = np.zeros(n_samples)
    
    for layer_name, layer_spike_mons in spike_monitors.items():
        for pop_name in ['E', 'PV', 'SOM', 'VIP']:
            spike_key = f'{pop_name}_spikes'
            if spike_key not in layer_spike_mons:
                continue
            
            spike_mon = layer_spike_mons[spike_key]
            
            try:
                spike_times_ms = np.array(spike_mon.t / ms)
            except:
                try:
                    spike_times_ms = np.array(spike_mon.t) * 1000
                except:
                    spike_times_ms = np.array(spike_mon.t)
            
            if len(spike_times_ms) == 0:
                continue
            
            spike_hist, _ = np.histogram(spike_times_ms, bins=n_samples,
                                        range=(0, sim_duration_ms))
            
            if pop_name == 'E':
                filtered = np.convolve(spike_hist.astype(float), kernel_ampa, mode='same')
                total_ampa += filtered
            else:
                filtered = np.convolve(spike_hist.astype(float), kernel_gaba, mode='same')
                total_gaba += filtered
    
    # Apply delay to AMPA
    ampa_delayed = np.zeros_like(total_ampa)
    ampa_delayed[delay_samples:] = total_ampa[:-delay_samples]
    
    # RWS formula
    lfp_global = alpha * ampa_delayed - total_gaba
    
    # Also return per-electrode (all same for this simple version, but with sign flipping)
    lfp_signals = {}
    for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
        # Simple sign based on depth (assuming 0 is middle)
        sign = 1 if ez > 0 else -1
        lfp_signals[elec_idx] = sign * lfp_global
    
    return lfp_signals, time_array, lfp_global


def calculate_lfp_sum_abs_currents(spike_monitors, neuron_groups, layer_configs,
                                    electrode_positions, fs=10000, sim_duration_ms=1000,
                                    tau_ampa=2.0, tau_gaba=5.0):
    """
    Alternative LFP proxy: Sum of absolute values of synaptic currents (|AMPA| + |GABA|).
    
    This was the second-best proxy in Mazzoni et al. (R² ~ 0.83), simpler than RWS.
    Equivalent to: |AMPA| + |GABA| = AMPA - GABA (since GABA is negative by convention)
    
    Good for quick estimates when you don't need maximum accuracy.
    """
    
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
    # Simple exponential kernels
    kernel_length = int(50 / dt_ms)
    kernel_t = np.arange(kernel_length) * dt_ms
    
    kernel_ampa = np.exp(-kernel_t / tau_ampa)
    kernel_ampa = kernel_ampa / kernel_ampa.sum()
    
    kernel_gaba = np.exp(-kernel_t / tau_gaba)
    kernel_gaba = kernel_gaba / kernel_gaba.sum()
    
    total_ampa = np.zeros(n_samples)
    total_gaba = np.zeros(n_samples)
    
    for layer_name, layer_spike_mons in spike_monitors.items():
        for pop_name in ['E', 'PV', 'SOM', 'VIP']:
            spike_key = f'{pop_name}_spikes'
            if spike_key not in layer_spike_mons:
                continue
            
            spike_mon = layer_spike_mons[spike_key]
            
            try:
                spike_times_ms = np.array(spike_mon.t / ms)
            except:
                try:
                    spike_times_ms = np.array(spike_mon.t) * 1000
                except:
                    spike_times_ms = np.array(spike_mon.t)
            
            if len(spike_times_ms) == 0:
                continue
            
            spike_hist, _ = np.histogram(spike_times_ms, bins=n_samples,
                                        range=(0, sim_duration_ms))
            
            if pop_name == 'E':
                filtered = np.convolve(spike_hist.astype(float), kernel_ampa, mode='same')
                total_ampa += filtered
            else:
                filtered = np.convolve(spike_hist.astype(float), kernel_gaba, mode='same')
                total_gaba += filtered
    
    # Sum of absolute values = AMPA + |GABA| = AMPA - GABA (with our sign convention)
    lfp_global = total_ampa - total_gaba
    
    lfp_signals = {}
    for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
        sign = 1 if ez > 0 else -1
        lfp_signals[elec_idx] = sign * lfp_global
    
    return lfp_signals, time_array


# Helper to try importing brian2 ms unit
try:
    from brian2 import ms
except ImportError:
    ms = 0.001


def compute_power_spectrum_from_lfp(lfp_signal, fs=10000, nperseg=None):
    """
    Compute power spectrum from LFP signal using Welch's method.
    
    Parameters
    ----------
    lfp_signal : array
        LFP time series
    fs : float
        Sampling frequency in Hz
    nperseg : int, optional
        Length of each segment for Welch's method
    
    Returns
    -------
    freqs : array
        Frequency axis
    psd : array
        Power spectral density
    """
    if nperseg is None:
        nperseg = min(len(lfp_signal) // 4, fs)
    
    freqs, psd = signal.welch(lfp_signal, fs=fs, nperseg=nperseg)
    return freqs, psd


def compare_lfp_to_original(lfp_new, lfp_original, fs=10000):
    """
    Compare new LFP calculation to original, computing correlation and spectral similarity.
    """
    # Normalize both
    lfp_new_norm = (lfp_new - np.mean(lfp_new)) / (np.std(lfp_new) + 1e-10)
    lfp_orig_norm = (lfp_original - np.mean(lfp_original)) / (np.std(lfp_original) + 1e-10)
    
    # Correlation
    corr = np.corrcoef(lfp_new_norm, lfp_orig_norm)[0, 1]
    
    # Spectral comparison
    freqs_new, psd_new = compute_power_spectrum_from_lfp(lfp_new, fs)
    freqs_orig, psd_orig = compute_power_spectrum_from_lfp(lfp_original, fs)
    
    # Spectral correlation (in log space)
    log_psd_new = np.log10(psd_new + 1e-20)
    log_psd_orig = np.log10(psd_orig + 1e-20)
    spectral_corr = np.corrcoef(log_psd_new, log_psd_orig)[0, 1]
    
    return {
        'temporal_correlation': corr,
        'spectral_correlation': spectral_corr,
        'variance_ratio': np.var(lfp_new) / (np.var(lfp_original) + 1e-10)
    }
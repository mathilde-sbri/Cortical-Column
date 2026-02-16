
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def calculate_lfp_mazzoni(spike_monitors, neuron_groups, layer_configs,
                          electrode_positions, fs=10000, sim_duration_ms=1000,
                          tau_ampa_rise=0.4, tau_ampa_decay=2.0,
                          tau_gaba_rise=0.25, tau_gaba_decay=5.0,
                          alpha=1.65, delay_ampa_ms=6.0, delay_gaba_ms=0.0,
                          spatial_decay_mm=0.3):
  
    
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
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
    
    delay_ampa_samples = int(delay_ampa_ms / dt_ms)
    delay_gaba_samples = int(delay_gaba_ms / dt_ms)
    
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
        
        layer_rws_signals[layer_name] = alpha * ampa_delayed - gaba_delayed
    
    lfp_signals = {i: np.zeros(n_samples) for i in range(len(electrode_positions))}
    
    for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
        lfp = np.zeros(n_samples)
        
        for layer_name, layer_rws in layer_rws_signals.items():
            layer_config = layer_configs[layer_name]
            z_range = layer_config['coordinates']['z']
            z_min, z_max = z_range
            layer_center_z = (z_min + z_max) / 2
            layer_thickness = z_max - z_min
            
        
            rel_depth = ez - layer_center_z
            
            if z_min <= ez <= z_max:
                dist_to_boundary = min(abs(ez - z_min), abs(ez - z_max))
                amplitude = 1.0
            else:
                if ez > z_max:
                    dist_to_boundary = ez - z_max
                else:
                    dist_to_boundary = z_min - ez
                amplitude = np.exp(-dist_to_boundary / spatial_decay_mm)
            

            dipole_sign = np.sign(rel_depth) if rel_depth != 0 else 1
            
            weight = dipole_sign * amplitude
            
            lfp += weight * layer_rws
        
        lfp_signals[elec_idx] = lfp
    
    return lfp_signals, time_array


def calculate_lfp_mazzoni_simple(spike_monitors, neuron_groups, layer_configs,
                                  electrode_positions, fs=10000, sim_duration_ms=1000):

    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
    tau_ampa_rise, tau_ampa_decay = 0.4, 2.0
    tau_gaba_rise, tau_gaba_decay = 0.25, 5.0
    
    alpha = 1.65
    delay_ampa_ms = 6.0
    
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
    
    ampa_delayed = np.zeros_like(total_ampa)
    ampa_delayed[delay_samples:] = total_ampa[:-delay_samples]
    
    lfp_global = alpha * ampa_delayed - total_gaba
    
    lfp_signals = {}
    for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
        sign = 1 if ez > 0 else -1
        lfp_signals[elec_idx] = sign * lfp_global
    
    return lfp_signals, time_array, lfp_global


def calculate_lfp_sum_abs_currents(spike_monitors, neuron_groups, layer_configs,
                                    electrode_positions, fs=10000, sim_duration_ms=1000,
                                    tau_ampa=2.0, tau_gaba=5.0):
  
    
    n_samples = int(sim_duration_ms * fs / 1000)
    time_array = np.linspace(0, sim_duration_ms, n_samples)
    dt_ms = time_array[1] - time_array[0]
    
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
    
    lfp_global = total_ampa - total_gaba
    
    lfp_signals = {}
    for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
        sign = 1 if ez > 0 else -1
        lfp_signals[elec_idx] = sign * lfp_global
    
    return lfp_signals, time_array


try:
    from brian2 import ms
except ImportError:
    ms = 0.001


def compute_power_spectrum_from_lfp(lfp_signal, fs=10000, nperseg=None):
 
    if nperseg is None:
        nperseg = min(len(lfp_signal) // 4, fs)
    
    freqs, psd = signal.welch(lfp_signal, fs=fs, nperseg=nperseg)
    return freqs, psd


def compare_lfp_to_original(lfp_new, lfp_original, fs=10000):
    
    lfp_new_norm = (lfp_new - np.mean(lfp_new)) / (np.std(lfp_new) + 1e-10)
    lfp_orig_norm = (lfp_original - np.mean(lfp_original)) / (np.std(lfp_original) + 1e-10)
    
    corr = np.corrcoef(lfp_new_norm, lfp_orig_norm)[0, 1]
    

    freqs_new, psd_new = compute_power_spectrum_from_lfp(lfp_new, fs)
    freqs_orig, psd_orig = compute_power_spectrum_from_lfp(lfp_original, fs)

    log_psd_new = np.log10(psd_new + 1e-20)
    log_psd_orig = np.log10(psd_orig + 1e-20)
    spectral_corr = np.corrcoef(log_psd_new, log_psd_orig)[0, 1]
    
    return {
        'temporal_correlation': corr,
        'spectral_correlation': spectral_corr,
        'variance_ratio': np.var(lfp_new) / (np.var(lfp_original) + 1e-10)
    }
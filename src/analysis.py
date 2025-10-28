"""
Analysis functions for neural data. to do  : check lfp method + spike analysis
"""
import numpy as np
from scipy.signal import welch, spectrogram
from brian2 import *
from scipy.ndimage import gaussian_filter1d
from spectrum import pmtm

def calculate_lfp(monitor, neuron_type='E'):
    """Calculate LFP using current inputs into E neurons, inspired from the paper of Mazzoni"""
    ge = np.array(monitor.gE/nS)  
    gi = np.array(monitor.gI/nS)  
    V = np.array(monitor.v/mV)
    
    I_AMPA = np.abs(ge * (0 - V))  # Ee = 0mV
    I_GABA = np.abs(gi * (-80 - V))  # Ei = -80mV
    
    total_current = np.sum(I_AMPA + I_GABA, axis=0)
    
    return total_current


def process_lfp(monitor, start_time_ms=0):
    lfp = calculate_lfp(monitor)
    lfp_time = np.array(monitor.t/ms)
    
    start_idx = np.argmax(lfp_time >= start_time_ms)
    lfp_stable = lfp[start_idx:]
    time_stable = lfp_time[start_idx:]
    if np.std(lfp_stable) != 0 :
        lfp_stable = (lfp_stable - np.mean(lfp_stable))/np.std(lfp_stable)
    return time_stable, lfp_stable


def compute_power_spectrum(lfp_signal, fs=10000, method='welch', nperseg=None):
    if method == 'multitaper':
        
        if nperseg is None:
            nperseg = len(lfp_signal)
        
        NW = 2
        psd, freq = pmtm(lfp_signal[:nperseg], NW=NW, NFFT=nperseg)
        
        return freq, psd
    
    elif method == 'welch':
        if nperseg is None:
            nperseg = min(4096, len(lfp_signal) // 4)
        
        freq, psd = welch(lfp_signal, fs=fs, nperseg=nperseg, 
                        noverlap=nperseg//2, window='hann')
        return freq, psd
    
    else:
        raise ValueError("method must be 'multitaper' or 'welch'")


def compute_power_spectrum_epochs(lfp_signal, fs=10000, epoch_duration_ms=1000, method='multitaper'):
    epoch_samples = int(epoch_duration_ms * fs / 1000)
    n_epochs = len(lfp_signal) // epoch_samples
    
    psds = []
    for i in range(n_epochs):
        start_idx = i * epoch_samples
        end_idx = start_idx + epoch_samples
        epoch = lfp_signal[start_idx:end_idx]
        
        freq, psd = compute_power_spectrum(epoch, fs=fs, method=method, nperseg=epoch_samples)
        psds.append(psd)
    
    psd_mean = np.mean(psds, axis=0)
    psd_std = np.std(psds, axis=0)
    
    return freq, psd_mean, psd_std


def compute_spectrogram(lfp_signal, fs=10000, window_ms=500, overlap=0.75):
    nperseg = int(window_ms * fs / 1000)
    noverlap = int(nperseg * overlap)
    
    freq, time, Sxx = spectrogram(lfp_signal, fs=fs, nperseg=nperseg,
                                noverlap=noverlap, window='hann')
    
    return freq, time, Sxx


def peak_frequency_track(f_hz, Sxx, f_gamma=(20, 80)):
    fmask = (f_hz >= f_gamma[0]) & (f_hz <= f_gamma[1])
    if not np.any(fmask):
        return np.full(Sxx.shape[1], np.nan), np.full(Sxx.shape[1], np.nan)
    Sg = Sxx[fmask, :]
    idx = np.argmax(Sg, axis=0)
    freqs_in_band = f_hz[fmask]
    peak_freq = freqs_in_band[idx]
    peak_pow  = Sg[idx, np.arange(Sg.shape[1])]
    return peak_freq, peak_pow


def add_heterogeneity_to_layer(layer, config):
    for pop_name, neuron_group in layer.neuron_groups.items():
        n = len(neuron_group)
        base = config['intrinsic_params'][pop_name]
        
        neuron_group.C = base['C'] * np.abs(1 + np.random.randn(n) * 0.15)
        neuron_group.gL = base['gL'] * np.abs(1 + np.random.randn(n) * 0.12)
        neuron_group.tauw = base['tauw'] * np.abs(1 + np.random.randn(n) * 0.15)
        neuron_group.b = base['b'] * np.abs(1 + np.random.randn(n) * 0.20)
        neuron_group.a = base['a'] * np.abs(1 + np.random.randn(n) * 0.15)





def detailed_activity_check(layer_name, state_monitor, spike_monitor, simulation_time, time_window=None):
    print(f"\n{'='*50}")
    print(f"Layer: {layer_name}")
    if time_window:
        print(f"Time window: {time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms")
    print(f"{'='*50}")
    
    for pop in ['E', 'PV', 'SOM', 'VIP']:
        spike_key = f'{pop}_spikes'
        state_key = f'{pop}_state'
        
        if spike_key not in spike_monitor:
            continue
            
        spike_mon = spike_monitor[spike_key]
        state_mon = state_monitor[state_key]
        
        n_neurons = len(spike_mon.source)
        
        if time_window:
            mask = (spike_mon.t >= time_window[0]) & (spike_mon.t < time_window[1])
            spike_times = spike_mon.t[mask]
            spike_indices = spike_mon.i[mask]
            analysis_duration = (time_window[1] - time_window[0]) / second
        else:
            spike_times = spike_mon.t
            spike_indices = spike_mon.i
            analysis_duration = simulation_time / second
        
        n_spikes = len(spike_times)
        
        if n_spikes > 0:
            rate = n_spikes / (n_neurons * analysis_duration)
            active_neurons = len(np.unique(spike_indices))
            pct_active = 100 * active_neurons / n_neurons
        else:
            rate = 0
            active_neurons = 0
            pct_active = 0
        
        gE_mean = np.mean(state_mon.gE/nS)
        gE_max = np.max(state_mon.gE/nS)
        gI_mean = np.mean(state_mon.gI/nS)
        gI_max = np.max(state_mon.gI/nS)
        
        print(f"\n{pop} population (N={n_neurons}):")
        print(f"  Firing rate: {rate:.2f} Hz")
        print(f"  Active neurons: {active_neurons}/{n_neurons} ({pct_active:.1f}%)")
        print(f"  Total spikes: {n_spikes}")
        print(f"  gE: mean={gE_mean:.4f} nS, max={gE_max:.3f} nS")
        print(f"  gI: mean={gI_mean:.4f} nS, max={gI_max:.3f} nS")
        
        if gE_mean < 0.01 and gI_mean < 0.01:
            print(f" WARNING: Very low synaptic activity!")
        if pct_active < 10:
            print(f" WARNING: <10% of neurons are active!")





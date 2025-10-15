"""
Analysis functions for neural data. to do  : check lfp method + spike analysis
"""
import numpy as np
from scipy.signal import welch, spectrogram
from brian2 import *
from scipy.ndimage import gaussian_filter1d

class LFPAnalysis:
    
    @staticmethod
    def calculate_lfp(monitor, neuron_type='E'):
        """Calculate LFP using current inputs, inspired from the paper of Mazzoni"""
        ge = np.array(monitor.gE/nS)  
        gi = np.array(monitor.gI/nS)  
        V = np.array(monitor.v/mV)
        
        I_AMPA = np.abs(ge * (0 - V))  # Ee = 0mV
        I_GABA = np.abs(gi * (-80 - V))  # Ei = -80mV
        
        total_current = np.sum(I_AMPA + I_GABA, axis=0)
        return total_current
    
    @staticmethod
    def process_lfp(monitor, start_time_ms=0):
        lfp = LFPAnalysis.calculate_lfp(monitor)
        lfp_time = np.array(monitor.t/ms)
        
        start_idx = np.argmax(lfp_time >= start_time_ms)
        lfp_stable = lfp[start_idx:]
        time_stable = lfp_time[start_idx:]
        
        lfp_stable = (lfp_stable - np.mean(lfp_stable)) / np.std(lfp_stable)
        
        return time_stable, lfp_stable
    
    @staticmethod
    def compute_power_spectrum(lfp_signal, fs=10000, nperseg=4096):
        freq, psd = welch(lfp_signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

        return freq, psd
    
    @staticmethod
    def power_spectrum_loglog(lfp_signal, time_ms, fmin=1, fmax=500, nperseg=4096, ax=None):
        
        t_s = np.asarray(time_ms) / 1000.0
        fs = 1.0 / np.median(np.diff(t_s))

        freq, psd = welch(lfp_signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

        return freq, psd

    
    @staticmethod
    def compute_spectrogram(time_ms, lfp, fmax=100, win_ms=500, step_ms=10, nfft=None):

        t_s = np.asarray(time_ms)/1000.0
        dt = float(np.median(np.diff(t_s)))
        fs = 1.0/dt

        nperseg = max(16, int(round(win_ms/1000.0 * fs)))
        step = max(1, int(round(step_ms/1000.0 * fs)))
        noverlap = max(0, nperseg - step)
        if nfft is None:
            pow2 = int(2**np.ceil(np.log2(nperseg)))
            nfft = max(pow2, nperseg)

        nperseg = min(nperseg, len(lfp))
        noverlap = min(noverlap, nperseg-1)
        if len(lfp) < nperseg:
            raise ValueError("lfp shorter than window")

        f, t_spec, Sxx = spectrogram(
            lfp, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            window='hann', detrend='constant', scaling='density', mode='psd'
        )
        mask = f <= fmax
        return t_spec, f[mask], Sxx[mask, :]


    @staticmethod
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




def build_psth_from_spikemon(
    spike_mons, t_stim, t_pre=0.200, t_post=0.500, binW=0.002, smooth_sigma=0.010
):

    edges   = np.arange(-t_pre, t_post + 1e-12, binW)
    centers = edges[:-1] + binW/2

    per_mon_meancounts = []

    for mon in spike_mons:
        t_rel = np.array(mon.t/second) - float(t_stim)

        try:
            n_units = int(mon.source.N)         
        except Exception:
            n_units = int(np.max(mon.i) + 1)   

        counts, _ = np.histogram(t_rel, bins=edges)

        meanc = counts.astype(float) / max(n_units, 1)
        per_mon_meancounts.append(meanc)

    if len(per_mon_meancounts) == 0:
        return centers, np.zeros_like(centers)

    mean_counts_per_neuron = np.mean(np.vstack(per_mon_meancounts), axis=0)

    psth_hz = mean_counts_per_neuron / binW

    sigma_bins = smooth_sigma / binW
    psth_hz = gaussian_filter1d(psth_hz, sigma=sigma_bins, mode='nearest')

    return centers, psth_hz

def baseline_subtract(lfp_time_rel, lfp_traces, t_pre=0.200):

    t = np.asarray(lfp_time_rel)
    pre_mask = (t >= -t_pre) & (t < 0)
    X = np.asarray(lfp_traces) 
    if X.ndim == 1:
        X = X[None, :]
    base = np.nanmean(X[:, pre_mask], axis=1, keepdims=True)
    Xb = X - base
    erp = np.nanmean(Xb, axis=0)
    return t, erp

def spectrogram_db_rel_baseline(x, fs, wlen_s=0.200, step_s=0.020):
    from scipy.signal import spectrogram
    nperseg = int(round(wlen_s * fs))
    step = int(round(step_s * fs))
    noverlap = max(0, nperseg - step)
    f, T, P = spectrogram(x - np.mean(x), fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')

    return f, T, P

def to_db_relative(P, Trel):
    pre_cols = np.where(Trel < 0)[0]
    if pre_cols.size == 0:
        base = np.mean(P, axis=1, keepdims=True)
    else:
        base = np.mean(P[:, pre_cols], axis=1, keepdims=True)
    dB = 10.0 * np.log10((P + 1e-20) / (base + 1e-20))
    return dB

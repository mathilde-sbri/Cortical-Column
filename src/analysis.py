"""
Analysis functions for neural data. to do  : check lfp method + spike analysis
"""
import numpy as np
from scipy.signal import welch, spectrogram
from brian2 import *
from scipy.ndimage import gaussian_filter1d
from spectrum import pmtm

class LFPAnalysis:
    
    @staticmethod
    def calculate_lfp(monitor, neuron_type='E'):
        """Calculate LFP using current inputs into E neurons, inspired from the paper of Mazzoni"""
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
        if np.std(lfp_stable) != 0 :
            lfp_stable = (lfp_stable - np.mean(lfp_stable))/np.std(lfp_stable)
        return time_stable, lfp_stable
    
    @staticmethod
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

    @staticmethod
    def compute_power_spectrum_epochs(lfp_signal, fs=10000, epoch_duration_ms=1000, method='multitaper'):
        epoch_samples = int(epoch_duration_ms * fs / 1000)
        n_epochs = len(lfp_signal) // epoch_samples
        
        psds = []
        for i in range(n_epochs):
            start_idx = i * epoch_samples
            end_idx = start_idx + epoch_samples
            epoch = lfp_signal[start_idx:end_idx]
            
            freq, psd = LFPAnalysis.compute_power_spectrum(epoch, fs=fs, method=method, nperseg=epoch_samples)
            psds.append(psd)
        
        psd_mean = np.mean(psds, axis=0)
        psd_std = np.std(psds, axis=0)
        
        return freq, psd_mean, psd_std

    @staticmethod
    def compute_spectrogram(lfp_signal, fs=10000, window_ms=50, overlap=0.9):
        nperseg = int(window_ms * fs / 1000)
        noverlap = int(nperseg * overlap)
        
        freq, time, Sxx = spectrogram(lfp_signal, fs=fs, nperseg=nperseg,
                                    noverlap=noverlap, window='hann')
        
        return freq, time, Sxx


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




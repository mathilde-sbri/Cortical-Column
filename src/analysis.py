"""
Analysis functions for neural data. to do  : check lfp method + spike analysis
"""
import numpy as np
from scipy.signal import welch
from brian2 import *

class LFPAnalysis:
    
    @staticmethod
    def calculate_lfp(monitor, neuron_type='E'):
        """Calculate LFP using current inputs, inspired from the paper of Mazzoni but need to check if it's exactly the same"""
        ge = np.array(monitor.ge/nS)  
        gi = np.array(monitor.gi/nS)  
        V = np.array(monitor.v/mV)
        
        I_AMPA = np.abs(ge * (0 - V))  # Ee = 0mV
        I_GABA = np.abs(gi * (-80 - V))  # Ei = -80mV
        
        total_current = np.sum(I_AMPA + I_GABA, axis=0)
        return total_current
    
    @staticmethod
    def process_lfp(monitor, start_time_ms=3000):
        lfp = LFPAnalysis.calculate_lfp(monitor)
        lfp_time = np.array(monitor.t/ms)
        
        start_idx = np.argmax(lfp_time >= start_time_ms)
        lfp_stable = lfp[start_idx:]
        time_stable = lfp_time[start_idx:]
        
        lfp_stable = (lfp_stable - np.mean(lfp_stable)) / np.std(lfp_stable)
        
        return time_stable, lfp_stable
    
    @staticmethod
    def compute_power_spectrum(lfp_signal, fs=10000, nperseg=4096):
        freq, psd = welch(lfp_signal, fs=fs, nperseg=nperseg)
        return freq, psd

class SpikeAnalysis:

    @staticmethod
    def bin_population_rate(spike_monitor, n_neurons, bin_ms=1.0, t_stop_s=None):
        t_s = np.array(spike_monitor.t)  
        if t_stop_s is None:
            t_stop_s = float(t_s.max()) if t_s.size else 0.0
        bins = np.arange(0.0, t_stop_s + 1e-9, bin_ms/1000.0)
        counts, _ = np.histogram(t_s, bins=bins)
        rate = counts / (n_neurons * (bin_ms/1000.0))  
        centers = 0.5*(bins[:-1] + bins[1:])
        return centers, rate

    @staticmethod
    def power_spectrum_from_rate(rate, fs_hz, normalize_dc=True, eps=1e-12):
        r_raw = np.asarray(rate)
        n = len(r_raw)
        if n == 0:
            return np.array([]), np.array([])

        psd_raw = (np.abs(np.fft.rfft(r_raw))**2) / n
        dc = psd_raw[0]
        r = r_raw - np.mean(r_raw)
        psd = (np.abs(np.fft.rfft(r))**2) / n

        if normalize_dc:
            psd = psd / max(dc, eps)

        freqs = np.fft.rfftfreq(n, d=1.0/fs_hz)
        return freqs, psd


    @staticmethod
    def gamma_peak_and_power(freqs, psd, f_lo=30.0, f_hi=50.0):
        if freqs.size == 0:
            return np.nan, np.nan
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(band):
            return np.nan, np.nan
        idx = np.argmax(psd[band])
        f_peak = freqs[band][idx]
        p_peak = psd[band][idx]
        return f_peak, p_peak

    @staticmethod
    def coherence_between_rates(rate_a, rate_b, fs_hz):
        n = min(len(rate_a), len(rate_b))
        if n == 0:
            return np.array([]), np.array([])
        a = rate_a[:n] - np.mean(rate_a[:n])
        b = rate_b[:n] - np.mean(rate_b[:n])
        A = np.fft.rfft(a)
        B = np.fft.rfft(b)
        Sxx = (A*np.conj(A)).real
        Syy = (B*np.conj(B)).real
        Sxy = A*np.conj(B)
        coh = (np.abs(Sxy)**2) / (Sxx*Syy + 1e-18)
        freqs = np.fft.rfftfreq(n, d=1.0/fs_hz)
        return freqs, coh

    @staticmethod
    def firing_rate_summary(spike_monitor, n_neurons, t_window_s=None, bin_ms=50.0):
        t_s = np.array(spike_monitor.t)
        if t_s.size == 0:
            return 0.0
        t_stop_s = float(t_s.max()) if t_window_s is None else t_window_s
        centers, rate = SpikeAnalysis.bin_population_rate(spike_monitor, n_neurons, bin_ms=bin_ms, t_stop_s=t_stop_s)
        return float(np.mean(rate))  

    @staticmethod
    def gamma_metrics_from_spikes(spike_monitor, n_neurons, discard_ms=500.0, bin_ms=1.0, gamma_band=(30.0, 50.0)):
        t_s = np.array(spike_monitor.t)
        if t_s.size == 0:
            return {'freqs': np.array([]), 'psd': np.array([]), 'f_peak': np.nan, 'gamma_power': np.nan,
                    'centers': np.array([]), 'rate': np.array([])}

        t_stop_s = float(t_s.max())
        centers, rate = SpikeAnalysis.bin_population_rate(spike_monitor, n_neurons, bin_ms=bin_ms, t_stop_s=t_stop_s)

        if discard_ms > 0:
            keep = centers*1000.0 >= discard_ms
            centers = centers[keep]
            rate = rate[keep]

        if centers.size < 8:
            return {'freqs': np.array([]), 'psd': np.array([]), 'f_peak': np.nan, 'gamma_power': np.nan,
                    'centers': centers, 'rate': rate}

        fs = 1.0 / np.diff(centers).mean()
        freqs, psd = SpikeAnalysis.power_spectrum_from_rate(rate, fs, normalize_dc=True)
        f_peak, g_pow = SpikeAnalysis.gamma_peak_and_power(freqs, psd, f_lo=gamma_band[0], f_hi=gamma_band[1])

        return {
            'freqs': freqs, 'psd': psd, 'f_peak': f_peak, 'gamma_power': g_pow,
            'centers': centers, 'rate': rate
        }

    @staticmethod
    def gamma_coherence_between_sites(spike_monitor_a, n_e_a, spike_monitor_b, n_e_b,
                                      discard_ms=500.0, bin_ms=1.0, ref_peak_from=None):
        out_a = SpikeAnalysis.gamma_metrics_from_spikes(spike_monitor_a, n_e_a, discard_ms=discard_ms, bin_ms=bin_ms)
        out_b = SpikeAnalysis.gamma_metrics_from_spikes(spike_monitor_b, n_e_b, discard_ms=discard_ms, bin_ms=bin_ms)
        if out_a['centers'].size == 0 or out_b['centers'].size == 0:
            return {'freqs': np.array([]), 'coh': np.array([]), 'coh_at_peak': np.nan, 'f_ref': np.nan}

        fs = 1.0 / np.diff(out_a['centers']).mean()
        n = min(len(out_a['rate']), len(out_b['rate']))
        freqs, coh = SpikeAnalysis.coherence_between_rates(out_a['rate'][:n], out_b['rate'][:n], fs)

        f_ref = out_a['f_peak'] if (ref_peak_from is None or np.isnan(ref_peak_from)) else float(ref_peak_from)
        if freqs.size == 0 or np.isnan(f_ref):
            return {'freqs': freqs, 'coh': coh, 'coh_at_peak': np.nan, 'f_ref': f_ref}

        idx = np.argmin(np.abs(freqs - f_ref))
        return {'freqs': freqs, 'coh': coh, 'coh_at_peak': float(coh[idx]), 'f_ref': f_ref}
"""
Analysis functions for neural data. to do  : check lfp method + spike analysis
"""
import numpy as np
from scipy.signal import welch, spectrogram
from brian2 import *

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
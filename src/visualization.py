"""
Visualization functions
"""
import matplotlib.pyplot as plt
import numpy as np
from .analysis import *
from brian2 import *
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

class NetworkVisualizer:
    @staticmethod
    def _smooth(y, window=11, poly=3):
        y = np.asarray(y)
        if y.size < 3:
            return y
        if _HAS_SAVGOL:
            window = max(3, min(window | 1, len(y) - (1 - len(y) % 2)))
            poly = min(poly, window - 1)
            return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
        else:
            w = max(3, window)
            k = np.ones(w) / w
            return np.convolve(y, k, mode="same")
    
    @staticmethod
    def plot_raster(spike_monitors, layer_configs, figsize=(15, 10)):
        fig, axes = plt.subplots(len(spike_monitors), 1, figsize=figsize)
        if len(spike_monitors) == 1:
            axes = [axes]
        
        for i, (layer_name, monitors) in enumerate(spike_monitors.items()):
            ax = axes[i]
            config = layer_configs[layer_name]
            
            if 'E_spikes' in monitors:
                ax.scatter(monitors['E_spikes'].t/second, monitors['E_spikes'].i,
                          color='green', s=0.5, alpha=0.6, label="E")
            
            if 'SOM_spikes' in monitors:
                ax.scatter(monitors['SOM_spikes'].t/second, 
                          monitors['SOM_spikes'].i + config['neuron_counts']['E'],
                          color='blue', s=0.5, alpha=0.8, label="SOM")
            
            if 'PV_spikes' in monitors:
                ax.scatter(monitors['PV_spikes'].t/second,
                          monitors['PV_spikes'].i + config['neuron_counts']['E'] + config['neuron_counts']['SOM'],
                          color='red', s=0.5, alpha=0.8, label="PV")
                
            if 'VIP_spikes' in monitors:
                ax.scatter(monitors['VIP_spikes'].t/second,
                          monitors['VIP_spikes'].i + config['neuron_counts']['E'] + config['neuron_counts']['SOM'] + config['neuron_counts']['PV'],
                          color='gold', s=0.5, alpha=0.8, label="VIP")
            
            ax.set_xlim(0, 4)
            ax.set_ylabel('Neuron index')
            ax.set_title(f'{layer_name} Spike Raster Plot')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_population_psd(pop_rate_dict, fmax=100, title='Population PSDs', figsize=(10, 5)):
        plt.figure(figsize=figsize)
        for name, (freqs, psd) in pop_rate_dict.items():
            mask = freqs <= fmax
            plt.plot(freqs[mask], psd[mask], label=name)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized power')
        plt.title(title)
        plt.legend(frameon=False)
        plt.tight_layout()
    
    @staticmethod
    def plot_lfp(state_monitors, layer_configs, figsize=(15, 12)):
        fig, axes = plt.subplots(len(state_monitors), 1, figsize=figsize)
        if len(state_monitors) == 1:
            axes = [axes]
        
        for i, (layer_name, monitors) in enumerate(state_monitors.items()):
            if 'E_state' in monitors:
                time_stable, lfp_stable = LFPAnalysis.process_lfp(monitors['E_state'])
                
                ax = axes[i]
                ax.plot(time_stable, lfp_stable, 'b-', linewidth=0.5)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('LFP (norm)')
                ax.set_title(f'{layer_name} Local Field Potential')
                ax.set_xlim(0, 4000)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rate(
        rate_monitors,
        layer_configs,
        figsize=(15, 10),
        smooth=True,
        smooth_window=51,
        smooth_poly=3,
        x_in_seconds=True,
        linewidth=1.6,
    ):

        color_map = {
            "E": "green",
            "SOM": "blue",
            "PV": "red",
            "VIP": "gold",
        }

        n_layers = len(rate_monitors)
        fig, axes = plt.subplots(n_layers, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, (layer_name, monitors) in enumerate(rate_monitors.items()):
            ax = axes[i]

            ordered_keys = [k for k in ["E_rate", "SOM_rate", "PV_rate", "VIP_rate"] if k in monitors]
            ordered_keys += [k for k in monitors.keys() if k not in ordered_keys]

            plotted_any = False
            for key in ordered_keys:
                mon = monitors[key]
                if not hasattr(mon, "t"):
                    continue

                if x_in_seconds:
                    t = mon.t / second
                    xlabel = "Time (s)"
                else:
                    t = mon.t / ms
                    xlabel = "Time (ms)"

                if hasattr(mon, "rate"):
                    y = mon.rate / Hz
                elif hasattr(mon, "rates"):  
                    y = mon.rates / Hz
                else:
                    continue

                if smooth and len(y) > 5:
                    y_plot = NetworkVisualizer._smooth(y, window=smooth_window, poly=smooth_poly)
                else:
                    y_plot = y

                pop = key.split("_")[0].upper()
                color = color_map.get(pop, None)

                ax.plot(t, y_plot, label=pop, linewidth=linewidth, color=color)
                plotted_any = True

            ax.set_title(f"{layer_name} Population Firing Rate")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Rate (Hz)")
            ax.grid(True, alpha=0.3)
            if plotted_any:
                ax.legend(frameon=False, ncols=4 if len(ordered_keys) >= 4 else 2)

            try:
                if x_in_seconds:
                    ax.set_xlim(0, 4)
                else:
                    ax.set_xlim(0, 4000)
            except Exception:
                pass

        plt.tight_layout()
        return fig


    

    

    def plot_power_spectra(
        state_monitors,
        layer_configs=None,             
        max_hz=50,                     
        smooth=True,
        smooth_window=11,
        smooth_poly=3,
        figsize=(10, 6),
        show_peaks=True,
        linewidth=2.2,):

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=figsize)

        for layer_name, monitors in state_monitors.items():
            if "E_state" not in monitors:
                continue

            _, lfp_stable = LFPAnalysis.process_lfp(monitors["E_state"])
            freq, psd = LFPAnalysis.compute_power_spectrum(lfp_stable)

            mask = freq <= max_hz
            f = freq[mask]
            p = psd[mask]

            if smooth and len(p) > 5:
                p_smooth = NetworkVisualizer._smooth(p, window=smooth_window, poly=smooth_poly)
            else:
                p_smooth = p

            line, = ax.plot(f, p_smooth, label=layer_name, linewidth=linewidth)

            if show_peaks and len(p) > 0:
                peak_idx = int(np.argmax(p))
                peak_f = freq[peak_idx]
                if peak_f <= max_hz:
                    ax.axvline(peak_f, linestyle="--", alpha=0.35)
                    ax.annotate(
                        f"{peak_f:.1f} Hz",
                        xy=(peak_f, p_smooth[min(peak_idx, len(p_smooth)-1)]),
                        xytext=(5, 8),
                        textcoords="offset points",
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                    )

        ax.set_xlabel("Frequency (Hz)", fontsize=13)
        ax.set_ylabel("Power", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=11, ncols=2)
        fig.tight_layout()
        return fig

    @staticmethod

    def plot_power_spectra_loglog(state_monitors, layer_configs, figsize=(12, 12)):

        n_layers = len(state_monitors)
        fig, axes = plt.subplots(n_layers, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        colors = ['b', 'g', 'purple', 'orange', 'red']
        fmin, fmax = 1, 200

        for i, (layer_name, monitors) in enumerate(state_monitors.items()):
            if 'E_state' not in monitors:
                continue
            
            time_stable, lfp_stable = LFPAnalysis.process_lfp(monitors['E_state'])
            freq, psd = LFPAnalysis.power_spectrum_loglog(lfp_stable, time_stable)

            mask = (freq >= fmin) & (freq <= fmax)
            ax = axes[i]
            color = colors[i % len(colors)]

            ax.loglog(freq[mask], psd[mask], lw=2, color=color, label=layer_name)

            f_ref = np.linspace(fmin, fmax, 500)
            one_over_f = 1 / f_ref
            scale_factor = np.max(psd[mask]) / np.max(one_over_f)
            ax.loglog(f_ref, scale_factor * one_over_f, 'k--', lw=1.5, label='1/f reference')

            ax.set_xlim(fmin, fmax)
            ax.set_xlabel("Frequency (Hz)", fontsize=12)
            ax.set_ylabel("Power (a.u.)", fontsize=12)
            ax.set_title(f"LFP Power Spectrum — {layer_name}", fontsize=14)
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend()

            ax.set_box_aspect(1)

        plt.tight_layout()
        return fig

    
    @staticmethod
    

    
    @staticmethod
    def plot_spectrogram(state_monitors, layer_configs, fmax=100, win_ms=250, step_ms=25,
                         light_window=(2.0, 4.0), figsize=(12, 5)):

        import matplotlib.pyplot as plt
        figs = []
        for layer_name, monitors in state_monitors.items():
            if 'E_state' not in monitors:
                continue
            time_ms, lfp = LFPAnalysis.process_lfp(monitors['E_state'])
            t_spec, f, Sxx = LFPAnalysis.compute_spectrogram(time_ms, lfp, fmax=fmax,
                                                             win_ms=win_ms, step_ms=step_ms)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            im = ax.imshow(10*np.log10(Sxx + 1e-20), origin='lower', aspect='auto',
                           extent=[t_spec[0], t_spec[-1], f[0], f[-1]])
            ax.set_title(f'{layer_name} LFP Spectrogram')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Power (dB)')
            if light_window is not None:
                on, off = light_window
                ax.axvline(on, color='w', ls='--', lw=1)
                ax.axvline(off, color='w', ls='--', lw=1)
            figs.append(fig)
        return figs

    @staticmethod
    def plot_peak_freq_track(state_monitors, layer_configs, f_gamma=(20, 80), fmax=100,
                             win_ms=250, step_ms=25, light_window=(2.0, 4.0), figsize=(12, 4)):

        import matplotlib.pyplot as plt
        figs = []
        for layer_name, monitors in state_monitors.items():
            if 'E_state' not in monitors:
                continue
            time_ms, lfp = LFPAnalysis.process_lfp(monitors['E_state'])
            t_spec, f, Sxx = LFPAnalysis.compute_spectrogram(time_ms, lfp, fmax=fmax,
                                                             win_ms=win_ms, step_ms=step_ms)
            peak_f, peak_p = LFPAnalysis.peak_frequency_track(f, Sxx, f_gamma=f_gamma)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(t_spec, peak_f, lw=2)
            ax.set_title(f'{layer_name} Gamma Peak Frequency ({f_gamma[0]}–{f_gamma[1]} Hz)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Peak freq (Hz)')
            if light_window is not None:
                on, off = light_window
                ax.axvline(on, color='k', ls='--', lw=1)
                ax.axvline(off, color='k', ls='--', lw=1)
            ax.set_ylim(f_gamma[0], f_gamma[1])
            ax.grid(alpha=0.3)
            figs.append(fig)
        return figs

        
class MatlabMatchPlots:
    @staticmethod
    def plot_stim_locked_lfp_and_spikes(all_state_monitors, all_spike_monitors,
                                        t_stim, t_pre=0.200, t_post=0.500,
                                        binW=0.002, smooth_sigma=0.010):


        lfp_traces = []
        for mon in all_state_monitors:
            t_ms, lfp = LFPAnalysis.process_lfp(mon)  
            t_s = t_ms / 1000.0
            t_rel = t_s - float(t_stim)            
            mask = (t_rel >= -t_pre) & (t_rel <= t_post)
            if np.any(mask):
                lfp_traces.append(np.interp(
                    np.linspace(-t_pre, t_post, np.sum(mask)),
                    t_rel[mask],
                    lfp[mask]
                ))
        if len(lfp_traces) == 0:
            raise RuntimeError("No LFP traces found in analysis window.")
        t_common = np.linspace(-t_pre, t_post, len(lfp_traces[0]))
        tERP, erp = baseline_subtract(t_common, np.vstack(lfp_traces), t_pre=t_pre)

        pooled_spike_mons = []
        if isinstance(all_spike_monitors, dict):
            for mon in all_spike_monitors.values():
                pooled_spike_mons.append(mon)
        elif isinstance(all_spike_monitors, list):
            pooled_spike_mons = list(all_spike_monitors)
        else:
            pooled_spike_mons = [all_spike_monitors]
        tCenters, psth_hz = build_psth_from_spikemon(
            pooled_spike_mons, t_stim, t_pre=t_pre, t_post=t_post, binW=binW, smooth_sigma=smooth_sigma
        )

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        ax = axes[0,0]
        ax.plot(tERP, erp, lw=1.5)
        ax.axvline(0, color='k', ls=':')
        ax.axhline(0, color='k', ls=':')
        ax.set_title('LFP ERP (stim-locked)')
        ax.set_xlabel('Time from stimulus (s)')
        ax.set_ylabel('LFP (baseline-sub)')

        ax = axes[1,0]
        ax.plot(tCenters, psth_hz, lw=1.5)
        ax.axvline(0, color='k', ls=':')
        ax.set_title(f'Spike PSTH (bin={binW*1e3:.0f} ms, σ={smooth_sigma*1e3:.0f} ms)')
        ax.set_xlabel('Time from stimulus (s)')
        ax.set_ylabel('Firing rate (Hz)')

        axes[0,1].axis('off')
        axes[1,1].axis('off')

        fig.suptitle('Stimulus-locked responses (pooled, MATLAB-like)')
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_spectrogram_db(state_monitors, t_stim, t_pre=0.200, t_post=0.500,
                             wlen=0.200, step=0.020, fmax=150):
        import matplotlib.pyplot as plt
        lfp_traces = []
        for mon in state_monitors:
            t_ms, lfp = LFPAnalysis.process_lfp(mon)
            t_s = t_ms / 1000.0
            t_rel = t_s - float(t_stim)
            mask = (t_rel >= -t_pre) & (t_rel <= t_post)
            if np.any(mask):
                lfp_traces.append(np.interp(
                    np.linspace(-t_pre, t_post, np.sum(mask)),
                    t_rel[mask],
                    lfp[mask]
                ))
        X = np.vstack(lfp_traces)
        x_avg = X.mean(axis=0)
        fs = (len(x_avg) - 1) / (t_post + t_pre)

        from scipy.signal import spectrogram
        nperseg = int(round(wlen * fs))
        nover = int(round((wlen - step) * fs))
        f, T, P = spectrogram(x_avg - np.mean(x_avg), fs=fs, nperseg=nperseg, noverlap=nover, detrend='constant')
        Trel = T - t_pre
        dB = to_db_relative(P, Trel)
        maskf = f <= fmax

        fig, ax = plt.subplots(1,1, figsize=(8,5))
        im = ax.imshow(dB[maskf,:], origin='lower', aspect='auto',
                       extent=[Trel[0], Trel[-1], f[maskf][0], f[maskf][-1]])
        ax.axvline(0, color='k', ls=':', lw=1)
        ax.set_xlabel('Time from stimulus (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('LFP spectrogram (dB vs pre-stim)')
        cb = plt.colorbar(im, ax=ax); cb.set_label('Power change (dB)')
        return fig

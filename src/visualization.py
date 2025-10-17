"""
Visualization functions
"""
import matplotlib.pyplot as plt
import numpy as np
from .analysis import LFPAnalysis
from brian2 import *

class NetworkVisualizer:
    
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
    def plot_power_spectra(state_monitors, layer_configs, figsize=(10, 12)):
        fig, axes = plt.subplots(len(state_monitors), 1, figsize=figsize)
        if len(state_monitors) == 1:
            axes = [axes]
        
        colors = ['b', 'g', 'purple', 'orange', 'red']
        
        for i, (layer_name, monitors) in enumerate(state_monitors.items()):
            if 'E_state' in monitors:
                _, lfp_stable = LFPAnalysis.process_lfp(monitors['E_state'])
                freq, psd = LFPAnalysis.compute_power_spectrum(lfp_stable)
                
                ax = axes[i]
                color = colors[i % len(colors)]
                ax.plot(freq[:50], psd[:50], color=color, 
                       label=layer_name, linewidth=2.5)
                ax.set_ylabel('Power', fontsize=18)
                ax.grid(True)
                
                peak_idx = np.argmax(psd[:50])
                ax.axvline(freq[peak_idx], color='r', linestyle='--')
                ax.legend(fontsize=20, loc='upper center')
                ax.tick_params(axis='both', which='major', labelsize=14)
        
        axes[-1].set_xlabel('Frequency (Hz)', fontsize=18)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_power_spectra_loglog(state_monitors, layer_configs, figsize=(10,12)):
        fig, axes = plt.subplots(len(state_monitors), 1, figsize=figsize)
        if len(state_monitors) == 1:
            axes = [axes]
        
        colors = ['b', 'g', 'purple', 'orange', 'red']
        fmin=1
        fmax=200

        for i, (layer_name, monitors) in enumerate(state_monitors.items()):
            if 'E_state' in monitors:
                time_stable , lfp_stable = LFPAnalysis.process_lfp(monitors['E_state'])
                freq, psd = LFPAnalysis.power_spectrum_loglog(lfp_stable, time_stable)
                
                ax = axes[i]
                color = colors[i % len(colors)]
                mask = (freq >= fmin) & (freq <= fmax)
                ax.loglog(freq[mask], psd[mask], lw=2, color=color, label=layer_name)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power (a.u.)")
                ax.set_title("LFP Power Spectrum (log–log)")
                ax.grid(True, which="both", ls="--", alpha=0.5)
        
        axes[-1].set_xlabel('Frequency (Hz)', fontsize=18)
        plt.tight_layout()
        return fig

    
    @staticmethod

    def plot_rate(rate_monitors, layer_configs, figsize=(10, 12)):

        layer_names = list(layer_configs.keys()) if isinstance(layer_configs, dict) else list(rate_monitors.keys())
        n_layers = len(layer_names) if layer_names else len(rate_monitors)

        if n_layers == 0:
            # Nothing to plot
            fig = plt.figure(figsize=figsize)
            fig.suptitle("Population Rates (no data)")
            return fig

        fig, axes = plt.subplots(n_layers, 1, sharex=True, figsize=figsize)
        if n_layers == 1:
            axes = [axes]

        def _to_seconds(t):
            try:
                return (t / second)
            except Exception:
                return t

        def _to_hz(r):
            try:
                return (r / Hz)
            except Exception:
                return r

        for ax, layer_name in zip(axes, layer_names):
            layer_rates = rate_monitors.get(layer_name, {})
            plotted_any = False

            for pop_key in sorted(layer_rates.keys()):
                mon = layer_rates[pop_key]
                try:
                    t = _to_seconds(mon.t)
                    r = _to_hz(mon.rate)
                    ax.plot(t, r, label=pop_key)
                    plotted_any = True
                except Exception as e:
                    ax.text(0.01, 0.9, f"Error plotting {pop_key}: {e}", transform=ax.transAxes, fontsize=8, color="red")

            ax.set_ylabel("Rate (Hz)")
            title = f"Layer {layer_name} — Population Rates" if layer_name is not None else "Population Rates"
            ax.set_title(title)
            if plotted_any:
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout(h_pad=1.0)
        return fig
    
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

        
        
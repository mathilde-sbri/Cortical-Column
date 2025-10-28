"""
Visualization functions
"""
import matplotlib.pyplot as plt
import numpy as np
from .analysis import *
from brian2 import *
from scipy import signal
from scipy.ndimage import gaussian_filter1d
try:
    from scipy.signal import savgol_filter, butter, filtfilt
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False


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
        
        ax.set_xlim(0.8, 1.6)
        ax.set_ylabel('Neuron index')
        ax.set_title(f'{layer_name} Spike Raster Plot')
        ax.legend()
    
    plt.tight_layout()
    return fig



def plot_lfp(state_monitors, layer_configs, figsize=(15, 12)):
    fig, axes = plt.subplots(len(state_monitors), 1, figsize=figsize)
    if len(state_monitors) == 1:
        axes = [axes]
    
    for i, (layer_name, monitors) in enumerate(state_monitors.items()):
        if 'E_state' in monitors:
            time_stable, lfp_stable = process_lfp(monitors['E_state'])
            
            ax = axes[i]
            ax.plot(time_stable, lfp_stable, 'b-', linewidth=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('LFP (norm)')
            ax.set_title(f'{layer_name} Local Field Potential')
            ax.set_xlim(800, 1600)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_power_spectra(state_monitors, layer_configs, figsize=(10, 12)):
    fig, axes = plt.subplots(len(state_monitors), 1, figsize=figsize)
    if len(state_monitors) == 1:
        axes = [axes]
    
    colors = ['b', 'g', 'purple', 'orange', 'red']
    
    for i, (layer_name, monitors) in enumerate(state_monitors.items()):
        if 'E_state' in monitors:
            _, lfp_stable = process_lfp(monitors['E_state'])
            freq, psd = compute_power_spectrum(lfp_stable)
            
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


    for ax, layer_name in zip(axes, layer_names):
        layer_rates = rate_monitors.get(layer_name, {})
        plotted_any = False

        for pop_key in sorted(layer_rates.keys()):
            mon = layer_rates[pop_key]
            try:

                t = mon.t / ms
                r = mon.smooth_rate(window='flat', width=10.1*ms) / Hz

                ax.plot(t, r, label=pop_key)
                ax.set_xlim(800, 1600)
                #ax.set_ylim(0, 50)
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


def plot_spectrogram(state_monitors, layer_configs, fmax=100, win_ms=250, step_ms=25,
                    light_window=(2.0, 4.0), figsize=(12, 5)):
    import matplotlib.pyplot as plt
    figs = []
    
    for layer_name, monitors in state_monitors.items():
        if 'E_state' not in monitors:
            continue
        
        time_ms, lfp = process_lfp(monitors['E_state'])
        
        f, t_spec, Sxx = compute_spectrogram(lfp, fs=10000, window_ms=50, overlap=0.85)
        
        Sxx_db = 10 * np.log10(Sxx + 1e-20)
        
        fmask = f <= fmax
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(Sxx_db[fmask, :], origin='lower', aspect='auto',
                    extent=[t_spec[0], t_spec[-1], f[fmask][0], f[fmask][-1]],
                    cmap='viridis')
        ax.set_title(f'{layer_name} LFP Spectrogram')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim([0, fmax])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        
        if light_window is not None:
            on, off = light_window
            ax.axvline(on, color='w', ls='--', lw=1)
            ax.axvline(off, color='w', ls='--', lw=1)
        
        figs.append(fig)
    
    return figs


def plot_peak_freq_track(state_monitors, layer_configs, f_gamma=(20, 80), fmax=100,
                            win_ms=250, step_ms=25, light_window=(2.0, 4.0), figsize=(12, 4)):

    import matplotlib.pyplot as plt
    figs = []
    for layer_name, monitors in state_monitors.items():
        if 'E_state' not in monitors:
            continue
        time_ms, lfp = process_lfp(monitors['E_state'])
        t_spec, f, Sxx = compute_spectrogram(time_ms, lfp, fmax=fmax,
                                                            win_ms=win_ms, step_ms=step_ms)
        peak_f, peak_p = peak_frequency_track(f, Sxx, f_gamma=f_gamma)

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

    


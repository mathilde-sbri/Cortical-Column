"""
Visualization functions
"""
import matplotlib.pyplot as plt
import numpy as np
from .analysis import LFPAnalysis
from brian2 import *
from scipy import signal
from scipy.ndimage import gaussian_filter1d
try:
    from scipy.signal import savgol_filter, butter, filtfilt
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
            
            ax.set_xlim(0, 1)
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
                ax.set_xlim(0, 1000)
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
                    ax.set_xlim(0, 1)
                else:
                    ax.set_xlim(0, 1000)
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




def compute_psth(spike_monitor, bin_width=2*ms, sigma=10*ms, simulation_time=1000*ms):
    """
    Compute PSTH (like MATLAB buildPSTH + smoothing)
    
    Parameters:
    -----------
    spike_monitor : Brian2 SpikeMonitor
    bin_width : time bin width (MATLAB: 2ms)
    sigma : Gaussian smoothing width (MATLAB: 10ms)
    simulation_time : total simulation duration
    
    Returns:
    --------
    t_centers : time bin centers (seconds)
    psth : firing rate in Hz
    """
    # Convert to base units
    bw = float(bin_width / ms) / 1000  # seconds
    sim_time = float(simulation_time / second)
    
    # Create time bins
    t_edges = np.arange(0, sim_time + bw, bw)
    t_centers = t_edges[:-1] + bw/2
    
    # Get spike times
    spike_times = spike_monitor.t / second
    
    # Histogram spike counts
    counts, _ = np.histogram(spike_times, bins=t_edges)
    
    # Convert to firing rate (Hz)
    # counts/bin → counts/second, then divide by number of neurons
    n_neurons = spike_monitor.source.N
    psth = counts / (bw * n_neurons)
    
    # Smooth with Gaussian (MATLAB uses conv with gaussian kernel)
    sigma_bins = float(sigma / ms) / 1000 / bw  # sigma in bins
    psth_smooth = gaussian_filter1d(psth, sigma=sigma_bins, mode='nearest')
    
    return t_centers, psth_smooth


def compute_population_lfp_proxy(
    spike_monitors_dict,
    pops=('E', 'PV', 'SOM', 'VIP'),
    weights_dict=None,
    dt_ms=0.1,
    t_start_ms=None,
    t_stop_ms=None,
    normalize_by_neurons=False,
    lowpass_cutoff_hz=300,
    butter_order=4,
):


    if weights_dict is None:
        weights_dict = {'E': 0.3, 'PV': 1.0, 'SOM': 1.0, 'VIP': 0.8}

    def _get_mon(d, pop):
        if f'{pop}_spikes' in d:
            return d[f'{pop}_spikes']
        return d.get(pop, None)

    def _extract_times_ms_and_n(mon):
        if mon is None:
            return np.array([], dtype=float), 0
        try:
            sts = mon.spike_trains()
            times = np.concatenate([np.asarray(t) for t in sts.values()]) if len(sts) else np.array([])
            try:
                times_ms = np.array([float(tt/second)*1000.0 for tt in times])
            except Exception:
                times_ms = np.asarray(times, dtype=float) * 1000.0
            return times_ms, len(sts)
        except Exception:
            pass

        try:
            times_ms = (np.asarray(mon.t) / second) * 1000.0
        except Exception:
            times_ms = np.asarray(mon.t, dtype=float) * 1000.0

        n_cells = getattr(mon, 'N', 0)
        return times_ms, int(n_cells) if n_cells is not None else 0

    all_times = []
    for pop in pops:
        mon = _get_mon(spike_monitors_dict, pop)
        times_ms, _ = _extract_times_ms_and_n(mon)
        if times_ms.size:
            all_times.append(times_ms)

    if not all_times and (t_start_ms is None or t_stop_ms is None):
        raise ValueError("No spikes found and no explicit t_start_ms/t_stop_ms provided.")

    if t_start_ms is None:
        t_start_ms = min(t.min() for t in all_times) if all_times else 0.0
    if t_stop_ms is None:
        t_stop_ms = max(t.max() for t in all_times) if all_times else (t_start_ms + 1000.0)

    edges = np.arange(t_start_ms, t_stop_ms + dt_ms, dt_ms)
    centers_ms = 0.5 * (edges[:-1] + edges[1:])
    lfp_proxy = np.zeros_like(centers_ms, dtype=float)

    for pop in pops:
        mon = _get_mon(spike_monitors_dict, pop)
        times_ms, n_cells = _extract_times_ms_and_n(mon)
        if times_ms.size == 0:
            continue
        counts, _ = np.histogram(times_ms, bins=edges)
        if normalize_by_neurons and n_cells > 0:
            counts = counts / float(n_cells)
        weight = weights_dict.get(pop, 1.0)
        lfp_proxy += weight * counts

    fs = 1000.0 / dt_ms  # Hz
    nyq = fs / 2.0
    cutoff = min(lowpass_cutoff_hz, nyq * 0.99) 
    b, a = butter(butter_order, cutoff / nyq, btype='low')
    lfp_filtered = filtfilt(b, a, lfp_proxy)

    return centers_ms / 1000.0, lfp_filtered



def compute_spectrogram(lfp_signal, fs=10000, window_ms=200, overlap_ms=180):
    """
    Compute spectrogram (matching MATLAB's spectrogram function)
    
    Parameters:
    -----------
    lfp_signal : 1D array of LFP proxy
    fs : sampling frequency (Hz)
    window_ms : window length in ms (MATLAB: 200ms)
    overlap_ms : overlap in ms (MATLAB: 180ms)
    
    Returns:
    --------
    f : frequency array
    t : time array
    Sxx_dB : spectrogram in dB relative to baseline
    """
    # Convert to samples
    nperseg = int(window_ms * fs / 1000)
    noverlap = int(overlap_ms * fs / 1000)
    nfft = 2**int(np.ceil(np.log2(nperseg * 2.5)))
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(lfp_signal, fs=fs, 
                                   window='hann',
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   nfft=nfft,
                                   scaling='density')
    
    # Baseline normalization (MATLAB: first 200ms or t<0 if stim-locked)
    # Here assume first 200ms is baseline
    baseline_idx = t < 0.2
    if np.sum(baseline_idx) > 0:
        baseline_power = np.mean(Sxx[:, baseline_idx], axis=1, keepdims=True)
    else:
        baseline_power = np.mean(Sxx, axis=1, keepdims=True)
    
    # Convert to dB
    Sxx_dB = 10 * np.log10(Sxx / (baseline_power + 1e-10))
    
    return f, t, Sxx_dB


# ============================================
# EXAMPLE USAGE: Create plots matching MATLAB
# ============================================

def plot_layer_psth(
    spike_monitors,
    layer_configs,
    pops_to_include=('E','PV','SOM','VIP'),
    bin_width=2*ms,
    sigma=10*ms,
    mode='mean_per_neuron',  # 'mean_per_neuron' or 'population_rate'
    stim_onset=0.5           # seconds
):


    def _all_spike_times_seconds(mon):
        sts = mon.spike_trains()
        times = np.concatenate([np.asarray(t) for t in sts.values()])
        try:
            times = np.array([float(t) for t in times]) 
        except Exception:
            times = np.asarray(times, dtype=float)
        return times, len(sts)


    t_min, t_max = None, None
    for layer_name, monitors in spike_monitors.items():
        for pop in pops_to_include:
            key = f'{pop}_spikes'
            if key in monitors:
                ts, _ = _all_spike_times_seconds(monitors[key])
                if ts.size:
                    mn, mx = ts.min(), ts.max()
                    t_min = mn if t_min is None else min(t_min, mn)
                    t_max = mx if t_max is None else max(t_max, mx)
    if t_min is None or t_max is None:
        raise ValueError("No spikes found in the provided monitors.")

    bin_w_s = float(bin_width/second) 
    edges = np.arange(t_min, t_max + bin_w_s, bin_w_s)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    layers = ['L1', 'L23', 'L4', 'L5', 'L6']

    for i, layer_name in enumerate(layers):
        ax = axes[i]
        monitors = spike_monitors.get(layer_name, {})
        total_counts = np.zeros(len(edges)-1, dtype=float)
        total_neurons = 0

        any_data = False
        for pop in pops_to_include:
            key = f'{pop}_spikes'
            if key not in monitors:
                continue
            ts, n_cells = _all_spike_times_seconds(monitors[key])
            if ts.size:
                any_data = True
                counts, _ = np.histogram(ts, bins=edges)
                total_counts += counts
                total_neurons += n_cells

        if not any_data:
            ax.text(0.5, 0.5, f'{layer_name}: no spikes', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if mode == 'mean_per_neuron':
            denom = (total_neurons * bin_w_s) if total_neurons > 0 else bin_w_s
        elif mode == 'population_rate':
            denom = bin_w_s
        else:
            raise ValueError("mode must be 'mean_per_neuron' or 'population_rate'")
        rate = total_counts / denom


        config = layer_configs.get(layer_name, layer_name)
        ax.plot(centers, rate, linewidth=1.8)
        ax.axvline(stim_onset, linestyle=':', alpha=0.5)
        ylabel = 'Rate (Hz per neuron)' if mode == 'mean_per_neuron' else 'Population rate (Hz)'
        ax.set_ylabel(ylabel)
        ax.set_title(f'{config} — {layer_name}')

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.suptitle('Layer PSTHs', y=1.02)
    return fig


def plot_layer_spectrograms(
    spike_monitors,
    layer_configs,
    pops_to_include=('E', 'PV', 'SOM', 'VIP'),
    fmax=150,
    vmin=-10,
    vmax=10,
    stim_onset=0.5
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    layers = ['L1', 'L23', 'L4', 'L5', 'L6']

    for i, layer in enumerate(layers):
        ax = axes[i]

        if layer not in spike_monitors:
            ax.text(0.5, 0.5, f'{layer}: no data', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        try:
            t_lfp, lfp = compute_population_lfp_proxy(
                spike_monitors[layer],
                pops=pops_to_include
            )
        except Exception as e:
            ax.text(0.5, 0.5, f'{layer}: error computing LFP\n{e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        fs = 1 / (t_lfp[1] - t_lfp[0]) 
        f, t_spec, Sxx_dB = compute_spectrogram(lfp, fs=fs)

        im = ax.pcolormesh(t_spec, f, Sxx_dB, shading='auto', cmap='RdBu_r',
                           vmin=vmin, vmax=vmax)
        ax.axvline(stim_onset, color='k', linestyle=':', linewidth=1)
        ax.set_ylabel('Freq (Hz)')
        ax.set_ylim([0, fmax])

        config = layer_configs.get(layer, layer)
        ax.set_title(f'{config} — {layer} LFP Spectrogram')

        plt.colorbar(im, ax=ax, label='Power change (dB)')

        if i == len(layers) - 1:
            ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.suptitle('Layer-wise LFP Spectrograms', y=1.02)
    return fig


        
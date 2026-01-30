"""
Comprehensive Diagnostic Plots for Cortical Column Model
=========================================================
This script generates multiple diagnostic plots to help identify
issues with network dynamics, E/I balance, and oscillatory behavior.

Add this to your project and call the functions after simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.stats import pearsonr
from brian2 import ms, Hz, second


def compute_firing_rates(spike_monitor, time_window_ms=50, dt_ms=0.1):
    """Compute instantaneous firing rates with sliding window."""
    spike_trains = spike_monitor.spike_trains()
    n_neurons = len(spike_trains)
    
    if n_neurons == 0:
        return None, None
    
    # Get time range
    all_times = []
    for times in spike_trains.values():
        if len(times) > 0:
            all_times.extend(times / ms)
    
    if len(all_times) == 0:
        return None, None
    
    t_max = max(all_times)
    time_bins = np.arange(0, t_max + dt_ms, dt_ms)
    
    # Compute population rate
    all_spikes = np.concatenate([times / ms for times in spike_trains.values() if len(times) > 0])
    hist, _ = np.histogram(all_spikes, bins=time_bins)
    
    # Smooth with Gaussian kernel
    kernel_width = int(time_window_ms / dt_ms)
    kernel = signal.windows.gaussian(kernel_width * 4, kernel_width)
    kernel /= kernel.sum()
    
    rate = np.convolve(hist, kernel, mode='same') * (1000 / dt_ms) / n_neurons
    
    return time_bins[:-1], rate


def plot_ei_balance_diagnostics(spike_monitors, state_monitors, config, baseline_time, 
                                 stimuli_time, save_path=None):
    """
    Plot E/I balance metrics across layers.
    
    This helps identify if inhibition is too strong/weak.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    layers = list(spike_monitors.keys())
    colors = {'E': 'royalblue', 'PV': 'orangered', 'SOM': 'forestgreen', 'VIP': 'gray'}
    
    # 1. E/I Firing Rate Ratio Over Time
    ax1 = fig.add_subplot(gs[0, :])
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        
        # Get E and total I rates
        e_mon = monitors.get('E_spikes')
        pv_mon = monitors.get('PV_spikes')
        som_mon = monitors.get('SOM_spikes')
        
        if e_mon is None:
            continue
            
        t_e, rate_e = compute_firing_rates(e_mon)
        
        if t_e is None:
            continue
        
        # Sum inhibitory rates
        rate_i = np.zeros_like(rate_e)
        for i_mon in [pv_mon, som_mon]:
            if i_mon is not None:
                t_i, r_i = compute_firing_rates(i_mon)
                if t_i is not None and len(r_i) == len(rate_i):
                    rate_i += r_i
        
        # Compute ratio (avoid division by zero)
        ratio = rate_e / (rate_i + 0.1)
        
        ax1.plot(t_e, ratio, label=layer_name, alpha=0.7)
    
    ax1.axvline(baseline_time, color='red', linestyle='--', label='Stimulus onset')
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('E/I Rate Ratio')
    ax1.set_title('Excitation/Inhibition Balance Over Time')
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, 2])
    
    # 2. Pre vs Post Stimulus Firing Rates by Cell Type
    ax2 = fig.add_subplot(gs[1, 0])
    
    cell_types = ['E', 'PV', 'SOM', 'VIP']
    x_positions = np.arange(len(layers))
    width = 0.18
    
    for i, cell_type in enumerate(cell_types):
        pre_rates = []
        post_rates = []
        
        for layer_name in layers:
            monitors = spike_monitors[layer_name]
            mon = monitors.get(f'{cell_type}_spikes')
            
            if mon is None:
                pre_rates.append(0)
                post_rates.append(0)
                continue
            
            spike_trains = mon.spike_trains()
            n_neurons = len(spike_trains)
            
            if n_neurons == 0:
                pre_rates.append(0)
                post_rates.append(0)
                continue
            
            # Count spikes in pre and post periods
            pre_count = 0
            post_count = 0
            
            for times in spike_trains.values():
                times_ms = times / ms
                pre_count += np.sum((times_ms > 500) & (times_ms < baseline_time))
                post_count += np.sum((times_ms > baseline_time + 500) & 
                                    (times_ms < baseline_time + stimuli_time))
            
            pre_duration = (baseline_time - 500) / 1000  # seconds
            post_duration = (stimuli_time - 500) / 1000
            
            pre_rates.append(pre_count / (n_neurons * pre_duration))
            post_rates.append(post_count / (n_neurons * post_duration))
        
        offset = (i - 1.5) * width
        bars1 = ax2.bar(x_positions + offset - width/4, pre_rates, width/2, 
                       label=f'{cell_type} pre' if i == 0 else '', 
                       color=colors[cell_type], alpha=0.5)
        bars2 = ax2.bar(x_positions + offset + width/4, post_rates, width/2,
                       label=f'{cell_type} post' if i == 0 else '',
                       color=colors[cell_type], alpha=1.0)
    
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(layers)
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_title('Firing Rates: Pre (faded) vs Post (solid) Stimulus')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[ct], label=ct) for ct in cell_types]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # 3. PV/E Rate Ratio (key for gamma)
    ax3 = fig.add_subplot(gs[1, 1])
    
    pre_ratios = []
    post_ratios = []
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        e_mon = monitors.get('E_spikes')
        pv_mon = monitors.get('PV_spikes')
        
        if e_mon is None or pv_mon is None:
            pre_ratios.append(0)
            post_ratios.append(0)
            continue
        
        # Get rates
        e_trains = e_mon.spike_trains()
        pv_trains = pv_mon.spike_trains()
        
        n_e = len(e_trains)
        n_pv = len(pv_trains)
        
        if n_e == 0 or n_pv == 0:
            pre_ratios.append(0)
            post_ratios.append(0)
            continue
        
        # Pre-stimulus
        e_pre = sum(np.sum((t/ms > 500) & (t/ms < baseline_time)) for t in e_trains.values())
        pv_pre = sum(np.sum((t/ms > 500) & (t/ms < baseline_time)) for t in pv_trains.values())
        
        # Post-stimulus
        e_post = sum(np.sum((t/ms > baseline_time + 500)) for t in e_trains.values())
        pv_post = sum(np.sum((t/ms > baseline_time + 500)) for t in pv_trains.values())
        
        # Rates
        pre_dur = (baseline_time - 500) / 1000
        post_dur = (stimuli_time - 500) / 1000
        
        e_rate_pre = e_pre / (n_e * pre_dur) if pre_dur > 0 else 0
        pv_rate_pre = pv_pre / (n_pv * pre_dur) if pre_dur > 0 else 0
        e_rate_post = e_post / (n_e * post_dur) if post_dur > 0 else 0
        pv_rate_post = pv_post / (n_pv * post_dur) if post_dur > 0 else 0
        
        pre_ratios.append(pv_rate_pre / (e_rate_pre + 0.1))
        post_ratios.append(pv_rate_post / (e_rate_post + 0.1))
    
    x = np.arange(len(layers))
    ax3.bar(x - 0.2, pre_ratios, 0.35, label='Pre-stimulus', color='steelblue', alpha=0.7)
    ax3.bar(x + 0.2, post_ratios, 0.35, label='Post-stimulus', color='coral', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers)
    ax3.set_ylabel('PV/E Rate Ratio')
    ax3.set_title('PV to E Firing Rate Ratio (high = strong inhibition)')
    ax3.legend()
    ax3.axhline(10, color='red', linestyle=':', alpha=0.5, label='Typical upper bound')
    
    # 4. Coefficient of Variation of ISI (regularity measure)
    ax4 = fig.add_subplot(gs[2, 0])
    
    cv_values = {ct: [] for ct in cell_types}
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        
        for cell_type in cell_types:
            mon = monitors.get(f'{cell_type}_spikes')
            
            if mon is None:
                cv_values[cell_type].append(np.nan)
                continue
            
            spike_trains = mon.spike_trains()
            cvs = []
            
            for neuron_id, times in spike_trains.items():
                times_ms = times / ms
                # Only post-stimulus
                times_post = times_ms[(times_ms > baseline_time + 500)]
                
                if len(times_post) > 3:
                    isis = np.diff(times_post)
                    if np.mean(isis) > 0:
                        cv = np.std(isis) / np.mean(isis)
                        cvs.append(cv)
            
            cv_values[cell_type].append(np.nanmean(cvs) if cvs else np.nan)
    
    x = np.arange(len(layers))
    for i, cell_type in enumerate(cell_types):
        offset = (i - 1.5) * 0.2
        ax4.bar(x + offset, cv_values[cell_type], 0.18, 
               label=cell_type, color=colors[cell_type])
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers)
    ax4.set_ylabel('CV of ISI')
    ax4.set_title('Spike Regularity (CV < 1: regular, CV ≈ 1: Poisson, CV > 1: bursty)')
    ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax4.legend(loc='upper right')
    
    # 5. Population synchrony (spike count correlation)
    ax5 = fig.add_subplot(gs[2, 1])
    
    sync_e = []
    sync_pv = []
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        
        for cell_type, sync_list in [('E', sync_e), ('PV', sync_pv)]:
            mon = monitors.get(f'{cell_type}_spikes')
            
            if mon is None:
                sync_list.append(np.nan)
                continue
            
            spike_trains = mon.spike_trains()
            
            if len(spike_trains) < 10:
                sync_list.append(np.nan)
                continue
            
            # Bin spikes (10ms bins)
            t_max = baseline_time + stimuli_time
            bins = np.arange(0, t_max, 10)
            
            spike_counts = []
            for neuron_id, times in list(spike_trains.items())[:50]:  # Sample 50 neurons
                counts, _ = np.histogram(times/ms, bins=bins)
                spike_counts.append(counts)
            
            if len(spike_counts) < 2:
                sync_list.append(np.nan)
                continue
            
            spike_counts = np.array(spike_counts)
            
            # Compute mean pairwise correlation
            correlations = []
            for i in range(min(20, len(spike_counts))):
                for j in range(i+1, min(20, len(spike_counts))):
                    if np.std(spike_counts[i]) > 0 and np.std(spike_counts[j]) > 0:
                        r, _ = pearsonr(spike_counts[i], spike_counts[j])
                        if not np.isnan(r):
                            correlations.append(r)
            
            sync_list.append(np.mean(correlations) if correlations else np.nan)
    
    x = np.arange(len(layers))
    ax5.bar(x - 0.2, sync_e, 0.35, label='E cells', color='royalblue', alpha=0.7)
    ax5.bar(x + 0.2, sync_pv, 0.35, label='PV cells', color='orangered', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(layers)
    ax5.set_ylabel('Mean Pairwise Correlation')
    ax5.set_title('Population Synchrony (high = synchronized firing)')
    ax5.legend()
    
    plt.suptitle('E/I Balance Diagnostics', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_spectral_diagnostics(spike_monitors, config, baseline_time, stimuli_time, 
                              save_path=None):
    """
    Plot spectral analysis to understand oscillation frequencies.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    layers = list(spike_monitors.keys())
    
    # 1-2. Power spectra for E and PV populations
    for idx, cell_type in enumerate(['E', 'PV']):
        ax = fig.add_subplot(gs[0, idx])
        
        for layer_name in layers:
            monitors = spike_monitors[layer_name]
            mon = monitors.get(f'{cell_type}_spikes')
            
            if mon is None:
                continue
            
            t, rate = compute_firing_rates(mon, time_window_ms=20)
            
            if t is None or len(rate) < 1000:
                continue
            
            # Post-stimulus only
            post_mask = t > baseline_time + 200
            rate_post = rate[post_mask]
            
            if len(rate_post) < 500:
                continue
            
            # Compute power spectrum
            fs = 10000  # 0.1ms bins = 10kHz
            f, psd = signal.welch(rate_post, fs=fs, nperseg=min(2048, len(rate_post)//2))
            
            # Plot only 1-100 Hz
            mask = (f > 1) & (f < 100)
            ax.semilogy(f[mask], psd[mask], label=layer_name, alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'{cell_type} Population Power Spectrum (post-stim)')
        ax.legend(fontsize=8)
        ax.axvline(10, color='green', linestyle=':', alpha=0.5, label='Alpha')
        ax.axvline(40, color='purple', linestyle=':', alpha=0.5, label='Gamma')
    
    # 3. Pre vs Post spectrum comparison for L4C
    ax3 = fig.add_subplot(gs[0, 2])
    
    if 'L4C' in spike_monitors:
        mon = spike_monitors['L4C'].get('E_spikes')
        if mon is not None:
            t, rate = compute_firing_rates(mon, time_window_ms=20)
            
            if t is not None:
                # Pre-stimulus
                pre_mask = (t > 500) & (t < baseline_time - 100)
                rate_pre = rate[pre_mask]
                
                # Post-stimulus
                post_mask = (t > baseline_time + 500)
                rate_post = rate[post_mask]
                
                fs = 10000
                
                if len(rate_pre) > 500:
                    f_pre, psd_pre = signal.welch(rate_pre, fs=fs, 
                                                   nperseg=min(2048, len(rate_pre)//2))
                    mask = (f_pre > 1) & (f_pre < 100)
                    ax3.semilogy(f_pre[mask], psd_pre[mask], 'b-', 
                                label='Pre-stimulus', alpha=0.7)
                
                if len(rate_post) > 500:
                    f_post, psd_post = signal.welch(rate_post, fs=fs,
                                                     nperseg=min(2048, len(rate_post)//2))
                    mask = (f_post > 1) & (f_post < 100)
                    ax3.semilogy(f_post[mask], psd_post[mask], 'r-',
                                label='Post-stimulus', alpha=0.7)
    
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_title('L4C E cells: Pre vs Post Stimulus Spectrum')
    ax3.legend()
    ax3.axvline(10, color='green', linestyle=':', alpha=0.3)
    ax3.axvline(40, color='purple', linestyle=':', alpha=0.3)
    
    # 4. Spectrogram for E cells (one representative layer)
    ax4 = fig.add_subplot(gs[1, :2])
    
    if 'L4C' in spike_monitors:
        mon = spike_monitors['L4C'].get('E_spikes')
        if mon is not None:
            t, rate = compute_firing_rates(mon, time_window_ms=10)
            
            if t is not None and len(rate) > 1000:
                fs = 10000
                f, t_spec, Sxx = signal.spectrogram(rate, fs=fs, nperseg=512, 
                                                     noverlap=480, nfft=1024)
                
                # Limit to 1-100 Hz
                freq_mask = (f > 1) & (f < 100)
                
                im = ax4.pcolormesh(t_spec * 1000, f[freq_mask], 
                                    10 * np.log10(Sxx[freq_mask, :] + 1e-10),
                                    shading='gouraud', cmap='viridis')
                ax4.axvline(baseline_time, color='red', linestyle='--', linewidth=2)
                ax4.set_xlabel('Time (ms)')
                ax4.set_ylabel('Frequency (Hz)')
                ax4.set_title('L4C E Population Spectrogram')
                plt.colorbar(im, ax=ax4, label='Power (dB)')
                
                # Mark frequency bands
                ax4.axhline(10, color='white', linestyle=':', alpha=0.5)
                ax4.axhline(40, color='white', linestyle=':', alpha=0.5)
    
    # 5. Peak frequency by layer
    ax5 = fig.add_subplot(gs[1, 2])
    
    pre_peaks = []
    post_peaks = []
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        mon = monitors.get('E_spikes')
        
        if mon is None:
            pre_peaks.append(np.nan)
            post_peaks.append(np.nan)
            continue
        
        t, rate = compute_firing_rates(mon, time_window_ms=20)
        
        if t is None:
            pre_peaks.append(np.nan)
            post_peaks.append(np.nan)
            continue
        
        fs = 10000
        
        # Pre
        pre_mask = (t > 500) & (t < baseline_time - 100)
        rate_pre = rate[pre_mask]
        
        if len(rate_pre) > 500:
            f, psd = signal.welch(rate_pre, fs=fs, nperseg=min(2048, len(rate_pre)//2))
            mask = (f > 5) & (f < 100)
            if np.sum(mask) > 0:
                pre_peaks.append(f[mask][np.argmax(psd[mask])])
            else:
                pre_peaks.append(np.nan)
        else:
            pre_peaks.append(np.nan)
        
        # Post
        post_mask = t > baseline_time + 500
        rate_post = rate[post_mask]
        
        if len(rate_post) > 500:
            f, psd = signal.welch(rate_post, fs=fs, nperseg=min(2048, len(rate_post)//2))
            mask = (f > 5) & (f < 100)
            if np.sum(mask) > 0:
                post_peaks.append(f[mask][np.argmax(psd[mask])])
            else:
                post_peaks.append(np.nan)
        else:
            post_peaks.append(np.nan)
    
    x = np.arange(len(layers))
    ax5.bar(x - 0.2, pre_peaks, 0.35, label='Pre-stimulus', color='steelblue')
    ax5.bar(x + 0.2, post_peaks, 0.35, label='Post-stimulus', color='coral')
    ax5.set_xticks(x)
    ax5.set_xticklabels(layers)
    ax5.set_ylabel('Peak Frequency (Hz)')
    ax5.set_title('E Population Peak Frequency by Layer')
    ax5.legend()
    
    # Reference lines
    ax5.axhline(10, color='green', linestyle=':', alpha=0.5, label='Alpha band')
    ax5.axhline(40, color='purple', linestyle=':', alpha=0.5, label='Gamma band')
    
    plt.suptitle('Spectral Diagnostics', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_connectivity_analysis(config, save_path=None):
    """
    Visualize connectivity parameters to identify E/I balance issues.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    layers_config = config['layers']
    layers = [l for l in layers_config.keys() if l != 'L1']
    
    # 1. E->X connection probabilities
    ax1 = fig.add_subplot(gs[0, 0])
    
    targets = ['E', 'PV', 'SOM', 'VIP']
    x = np.arange(len(layers))
    width = 0.2
    colors = ['royalblue', 'orangered', 'forestgreen', 'gray']
    
    for i, target in enumerate(targets):
        probs = []
        for layer in layers:
            conn_prob = layers_config[layer].get('connection_prob', {})
            key = f'E_{target}'
            probs.append(conn_prob.get(key, 0))
        
        ax1.bar(x + (i - 1.5) * width, probs, width, label=f'E→{target}', color=colors[i])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_ylabel('Connection Probability')
    ax1.set_title('E Cell Output Connection Probabilities')
    ax1.legend()
    
    # 2. E->E vs E->PV effective strength
    ax2 = fig.add_subplot(gs[0, 1])
    
    ee_eff = []
    epv_eff = []
    
    for layer in layers:
        conn_prob = layers_config[layer].get('connection_prob', {})
        cond = layers_config[layer].get('conductance', {})
        
        p_ee = conn_prob.get('E_E', 0)
        p_epv = conn_prob.get('E_PV', 0)
        g_ee_ampa = cond.get('E_E_AMPA', 0)
        g_ee_nmda = cond.get('E_E_NMDA', 0)
        g_epv_ampa = cond.get('E_PV_AMPA', 0)
        g_epv_nmda = cond.get('E_PV_NMDA', 0)
        
        ee_eff.append(p_ee * (g_ee_ampa + g_ee_nmda))
        epv_eff.append(p_epv * (g_epv_ampa + g_epv_nmda))
    
    ax2.bar(x - 0.2, ee_eff, 0.35, label='E→E', color='royalblue')
    ax2.bar(x + 0.2, epv_eff, 0.35, label='E→PV', color='orangered')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_ylabel('Effective Conductance (prob × g)')
    ax2.set_title('E→E vs E→PV Effective Strength')
    ax2.legend()
    
    # 3. E->PV / E->E Ratio
    ax3 = fig.add_subplot(gs[1, 0])
    
    ratios = [epv / (ee + 1e-6) for ee, epv in zip(ee_eff, epv_eff)]
    
    colors_ratio = ['green' if r < 4 else 'orange' if r < 6 else 'red' for r in ratios]
    ax3.bar(x, ratios, color=colors_ratio)
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers)
    ax3.set_ylabel('E→PV / E→E Ratio')
    ax3.set_title('E→PV to E→E Ratio (target: 2-4, yours shown)')
    ax3.axhline(4, color='orange', linestyle='--', label='Warning threshold')
    ax3.axhline(6, color='red', linestyle='--', label='Problem threshold')
    ax3.legend()
    
    # Add text annotations
    for i, r in enumerate(ratios):
        ax3.text(i, r + 0.3, f'{r:.1f}', ha='center', fontsize=10)
    
    # 4. Neuron counts
    ax4 = fig.add_subplot(gs[1, 1])
    
    cell_types = ['E', 'PV', 'SOM', 'VIP']
    colors_ct = ['royalblue', 'orangered', 'forestgreen', 'gray']
    
    bottom = np.zeros(len(layers))
    
    for ct, color in zip(cell_types, colors_ct):
        counts = [layers_config[layer]['neuron_counts'].get(ct, 0) for layer in layers]
        ax4.bar(x, counts, 0.6, bottom=bottom, label=ct, color=color)
        bottom += counts
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers)
    ax4.set_ylabel('Neuron Count')
    ax4.set_title('Neuron Population Sizes')
    ax4.legend()
    
    plt.suptitle('Connectivity Analysis', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_temporal_dynamics(spike_monitors, baseline_time, stimuli_time, save_path=None):
    """
    Plot detailed temporal dynamics around stimulus onset.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    layers = list(spike_monitors.keys())
    cell_types = ['E', 'PV', 'SOM', 'VIP']
    colors = {'E': 'royalblue', 'PV': 'orangered', 'SOM': 'forestgreen', 'VIP': 'gray'}
    
    # Focus window around stimulus
    t_pre = 500  # ms before stimulus
    t_post = 1000  # ms after stimulus
    
    # 1-4. Rate traces for each layer around stimulus onset
    for idx, layer_name in enumerate(layers[:4]):  # First 4 layers
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        monitors = spike_monitors[layer_name]
        
        for cell_type in cell_types:
            mon = monitors.get(f'{cell_type}_spikes')
            
            if mon is None:
                continue
            
            t, rate = compute_firing_rates(mon, time_window_ms=25)
            
            if t is None:
                continue
            
            # Center time on stimulus onset
            t_centered = t - baseline_time
            
            # Focus on window around stimulus
            mask = (t_centered > -t_pre) & (t_centered < t_post)
            
            ax.plot(t_centered[mask], rate[mask], label=cell_type, 
                   color=colors[cell_type], alpha=0.8)
        
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Stimulus')
        ax.axvspan(0, t_post, color='red', alpha=0.1)
        ax.set_xlabel('Time relative to stimulus (ms)')
        ax.set_ylabel('Rate (Hz)')
        ax.set_title(f'{layer_name} - Response Dynamics')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim([-t_pre, t_post])
    
    # 5. Response latency analysis
    ax5 = fig.add_subplot(gs[2, 0])
    
    latencies = {ct: [] for ct in cell_types}
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        
        for cell_type in cell_types:
            mon = monitors.get(f'{cell_type}_spikes')
            
            if mon is None:
                latencies[cell_type].append(np.nan)
                continue
            
            t, rate = compute_firing_rates(mon, time_window_ms=20)
            
            if t is None:
                latencies[cell_type].append(np.nan)
                continue
            
            # Find when rate exceeds 2x baseline
            t_centered = t - baseline_time
            
            # Baseline rate
            baseline_mask = (t_centered > -500) & (t_centered < -100)
            if np.sum(baseline_mask) == 0:
                latencies[cell_type].append(np.nan)
                continue
            
            baseline_rate = np.mean(rate[baseline_mask])
            threshold = baseline_rate * 1.5 + 1  # 50% increase or +1 Hz
            
            # Find first crossing after stimulus
            post_mask = t_centered > 0
            post_times = t_centered[post_mask]
            post_rates = rate[post_mask]
            
            crossings = np.where(post_rates > threshold)[0]
            
            if len(crossings) > 0:
                latencies[cell_type].append(post_times[crossings[0]])
            else:
                latencies[cell_type].append(np.nan)
    
    x = np.arange(len(layers))
    width = 0.2
    
    for i, cell_type in enumerate(cell_types):
        offset = (i - 1.5) * width
        ax5.bar(x + offset, latencies[cell_type], width, 
               label=cell_type, color=colors[cell_type])
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(layers)
    ax5.set_ylabel('Response Latency (ms)')
    ax5.set_title('Response Latency by Cell Type (time to 50% rate increase)')
    ax5.legend()
    
    # 6. Rate change magnitude
    ax6 = fig.add_subplot(gs[2, 1])
    
    rate_changes = {ct: [] for ct in cell_types}
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        
        for cell_type in cell_types:
            mon = monitors.get(f'{cell_type}_spikes')
            
            if mon is None:
                rate_changes[cell_type].append(np.nan)
                continue
            
            t, rate = compute_firing_rates(mon, time_window_ms=50)
            
            if t is None:
                rate_changes[cell_type].append(np.nan)
                continue
            
            t_centered = t - baseline_time
            
            # Baseline vs steady-state post
            baseline_mask = (t_centered > -800) & (t_centered < -100)
            post_mask = (t_centered > 300) & (t_centered < 1000)
            
            if np.sum(baseline_mask) == 0 or np.sum(post_mask) == 0:
                rate_changes[cell_type].append(np.nan)
                continue
            
            baseline_rate = np.mean(rate[baseline_mask])
            post_rate = np.mean(rate[post_mask])
            
            # Percent change
            if baseline_rate > 0.1:
                change = (post_rate - baseline_rate) / baseline_rate * 100
            else:
                change = post_rate * 100  # If baseline near 0, just use post rate
            
            rate_changes[cell_type].append(change)
    
    for i, cell_type in enumerate(cell_types):
        offset = (i - 1.5) * width
        ax6.bar(x + offset, rate_changes[cell_type], width,
               label=cell_type, color=colors[cell_type])
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(layers)
    ax6.set_ylabel('Rate Change (%)')
    ax6.set_title('Firing Rate Change with Stimulus')
    ax6.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax6.legend()
    
    plt.suptitle('Temporal Dynamics Analysis', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_all_diagnostics(spike_monitors, state_monitors, config, baseline_time, 
                         stimuli_time, output_dir='.'):
    """
    Generate all diagnostic plots and save them.
    
    Usage:
        plot_all_diagnostics(spike_monitors, state_monitors, CONFIG, 
                            baseline_time, stimuli_time, output_dir='diagnostics/')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating E/I balance diagnostics...")
    fig1 = plot_ei_balance_diagnostics(spike_monitors, state_monitors, config,
                                        baseline_time, stimuli_time,
                                        save_path=f'{output_dir}/ei_balance.png')
    
    print("Generating spectral diagnostics...")
    fig2 = plot_spectral_diagnostics(spike_monitors, config, baseline_time, stimuli_time,
                                      save_path=f'{output_dir}/spectral.png')
    
    print("Generating connectivity analysis...")
    fig3 = plot_connectivity_analysis(config, save_path=f'{output_dir}/connectivity.png')
    
    print("Generating temporal dynamics...")
    fig4 = plot_temporal_dynamics(spike_monitors, baseline_time, stimuli_time,
                                   save_path=f'{output_dir}/temporal.png')
    
    print(f"\nDiagnostic plots saved to {output_dir}/")
    print("  - ei_balance.png: E/I ratio, firing rates, synchrony")
    print("  - spectral.png: Power spectra, spectrogram, peak frequencies")
    print("  - connectivity.png: Connection strengths, E→PV/E→E ratio")
    print("  - temporal.png: Response dynamics, latencies, rate changes")
    
    return fig1, fig2, fig3, fig4


# =============================================================================
# QUICK SUMMARY FUNCTION
# =============================================================================

def print_quick_summary(spike_monitors, config, baseline_time, stimuli_time):
    """
    Print a quick text summary of key metrics.
    """
    print("\n" + "="*70)
    print("QUICK DIAGNOSTIC SUMMARY")
    print("="*70)
    
    layers = list(spike_monitors.keys())
    layers_config = config['layers']
    
    print("\n1. FIRING RATES (Hz)")
    print("-"*50)
    print(f"{'Layer':<8} {'E pre':>8} {'E post':>8} {'PV pre':>8} {'PV post':>8}")
    
    for layer_name in layers:
        monitors = spike_monitors[layer_name]
        
        rates = {}
        for ct in ['E', 'PV']:
            mon = monitors.get(f'{ct}_spikes')
            if mon is None:
                rates[f'{ct}_pre'] = 'N/A'
                rates[f'{ct}_post'] = 'N/A'
                continue
            
            spike_trains = mon.spike_trains()
            n = len(spike_trains)
            
            if n == 0:
                rates[f'{ct}_pre'] = 'N/A'
                rates[f'{ct}_post'] = 'N/A'
                continue
            
            pre_count = sum(np.sum((t/ms > 500) & (t/ms < baseline_time)) 
                          for t in spike_trains.values())
            post_count = sum(np.sum((t/ms > baseline_time + 500)) 
                           for t in spike_trains.values())
            
            pre_dur = (baseline_time - 500) / 1000
            post_dur = (stimuli_time - 500) / 1000
            
            rates[f'{ct}_pre'] = f"{pre_count / (n * pre_dur):.1f}"
            rates[f'{ct}_post'] = f"{post_count / (n * post_dur):.1f}"
        
        print(f"{layer_name:<8} {rates['E_pre']:>8} {rates['E_post']:>8} "
              f"{rates['PV_pre']:>8} {rates['PV_post']:>8}")
    
    print("\n2. E→PV / E→E RATIO (should be 2-4)")
    print("-"*50)
    
    for layer_name in [l for l in layers if l in layers_config]:
        lc = layers_config[layer_name]
        conn_prob = lc.get('connection_prob', {})
        cond = lc.get('conductance', {})
        
        p_ee = conn_prob.get('E_E', 0)
        p_epv = conn_prob.get('E_PV', 0)
        g_ee = cond.get('E_E_AMPA', 0) + cond.get('E_E_NMDA', 0)
        g_epv = cond.get('E_PV_AMPA', 0) + cond.get('E_PV_NMDA', 0)
        
        eff_ee = p_ee * g_ee
        eff_epv = p_epv * g_epv
        ratio = eff_epv / (eff_ee + 1e-6)
        
        status = "OK" if ratio < 4 else "HIGH" if ratio < 6 else "PROBLEM!"
        print(f"  {layer_name}: {ratio:.1f}x  [{status}]")
    
    print("\n3. KEY OBSERVATIONS")
    print("-"*50)
    print("  Check the diagnostic plots for detailed analysis.")
    print("  Look for:")
    print("    - E/I ratio changes with stimulus")
    print("    - Peak frequency shifts (should see gamma increase)")
    print("    - PV response latency (should respond quickly after E)")
    print("="*70 + "\n")

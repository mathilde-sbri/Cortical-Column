import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config_test import CONFIG
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(bin_width, smooth_sigma=0.010, nsigma=5):
    sigma_bins = smooth_sigma / bin_width
    half = int(np.ceil(nsigma * sigma_bins))
    t = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma_bins) ** 2)
    k /= k.sum()
    return k

def smooth_psth(psth, bin_width, smooth_sigma=0.010):
    k = gaussian_kernel(bin_width, smooth_sigma, nsigma=5)
    return np.apply_along_axis(lambda x: np.convolve(x, k, mode='same'), 0, psth)

def _merge_ranges(ranges):
    if not ranges:
        return None
    zmins = [min(a, b) for a, b in ranges]
    zmaxs = [max(a, b) for a, b in ranges]
    return (min(zmins), max(zmaxs))

def get_layer_bounds_from_config(CONFIG):

    layers = CONFIG.get('layers', {})
    sg_ranges = []
    if 'L23' in layers: sg_ranges.append(tuple(layers['L23']['coordinates']['z']))
    if 'L1'  in layers: sg_ranges.append(tuple(layers['L1']['coordinates']['z'])) 
    g_ranges  = []
    if 'L4' in layers:  g_ranges.append(tuple(layers['L4']['coordinates']['z']))

    ig_ranges = []
    if 'L5' in layers:  ig_ranges.append(tuple(layers['L5']['coordinates']['z']))
    if 'L6' in layers:  ig_ranges.append(tuple(layers['L6']['coordinates']['z']))

    bounds = {
        'SG': _merge_ranges(sg_ranges),
        'G' : _merge_ranges(g_ranges),
        'IG': _merge_ranges(ig_ranges),
    }
    return bounds

def assign_layers_for_channels(z_coords, bounds):
    masks = {k: np.zeros(len(z_coords), dtype=bool) for k in ['SG','G','IG']}
    for k in ['SG','G','IG']:
        br = bounds.get(k)
        if br is None:
            continue
        zmin, zmax = min(br), max(br)
        masks[k] = (z_coords >= zmin) & (z_coords <= zmax)
    return masks

def assign_layers_for_bipolar_channels(z_coords_bipolar, bounds):
    masks = {k: np.zeros(len(z_coords_bipolar), dtype=bool) for k in ['SG','G','IG']}
    for k in ['SG','G','IG']:
        br = bounds.get(k)
        if br is None:
            continue
        zmin, zmax = min(br), max(br)
        masks[k] = (z_coords_bipolar >= zmin) & (z_coords_bipolar <= zmax)
    return masks

def get_original_and_bipolar_depths_mm(probe_coords_mm):
    depths_original_mm = np.asarray(probe_coords_mm)[:, 2]
    depths_bipolar_mm  = 0.5 * (depths_original_mm[:-1] + depths_original_mm[1:])
    return depths_original_mm, depths_bipolar_mm


def plot_layered_psth_3x1(psth, t_centers, masks_original, title_prefix="PSTH by layer",
                          line_color='w', event_ms=0, figsize=(8, 9)):

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    layer_order = ['SG','G','IG']
    for i, lab in enumerate(layer_order):
        ax = axs[i]
        m = masks_original[lab]
        if np.any(m):
            y = psth[:, m].mean(axis=1)
            ax.plot(t_centers * 1000, y, '-', lw=1.8, color=line_color)
            n = int(m.sum())
        else:
            n = 0
        ax.axvline(event_ms, ls='--', lw=0.8, color=(0.8,0.8,0.8))
        ax.set_ylabel('Rate (Hz)')
        ax.set_title(f'{title_prefix} — {lab} (n={n})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time from stimulus (ms)')
    return fig, axs

def plot_layered_lfp_3x1(lfp_bipolar, lfp_time, masks_bipolar, title_prefix="Bipolar LFP by layer",
                         line_color='w', event_ms=0, figsize=(8, 9)):

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    layer_order = ['SG','G','IG']
    t_ms = lfp_time * 1000.0
    for i, lab in enumerate(layer_order):
        ax = axs[i]
        m = masks_bipolar[lab]
        if np.any(m):
            y = np.nanmean(lfp_bipolar[:, m], axis=1)
            ax.plot(t_ms, y, '-', lw=1.5, color=line_color)
            n = int(m.sum())
        else:
            n = 0
        ax.axvline(event_ms, ls='--', lw=0.8, color=(0.8,0.8,0.8))
        ax.set_ylabel('LFP (μV)')
        ax.set_title(f'{title_prefix} — {lab} (n={n})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time from stimulus (ms)')
    return fig, axs

def bipolar_rereference_lfp(lfp_signal, probe):

    n_channels = probe.n
    # LFPb[i] = LFP[i] - LFP[i+1]
    lfp_bipolar = np.diff(lfp_signal, axis=1)
    return lfp_bipolar



def plot_channel_psth_and_lfp(psth, t_centers, lfp_bipolar, lfp_time, 
                               probe_coords, event_times=None, 
                               figsize=(14, 10)):
    n_channels_bipolar = lfp_bipolar.shape[1]
    
    fig, axes = plt.subplots(n_channels_bipolar, 2, 
                            figsize=figsize, 
                            sharex='col',
                            constrained_layout=True)
    
    depths_original = probe_coords[:, 2] 
    depths_bipolar = (depths_original[:-1] + depths_original[1:]) / 2
    
    for ch in range(n_channels_bipolar):
        ax_psth = axes[ch, 0]
        ax_psth.plot(t_centers * 1000, psth[:, ch], 'k-', linewidth=1)
        ax_psth.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        if event_times is not None:
            for et in event_times:
                if et != 0:  
                    ax_psth.axvline(et * 1000, color='orange', 
                                   linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax_psth.set_ylabel(f'Ch {ch+1}\n({depths_bipolar[ch]*1000:.0f} μm)\nRate (Hz)', 
                          fontsize=8)
        ax_psth.spines['top'].set_visible(False)
        ax_psth.spines['right'].set_visible(False)
        
        if ch == 0:
            ax_psth.set_title('PSTH', fontsize=10, fontweight='bold')
        if ch == n_channels_bipolar - 1:
            ax_psth.set_xlabel('Time from stimulus (ms)', fontsize=9)
        
        ax_lfp = axes[ch, 1]
        ax_lfp.plot(lfp_time * 1000, lfp_bipolar[:, ch], 'k-', linewidth=0.5)
        ax_lfp.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        if event_times is not None:
            for et in event_times:
                if et != 0:
                    ax_lfp.axvline(et * 1000, color='orange', 
                                  linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax_lfp.set_ylabel(f'LFP (μV)', fontsize=8)
        ax_lfp.spines['top'].set_visible(False)
        ax_lfp.spines['right'].set_visible(False)
        
        if ch == 0:
            ax_lfp.set_title('Bipolar LFP', fontsize=10, fontweight='bold')
        if ch == n_channels_bipolar - 1:
            ax_lfp.set_xlabel('Time from stimulus (ms)', fontsize=9)
    
    return fig, axes




def build_psth_from_mua(mua, probe, t_pre, t_post, bin_width, 
                        event_times, channels=None, trials=None):
 
    t_edges = np.arange(-t_pre, t_post + bin_width, bin_width)
    t_centers = t_edges[:-1] + bin_width / 2
    
    if channels is None:
        channels = np.arange(probe.n)
    
    if trials is None:
        trials = np.arange(len(event_times))
    
    n_channels = len(channels)
    n_trials = len(trials)
    n_bins = len(t_edges) - 1
    
    spike_times_sec = mua.t / b2.second
    spike_channels = mua.i
    
    counts = np.zeros((n_bins, n_channels, n_trials))
    
    for ch_idx, channel in enumerate(channels):
        channel_mask = (spike_channels == channel)
        channel_spike_times = spike_times_sec[channel_mask]
        
        for tr_idx, trial in enumerate(trials):
            event_time = event_times[trial] / b2.second 
            time_mask = (channel_spike_times >= event_time - t_pre) & \
                       (channel_spike_times < event_time + t_post)
            
            if np.any(time_mask):
                spike_times_rel = channel_spike_times[time_mask] - event_time
                counts[:, ch_idx, tr_idx] = np.histogram(spike_times_rel, bins=t_edges)[0]
 
    psth = np.mean(counts, axis=2) / bin_width
    
    return psth, t_centers


def analyze_and_plot_laminar_recording_mua(sim, column, probe, tklfp, mua, 
                                           stim_onset_time=1000*b2.ms):

    lfp_raw = tklfp.lfp / b2.uvolt 
    lfp_bipolar = bipolar_rereference_lfp(lfp_raw, probe)
 
    t_pre = 0.2  
    t_post = 0.5 
    bin_width = 0.002 
    
    event_times = np.array([stim_onset_time / b2.second]) * b2.second
    
    psth, t_centers = build_psth_from_mua(
        mua, probe, t_pre, t_post, bin_width, event_times
    )
    psth = smooth_psth(psth, bin_width=0.01, smooth_sigma=0.010)  

    lfp_time = (tklfp.t / b2.second) - (stim_onset_time / b2.second)
    
    stim_time_sec = stim_onset_time / b2.second
    time_mask = (lfp_time >= -t_pre) & (lfp_time <= t_post)
    
    fig, axes = plot_channel_psth_and_lfp(
        psth, t_centers,
        lfp_bipolar[time_mask, :],  
        lfp_time[time_mask],
        probe.coords / b2.mm, 
        event_times=[0] 
    )
    
    bounds = get_layer_bounds_from_config(CONFIG)

    depths_orig_mm, depths_bip_mm = get_original_and_bipolar_depths_mm(probe.coords / b2.mm)

    masks_psth  = assign_layers_for_channels(depths_orig_mm, bounds)     
    masks_lfpb  = assign_layers_for_bipolar_channels(depths_bip_mm, bounds) 


    fig_psth_layers, _ = plot_layered_psth_3x1(
        psth, t_centers, masks_psth,
        title_prefix="PSTH (MUA) by layer",
        line_color='black', event_ms=0, figsize=(8, 9)
    )

    fig_lfp_layers, _ = plot_layered_lfp_3x1(
        lfp_bipolar[ (lfp_time>=t_centers[0]) & (lfp_time<=t_centers[-1]) , : ] if lfp_bipolar.shape[0] != len(lfp_time) else lfp_bipolar,
        lfp_time,
        masks_lfpb,
        title_prefix="Bipolar LFP by layer",
        line_color='black', event_ms=0, figsize=(8, 9)
    )

    return fig, axes, psth, lfp_bipolar



import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import brian2 as b2
from brian2 import *

from config.config_test2 import CONFIG
from src.column import CorticalColumn
from src.analysis import add_heterogeneity_to_layer, calculate_lfp

from scipy.signal import welch 




def process_lfp(monitor, start_time_ms=0):
    lfp = calculate_lfp(monitor)
    lfp_time = np.array(monitor.t/ms)

    start_idx = np.argmax(lfp_time >= start_time_ms)
    lfp_stable = lfp[start_idx:]
    time_stable = lfp_time[start_idx:]
    if np.std(lfp_stable) != 0 :
        lfp_stable = (lfp_stable - np.mean(lfp_stable))/np.std(lfp_stable)
    return time_stable, lfp_stable


def compute_power_spectrum(lfp_signal, fs=10000, method='welch', nperseg=None):
    if method == 'welch':
        if nperseg is None:
            nperseg = min(4096, len(lfp_signal) // 4)

        freq, psd = welch(lfp_signal, fs=fs, nperseg=nperseg,
                          noverlap=nperseg//2, window='hann')
        return freq, psd
    else:
        raise ValueError("method must be 'multitaper' or 'welch'")


def build_and_run_column(config, warmup=400*ms, simtime=800*ms):

    b2.start_scope()
    np.random.seed(config['simulation']['RANDOM_SEED'])
    b2.defaultclock.dt = config['simulation']['DT']

    column = CorticalColumn(column_id=0, config=config)
    for _, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, config)

    all_monitors = column.get_all_monitors()

    column.network.run(warmup)
    
    column.network.run(simtime)

    return all_monitors


def extract_l4c_e_lfp_psd(all_monitors, start_time_ms=0, fmax=150):

    l4c_mons = all_monitors.get('L4C', {})
    if 'E_state' not in l4c_mons:
        raise RuntimeError(
            f"No 'E_state' in L4C monitors. Available keys: {list(l4c_mons.keys())}"
        )

    mon = l4c_mons['E_state']
    _, lfp_stable = process_lfp(mon, start_time_ms=start_time_ms)

    dt_sec = float(b2.defaultclock.dt/second)
    fs = 1.0 / dt_sec

    freqs, psd = compute_power_spectrum(lfp_stable, fs=fs)

    mask = freqs <= fmax
    return freqs[mask], psd[mask]


def pv_sweep_lfp_psd(PV_counts,
                     base_config,
                     warmup=400*ms, simtime=800*ms,
                     start_time_ms=0,
                     fmax=150,
                     rescale_total_inhibition=False):

    psd_list = []
    freqs_ref = None

    base_PV = base_config['layers']['L4C']['neuron_counts']['PV']

    for N_PV in PV_counts:
        print(f"Running PV sweep: N_PV={N_PV}")

        config = deepcopy(base_config)
        config['layers']['L4C']['neuron_counts']['PV'] = int(N_PV)

        if rescale_total_inhibition:
            scale = float(base_PV) / float(N_PV)
            config['layers']['L4C']['conductance']['PV_E'] *= scale
            config['layers']['L4C']['conductance']['PV_PV'] *= scale

        all_monitors = build_and_run_column(
            config,
            warmup=warmup, simtime=simtime,
            # stim_rate_E=config['layers']['L4C']['input_rate'],
            # stim_rate_PV=6*Hz
        )

        freqs, psd = extract_l4c_e_lfp_psd(
            all_monitors,
            start_time_ms=start_time_ms,
            fmax=fmax
        )

        if freqs_ref is None:
            freqs_ref = freqs
        psd_list.append(psd)

    psd_mat = np.vstack(psd_list)
    return freqs_ref, psd_mat



def input_sweep_lfp_psd(input_rates_Hz,
                        base_config,
                        warmup=400*ms, simtime=800*ms,
                        start_time_ms=0,
                        fmax=150):

    psd_list = []
    freqs_ref = None

    for r in input_rates_Hz:
        print(f"Running input sweep: stim_rate_E={r} Hz")

        config = deepcopy(base_config)
        config['layers']['L4C']['input_rate'] = r*Hz
        
        all_monitors = build_and_run_column(config, warmup, simtime)

        freqs, psd = extract_l4c_e_lfp_psd(
            all_monitors,
            start_time_ms=start_time_ms,
            fmax=fmax
        )

        if freqs_ref is None:
            freqs_ref = freqs
        psd_list.append(psd)

    psd_mat = np.vstack(psd_list)
    return freqs_ref, psd_mat

def plot_psd_curves(freqs, psd_mat, yvals, ylabel, title):

    plt.figure(figsize=(10,12))
    
    cmap = plt.get_cmap("viridis")
    
    N = len(yvals)
    for i, y in enumerate(yvals):
        color = cmap(i / (N-1)) 
        plt.plot(freqs, psd_mat[i], color=color, alpha=0.9, linewidth=2,
                 label=f"{y}")
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (a.u.)")
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()

    plt.legend(title=ylabel)

def plot_param_spectrogram(freqs, psd_mat, yvals, ylabel, title):
    plt.figure(figsize=(7,5))
    plt.imshow(
        psd_mat,
        aspect='auto',
        origin='lower',
        extent=[freqs[0], freqs[-1], yvals[0], yvals[-1]]
    )
    plt.colorbar(label="PSD (a.u.)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()


def main():

    SWEEP_TYPE = "input"     

    warmup = 400*ms
    simtime = 800*ms
    start_time_ms = 300
    fmax = 150

    if SWEEP_TYPE == "pv":
        PV_counts = np.arange(50, 801, 50)

        freqs, psd_mat = pv_sweep_lfp_psd(
            PV_counts=PV_counts,
            base_config=CONFIG,
            warmup=warmup, simtime=simtime,
            start_time_ms=start_time_ms,
            fmax=fmax,
            rescale_total_inhibition=False
        )

        plot_psd_curves(
            freqs, psd_mat, PV_counts,
            ylabel="Number of PV neurons",
            title="L4C E LFP PSD vs PV count"
        )
        plot_param_spectrogram(
            freqs, psd_mat, PV_counts,
            ylabel="Number of PV neurons",
            title="Frequency content as PV count varies (LFP-based)"
        )

    elif SWEEP_TYPE == "input":
        input_rates = np.arange(2, 41, 2)

        freqs, psd_mat = input_sweep_lfp_psd(
            input_rates_Hz=input_rates,
            base_config=CONFIG,
            warmup=warmup, simtime=simtime,
            start_time_ms=start_time_ms,
            fmax=fmax
        )

        plot_psd_curves(
            freqs, psd_mat, input_rates,
            ylabel="Stim rate to E (Hz)",
            title="L4C E LFP PSD vs external input"
        )
        plot_param_spectrogram(
            freqs, psd_mat, input_rates,
            ylabel="Stim rate to E (Hz)",
            title="Frequency content as external input varies (LFP-based)"
        )

    else:
        raise ValueError("SWEEP_TYPE must be 'pv' or 'input'")

    plt.show()


if __name__ == "__main__":
    main()

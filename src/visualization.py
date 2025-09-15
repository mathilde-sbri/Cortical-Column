"""
Visualization functions
"""
import matplotlib.pyplot as plt
import numpy as np
from .analysis import LFPAnalysis, SpikeAnalysis
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
                ax.set_xlim(3000, 4000)
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
    def _get_NE(layer_configs, layer_name):
        return layer_configs[layer_name]['neuron_counts']['E']

    @staticmethod
    def plot_E_population_rates(spike_monitors, layer_configs, bin_ms=1.0, discard_ms=500.0,
                                figsize=(12, 3.5)):

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for layer_name, monitors in spike_monitors.items():
            if 'E_spikes' not in monitors:
                continue
            nE = NetworkVisualizer._get_NE(layer_configs, layer_name)
            centers, rate = SpikeAnalysis.bin_population_rate(
                monitors['E_spikes'], nE, bin_ms=bin_ms,
                t_stop_s=float(np.array(monitors['E_spikes'].t).max()) if monitors['E_spikes'].num_spikes > 0 else 0.0
            )
            if discard_ms > 0 and centers.size:
                keep = centers * 1000.0 >= discard_ms
                centers, rate = centers[keep], rate[keep]

            ax.plot(centers, rate, label=f'{layer_name}')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('E pop. rate (Hz/neuron)')
        ax.set_title('Population firing rate (E)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_E_population_psd(spike_monitors, layer_configs, bin_ms=1.0, discard_ms=500.0,
                              fmax=100, normalize_dc=True, figsize=(10, 5)):

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        summary_rows = []

        for layer_name, monitors in spike_monitors.items():
            if 'E_spikes' not in monitors:
                continue
            nE = NetworkVisualizer._get_NE(layer_configs, layer_name)
            out = SpikeAnalysis.gamma_metrics_from_spikes(
                monitors['E_spikes'], nE, discard_ms=discard_ms, bin_ms=bin_ms, gamma_band=(30.0, 50.0)
            )
            freqs, psd = out['freqs'], out['psd']
            if freqs.size == 0:
                continue

            mask = freqs <= fmax
            ax.plot(freqs[mask], psd[mask], label=f'{layer_name}')
            if not np.isnan(out['f_peak']):
                ax.axvline(out['f_peak'], linestyle='--', alpha=0.6)
                summary_rows.append((layer_name, out['f_peak'], out['gamma_power']))

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized power' if normalize_dc else 'Power')
        ax.set_title('E population rate — PSD (1 ms bins)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        if summary_rows:
            print("Gamma peaks:")
            for name, fpk, gpow in summary_rows:
                print(f"  {name:>16s}: {fpk:5.1f} Hz | power {gpow:.3f}")

        return fig

    @staticmethod
    def plot_gamma_peaks(spike_monitors, layer_configs, bin_ms=1.0, discard_ms=500.0,
                         figsize=(8, 4)):

        labels, powers, peaks = [], [], []
        for layer_name, monitors in spike_monitors.items():
            if 'E_spikes' not in monitors:
                continue
            nE = NetworkVisualizer._get_NE(layer_configs, layer_name)
            out = SpikeAnalysis.gamma_metrics_from_spikes(
                monitors['E_spikes'], nE, discard_ms=discard_ms, bin_ms=bin_ms, gamma_band=(30.0, 50.0)
            )
            if np.isnan(out['gamma_power']):
                continue
            labels.append(layer_name)
            powers.append(out['gamma_power'])
            peaks.append(out['f_peak'])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(np.arange(len(powers)), powers)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=0)
        ax.set_ylabel('Gamma power (norm.)')
        ax.set_title('Gamma peak power (30–50 Hz)')

        for i, fpk in enumerate(peaks):
            if not np.isnan(fpk):
                ax.text(i, powers[i], f'{fpk:.1f} Hz', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_gamma_coherence(spike_monitors, layer_configs, layer_pairs,
                             bin_ms=1.0, discard_ms=500.0, fmax=100, figsize=(10, 4.5)):

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for (la, lb) in layer_pairs:
            if la not in spike_monitors or lb not in spike_monitors:
                continue
            if 'E_spikes' not in spike_monitors[la] or 'E_spikes' not in spike_monitors[lb]:
                continue

            nEa = NetworkVisualizer._get_NE(layer_configs, la)
            nEb = NetworkVisualizer._get_NE(layer_configs, lb)

            coh_out = SpikeAnalysis.gamma_coherence_between_sites(
                spike_monitors[la]['E_spikes'], nEa,
                spike_monitors[lb]['E_spikes'], nEb,
                discard_ms=discard_ms, bin_ms=bin_ms, ref_peak_from=None  
            )

            freqs, coh = coh_out['freqs'], coh_out['coh']
            if freqs.size == 0:
                continue

            mask = freqs <= fmax
            ax.plot(freqs[mask], coh[mask], label=f'{la} ↔ {lb}  (C@{coh_out["f_ref"]:.1f}Hz={coh_out["coh_at_peak"]:.2f})')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_ylim(0, 1.02)
        ax.set_title('Gamma coherence between sites (E population rates)')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig


    @staticmethod
    def build_psd_inputs_from_spikes(spike_monitors, layer_configs, bin_ms=1.0, discard_ms=500.0):

        out = {}
        for layer_name, monitors in spike_monitors.items():
            if 'E_spikes' not in monitors:
                continue
            nE = NetworkVisualizer._get_NE(layer_configs, layer_name)
            res = SpikeAnalysis.gamma_metrics_from_spikes(
                monitors['E_spikes'], nE, discard_ms=discard_ms, bin_ms=bin_ms, gamma_band=(30.0, 50.0)
            )
            if res['freqs'].size:
                out[layer_name] = (res['freqs'], res['psd'])
        return out


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
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
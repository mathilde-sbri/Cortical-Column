import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config_test import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *

class LFPRecorder:
    """
    Records Local Field Potential based on Mazzoni et al. (2008)
    LFP = sum of |I_AMPA| + |I_GABA| from pyramidal (E) neurons
    """
    def __init__(self, layer, dt=0.1*ms):
        self.layer = layer
        self.dt = dt
        self.E_group = layer.neuron_groups.get('E')
        
        if self.E_group is None:
            raise ValueError(f"No excitatory neurons found in layer {layer.name}")
        
        # We'll compute LFP by monitoring synaptic currents
        # The LFP is approximated as sum of |I_E| + |I_I| across all E neurons
        self.lfp_values = []
        self.times = []
        
    def record_lfp(self, state_monitor):
        """
        Compute LFP from state monitor data
        state_monitor should have recorded IsynE and IsynI (or gE, gI)
        """
        # Get the synaptic currents for all E neurons
        if hasattr(state_monitor, 'IsynE'):
            # Direct current recording
            I_E = np.abs(state_monitor.IsynE / pA)  # Convert to pA and take absolute
            I_I_PV = np.abs(state_monitor.IsynIPV / pA)
            I_I_SOM = np.abs(state_monitor.IsynISOM / pA)
            I_I_VIP = np.abs(state_monitor.IsynIVIP / pA)
            I_I = I_I_PV + I_I_SOM + I_I_VIP
        else:
            # Reconstruct from conductances if currents not directly available
            # This won't work perfectly but is an approximation
            print("Warning: Using conductance approximation for LFP")
            I_E = np.abs(state_monitor.gE / nS)
            I_I = np.abs(state_monitor.gI / nS)
        
        # Sum across all pyramidal neurons to get population LFP
        # Shape: (n_neurons, n_timepoints) -> (n_timepoints,)
        lfp = np.sum(I_E, axis=0) + np.sum(I_I, axis=0)
        
        return lfp, state_monitor.t / ms

def compute_layer_lfps(column, all_monitors):
    """
    Compute LFP for each layer in the column
    Returns dict of {layer_name: (lfp_signal, times)}
    """
    layer_lfps = {}
    
    for layer_name, monitors in all_monitors.items():
        # Get the E population state monitor
        e_state_key = [k for k in monitors.keys() if 'E_state' in k]
        
        if not e_state_key:
            print(f"Warning: No E state monitor found for {layer_name}")
            continue
            
        state_mon = monitors[e_state_key[0]]
        
        # Create LFP recorder for this layer
        layer = column.get_layer(layer_name)
        recorder = LFPRecorder(layer)
        
        # Compute LFP
        lfp, times = recorder.record_lfp(state_mon)
        layer_lfps[layer_name] = (lfp, times)
        
        print(f"Computed LFP for {layer_name}: {len(lfp)} timepoints")
    
    return layer_lfps

def plot_layer_lfps(layer_lfps, figsize=(15, 10)):
    """
    Plot LFPs from all layers
    """
    n_layers = len(layer_lfps)
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=True)
    
    if n_layers == 1:
        axes = [axes]
    
    for idx, (layer_name, (lfp, times)) in enumerate(sorted(layer_lfps.items())):
        ax = axes[idx]
        
        # High-pass filter at 1 Hz (as in the paper)
        from scipy import signal
        fs = 1000 / np.mean(np.diff(times))  # Sampling frequency in Hz
        sos = signal.butter(4, 1, 'highpass', fs=fs, output='sos')
        lfp_filtered = signal.sosfilt(sos, lfp)
        
        ax.plot(times, lfp_filtered, 'k', linewidth=0.5)
        ax.set_ylabel(f'{layer_name}\nLFP (a.u.)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add scale bar
        if idx == 0:
            ax.set_title('Layer-wise Local Field Potentials', fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel('Time (ms)', fontsize=10)
    plt.tight_layout()
    return fig


def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']
    
    print("Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)
    
    all_monitors = column.get_all_monitors()
    
    print("Running baseline simulation...")
    column.network.run(500*ms)
    
    ############## CREATING STIM INPUT ##############
    print("Adding stimulus input...")
    w_ext = CONFIG['synapses']['Q']['EXT']
    
    # L4 stimulus
    L4 = column.layers['L4']
    cfg_L4 = CONFIG['layers']['L4']
    L4_E_grp = L4.neuron_groups['E']
    N_stim_E = int(cfg_L4['poisson_inputs']['E']['N'] / 2)
    stim_rate_E = 8*Hz
    L4_E_stim = PoissonInput(L4_E_grp, 'gE', N=N_stim_E, rate=stim_rate_E, weight=w_ext)
    
    L4_PV_grp = L4.neuron_groups['PV']
    N_stim_PV = int(cfg_L4['poisson_inputs']['PV']['N'] / 2)
    stim_rate_PV = 6*Hz
    L4_PV_stim = PoissonInput(L4_PV_grp, 'gE', N=N_stim_PV, rate=stim_rate_PV, weight=w_ext)
    
    # L6 stimulus
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    stim_rate_L6_E = 6*Hz
    L6_E_stim = PoissonInput(L6_E_grp, 'gE', N=N_stim_L6_E, rate=stim_rate_L6_E, weight=w_ext)
    
    N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'])
    stim_rate_L6_PV = 4*Hz
    L6_PV_stim = PoissonInput(L6.neuron_groups['PV'], 'gE', N=N_stim_L6_PV, rate=stim_rate_L6_PV, weight=w_ext)
    
    column.network.add(L4_E_stim, L4_PV_stim, L6_E_stim, L6_PV_stim)
    
    print(" Running stimulation simulation...")
    column.network.run(500*ms)
    
    print("Simulation complete")
    
    # Organize monitors
    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    
    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        state_monitors[layer_name] = {k: v for k, v in monitors.items() if 'state' in k}
        rate_monitors[layer_name] = {k: v for k, v in monitors.items() if 'rate' in k}
    
    ############## LFP ANALYSIS ##############
    print(" Computing LFPs for each layer...")
    layer_lfps = compute_layer_lfps(column, all_monitors)
    
    print("Plotting LFPs...")
    fig_lfp = plot_layer_lfps(layer_lfps)
    

    
    print(" All analyses complete!")
    plt.show()

if __name__ == "__main__":
    main()
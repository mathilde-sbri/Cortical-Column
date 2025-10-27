import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config_test import CONFIG
from src.column import CorticalColumn
from src.visualization import NetworkVisualizer, plot_layer_psth, plot_layer_spectrograms
from src.analysis import LFPAnalysis

def add_heterogeneity_to_layer(layer, config):
    for pop_name, neuron_group in layer.neuron_groups.items():
        n = len(neuron_group)
        base = config['intrinsic_params'][pop_name]
        
        neuron_group.C = base['C'] * np.abs(1 + np.random.randn(n) * 0.15)
        neuron_group.gL = base['gL'] * np.abs(1 + np.random.randn(n) * 0.12)
        neuron_group.tauw = base['tauw'] * np.abs(1 + np.random.randn(n) * 0.15)
        neuron_group.b = base['b'] * np.abs(1 + np.random.randn(n) * 0.20)
        neuron_group.a = base['a'] * np.abs(1 + np.random.randn(n) * 0.15)


def detailed_activity_check(layer_name, state_monitor, spike_monitor, simulation_time, time_window=None):
    print(f"\n{'='*50}")
    print(f"Layer: {layer_name}")
    if time_window:
        print(f"Time window: {time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms")
    print(f"{'='*50}")
    
    for pop in ['E', 'PV', 'SOM', 'VIP']:
        spike_key = f'{pop}_spikes'
        state_key = f'{pop}_state'
        
        if spike_key not in spike_monitor:
            continue
            
        spike_mon = spike_monitor[spike_key]
        state_mon = state_monitor[state_key]
        
        n_neurons = len(spike_mon.source)
        
        if time_window:
            mask = (spike_mon.t >= time_window[0]) & (spike_mon.t < time_window[1])
            spike_times = spike_mon.t[mask]
            spike_indices = spike_mon.i[mask]
            analysis_duration = (time_window[1] - time_window[0]) / second
        else:
            spike_times = spike_mon.t
            spike_indices = spike_mon.i
            analysis_duration = simulation_time / second
        
        n_spikes = len(spike_times)
        
        if n_spikes > 0:
            rate = n_spikes / (n_neurons * analysis_duration)
            active_neurons = len(np.unique(spike_indices))
            pct_active = 100 * active_neurons / n_neurons
        else:
            rate = 0
            active_neurons = 0
            pct_active = 0
        
        gE_mean = np.mean(state_mon.gE/nS)
        gE_max = np.max(state_mon.gE/nS)
        gI_mean = np.mean(state_mon.gI/nS)
        gI_max = np.max(state_mon.gI/nS)
        
        print(f"\n{pop} population (N={n_neurons}):")
        print(f"  Firing rate: {rate:.2f} Hz")
        print(f"  Active neurons: {active_neurons}/{n_neurons} ({pct_active:.1f}%)")
        print(f"  Total spikes: {n_spikes}")
        print(f"  gE: mean={gE_mean:.4f} nS, max={gE_max:.3f} nS")
        print(f"  gI: mean={gI_mean:.4f} nS, max={gI_max:.3f} nS")
        
        if gE_mean < 0.01 and gI_mean < 0.01:
            print(f"  ⚠️  WARNING: Very low synaptic activity!")
        if pct_active < 10:
            print(f"  ⚠️  WARNING: <10% of neurons are active!")


def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)
    
    all_monitors = column.get_all_monitors()
    
    column.network.run(1000*ms)

    w_ext = CONFIG['synapses']['Q']['EXT']
    

    L4 = column.layers['L4']
    cfg_L4 = CONFIG['layers']['L4']
    
    L4_E_grp = L4.neuron_groups['E']

    
    N_stim_E = int(cfg_L4['poisson_inputs']['E']['N'] * 2)
    stim_rate_E = 10*Hz  


    
    L4_E_stim = PoissonInput(L4_E_grp, 'gE', 
                             N=N_stim_E, rate=stim_rate_E, weight=w_ext)
    
    L4_PV_grp = L4.neuron_groups['PV']
    N_stim_PV = int(cfg_L4['poisson_inputs']['PV']['N'] * 1.5)
    stim_rate_PV = 8*Hz
    
    L4_PV_stim = PoissonInput(L4_PV_grp, 'gE', 
                              N=N_stim_PV, rate=stim_rate_PV, weight=w_ext)
    
    # L6 receives weaker but concurrent thalamic input
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    
    L6_E_grp = L6.neuron_groups['E']
    
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'] * 1.3)
    stim_rate_L6_E = 8*Hz
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE',
                             N=N_stim_L6_E, rate=stim_rate_L6_E, weight=w_ext)
    
    N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'] * 1.2)
    stim_rate_L6_PV = 6*Hz
    
    L6_PV_stim = PoissonInput(L6.neuron_groups['PV'], 'gE',
                              N=N_stim_L6_PV, rate=stim_rate_L6_PV, weight=w_ext)
    
    column.network.add(L4_E_stim, L4_PV_stim, L6_E_stim, L6_PV_stim)


    column.network.run(3000*ms)

    print("Simulation complete")
    

    print("\n" + "="*60)
    print("NETWORK ACTIVITY ANALYSIS")
    print("="*60)
    
    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    
    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'state' in k
        }
        rate_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }
    
    print("\n" + "─"*60)
    print("SPONTANEOUS ACTIVITY (0-1000ms)")
    print("─"*60)
    for layer in ['L4', 'L23', 'L5', 'L6']:
        detailed_activity_check(layer, state_monitors[layer], 
                              spike_monitors[layer], 1000*ms,
                              time_window=(0*ms, 1000*ms))
    
    print("\n" + "─"*60)
    print("STIMULUS PERIOD (1000-4000ms)")
    print("─"*60)
    for layer in ['L4', 'L23', 'L5', 'L6']:
        detailed_activity_check(layer, state_monitors[layer], 
                              spike_monitors[layer], 3000*ms,
                              time_window=(1000*ms, 4000*ms))
    

    
    fig1 = NetworkVisualizer.plot_raster(spike_monitors, CONFIG['layers'])
    fig2 = NetworkVisualizer.plot_lfp(state_monitors, CONFIG['layers'])
    fig3 = NetworkVisualizer.plot_power_spectra(state_monitors, CONFIG['layers'])
    fig4 = NetworkVisualizer.plot_spectrogram(state_monitors, CONFIG['layers'])
    fig5 = NetworkVisualizer.plot_rate(rate_monitors, CONFIG['layers'])
    
    
    plt.show()


if __name__ == "__main__":
    main()
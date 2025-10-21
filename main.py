"""
Main simulation runner
"""
import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config_test import CONFIG
from src.column import CorticalColumn
from src.visualization import NetworkVisualizer
from src.analysis import LFPAnalysis

def main():
  
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']
    
    print("Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    all_monitors = column.get_all_monitors()
    
    print("Running simulation...")
    column.network.run(1000*ms)

    L4 = column.layers['L4']
    cfg = CONFIG['layers']['L4']
    NE = cfg['neuron_counts']['E']
    N = cfg['poisson_inputs']['E']['N'] 
    w = CONFIG['synapses']['Q']['EXT']

    
    E_grp = L4.neuron_groups['E']
    E20 = PoissonInput(E_grp, 'gE', N=N, rate=10*Hz, weight=w)

    L4.poisson_inputs['E_20'] = E20
    column.network.add(E20)

    NEpv = cfg['neuron_counts']['PV']
    Npv = cfg['poisson_inputs']['PV']['N'] 
    w = CONFIG['synapses']['Q']['EXT']
    PV20 = PoissonInput(L4.neuron_groups['PV'], 'gE', N=Npv, rate=10*Hz, weight=w)
    L4.poisson_inputs['PV_20'] = PV20
    column.network.add(PV20)

    column.network.run(CONFIG['simulation']['SIMULATION_TIME'] - 1000*ms)

    print("Simulation complete")
    
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
    
    def detailed_activity_check(layer_name, state_monitor, spike_monitor, simulation_time):
        print(f"\n{'='*50}")
        print(f"Layer: {layer_name}")
        print(f"{'='*50}")
        
        for pop in ['E', 'PV', 'SOM', 'VIP']:
            spike_key = f'{pop}_spikes'
            state_key = f'{pop}_state'
            
            if spike_key not in monitors:
                continue
                
            spike_mon = spike_monitor[spike_key]
            state_mon = state_monitor[state_key]
            
            n_neurons = len(spike_mon.source)
            n_spikes = len(spike_mon.t)
            
            if n_spikes > 0:
                rate = n_spikes / (n_neurons * simulation_time/second)
                active_neurons = len(np.unique(spike_mon.i))
                pct_active = 100 * active_neurons / n_neurons
            else:
                rate = 0
                active_neurons = 0
                pct_active = 0
            
            # Check conductance statistics over time
            gE_mean = np.mean(state_mon.gE/nS)
            gE_max = np.max(state_mon.gE/nS)
            gI_mean = np.mean(state_mon.gI/nS)
            gI_max = np.max(state_mon.gI/nS)
            
            print(f"\n{pop} population (N={n_neurons}):")
            print(f"  Firing rate: {rate:.2f} Hz")
            print(f"  Active neurons: {active_neurons}/{n_neurons} ({pct_active:.1f}%)")
            print(f"  gE: mean={gE_mean:.4f} nS, max={gE_max:.3f} nS")
            print(f"  gI: mean={gI_mean:.4f} nS, max={gI_max:.3f} nS")
            
            if gE_mean < 0.01 and gI_mean < 0.01:
                print(f"  ⚠️  WARNING: Very low synaptic activity!")
            if pct_active < 10:
                print(f"  ⚠️  WARNING: <10% of neurons are active!")

    detailed_activity_check('L5', state_monitors['L5'], spike_monitors['L5'], 4000*ms)
    # E_rate = rate_monitors["L4"]["E_rate"]

    # t_ms = E_rate.t / ms
    # r_hz = E_rate.smooth_rate(window='flat', width=10.1*ms) / Hz

    # plt.figure()
    # plt.plot(t_ms, r_hz, linewidth=3)
    # plt.xlabel('Time (ms)')

    # plt.ylabel('Rate (Hz)')
    # plt.tight_layout()
    # plt.show()
        
    fig1 = NetworkVisualizer.plot_raster(spike_monitors, CONFIG['layers'])

    
    fig2 = NetworkVisualizer.plot_lfp(state_monitors, CONFIG['layers'])
    
    fig3 = NetworkVisualizer.plot_power_spectra(state_monitors, CONFIG['layers'])
    fig4 = NetworkVisualizer.plot_power_spectra_loglog(state_monitors, CONFIG['layers'])
    fig5 = NetworkVisualizer.plot_rate(rate_monitors, CONFIG['layers'])



    
    plt.show()

if __name__ == "__main__":
    main()
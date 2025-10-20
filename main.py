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
    E20 = PoissonInput(E_grp, 'gE', N=N, rate=20*Hz, weight=w)

    L4.poisson_inputs['E_20'] = E20
    column.network.add(E20)

    NEpv = cfg['neuron_counts']['PV']
    Npv = cfg['poisson_inputs']['PV']['N'] 
    w = CONFIG['synapses']['Q']['EXT']
    PV20 = PoissonInput(L4.neuron_groups['PV'], 'gE', N=Npv, rate=20*Hz, weight=w)
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
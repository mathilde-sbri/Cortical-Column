"""
Main simulation runner
"""
import numpy as np
import brian2 as b2
from brian2 import *

from config.config import CONFIG
from src.column import CorticalColumn
from src.visualization import NetworkVisualizer
from src.analysis import LFPAnalysis

def main():
  
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']
    
    print("Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    # Not sure this is actually useful, to check
    all_monitors = column.get_all_monitors()
    
    print("Running simulation...")
    column.network.run(CONFIG['simulation']['SIMULATION_TIME'])
    print("Simulation complete")
    
    spike_monitors = {}
    state_monitors = {}
    
    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'state' in k
        }
    

    
    fig1 = NetworkVisualizer.plot_raster(spike_monitors, CONFIG['layers'])
    fig_rates = NetworkVisualizer.plot_E_population_rates(spike_monitors, CONFIG['layers'])
    fig_psd   = NetworkVisualizer.plot_E_population_psd(spike_monitors, CONFIG['layers'])
    fig_gamma = NetworkVisualizer.plot_gamma_peaks(spike_monitors, CONFIG['layers'])

    
    fig2 = NetworkVisualizer.plot_lfp(state_monitors, CONFIG['layers'])
    
    fig3 = NetworkVisualizer.plot_power_spectra(state_monitors, CONFIG['layers'])
    
    plt.show()

if __name__ == "__main__":
    main()

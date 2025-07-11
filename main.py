"""
Main simulation runner
"""
import numpy as np
import brian2 as b2
from brian2 import *

from src.parameters import *
from src.column import CorticalColumn
from src.visualization import NetworkVisualizer
from src.analysis import LFPAnalysis
from config.layer_configs import LAYER_CONFIGS

def main():
  
    np.random.seed(RANDOM_SEED)
    b2.start_scope()
    b2.defaultclock.dt = DT
    
    print("Creating cortical column...")
    column = CorticalColumn(column_id=0)
    
    # Not sure this is actually useful, to check
    all_monitors = column.get_all_monitors()
    
    print("Running simulation...")
    column.network.run(SIMULATION_TIME)
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
    

    
    fig1 = NetworkVisualizer.plot_raster(spike_monitors, LAYER_CONFIGS)
    
    fig2 = NetworkVisualizer.plot_lfp(state_monitors, LAYER_CONFIGS)
    
    fig3 = NetworkVisualizer.plot_power_spectra(state_monitors, LAYER_CONFIGS)
    
    plt.show()

if __name__ == "__main__":
    main()
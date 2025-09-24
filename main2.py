"""
Main simulation runner
"""
import numpy as np
import brian2 as b2
from brian2 import *

from config.config_1layer import CONFIG
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

    prot = CONFIG['light_protocol'] 
    layer = column.layers['L23']
    vip=layer.neuron_groups['VIP']
    vip.I_bias_VIP = prot['I_bias_VIP']
    vip.I_light    = prot['I_light']
    vip.light_level = 0.0
    print("Running simulation...")

    column.network.run(prot['onset'])

    vip.light_level = 1.0
    column.network.run(prot['offset'] - prot['onset'])

    vip.light_level = 0.0
    
    column.network.run(CONFIG['simulation']['SIMULATION_TIME'] - prot['offset'])
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
    

    
    fig1 = NetworkVisualizer.plot_raster(spike_monitors, CONFIG['layers'])

    
    fig2 = NetworkVisualizer.plot_lfp(state_monitors, CONFIG['layers'])
    
    fig3 = NetworkVisualizer.plot_power_spectra(state_monitors, CONFIG['layers'])
    fig4 = NetworkVisualizer.plot_rate(rate_monitors, CONFIG['layers'])

    fig5 = NetworkVisualizer.plot_spectrogram(
        state_monitors, CONFIG['layers'],
        fmax=100, win_ms=250, step_ms=25, light_window=(2.0, 4.0)
    )

    fig6 = NetworkVisualizer.plot_peak_freq_track(
        state_monitors, CONFIG['layers'],
        f_gamma=(20, 80), fmax=100, win_ms=250, step_ms=25, light_window=(2.0, 4.0)
    )

    
    plt.show()

if __name__ == "__main__":
    main()

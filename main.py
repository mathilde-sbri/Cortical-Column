import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config_test3 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *



def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)
    
    all_monitors = column.get_all_monitors()
    
    column.network.run(500*ms)

    # ############## CREATING STIM INPUT ##############
    # w_ext = CONFIG['synapses']['Q']['EXT']
    

    # L4 = column.layers['L4']
    # cfg_L4 = CONFIG['layers']['L4']
    
    # L4_E_grp = L4.neuron_groups['E']

    
    # N_stim_E = int(cfg_L4['poisson_inputs']['E']['N'] /2)
    # stim_rate_E = 8*Hz  


    
    # L4_E_stim = PoissonInput(L4_E_grp, 'gE', 
    #                          N=N_stim_E, rate=stim_rate_E, weight=w_ext)
    
    # L4_PV_grp = L4.neuron_groups['PV']
    # N_stim_PV = int(cfg_L4['poisson_inputs']['PV']['N']/2)
    # stim_rate_PV = 6*Hz
    
    # L4_PV_stim = PoissonInput(L4_PV_grp, 'gE', 
    #                           N=N_stim_PV, rate=stim_rate_PV, weight=w_ext)
    
    # # L6 receives weaker but concurrent thalamic input
    # L6 = column.layers['L6']
    # cfg_L6 = CONFIG['layers']['L6']
    
    # L6_E_grp = L6.neuron_groups['E']
    
    # N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'] )
    # stim_rate_L6_E = 6*Hz
    
    # L6_E_stim = PoissonInput(L6_E_grp, 'gE',
    #                          N=N_stim_L6_E, rate=stim_rate_L6_E, weight=w_ext)
    
    # N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'] )
    # stim_rate_L6_PV = 4*Hz
    
    # L6_PV_stim = PoissonInput(L6.neuron_groups['PV'], 'gE',
    #                           N=N_stim_L6_PV, rate=stim_rate_L6_PV, weight=w_ext)
    
    # column.network.add(L4_E_stim, L4_PV_stim, L6_E_stim, L6_PV_stim)

    #################################################


    # column.network.run(500*ms)

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

    
    fig1 = plot_raster(spike_monitors, CONFIG['layers'])
    fig2 = plot_lfp(state_monitors, CONFIG['layers'])
    fig3 = plot_power_spectra(state_monitors, CONFIG['layers'])
    fig5 = plot_rate(rate_monitors, CONFIG['layers'])
    
    
    plt.show()


if __name__ == "__main__":
    main()
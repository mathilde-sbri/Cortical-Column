import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *


def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    # for layer_name, layer in column.layers.items():
    #     add_heterogeneity_to_layer(layer, CONFIG) # optional
    
    all_monitors = column.get_all_monitors()
    
    baseline_time = 1000
    column.network.run(baseline_time*ms)

    ##############  FEEDFORWARD STIM INPUT ##############
    # w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    

    # L4C = column.layers['L4C']
    # cfg_L4C = CONFIG['layers']['L4C']
    
    # L4C_E_grp = L4C.neuron_groups['E']

    
    # N_stim_E = int(cfg_L4C['poisson_inputs']['E']['N'])
    # stim_rate_E = 30*Hz  


    
    # L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA', 
    #                          N=N_stim_E, rate=stim_rate_E, weight=w_ext_AMPA/2)
  
    # L4C_PV_grp = L4C.neuron_groups['PV']
    # N_stim_PV = int(cfg_L4C['poisson_inputs']['PV']['N'])
    # stim_rate_PV = 30*Hz
    
    # L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA', 
    #                           N=N_stim_PV, rate=stim_rate_PV, weight=w_ext_AMPA)
    
    
    
    # # L6 receives weaker halamic input
    # L6 = column.layers['L6']
    # cfg_L6 = CONFIG['layers']['L6']
    
    # L6_E_grp = L6.neuron_groups['E']
    
    # N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'] )
    # stim_rate_L6_E = 3*Hz
    
    # L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
    #                          N=N_stim_L6_E, rate=stim_rate_L6_E, weight=w_ext_AMPA/3)
    

    

    # column.network.add(L4C_E_stimAMPA, L4C_PV_stim, L6_E_stim)

    ##############################################

     ############## CREATING FEEDBACK STIM INPUT ##############
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    

    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    
    L6_SOM_grp = L6.neuron_groups['SOM']

    
    N_stim_SOM = int(cfg_L6['poisson_inputs']['SOM']['N']/2)
    stim_rate_SOM = 5*Hz  


    
    L6_SOM_stimAMPA = PoissonInput(L6_SOM_grp, 'gE_AMPA', 
                             N=N_stim_SOM, rate=stim_rate_SOM, weight=w_ext_AMPA)

    
    column.network.add(L6_SOM_stimAMPA)

    column.network.run(500*ms)

    print("Simulation complete")
    
    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    neuron_groups = {}
    
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
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups
    
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

    # electrode_positions = CONFIG['electrode_positions']

    # print("computing LFP using kernel method")
    # lfp_signals, time_array = calculate_lfp_kernel_method(
    #     spike_monitors, 
    #     neuron_groups,
    #     CONFIG['layers'],
    #     electrode_positions,
    #     fs=10000,
    #     sim_duration_ms=1500
    # )

    # bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
    #     lfp_signals, 
    #     electrode_positions
    # )


    # fig_comparison = plot_lfp_comparison(lfp_signals, bipolar_signals, time_array,
    #                                      electrode_positions, channel_labels, 
    #                                      channel_depths, time_range=(800, 1200))


    fig_raster = plot_raster(spike_monitors, CONFIG['layers'])

    fig_power = plot_power_spectra(state_monitors, CONFIG['layers'])

    fig_rate = plot_rate(rate_monitors, CONFIG['layers'], baseline_time,
                 smooth_window=15*ms, 
                 ylim_max=80,      
                 show_stats=True)  

    
    plt.show()


if __name__ == "__main__":
    main()
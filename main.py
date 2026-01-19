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

    
    baseline_time = 1000 # In ms, time during which to run the baseline simulation
    stimuli_time = 500 # In ms, time during which to run the simulation after adding the stimuli

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    # for layer_name, layer in column.layers.items():
    #     add_heterogeneity_to_layer(layer, CONFIG) # optional
    
    all_monitors = column.get_all_monitors()
    
    
    

    ##############  FEEDBACK STIM INPUT ##############
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    
    L23 = column.layers['L23']
    cfg_L23 = CONFIG['layers']['L23']
    L23_E_grp = L23.neuron_groups['E']
    N_fb_L23 = int(cfg_L23['poisson_inputs']['E']['N'])
    
    L23_feedback = PoissonInput(L23_E_grp, 'gE_AMPA',
                                N=N_fb_L23, 
                                rate=3*Hz, 
                                weight=w_ext_AMPA*0.8)
    column.network.add(L23_feedback)
    
    L23_SOM_grp = L23.neuron_groups['SOM']
    N_fb_L23_SOM = int(cfg_L23['poisson_inputs']['SOM']['N'])
    L23_SOM_feedback = PoissonInput(L23_SOM_grp, 'gE_AMPA',
                                    N=N_fb_L23_SOM,
                                    rate=2*Hz,
                                    weight=w_ext_AMPA*0.6)
    column.network.add(L23_SOM_feedback)
   
    L5 = column.layers['L5']
    cfg_L5 = CONFIG['layers']['L5']
    L5_E_grp = L5.neuron_groups['E']
    N_fb_L5 = int(cfg_L5['poisson_inputs']['E']['N'])
    
    L5_feedback = PoissonInput(L5_E_grp, 'gE_AMPA',
                                N=N_fb_L5,
                                rate=4*Hz,
                                weight=w_ext_AMPA)
    column.network.add(L5_feedback)
    
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    
    L6_E_grp = L6.neuron_groups['E']
    N_fb_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    L6_E_feedback = PoissonInput(L6_E_grp, 'gE_AMPA',
                                 N=N_fb_L6_E,
                                 rate=3*Hz,
                                 weight=w_ext_AMPA)
    
    L6_SOM_grp = L6.neuron_groups['SOM']
    N_stim_SOM = int(cfg_L6['poisson_inputs']['SOM']['N'])
    L6_SOM_feedback = PoissonInput(L6_SOM_grp, 'gE_AMPA', 
                                   N=N_stim_SOM, 
                                   rate=2*Hz,  
                                   weight=w_ext_AMPA)
    
    column.network.add(L6_E_feedback, L6_SOM_feedback)

    column.network.run(baseline_time*ms)

    
    L4C = column.layers['L4C']
    cfg_L4C = CONFIG['layers']['L4C']
    
    L4C_E_grp = L4C.neuron_groups['E']
    N_stim_E = int(cfg_L4C['poisson_inputs']['E']['N'])
    stim_rate_E = 10*Hz  
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA', 
                                  N=N_stim_E, 
                                  rate=stim_rate_E, 
                                  weight=w_ext_AMPA/2)  
    
    L4C_PV_grp = L4C.neuron_groups['PV']
    N_stim_PV = int(cfg_L4C['poisson_inputs']['PV']['N'])
    stim_rate_PV = 10*Hz 
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA', 
                               N=N_stim_PV, 
                               rate=stim_rate_PV, 
                               weight=w_ext_AMPA)  
    
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    stim_rate_L6_E = 3*Hz  
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=N_stim_L6_E, 
                             rate=stim_rate_L6_E, 
                             weight=w_ext_AMPA/3)
    
    column.network.add(L4C_E_stimAMPA, L4C_PV_stim, L6_E_stim)

    ##############################################

    #  ############## CREATING FEEDBACK STIM INPUT ##############
    # w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    

    # L6 = column.layers['L6']
    # cfg_L6 = CONFIG['layers']['L6']
    
    # L6_SOM_grp = L6.neuron_groups['SOM']

    
    # N_stim_SOM = int(cfg_L6['poisson_inputs']['SOM']['N']/2)
    # stim_rate_SOM = 5*Hz  


    
    # L6_SOM_stimAMPA = PoissonInput(L6_SOM_grp, 'gE_AMPA', 
    #                          N=N_stim_SOM, rate=stim_rate_SOM, weight=w_ext_AMPA)

    
    # column.network.add(L6_SOM_stimAMPA)

    column.network.run(stimuli_time*ms)

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


    fig_raster = plot_raster(spike_monitors, baseline_time, stimuli_time, CONFIG['layers'])

    # fig_power_lfp = plot_lfp_power_comparison(
    #                     state_monitors, 
    #                     CONFIG['layers'],
    #                     baseline_time=baseline_time,
    #                     pre_stim_duration=500,
    #                     post_stim_duration=500
    #                 )


    fig_rate = plot_rate(rate_monitors, CONFIG['layers'], baseline_time, stimuli_time,
                 smooth_window=15*ms, 
                 ylim_max=80,      
                 show_stats=True)  

    
    plt.show()


if __name__ == "__main__":
    main()
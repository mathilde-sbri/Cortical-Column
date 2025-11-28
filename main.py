import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config_test2 import CONFIG
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
    
    column.network.run(1000*ms)

    # ############## CREATING STIM INPUT ##############
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    w_ext_NMDA = CONFIG['synapses']['Q']['EXT_NMDA']
    

    L4C = column.layers['L4C']
    cfg_L4C = CONFIG['layers']['L4C']
    
    L4C_E_grp = L4C.neuron_groups['E']

    
    N_stim_E = int(cfg_L4C['poisson_inputs']['E']['N'])
    stim_rate_E = 10*Hz  


    
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA', 
                             N=N_stim_E, rate=stim_rate_E, weight=w_ext_AMPA)
    L4C_E_stimNMDA = PoissonInput(L4C_E_grp, 'gE_NMDA', 
                             N=N_stim_E, rate=stim_rate_E, weight=w_ext_NMDA)
    
    L4C_PV_grp = L4C.neuron_groups['PV']
    N_stim_PV = int(cfg_L4C['poisson_inputs']['PV']['N'])
    stim_rate_PV = 3*Hz
    
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA', 
                              N=N_stim_PV, rate=stim_rate_PV, weight=w_ext_AMPA)
    
    # L6 receives weaker but concurrent thalamic input
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    
    L6_E_grp = L6.neuron_groups['E']
    
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'] )
    stim_rate_L6_E = 8*Hz
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=N_stim_L6_E, rate=stim_rate_L6_E, weight=w_ext_AMPA)
    
    N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'] )
    stim_rate_L6_PV = 4*Hz
    
    L6_PV_stim = PoissonInput(L6.neuron_groups['PV'], 'gE_AMPA',
                              N=N_stim_L6_PV, rate=stim_rate_L6_PV, weight=w_ext_AMPA)
    
 

    ################################################

    column.network.add(L4C_E_stimAMPA, L4C_E_stimNMDA, L4C_PV_stim,  L6_E_stim, L6_PV_stim)
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

    electrode_positions = [
        (0, 0, -0.94),
        (0, 0, -0.79),
        (0, 0, -0.64),
        (0, 0, -0.49),
        (0, 0, -0.34),
        (0, 0, -0.19),
        (0, 0, -0.04),
        (0, 0, 0.1),
        (0, 0, 0.26),
        (0, 0, 0.4),
        (0, 0, 0.56),
        (0, 0, 0.7), 
        (0, 0, 0.86),
        (0, 0, 1.0),  
        (0, 0, 1.16),  
        (0, 0, 1.3),   

    ]
    
    # print("LFP using kernel method")
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors, 
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        fs=10000,
        sim_duration_ms=2000
    )
    # print("Computing CSD from monopolar LFP...")
    # csd, csd_depths, csd_sort_idx = compute_csd_from_lfp(
    #     lfp_signals,
    #     electrode_positions,
    #     sigma=0.3,     
    #     vaknin=True
    # )

    


    # print(" bipolar LFP now")
    # bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
    #     lfp_signals, 
    #     electrode_positions
    # )

    fig1 = plot_raster(spike_monitors, CONFIG['layers'])
    # fig2 = plot_lfp(state_monitors, CONFIG['layers'])  # old Mazzoni method
    
    # # kernel method plots
    # fig_kernel = plot_lfp_kernel(lfp_signals, time_array, electrode_positions)
    # fig_bipolar = plot_bipolar_lfp(bipolar_signals, channel_labels, channel_depths, 
    #                                time_array, time_range=(0, 1000))
    # fig_comparison = plot_lfp_comparison(lfp_signals, bipolar_signals, time_array,
    #                                     electrode_positions, channel_labels, 
    #                                     channel_depths, time_range=(400, 600))
    # fig_bipolar_psd = plot_bipolar_power_spectra(bipolar_signals, channel_labels, channel_depths,
    #                                              time_array, fmax=100)
    
    # fig3 = plot_power_spectra(state_monitors, CONFIG['layers'])
    fig4 = plot_power_spectra_stim(
        lfp_signals, 
        time_array,
        electrode_positions,
        stim_time=1000,
        pre_window=300,
        post_window=300
    )

    fig5 = plot_rate(rate_monitors, CONFIG['layers'], 
                 smooth_window=15*ms, 
                 ylim_max=80,      
                 show_stats=True)  
    # fig_csd = plot_csd(
    #     csd,
    #     time_array,
    #     csd_depths,
    #     time_range=(500, 800),  
    #     figsize=(8, 10),
    #     cmap='seismic'
    # )
    # fig_rate_fft = plot_rate_fft(rate_monitors, fmax=100)
    # fig_wvlt = plot_wavelet_transform(bipolar_signals, channel_labels, channel_depths, 
    #                                time_array, time_range=(300, 800))

    
    plt.show()


if __name__ == "__main__":
    main()
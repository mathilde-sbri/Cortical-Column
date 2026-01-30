"""
Main simulation script with comprehensive diagnostics.
Replace your main.py with this, or copy the diagnostic sections.
"""

import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *

# Import diagnostic functions (copy diagnostic_plots.py to your project)
from diagnostic_plots import (plot_all_diagnostics, print_quick_summary,
                              plot_ei_balance_diagnostics, plot_spectral_diagnostics,
                              plot_connectivity_analysis, plot_temporal_dynamics)


def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    baseline_time = 2000  # In ms
    stimuli_time = 1500   # In ms

    print("Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    all_monitors = column.get_all_monitors()
    
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']

    # ============ BASELINE PERIOD ============
    column.network.run(baseline_time * ms)

    # ============ STIMULUS PERIOD ============
    L4C = column.layers['L4C']
    cfg_L4C = CONFIG['layers']['L4C']
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    
    L4C_E_grp = L4C.neuron_groups['E']
    N_stim_E = int(cfg_L4C['poisson_inputs']['E']['N'])
    stim_rate_E = 18*Hz  
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA', 
                                  N=N_stim_E, 
                                  rate=stim_rate_E, 
                                  weight=w_ext_AMPA)  
    
    L4C_PV_grp = L4C.neuron_groups['PV']
    N_stim_PV = int(cfg_L4C['poisson_inputs']['PV']['N'])
    stim_rate_PV = 18*Hz 
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA', 
                               N=N_stim_PV, 
                               rate=stim_rate_PV, 
                               weight=w_ext_AMPA)  
    
    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    stim_rate_L6_E = 8*Hz  
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=N_stim_L6_E, 
                             rate=stim_rate_L6_E, 
                             weight=w_ext_AMPA/2)
    
    column.network.add(L4C_E_stimAMPA, L4C_PV_stim, L6_E_stim)

    column.network.run(stimuli_time * ms)

    print("Simulation complete")
    
    # ============ ORGANIZE MONITORS ============
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

    # ============ STANDARD PLOTS ============
    electrode_positions = CONFIG['electrode_positions']

    print("Computing LFP using kernel method...")
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        fs=10000,
        sim_duration_ms=baseline_time + stimuli_time
    )

    print("Computing bipolar LFP...")
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals,
        electrode_positions
    )

    fig_raster = plot_raster(spike_monitors, baseline_time, stimuli_time, CONFIG['layers'])

    fig_power_lfp = plot_lfp_power_comparison_kernel(
                        lfp_signals,
                        time_array,
                        electrode_positions,
                        baseline_time=baseline_time,
                        pre_stim_duration=1000,
                        post_stim_duration=1000,
                        transient_skip=500
                    )

    fig_power_bipolar = plot_bipolar_power_comparison_kernel(
                        bipolar_signals,
                        channel_labels,
                        channel_depths,
                        time_array,
                        baseline_time=baseline_time,
                        pre_stim_duration=1000,
                        post_stim_duration=1000,
                        transient_skip=500
                    )

    fig_rate = plot_rate(rate_monitors, CONFIG['layers'], baseline_time, stimuli_time,
                 smooth_window=15*ms, 
                 ylim_max=80,      
                 show_stats=True)

    # ============ DIAGNOSTIC PLOTS ============
    
    print("\n" + "="*70)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*70)
    
    # Quick text summary
    print_quick_summary(spike_monitors, CONFIG, baseline_time, stimuli_time)
    
    # Generate all diagnostic figures
    figs = plot_all_diagnostics(
        spike_monitors, 
        state_monitors, 
        CONFIG, 
        baseline_time, 
        stimuli_time,
        output_dir='diagnostics'
    )
    
    plt.show()


if __name__ == "__main__":
    main()
import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config_test import CONFIG
from src.column import CorticalColumn
import cleo
from cleo import ephys
from src.analysis import *
from src.cleo_plots import *
import sys
import pickle

def run_single_trial(seed=None, plot=False, save_path=None):
    """
    Run a single trial of the simulation
    
    Args:
        seed: Random seed for this trial (if None, uses CONFIG seed)
        plot: Whether to generate plots
        save_path: Path to save results (if None, doesn't save)
    
    Returns:
        Dictionary with trial results
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    print(f"Creating cortical column (seed={seed})...")
    column = CorticalColumn(column_id=0, config=CONFIG)
    
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)
    
    all_monitors = column.get_all_monitors()
    
    sim = cleo.CLSimulator(column.network)
    
    rwslfp = ephys.RWSLFPSignalFromPSCs()
    mua = ephys.MultiUnitActivity(threshold_sigma=5)
    ss = ephys.SortedSpiking()
    probe = column.electrode
    sim.set_io_processor(cleo.ioproc.RecordOnlyProcessor(sample_period=1 * b2.ms))
    probe.add_signals(mua, ss, rwslfp)

    for layer_name, layer in column.layers.items():
        if layer_name != 'L1':
            group = layer.neuron_groups['E']
            sim.inject(
                probe, group,
                Iampa_var_names=['IsynE'],
                Igaba_var_names=['IsynI'] 
            )

    print("Running baseline...")
    sim.run(500 * b2.ms)
    
    w_ext = CONFIG['synapses']['Q']['EXT']
    
    L4 = column.layers['L4']
    cfg_L4 = CONFIG['layers']['L4']
    L4_E_grp = L4.neuron_groups['E']
    N_stim_E = int(cfg_L4['poisson_inputs']['E']['N']/2)
    stim_rate_E = 8*Hz  
    L4_E_stim = PoissonInput(L4_E_grp, 'gE', 
                             N=N_stim_E, rate=stim_rate_E, weight=w_ext)
    
    L4_PV_grp = L4.neuron_groups['PV']
    N_stim_PV = int(cfg_L4['poisson_inputs']['PV']['N']/2)
    stim_rate_PV = 6*Hz
    L4_PV_stim = PoissonInput(L4_PV_grp, 'gE', 
                              N=N_stim_PV, rate=stim_rate_PV, weight=w_ext)
    
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    stim_rate_L6_E = 6*Hz
    L6_E_stim = PoissonInput(L6_E_grp, 'gE',
                             N=N_stim_L6_E, rate=stim_rate_L6_E, weight=w_ext)
    
    N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'])
    stim_rate_L6_PV = 4*Hz
    L6_PV_stim = PoissonInput(L6.neuron_groups['PV'], 'gE',
                              N=N_stim_L6_PV, rate=stim_rate_L6_PV, weight=w_ext)
    
    column.network.add(L4_E_stim, L4_PV_stim, L6_E_stim, L6_PV_stim)

    print("Running stimulus...")
    sim.run(500 * b2.ms)

    print("Analyzing...")
    results = analyze_and_plot_laminar_recording_mua(
        sim, column, probe, rwslfp, mua,
        stim_onset_time=500*b2.ms,
        plot=plot
    )
    
    if save_path is not None:
        print(f"Saving results to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run single trial simulation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    
    args = parser.parse_args()
    
    results = run_single_trial(seed=args.seed, plot=args.plot, save_path=args.save)
    
    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_single_trial import run_single_trial
from src.analysis import *
from src.cleo_plots import *


def run_multiple_trials(n_trials=50, base_seed=58879, save_dir='./trial_results'):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for trial in tqdm(range(n_trials), desc="Running trials"):
        seed = base_seed + trial
        save_path = f"{save_dir}/trial_{trial:03d}.pkl"
        
        print(f"\n=== Trial {trial+1}/{n_trials} (seed={seed}) ===")
        
        try:
            results = run_single_trial(seed=seed, plot=False, save_path=save_path)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in trial {trial}: {e}")
            continue
    
    return all_results


def average_results(all_results):

    if not all_results:
        raise ValueError("No results to average!")
    
    n_trials = len(all_results)
    print(f"\nAveraging {n_trials} trials...")
    
    psths = np.stack([r['psth'] for r in all_results], axis=0) 
    lfps = np.stack([r['lfp_bipolar'] for r in all_results], axis=0) 
    
    psth_avg = np.mean(psths, axis=0)
    psth_std = np.std(psths, axis=0)
    
    lfp_avg = np.mean(lfps, axis=0)
    lfp_std = np.std(lfps, axis=0)
    
    averaged = {
        'psth_mean': psth_avg,
        'psth_std': psth_std,
        'lfp_mean': lfp_avg,
        'lfp_std': lfp_std,
        't_centers': all_results[0]['t_centers'],
        'lfp_time': all_results[0]['lfp_time'],
        'masks_psth': all_results[0]['masks_psth'],
        'masks_lfpb': all_results[0]['masks_lfpb'],
        'n_trials': n_trials
    }
    
    return averaged


def plot_averaged_results(averaged, show_std=True, save_path=None):

    psth_mean = averaged['psth_mean']
    psth_std = averaged['psth_std']
    lfp_mean = averaged['lfp_mean']
    lfp_std = averaged['lfp_std']
    t_centers = averaged['t_centers']
    lfp_time = averaged['lfp_time']
    masks_psth = averaged['masks_psth']
    masks_lfpb = averaged['masks_lfpb']
    n_trials = averaged['n_trials']
    
    fig_psth, axs_psth = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    layer_order = ['SG', 'G', 'IG']
    
    for i, lab in enumerate(layer_order):
        ax = axs_psth[i]
        m = masks_psth[lab]
        
        if np.any(m):
            y_mean = psth_mean[:, m].mean(axis=1)
            y_std = psth_std[:, m].mean(axis=1) 
            
            ax.plot(t_centers * 1000, y_mean, '-', lw=1.8, color='black', label='Mean')
            
            if show_std:
                ax.fill_between(t_centers * 1000, 
                               y_mean - y_std, 
                               y_mean + y_std,
                               alpha=0.3, color='gray', label='±1 SD')
            n = int(m.sum())
        else:
            n = 0
        
        ax.axvline(0, ls='--', lw=0.8, color=(0.8, 0.8, 0.8))
        ax.set_ylabel('Rate (Hz)')
        ax.set_title(f'PSTH — {lab} (n_ch={n}, n_trials={n_trials})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0 and show_std:
            ax.legend(loc='upper right', fontsize=8)
        
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time from stimulus (ms)')
    
    # Plot layered LFP
    fig_lfp, axs_lfp = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    t_ms = lfp_time * 1000.0
    
    for i, lab in enumerate(layer_order):
        ax = axs_lfp[i]
        m = masks_lfpb[lab]
        
        if np.any(m):
            y_mean = np.nanmean(lfp_mean[:, m], axis=1)
            y_std = np.nanmean(lfp_std[:, m], axis=1)
            
            ax.plot(t_ms, y_mean, '-', lw=1.5, color='black', label='Mean')
            
            if show_std:
                ax.fill_between(t_ms,
                               y_mean - y_std,
                               y_mean + y_std,
                               alpha=0.3, color='gray', label='±1 SD')
            n = int(m.sum())
        else:
            n = 0
        
        ax.axvline(0, ls='--', lw=0.8, color=(0.8, 0.8, 0.8))
        ax.set_ylabel('LFP (μV)')
        ax.set_title(f'Bipolar LFP — {lab} (n_ch={n}, n_trials={n_trials})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0 and show_std:
            ax.legend(loc='upper right', fontsize=8)
        
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time from stimulus (ms)')
    
    if save_path:
        fig_psth.savefig(f"{save_path}_psth.png", dpi=300, bbox_inches='tight')
        fig_lfp.savefig(f"{save_path}_lfp.png", dpi=300, bbox_inches='tight')
        print(f"Figures saved to {save_path}_*.png")
    
    return fig_psth, fig_lfp


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run multiple trials and average')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--base_seed', type=int, default=58879, help='Base random seed')
    parser.add_argument('--save_dir', type=str, default='./trial_results', 
                       help='Directory to save individual trials')
    parser.add_argument('--output', type=str, default='./averaged_results',
                       help='Path prefix for output figures')
    parser.add_argument('--load_only', action='store_true',
                       help='Only load and average existing results')
    
    args = parser.parse_args()
    
    if not args.load_only:
        all_results = run_multiple_trials(
            n_trials=args.n_trials,
            base_seed=args.base_seed,
            save_dir=args.save_dir
        )
    else:
        print("Loading existing results...")
        all_results = []
        for trial_file in sorted(Path(args.save_dir).glob("trial_*.pkl")):
            with open(trial_file, 'rb') as f:
                all_results.append(pickle.load(f))
        print(f"Loaded {len(all_results)} trials")
    
    averaged = average_results(all_results)
    
    with open(f"{args.output}_averaged.pkl", 'wb') as f:
        pickle.dump(averaged, f)
    print(f"Averaged results saved to {args.output}_averaged.pkl")
    
    print("Generating plots...")
    fig_psth, fig_lfp = plot_averaged_results(averaged, show_std=True, save_path=args.output)
    
    plt.show()


if __name__ == "__main__":
    main()
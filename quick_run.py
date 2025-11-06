#!/usr/bin/env python
"""
Quick launcher for running trials with different configurations
"""

def run_quick_test(n_trials=5):
    """Run a quick test with few trials"""
    print("=== QUICK TEST MODE ===")
    from run_multi_trial import run_multiple_trials, average_results, plot_averaged_results
    import matplotlib.pyplot as plt
    
    results = run_multiple_trials(n_trials=n_trials, base_seed=58879, 
                                   save_dir='./test_trials')
    averaged = average_results(results)
    plot_averaged_results(averaged, show_std=True, save_path='./test_averaged')
    plt.show()


def run_full_experiment(n_trials=20):
    """Run full experiment with many trials"""
    print("=== FULL EXPERIMENT MODE ===")
    from run_multi_trial import run_multiple_trials, average_results, plot_averaged_results
    import matplotlib.pyplot as plt
    
    results = run_multiple_trials(n_trials=n_trials, base_seed=58879,
                                   save_dir='./full_trials')
    averaged = average_results(results)
    plot_averaged_results(averaged, show_std=True, save_path='./full_averaged')
    plt.show()


def compare_n_trials(trial_counts=[5, 10, 25, 50]):
    """Compare how results change with different numbers of trials"""
    print("=== COMPARISON MODE ===")
    from run_multi_trial import run_multiple_trials, average_results
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pickle
    
    # Run or load trials
    all_trials = []
    save_dir = './comparison_trials'
    Path(save_dir).mkdir(exist_ok=True)
    
    # Check if we already have results
    if Path(f"{save_dir}/all_trials.pkl").exists():
        print("Loading existing trials...")
        with open(f"{save_dir}/all_trials.pkl", 'rb') as f:
            all_trials = pickle.load(f)
    else:
        print(f"Running {max(trial_counts)} trials...")
        all_trials = run_multiple_trials(n_trials=max(trial_counts), 
                                         base_seed=58879,
                                         save_dir=save_dir)
        with open(f"{save_dir}/all_trials.pkl", 'wb') as f:
            pickle.dump(all_trials, f)
    
    # Compare different numbers of trials
    fig, axes = plt.subplots(len(trial_counts), 2, figsize=(12, 4*len(trial_counts)),
                            sharex='col', constrained_layout=True)
    
    for idx, n in enumerate(trial_counts):
        print(f"Averaging first {n} trials...")
        subset = all_trials[:n]
        averaged = average_results(subset)
        
        # PSTH plot
        ax_psth = axes[idx, 0] if len(trial_counts) > 1 else axes[0]
        psth_mean = averaged['psth_mean']
        t_centers = averaged['t_centers']
        masks = averaged['masks_psth']
        
        # Plot granular layer
        m = masks['G']
        if np.any(m):
            y = psth_mean[:, m].mean(axis=1)
            ax_psth.plot(t_centers * 1000, y, '-', lw=1.5, color='black')
        ax_psth.axvline(0, ls='--', color='gray', alpha=0.5)
        ax_psth.set_ylabel('Rate (Hz)')
        ax_psth.set_title(f'PSTH - Granular Layer (n={n} trials)')
        ax_psth.spines['top'].set_visible(False)
        ax_psth.spines['right'].set_visible(False)
        if idx == len(trial_counts) - 1:
            ax_psth.set_xlabel('Time from stimulus (ms)')
        
        # LFP plot
        ax_lfp = axes[idx, 1] if len(trial_counts) > 1 else axes[1]
        lfp_mean = averaged['lfp_mean']
        lfp_time = averaged['lfp_time']
        masks_lfp = averaged['masks_lfpb']
        
        m = masks_lfp['G']
        if np.any(m):
            y = np.nanmean(lfp_mean[:, m], axis=1)
            ax_lfp.plot(lfp_time * 1000, y, '-', lw=1.5, color='black')
        ax_lfp.axvline(0, ls='--', color='gray', alpha=0.5)
        ax_lfp.set_ylabel('LFP (μV)')
        ax_lfp.set_title(f'LFP - Granular Layer (n={n} trials)')
        ax_lfp.spines['top'].set_visible(False)
        ax_lfp.spines['right'].set_visible(False)
        if idx == len(trial_counts) - 1:
            ax_lfp.set_xlabel('Time from stimulus (ms)')
    
    plt.savefig('./comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison saved to comparison.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("\nUsage:")
        print("  python quick_run.py test [n_trials]    - Quick test (default: 5 trials)")
        print("  python quick_run.py full [n_trials]    - Full experiment (default: 50 trials)")
        print("  python quick_run.py compare            - Compare different trial counts")
        print()
        sys.exit(0)
    
    mode = sys.argv[1].lower()
    
    if mode == 'test':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_quick_test(n_trials=n)
    
    elif mode == 'full':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        run_full_experiment(n_trials=n)
    
    elif mode == 'compare':
        compare_n_trials()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'test', 'full', or 'compare'")
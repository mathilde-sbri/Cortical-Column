#!/usr/bin/env python3
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
FOLDER = "test_trials"                  # where trial_*.pkl live
PATTERN = os.path.join(FOLDER, "trial_*.pkl")

# ----------------------------
# Helpers
# ----------------------------
def compute_bipolar_lfp(lfp_mono):
    """
    lfp_mono: (time, n_channels)
    returns bipolar along adjacent pairs: (time, n_channels-1)
    """
    if lfp_mono.shape[1] < 2:
        raise ValueError("Need at least 2 channels to compute bipolar LFP.")
    return lfp_mono[:, 1:] - lfp_mono[:, :-1]

def _first_present(d, keys):
    """Return first key in `keys` that exists in dict d, else None."""
    for k in keys:
        if k in d:
            return k
    return None

def _fallback_equal_split(n, labels=("SG","G","IG")):
    """
    If we have no layer masks in the files, split channels into thirds as a fallback.
    Returns dict {label: bool_mask}
    """
    thirds = np.array_split(np.arange(n), 3)
    masks = {lab: np.zeros(n, dtype=bool) for lab in labels}
    for lab, idx in zip(labels, thirds):
        masks[lab][idx] = True
    return masks

def _ensure_1d_mask(m, n):
    """Make sure mask is 1D boolean of length n."""
    m = np.asarray(m).astype(bool).ravel()
    if m.size != n:
        raise ValueError(f"Mask length {m.size} does not match n={n}.")
    return m

def load_trials(pattern=PATTERN):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .pkl files found matching {pattern}")
    print(f"Found {len(files)} trial files.")
    return files

def make_fixed_layer_masks(n, IG=(1,5), G=(6,8), SG=(9,16)):
    """
    Build boolean masks of length n with 1-based inclusive channel ranges:
      IG: 1..5, G: 6..8, SG: 9..16  (clipped to n if needed)
    Returns dict with keys 'SG','G','IG' to match downstream expectations.
    """
    masks = {lab: np.zeros(n, dtype=bool) for lab in ("SG","G","IG")}

    def set_range(label, lo_inclusive_1b, hi_inclusive_1b):
        # convert 1-based inclusive to 0-based slice [start:stop)
        start = max(lo_inclusive_1b - 1, 0)
        stop  = min(hi_inclusive_1b, n)    # stop is exclusive already in 0-based slice
        if start < stop:
            masks[label][start:stop] = True

    set_range("IG", *IG)
    set_range("G",  *G)
    set_range("SG", *SG)
    return masks

def collect_and_average(files):
    psth_list, lfp_list = [], []
    t_centers = None
    lfp_time  = None

    # masks to extract (if present)
    masks_psth = masks_lfp = masks_lfp_bip = None
    # also accept alternative key spellings
    psth_mask_keys     = ["masks_psth"]
    lfp_mask_keys      = ["masks_lfp"]
    lfp_bip_mask_keys  = ["masks_lfp_bip", "masks_lfpbp", "masks_lfpb", "masks_lfp_bp"]

    for fpath in files:
        with open(fpath, "rb") as f:
            res = pickle.load(f)

        psth_list.append(res["psth"])     # (time_bins, n_channels)
        lfp = res.get("lfp")
        if t_centers is None:
            if "t_centers" in res:
                t_centers = np.asarray(res["t_centers"])
            else:
                # synthesize from shape + known PSTH params if present
                n_time   = res["psth"] .shape[0]
                bin_w    = float(res.get("bin_width", 0.001))  # seconds; fallback 1 ms
                t_pre    = float(res.get("t_pre", 0.5))        # seconds baseline; fallback 0.5 s
                # build centers from [-t_pre, +t_post] with uniform bins
                t_edges   = np.arange(-t_pre, -t_pre + n_time*bin_w + 1e-12, bin_w)
                t_centers = t_edges[:-1] + bin_w/2.0
                print(f"[{os.path.basename(fpath)}] Synthesized t_centers "
                    f"({t_centers[0]:.3f}..{t_centers[-1]:.3f} s, bin={bin_w*1e3:.1f} ms)")
                

            
    if lfp is not None:
        lfp_list.append(lfp)
        if lfp_time is None:
            if "lfp_time" in res:
                lfp_time = np.asarray(res["lfp_time"])
            else:
                # fallback: uniform sampling around stim
                fs = float(res.get("lfp_fs", 1000.0))  # Hz; fallback 1 kHz if unknown
                dt = 1.0 / fs
                n  = lfp.shape[0]
                t_pre = float(res.get("t_pre", 0.5))
                lfp_time = np.linspace(0, n*dt, n, endpoint=False) - t_pre
                print(f"[{os.path.basename(fpath)}] Synthesized lfp_time at {fs:.1f} Hz")
    else:
        print(f"Warning: 'lfp' not found in {fpath}, skipping LFP.")
 


        if t_centers is None:
            t_centers = np.asarray(res["t_centers"])
        if lfp_time is None:
            lfp_time = np.asarray(res["lfp_time"])

        # Grab masks (only once, assume consistent across trials)
        if masks_psth is None:
            k = _first_present(res, psth_mask_keys)
            if k is not None:
                masks_psth = {lab: np.asarray(m, dtype=bool).ravel() for lab, m in res[k].items()}
        if masks_lfp is None:
            k = _first_present(res, lfp_mask_keys)
            if k is not None:
                masks_lfp = {lab: np.asarray(m, dtype=bool).ravel() for lab, m in res[k].items()}
        if masks_lfp_bip is None:
            k = _first_present(res, lfp_bip_mask_keys)
            if k is not None:
                masks_lfp_bip = {lab: np.asarray(m, dtype=bool).ravel() for lab, m in res[k].items()}

    # Stack trials and average
    psth_all = np.stack(psth_list, axis=2)  # (time_bins, n_channels, n_trials)
    lfp_all  = np.stack(lfp_list,  axis=2)  # (time_points, n_channels, n_trials)

    mean_psth = np.mean(psth_all, axis=2)
    mean_lfp  = np.mean(lfp_all,  axis=2)

    # Compute bipolar LFP (difference of means == mean of differences)
    mean_lfp_bip = compute_bipolar_lfp(mean_lfp)

    # If masks are missing, create fallbacks so plots still run
    n_ch_psth = mean_psth.shape[1]


    n_ch_lfp  = mean_lfp.shape[1]
    n_ch_bip  = mean_lfp_bip.shape[1]


    masks_psth = make_fixed_layer_masks(n_ch_psth)
    masks_lfp = make_fixed_layer_masks(n_ch_lfp)

    masks_lfp_bip = make_fixed_layer_masks(n_ch_bip)

    # Validate mask lengths
    for lab in ("SG","G","IG"):
        masks_psth[lab]    = _ensure_1d_mask(masks_psth[lab], n_ch_psth)
        masks_lfp[lab]     = _ensure_1d_mask(masks_lfp[lab],  n_ch_lfp)
        masks_lfp_bip[lab] = _ensure_1d_mask(masks_lfp_bip[lab], n_ch_bip)

    return {
        "mean_psth": mean_psth,
        "mean_lfp": mean_lfp,
        "mean_lfp_bip": mean_lfp_bip,
        "t_centers": t_centers,
        "lfp_time": lfp_time,
        "masks_psth": masks_psth,
        "masks_lfp": masks_lfp,
        "masks_lfp_bip": masks_lfp_bip,
    }

# ----------------------------
# Plotting
# ----------------------------
def plot_per_channel(mean_psth, t_centers, mean_lfp, lfp_time, out_dir):
    n_channels_psth = mean_psth.shape[1]
    n_channels_lfp  = mean_lfp.shape[1]

    # PSTH per channel
    fig_psth, axs_psth = plt.subplots(
        n_channels_psth, 1, figsize=(6.6, 8.3),
        sharex=True, constrained_layout=True
    )
    if n_channels_psth == 1:
        axs_psth = np.array([axs_psth])

    t_ms = t_centers * 1000.0
    for ch in range(n_channels_psth):
        ax = axs_psth[ch]
        ax.plot(t_ms, mean_psth[:, ch], 'k-', lw=1.2)
        ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.7)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_ylim(0,220)
        # ax.set_title(f"PSTH — Channel {ch+1}", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ch < n_channels_psth - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time from stimulus (ms)")

    p_psth = os.path.join(out_dir, "mean_PSTH_per_channel_across_trials.png")
    fig_psth.savefig(p_psth, dpi=200, bbox_inches="tight")
    print(f"Saved: {p_psth}")

    # Mono LFP per channel
    fig_lfp, axs_lfp = plt.subplots(
        n_channels_lfp, 1, figsize=(6.6, 8.3),
        sharex=True, constrained_layout=True
    )
    if n_channels_lfp == 1:
        axs_lfp = np.array([axs_lfp])

    lfp_ms = lfp_time * 1000.0
    for ch in range(n_channels_lfp):
        ax = axs_lfp[ch]
        ax.plot(lfp_ms, mean_lfp[:, ch], 'k-', lw=0.9)
        ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.7)
        ax.set_ylabel("LFP (µV)", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ch < n_channels_lfp - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time from stimulus (ms)")

    p_lfp = os.path.join(out_dir, "mean_LFP_mono_per_channel_across_trials.png")
    fig_lfp.savefig(p_lfp, dpi=200, bbox_inches="tight")
    print(f"Saved: {p_lfp}")

def plot_per_channel_bipolar(mean_lfp_bip, lfp_time, out_dir):
    n_pairs = mean_lfp_bip.shape[1]
    fig_bp, axs_bp = plt.subplots(
        n_pairs, 1, figsize=(6.6, 8.3),
        sharex=True, constrained_layout=True
    )
    if n_pairs == 1:
        axs_bp = np.array([axs_bp])

    t_ms = lfp_time * 1000.0
    for p in range(n_pairs):
        ax = axs_bp[p]
        ax.plot(t_ms, mean_lfp_bip[:, p], 'k-', lw=0.8)
        ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.7)
        ax.set_ylabel("LFP (µV)", fontsize=9)
        ax.set_ylim(-30,30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if p < n_pairs - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time from stimulus (ms)")

    p_bp = os.path.join(out_dir, "mean_LFP_bipolar_per_pair_across_trials.png")
    fig_bp.savefig(p_bp, dpi=200, bbox_inches="tight")
    print(f"Saved: {p_bp}")

def plot_layer_averages(y, x, masks, ylabel, title_prefix, out_path):
    """
    y: (time, channels)
    x: (time,)
    masks: dict with 'SG','G','IG' -> bool masks
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True, constrained_layout=True)
    layer_order = ['SG','G','IG']
    x_ms = x * 1000.0

    for i, lab in enumerate(layer_order):
        ax = axs[i]
        m = masks.get(lab, None)
        n = 0
        if m is not None and np.any(m):
            n = int(np.sum(m))
            # nanmean for LFP robustness
            y_mean = np.nanmean(y[:, m], axis=1)
            ax.plot(x_ms, y_mean, '-', lw=1.4, color='black')
        ax.axvline(0, ls='--', lw=0.8, color=(0.8,0.8,0.8))
        ax.set_ylabel(ylabel)
        ax.set_ylim(-30,30)
        ax.set_title(f"{title_prefix} — {lab} (n={n})")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 2:
            ax.set_xlabel("Time from stimulus (ms)")
        else:
            ax.tick_params(labelbottom=False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(FOLDER, exist_ok=True)
    files = load_trials(PATTERN)
    data = collect_and_average(files)

    mean_psth     = data["mean_psth"]
    mean_lfp    = data["mean_lfp"]
    mean_lfp_bip    = data["mean_lfp_bip"]
    t_centers     = data["t_centers"]
    lfp_time      = data["lfp_time"]
    masks_psth    = data["masks_psth"]
    masks_lfp     = data["masks_lfp"]
    masks_lfp_bip = data["masks_lfp_bip"]

    # Per-channel plots (PSTH + mono LFP)
    plot_per_channel(mean_psth, t_centers, mean_lfp, lfp_time, FOLDER)


    # Per-pair bipolar LFP
    plot_per_channel_bipolar(mean_lfp_bip, lfp_time, FOLDER)

    # Layer-averaged plots
    plot_layer_averages(mean_psth, t_centers, masks_psth,
                        ylabel="Rate (Hz)",
                        title_prefix="PSTH (averaged across trials) by layer",
                        out_path=os.path.join(FOLDER, "layer_avg_PSTH_across_trials.png"))

    plot_layer_averages(mean_lfp, lfp_time, masks_lfp,
                        ylabel="LFP (µV)",
                        title_prefix="Monopolar LFP (averaged across trials) by layer",
                        out_path=os.path.join(FOLDER, "layer_avg_LFP_mono_across_trials.png"))

    plot_layer_averages(mean_lfp_bip, lfp_time, masks_lfp_bip,
                        ylabel="LFP (µV)",
                        title_prefix="Bipolar LFP (averaged across trials) by layer",
                        out_path=os.path.join(FOLDER, "layer_avg_LFP_bipolar_across_trials.png"))

    print("Done.")

if __name__ == "__main__":
    main()

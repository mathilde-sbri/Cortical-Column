"""
Test Script for Alpha Oscillation Generation
=============================================
This script tests whether Ih current in L5 E neurons generates alpha rhythms.
"""

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# Set up Brian2
start_scope()
defaultclock.dt = 0.1*ms
np.random.seed(58879)

print("=" * 60)
print("ALPHA OSCILLATION TEST - L5 with Ih current")
print("=" * 60)

# =============================================================================
# PARAMETERS
# =============================================================================

# Ih parameters - THE KEY ONES
gh_max = 15*nS          # INCREASE from 8nS
Eh = -30*mV             # Keep same
tau_h = 80*ms           # Keep same  
V_half_h = -75*mV       # CHANGE from -80mV
k_h = 6*mV            # Activation slope

# Other parameters
N_E = 400               # Number of E neurons (reduced for speed)
N_PV = 50               # Number of PV neurons
N_SOM = 40              # Number of SOM neurons

EL = -62*mV             # CHANGE from -68mV (more excitable)
VT = -50*mV             # Threshold
V_reset = -65*mV        # Reset potential
C = 97*pF               # Capacitance
gL = 4.2*nS             # Leak conductance

# Synaptic time constants
tau_AMPA = 5*ms
tau_NMDA = 100*ms
tau_PV = 6*ms
tau_SOM = 40*ms

# Reversal potentials
Ee = 0*mV
Ei = -80*mV

# =============================================================================
# NEURON EQUATIONS
# =============================================================================

E_eqs = '''
    dv/dt = (
        gL*(EL - v)
      + gL*DeltaT*exp((v - VT)/DeltaT)
      + gE_AMPA*(Ee - v)
      + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
      - gI*(v - Ei)
      + I_h
      - w + I_ext
    )/C : volt (unless refractory)
    
    # Ih current - creates alpha rhythm!
    I_h = gh * (Eh - v) : amp
    dgh/dt = (gh_inf - gh) / tau_h : siemens
    gh_inf = gh_max / (1 + exp((v - V_half_h) / k_h)) : siemens
    
    # Synaptic conductances
    dgE_AMPA/dt = -gE_AMPA / tau_AMPA : siemens
    dgE_NMDA/dt = -gE_NMDA / tau_NMDA : siemens
    dgI/dt = -gI / tau_I : siemens
    
    # Adaptation
    dw/dt = (a*(v - EL) - w) / tauw : amp
    
    I_ext : amp
    tau_I : second
    a : siemens
    b : amp
    DeltaT : volt
    tauw : second
'''

I_eqs = '''
    dv/dt = (
        gL_i*(EL_i - v)
      + gL_i*DeltaT_i*exp((v - VT)/DeltaT_i)
      + gE_AMPA*(Ee - v)
      - gI*(v - Ei)
      + I_ext
    )/C_i : volt (unless refractory)
    
    dgE_AMPA/dt = -gE_AMPA / tau_AMPA : siemens
    dgI/dt = -gI / tau_I : siemens
    
    I_ext : amp
    tau_I : second
'''

# =============================================================================
# CREATE NEURONS
# =============================================================================

print("\nCreating neurons...")

# E neurons with Ih
E_neurons = NeuronGroup(N_E, E_eqs,
                        threshold='v > -40*mV',
                        reset='v = V_reset; w += b',
                        refractory=5*ms,
                        method='euler')

E_neurons.v = '-68*mV + rand()*10*mV'
E_neurons.gh = gh_max * 0.3  # Start with some Ih active
E_neurons.gE_AMPA = 0*nS
E_neurons.gE_NMDA = 0*nS
E_neurons.gI = 0*nS
E_neurons.w = 0*pA
E_neurons.I_ext = 0*pA
E_neurons.tau_I = tau_SOM  # E cells receive SOM inhibition (slow)
E_neurons.a = 4*nS
E_neurons.b = 100*pA    # INCREASE from 80pA (stronger burst termination)
E_neurons.tauw = 80*ms 
E_neurons.DeltaT = 2*mV

# PV neurons (fast inhibition)
PV_neurons = NeuronGroup(N_PV, I_eqs,
                         threshold='v > -40*mV',
                         reset='v = V_reset',
                         refractory=2*ms,
                         method='euler',
                         namespace={'gL_i': 3.8*nS, 'EL_i': -68*mV, 
                                   'DeltaT_i': 0.5*mV, 'C_i': 38*pF})

PV_neurons.v = '-65*mV + rand()*5*mV'
PV_neurons.gE_AMPA = 0*nS
PV_neurons.gI = 0*nS
PV_neurons.I_ext = 0*pA
PV_neurons.tau_I = tau_PV

# SOM neurons (slow inhibition - important for alpha!)
SOM_neurons = NeuronGroup(N_SOM, I_eqs,
                          threshold='v > -40*mV',
                          reset='v = V_reset',
                          refractory=5*ms,
                          method='euler',
                          namespace={'gL_i': 2.3*nS, 'EL_i': -68*mV,
                                    'DeltaT_i': 1.5*mV, 'C_i': 37*pF})

SOM_neurons.v = '-65*mV + rand()*5*mV'
SOM_neurons.gE_AMPA = 0*nS
SOM_neurons.gI = 0*nS
SOM_neurons.I_ext = 0*pA
SOM_neurons.tau_I = tau_SOM

# =============================================================================
# SYNAPTIC CONNECTIONS
# =============================================================================

print("Creating synapses...")

# E -> E (recurrent excitation via AMPA and NMDA)
syn_EE_AMPA = Synapses(E_neurons, E_neurons, on_pre='gE_AMPA += 0.8*nS')
syn_EE_AMPA.connect(p=0.1)

syn_EE_NMDA = Synapses(E_neurons, E_neurons, on_pre='gE_NMDA += 0.3*nS')
syn_EE_NMDA.connect(p=0.1)

# E -> PV
syn_E_PV = Synapses(E_neurons, PV_neurons, on_pre='gE_AMPA += 3*nS')
syn_E_PV.connect(p=0.3)

# E -> SOM (important - drives the slow inhibitory rhythm)
syn_E_SOM = Synapses(E_neurons, SOM_neurons, on_pre='gE_AMPA += 2*nS')
syn_E_SOM.connect(p=0.3)

# PV -> E (fast inhibition)
syn_PV_E = Synapses(PV_neurons, E_neurons, on_pre='gI += 2*nS')
syn_PV_E.connect(p=0.4)

# SOM -> E (slow dendritic inhibition - KEY for alpha!)
syn_SOM_E = Synapses(SOM_neurons, E_neurons, on_pre='gI += 1.5*nS')
syn_SOM_E.connect(p=0.35)

# PV -> PV
syn_PV_PV = Synapses(PV_neurons, PV_neurons, on_pre='gI += 2*nS')
syn_PV_PV.connect(p=0.4)

# =============================================================================
# EXTERNAL INPUT
# =============================================================================

print("Setting up external input...")

# Background Poisson input to E cells
poisson_E = PoissonInput(E_neurons, 'gE_AMPA', N=50, rate=6*Hz, weight=1.2*nS)

# Background to PV cells
poisson_PV = PoissonInput(PV_neurons, 'gE_AMPA', N=20, rate=3*Hz, weight=1*nS)

# Background to SOM cells
poisson_SOM = PoissonInput(SOM_neurons, 'gE_AMPA', N=20, rate=3*Hz, weight=1*nS)

# =============================================================================
# MONITORS
# =============================================================================

print("Setting up monitors...")

# Spike monitors
spike_E = SpikeMonitor(E_neurons)
spike_PV = SpikeMonitor(PV_neurons)
spike_SOM = SpikeMonitor(SOM_neurons)

# State monitor for a few E cells (to see Ih dynamics)
state_E = StateMonitor(E_neurons, ['v', 'gh', 'I_h', 'w', 'gI'], record=[0, 1, 2, 3, 4])

# Population rate
rate_E = PopulationRateMonitor(E_neurons)
rate_PV = PopulationRateMonitor(PV_neurons)
rate_SOM = PopulationRateMonitor(SOM_neurons)

# =============================================================================
# RUN SIMULATION
# =============================================================================

print("\nRunning simulation (3 seconds)...")
run(3000*ms, report='text')
print("Done!")

# =============================================================================
# ANALYSIS AND PLOTTING
# =============================================================================

print("\nAnalyzing results...")

fig = plt.figure(figsize=(14, 16))

# 1. Raster plot
ax1 = fig.add_subplot(5, 1, 1)
ax1.plot(spike_E.t/ms, spike_E.i, '.', ms=1, color='blue', label='E')
ax1.plot(spike_PV.t/ms, spike_PV.i + N_E, '.', ms=1, color='red', label='PV')
ax1.plot(spike_SOM.t/ms, spike_SOM.i + N_E + N_PV, '.', ms=1, color='green', label='SOM')
ax1.set_ylabel('Neuron index')
ax1.set_title('Spike Raster')
ax1.legend(loc='upper right')
ax1.set_xlim([0, 3000])

# 2. Population firing rates
ax2 = fig.add_subplot(5, 1, 2)
# Smooth the rates
window = 20*ms
smoothed_E   = rate_E.smooth_rate(window='flat', width=window)
smoothed_PV  = rate_PV.smooth_rate(window='flat', width=window)
smoothed_SOM = rate_SOM.smooth_rate(window='flat', width=window)


ax2.plot(rate_E.t/ms, smoothed_E/Hz, 'b-', label='E', alpha=0.8)
ax2.plot(rate_PV.t/ms, smoothed_PV/Hz, 'r-', label='PV', alpha=0.8)
ax2.plot(rate_SOM.t/ms, smoothed_SOM/Hz, 'g-', label='SOM', alpha=0.8)
ax2.set_ylabel('Rate (Hz)')
ax2.set_title('Population Rates (smoothed)')
ax2.legend()
ax2.set_xlim([0, 3000])

# 3. Single E neuron membrane potential
ax3 = fig.add_subplot(5, 1, 3)
ax3.plot(state_E.t/ms, state_E.v[0]/mV, 'b-', lw=0.5)
ax3.set_ylabel('V (mV)')
ax3.set_title('E neuron membrane potential (showing Ih-induced sag and rebound)')
ax3.axhline(y=-80, color='gray', linestyle='--', alpha=0.5, label='Ih activation zone')
ax3.set_xlim([1000, 2000])  # Zoom to middle second

# 4. Ih conductance and current
ax4 = fig.add_subplot(5, 1, 4)
ax4.plot(state_E.t/ms, state_E.gh[0]/nS, 'purple', lw=0.8, label='gh (nS)')
ax4.set_ylabel('gh (nS)', color='purple')
ax4.tick_params(axis='y', labelcolor='purple')
ax4.set_xlim([1000, 2000])

ax4b = ax4.twinx()
ax4b.plot(state_E.t/ms, state_E.I_h[0]/pA, 'orange', lw=0.8, label='I_h (pA)')
ax4b.set_ylabel('I_h (pA)', color='orange')
ax4b.tick_params(axis='y', labelcolor='orange')
ax4.set_title('Ih conductance and current (should oscillate at ~10 Hz)')

# 5. Power spectrum of E population rate
ax5 = fig.add_subplot(5, 1, 5)

# Compute power spectrum
from scipy import signal

# Use the smoothed rate for spectrum
rate_signal = smoothed_E/Hz
fs = 10000  # sampling rate in Hz (0.1 ms dt)
# Downsample for spectrum
downsample_factor = 10
rate_downsampled = rate_signal[::downsample_factor]
fs_down = fs / downsample_factor

# Compute power spectrum
freqs, psd = signal.welch(rate_downsampled, fs=fs_down, nperseg=min(1024, len(rate_downsampled)//2))

ax5.semilogy(freqs, psd)
ax5.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='10 Hz (alpha)')
ax5.axvspan(8, 13, alpha=0.2, color='red', label='Alpha band')
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Power')
ax5.set_title('Power Spectrum of E Population Rate')
ax5.set_xlim([0, 100])
ax5.legend()

plt.tight_layout()
plt.savefig('alpha_test_results.png', dpi=150)
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"E neurons fired {len(spike_E.i)} spikes")
print(f"PV neurons fired {len(spike_PV.i)} spikes")
print(f"SOM neurons fired {len(spike_SOM.i)} spikes")
print(f"\nMean E rate: {len(spike_E.i) / (N_E * 3):.2f} Hz")
print(f"Mean PV rate: {len(spike_PV.i) / (N_PV * 3):.2f} Hz")
print(f"Mean SOM rate: {len(spike_SOM.i) / (N_SOM * 3):.2f} Hz")

# Find peak frequency in alpha band
alpha_mask = (freqs >= 5) & (freqs <= 20)
if np.any(alpha_mask):
    peak_idx = np.argmax(psd[alpha_mask])
    peak_freq = freqs[alpha_mask][peak_idx]
    print(f"\nPeak frequency in 5-20 Hz band: {peak_freq:.1f} Hz")
    
    # Check if there's a real alpha peak
    alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
    low_freq_power = np.mean(psd[(freqs >= 1) & (freqs <= 5)])
    gamma_power = np.mean(psd[(freqs >= 30) & (freqs <= 50)])
    
    print(f"Alpha (8-13 Hz) power: {alpha_power:.4f}")
    print(f"Low freq (1-5 Hz) power: {low_freq_power:.4f}")
    print(f"Gamma (30-50 Hz) power: {gamma_power:.4f}")
    
    if alpha_power > gamma_power and alpha_power > low_freq_power * 0.5:
        print("\n✓ ALPHA RHYTHM DETECTED!")
    else:
        print("\n✗ No clear alpha rhythm - may need parameter tuning")

print("\nResults saved to /mnt/user-data/outputs/alpha_test_results.png")

"""
Analysis functions for neural data. to do  : check lfp method + spike analysis
"""
import numpy as np
from scipy.signal import welch
from brian2 import *

class LFPAnalysis:
    
    @staticmethod
    def calculate_lfp(monitor, neuron_type='E'):
        """Calculate LFP using current inputs, inspired from the paper of Mazzoni but need to check if it's exactly the same"""
        ge = np.array(monitor.ge/nS)  
        gi = np.array(monitor.gi/nS)  
        V = np.array(monitor.v/mV)
        
        I_AMPA = np.abs(ge * (0 - V))  # Ee = 0mV
        I_GABA = np.abs(gi * (-80 - V))  # Ei = -80mV
        
        total_current = np.sum(I_AMPA + I_GABA, axis=0)
        return total_current
    
    @staticmethod
    def process_lfp(monitor, start_time_ms=3000):
        lfp = LFPAnalysis.calculate_lfp(monitor)
        lfp_time = np.array(monitor.t/ms)
        
        start_idx = np.argmax(lfp_time >= start_time_ms)
        lfp_stable = lfp[start_idx:]
        time_stable = lfp_time[start_idx:]
        
        lfp_stable = (lfp_stable - np.mean(lfp_stable)) / np.std(lfp_stable)
        
        return time_stable, lfp_stable
    
    @staticmethod
    def compute_power_spectrum(lfp_signal, fs=10000, nperseg=4096):
        freq, psd = welch(lfp_signal, fs=fs, nperseg=nperseg)
        return freq, psd

class SpikeAnalysis:
    
    @staticmethod
    def calculate_firing_rates(spike_monitor, window_size=100*ms):

        pass
    
    @staticmethod
    def calculate_synchrony(spike_monitors):

        pass
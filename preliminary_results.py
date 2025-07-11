import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
from brian2 import *
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d
from neurodsp.spectral import compute_spectrum, rotate_powerlaw
from neurodsp.utils import create_times
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series

np.random.seed(58879)



#  parameters
V_reset = -65.*mV
VT = -50.*mV
Ei = -80.*mV
Ee = 0.*mV
t_ref = 5*ms
tauw = 500*ms
t_simulation = 4000*ms


Eleaky_RS = -60*mV
Eleaky_FS = -60*mV  
Eleaky_SST = -55*mV

#  conductances
Q_PV_to_EPV = 6*nS     
Q_SOM_to_EPV = 5*nS     
Q_E_to_E = 1.25*nS      
Q_E_to_PV = 3.75*nS     
Q_E_to_SOM = 2.5*nS     
Q_Ext = 1.25*nS        

# Time constants
tau_e = 5*ms
tau_i = 5*ms
tau_e_pv = 1*ms
tau_e_som = 2*ms

#  equations
eqsE= """
dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
IsynE=ge*(Ee-v) : amp
IsynI=gi*(Ei-v) : amp
dge/dt = -ge/tau_e : siemens
dgi/dt = -gi/tau_i : siemens
dw/dt = (a*(v - EL) - w)/tauw : amp
taum= C/gL : second
I : amp
a : siemens
b : amp
DeltaT: volt
Vcut: volt
EL : volt
C : farad
gL : siemens
"""

eqsPV= """
dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
IsynE=ge*(Ee-v) : amp
IsynI=gi*(Ei-v) : amp
dge/dt = -ge/tau_e_pv : siemens
dgi/dt = -gi/tau_i : siemens
dw/dt = (a*(v - EL) - w)/tauw : amp
taum= C/gL : second
I : amp
a : siemens
b : amp
DeltaT: volt
Vcut: volt
EL : volt
C : farad
gL : siemens
"""

eqsSST= """
dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
IsynE=ge*(Ee-v) : amp
IsynI=gi*(Ei-v) : amp
dge/dt = -ge/tau_e_som : siemens
dgi/dt = -gi/tau_i : siemens
dw/dt = (a*(v - EL) - w)/tauw : amp
taum= C/gL : second
I : amp
a : siemens
b : amp
DeltaT: volt
Vcut: volt
EL : volt
C : farad
gL : siemens
"""

b2.start_scope()
b2.defaultclock.dt = 0.1*ms

#==============================================================================
# LAYER 2/3
#==============================================================================

L23_p = 0.034 # connection probability for layer 2/3
L23_rate = 5*Hz # rate of input

# Number of neurons for layer 2/3
L23_N_E = 4944
L23_N_PV = 260
L23_N_SOM = 188
L23_N_EXT = int(L23_p*L23_N_E)

L23_neuron_E = NeuronGroup(L23_N_E, eqsE, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L23_neuron_E.a = 4*nS
L23_neuron_E.b = 130*pA
L23_neuron_E.DeltaT = 2*mV
L23_neuron_E.Vcut = VT + 5*L23_neuron_E.DeltaT
L23_neuron_E.EL = Eleaky_RS
L23_neuron_E.C = 200*pF
L23_neuron_E.gL = 10*nS
L23_neuron_E.v = -60*mV
L23_neuron_E.ge = 0*nS
L23_neuron_E.gi = 0*nS
L23_neuron_E.w = 0*pA
L23_neuron_E.I = 0*pA  

L23_neuron_PV = NeuronGroup(L23_N_PV, eqsPV, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L23_neuron_PV.a = 0*nS
L23_neuron_PV.b = 0*pA
L23_neuron_PV.DeltaT = 0.5*mV
L23_neuron_PV.Vcut = VT + 5*L23_neuron_PV.DeltaT
L23_neuron_PV.EL = Eleaky_FS
L23_neuron_PV.C = 200*pF
L23_neuron_PV.gL = 10*nS
L23_neuron_PV.v = -60*mV
L23_neuron_PV.ge = 0*nS
L23_neuron_PV.gi = 0*nS
L23_neuron_PV.w = 0*pA
L23_neuron_PV.I = 0*pA  

L23_neuron_SST = NeuronGroup(L23_N_SOM, eqsSST, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L23_neuron_SST.a = 4*nS
L23_neuron_SST.b = 25*pA
L23_neuron_SST.DeltaT = 1.5*mV
L23_neuron_SST.Vcut = VT + 5*L23_neuron_SST.DeltaT
L23_neuron_SST.EL = Eleaky_SST
L23_neuron_SST.C = 200*pF
L23_neuron_SST.gL = 10*nS
L23_neuron_SST.v = -60*mV
L23_neuron_SST.ge = 0*nS
L23_neuron_SST.gi = 0*nS
L23_neuron_SST.w = 0*pA
L23_neuron_SST.I = 0*pA  

# inputs
L23_PoissonPV_Brian = PoissonInput(L23_neuron_PV, 'ge', N=L23_N_EXT, rate=L23_rate, weight=Q_Ext)
L23_PoissonE_Brian = PoissonInput(L23_neuron_E, 'ge', N=L23_N_EXT, rate=L23_rate, weight=Q_Ext)

# connections within the layer
L23_con_E_E = Synapses(L23_neuron_E, L23_neuron_E, on_pre=f'ge_post += {Q_E_to_E/nS}*nS')
L23_con_E_E.connect(p=L23_p)

L23_con_E_PV = Synapses(L23_neuron_E, L23_neuron_PV, on_pre=f'ge_post += {Q_E_to_PV/nS}*nS')
L23_con_E_PV.connect(p=2*L23_p)

L23_con_E_SST = Synapses(L23_neuron_E, L23_neuron_SST, on_pre=f'ge_post += {Q_E_to_SOM/nS}*nS')
L23_con_E_SST.connect(p=L23_p)

L23_con_PV_E = Synapses(L23_neuron_PV, L23_neuron_E, on_pre=f'gi_post += {Q_PV_to_EPV/nS}*nS')
L23_con_PV_E.connect(p=L23_p)

L23_con_PV_PV = Synapses(L23_neuron_PV, L23_neuron_PV, on_pre=f'gi_post += {Q_PV_to_EPV/nS}*nS')
L23_con_PV_PV.connect(p=L23_p)

L23_con_SST_E = Synapses(L23_neuron_SST, L23_neuron_E, on_pre=f'gi_post += {Q_SOM_to_EPV/nS}*nS')
L23_con_SST_E.connect(p=L23_p)

L23_con_SST_PV = Synapses(L23_neuron_SST, L23_neuron_PV, on_pre=f'gi_post += {Q_SOM_to_EPV/nS}*nS')
L23_con_SST_PV.connect(p=L23_p)

L23_mon_E = StateMonitor(L23_neuron_E, ['v', 'ge', 'gi'], record=True)
L23_mon_PV = StateMonitor(L23_neuron_PV, ['v', 'ge', 'gi'], record=True)
L23_mon_SST = StateMonitor(L23_neuron_SST, ['v', 'ge', 'gi'], record=True)

L23_spike_mon_E = SpikeMonitor(L23_neuron_E, variables='t')
L23_spike_mon_PV = SpikeMonitor(L23_neuron_PV, variables='t')
L23_spike_mon_SST = SpikeMonitor(L23_neuron_SST, variables='t')

L23_pop_rate_E = PopulationRateMonitor(L23_neuron_E)
L23_pop_rate_PV = PopulationRateMonitor(L23_neuron_PV)
L23_pop_rate_SST = PopulationRateMonitor(L23_neuron_SST)

#==============================================================================
# LAYER 4
#==============================================================================


L4_p = 0.034 # connection probability 
L4_rate = 5*Hz # rate of input

# number of neurons
L4_N_E = 4040   
L4_N_PV = 392
L4_N_SOM = 212
L4_N_EXT = int(L4_p*L4_N_E)

L4_neuron_E = NeuronGroup(L4_N_E, eqsE, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L4_neuron_E.a = 4*nS
L4_neuron_E.b = 130*pA
L4_neuron_E.DeltaT = 2*mV
L4_neuron_E.Vcut = VT + 5*L4_neuron_E.DeltaT
L4_neuron_E.EL = Eleaky_RS
L4_neuron_E.C = 200*pF
L4_neuron_E.gL = 10*nS
L4_neuron_E.v = -60*mV
L4_neuron_E.ge = 0*nS
L4_neuron_E.gi = 0*nS
L4_neuron_E.w = 0*pA
L4_neuron_E.I = 0*pA  

L4_neuron_PV = NeuronGroup(L4_N_PV, eqsPV, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L4_neuron_PV.a = 0*nS
L4_neuron_PV.b = 0*pA
L4_neuron_PV.DeltaT = 0.5*mV
L4_neuron_PV.Vcut = VT + 5*L4_neuron_PV.DeltaT
L4_neuron_PV.EL = Eleaky_FS
L4_neuron_PV.C = 200*pF
L4_neuron_PV.gL = 10*nS
L4_neuron_PV.v = -60*mV
L4_neuron_PV.ge = 0*nS
L4_neuron_PV.gi = 0*nS
L4_neuron_PV.w = 0*pA
L4_neuron_PV.I = 0*pA  

L4_neuron_SST = NeuronGroup(L4_N_SOM, eqsSST, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L4_neuron_SST.a = 4*nS
L4_neuron_SST.b = 25*pA
L4_neuron_SST.DeltaT = 1.5*mV
L4_neuron_SST.Vcut = VT + 5*L4_neuron_SST.DeltaT
L4_neuron_SST.EL = Eleaky_SST
L4_neuron_SST.C = 200*pF
L4_neuron_SST.gL = 10*nS
L4_neuron_SST.v = -60*mV
L4_neuron_SST.ge = 0*nS
L4_neuron_SST.gi = 0*nS
L4_neuron_SST.w = 0*pA
L4_neuron_SST.I = 0*pA  

# poisson inputs
L4_PoissonPV_Brian = PoissonInput(L4_neuron_PV, 'ge', N=L4_N_EXT, rate=L4_rate, weight=Q_Ext)
L4_PoissonE_Brian = PoissonInput(L4_neuron_E, 'ge', N=L4_N_EXT, rate=L4_rate, weight=Q_Ext)

# connections
L4_con_E_E = Synapses(L4_neuron_E, L4_neuron_E, on_pre=f'ge_post += {Q_E_to_E/nS}*nS')
L4_con_E_E.connect(p=L4_p)

L4_con_E_PV = Synapses(L4_neuron_E, L4_neuron_PV, on_pre=f'ge_post += {Q_E_to_PV/nS}*nS')
L4_con_E_PV.connect(p=2*L4_p)

L4_con_E_SST = Synapses(L4_neuron_E, L4_neuron_SST, on_pre=f'ge_post += {Q_E_to_SOM/nS}*nS')
L4_con_E_SST.connect(p=L4_p)

L4_con_PV_E = Synapses(L4_neuron_PV, L4_neuron_E, on_pre=f'gi_post += {Q_PV_to_EPV/nS}*nS')
L4_con_PV_E.connect(p=L4_p)

L4_con_PV_PV = Synapses(L4_neuron_PV, L4_neuron_PV, on_pre=f'gi_post += {Q_PV_to_EPV/nS}*nS')
L4_con_PV_PV.connect(p=L4_p)

L4_con_SST_E = Synapses(L4_neuron_SST, L4_neuron_E, on_pre=f'gi_post += {Q_SOM_to_EPV/nS}*nS')
L4_con_SST_E.connect(p=L4_p)

L4_con_SST_PV = Synapses(L4_neuron_SST, L4_neuron_PV, on_pre=f'gi_post += {Q_SOM_to_EPV/nS}*nS')
L4_con_SST_PV.connect(p=L4_p)

L4_mon_E = StateMonitor(L4_neuron_E, ['v', 'ge', 'gi'], record=True)
L4_mon_PV = StateMonitor(L4_neuron_PV, ['v', 'ge', 'gi'], record=True)
L4_mon_SST = StateMonitor(L4_neuron_SST, ['v', 'ge', 'gi'], record=True)

L4_spike_mon_E = SpikeMonitor(L4_neuron_E, variables='t')
L4_spike_mon_PV = SpikeMonitor(L4_neuron_PV, variables='t')
L4_spike_mon_SST = SpikeMonitor(L4_neuron_SST, variables='t')

L4_pop_rate_E = PopulationRateMonitor(L4_neuron_E)
L4_pop_rate_PV = PopulationRateMonitor(L4_neuron_PV)
L4_pop_rate_SST = PopulationRateMonitor(L4_neuron_SST)

#==============================================================================
# LAYER 5/6
#==============================================================================

L56_p = 0.017 # connection probability
L56_rate = 2*Hz # rate of input

# number of neurons
L56_N_E = 8016   
L56_N_PV = 260
L56_N_SOM = 1032
L56_N_EXT = int(L56_p*L56_N_E)

L56_neuron_E = NeuronGroup(L56_N_E, eqsE, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L56_neuron_E.a = 4*nS
L56_neuron_E.b = 130*pA
L56_neuron_E.DeltaT = 2*mV
L56_neuron_E.Vcut = VT + 5*L56_neuron_E.DeltaT
L56_neuron_E.EL = Eleaky_RS
L56_neuron_E.C = 200*pF
L56_neuron_E.gL = 10*nS
L56_neuron_E.v = -60*mV
L56_neuron_E.ge = 0*nS
L56_neuron_E.gi = 0*nS
L56_neuron_E.w = 0*pA
L56_neuron_E.I = 0*pA  

L56_neuron_PV = NeuronGroup(L56_N_PV, eqsPV, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L56_neuron_PV.a = 0*nS
L56_neuron_PV.b = 0*pA
L56_neuron_PV.DeltaT = 0.5*mV
L56_neuron_PV.Vcut = VT + 5*L56_neuron_PV.DeltaT
L56_neuron_PV.EL = Eleaky_FS
L56_neuron_PV.C = 200*pF
L56_neuron_PV.gL = 10*nS
L56_neuron_PV.v = -60*mV
L56_neuron_PV.ge = 0*nS
L56_neuron_PV.gi = 0*nS
L56_neuron_PV.w = 0*pA
L56_neuron_PV.I = 0*pA  

L56_neuron_SST = NeuronGroup(L56_N_SOM, eqsSST, threshold='v>Vcut', reset="v=V_reset; w+=b", refractory=t_ref)
L56_neuron_SST.a = 4*nS
L56_neuron_SST.b = 25*pA
L56_neuron_SST.DeltaT = 1.5*mV
L56_neuron_SST.Vcut = VT + 5*L56_neuron_SST.DeltaT
L56_neuron_SST.EL = Eleaky_SST
L56_neuron_SST.C = 200*pF
L56_neuron_SST.gL = 10*nS
L56_neuron_SST.v = -60*mV
L56_neuron_SST.ge = 0*nS
L56_neuron_SST.gi = 0*nS
L56_neuron_SST.w = 0*pA
L56_neuron_SST.I = 0*pA  

# poisson inputs
L56_PoissonPV_Brian = PoissonInput(L56_neuron_PV, 'ge', N=L56_N_EXT, rate=L56_rate, weight=Q_Ext)
L56_PoissonE_Brian = PoissonInput(L56_neuron_E, 'ge', N=L56_N_EXT, rate=L56_rate, weight=Q_Ext)

# connections
L56_con_E_E = Synapses(L56_neuron_E, L56_neuron_E, on_pre=f'ge_post += {Q_E_to_E/nS}*nS')
L56_con_E_E.connect(p=L56_p)

L56_con_E_PV = Synapses(L56_neuron_E, L56_neuron_PV, on_pre=f'ge_post += {Q_E_to_PV/nS}*nS')
L56_con_E_PV.connect(p=2*L56_p)

L56_con_E_SST = Synapses(L56_neuron_E, L56_neuron_SST, on_pre=f'ge_post += {Q_E_to_SOM/nS}*nS')
L56_con_E_SST.connect(p=L56_p)

L56_con_PV_E = Synapses(L56_neuron_PV, L56_neuron_E, on_pre=f'gi_post += {Q_PV_to_EPV/nS}*nS')
L56_con_PV_E.connect(p=L56_p)

L56_con_PV_PV = Synapses(L56_neuron_PV, L56_neuron_PV, on_pre=f'gi_post += {Q_PV_to_EPV/nS}*nS')
L56_con_PV_PV.connect(p=L56_p)

L56_con_SST_E = Synapses(L56_neuron_SST, L56_neuron_E, on_pre=f'gi_post += {Q_SOM_to_EPV/nS}*nS')
L56_con_SST_E.connect(p=L56_p)

L56_con_SST_PV = Synapses(L56_neuron_SST, L56_neuron_PV, on_pre=f'gi_post += {Q_SOM_to_EPV/nS}*nS')
L56_con_SST_PV.connect(p=L56_p)

L56_mon_E = StateMonitor(L56_neuron_E, ['v', 'ge', 'gi'], record=True)
L56_mon_PV = StateMonitor(L56_neuron_PV, ['v', 'ge', 'gi'], record=True)
L56_mon_SST = StateMonitor(L56_neuron_SST, ['v', 'ge', 'gi'], record=True)

L56_spike_mon_E = SpikeMonitor(L56_neuron_E, variables='t')
L56_spike_mon_PV = SpikeMonitor(L56_neuron_PV, variables='t')
L56_spike_mon_SST = SpikeMonitor(L56_neuron_SST, variables='t')

L56_pop_rate_E = PopulationRateMonitor(L56_neuron_E)
L56_pop_rate_PV = PopulationRateMonitor(L56_neuron_PV)
L56_pop_rate_SST = PopulationRateMonitor(L56_neuron_SST)

#==============================================================================
# Connection between the layers
#==============================================================================

# this needs to be changed with realistic connections between different populations of different layers. 
# here connections are made between excitatory populations of different layers only with very low probability (0.0001)
# also the synapse conductance used is the same as E-E connections within one layer, maybe it should be changed

# L23 E to L4 E connections
L23_L4_con_E_E = Synapses(L23_neuron_E, L4_neuron_E, on_pre=f'ge_post += {Q_E_to_E/nS}*nS')
L23_L4_con_E_E.connect(p=0.0001)



# L4 E to L56 E connections
L4_L56_con_E_E = Synapses(L4_neuron_E, L56_neuron_E, on_pre=f'ge_post += {Q_E_to_E/nS}*nS')
L4_L56_con_E_E.connect(p=0.0001)


# L56 E to L23 E connections 
L56_L23_con_E_E = Synapses(L56_neuron_E, L23_neuron_E, on_pre=f'ge_post += {Q_E_to_E/nS}*nS')
L56_L23_con_E_E.connect(p=0.0001)


#==============================================================================
# RUN SIMULATION
#==============================================================================

print("Running simulation...")
run(t_simulation)
print("Simulation complete")

#==============================================================================
# ANALYSIS FUNCTIONS
#==============================================================================

def calculate_lfp(monitor, neuron_type='E'):
    """
    Calculate LFP using current inputs of excitatory neurons
    """
    ge = np.array(monitor.ge/nS)  
    gi = np.array(monitor.gi/nS)  
    
    # convert conductances to currents with I = g*(E-V)
    V = np.array(monitor.v/mV)
    
    I_AMPA = np.abs(ge * (0 - V))  # Ee = 0mV
    I_GABA = np.abs(gi * (-80 - V))  # Ei = -80mV
    
    total_current = np.sum(I_AMPA + I_GABA, axis=0)
    
    return total_current

def process_lfp(monitor, layer_name):
    #for normalizing lfp
    lfp = calculate_lfp(monitor)
    lfp_time = np.array(monitor.t/ms)
    
    start_idx = np.argmax(lfp_time >= 3000)
    lfp_stable = lfp[start_idx:]
    time_stable = lfp_time[start_idx:]
    
    # normalize LFP
    lfp_stable = (lfp_stable - np.mean(lfp_stable)) / np.std(lfp_stable)
    
    return time_stable, lfp_stable

#  LFPs for all layers
L23_time_stable, L23_lfp_stable = process_lfp(L23_mon_E, "L23")
L4_time_stable, L4_lfp_stable = process_lfp(L4_mon_E, "L4")
L56_time_stable, L56_lfp_stable = process_lfp(L56_mon_E, "L56")

#==============================================================================
# PLOTS
#==============================================================================

plt.figure(figsize=(15, 18))

# 1. SPIKE RASTER PLOTS
# Layer 2/3
ax1 = plt.subplot(6, 1, 1)
ax1.scatter(L23_spike_mon_E.t/second, L23_spike_mon_E.i, 
            color='green', s=0.5, alpha=0.6, label="E")
ax1.scatter(L23_spike_mon_SST.t/second, L23_spike_mon_SST.i + L23_N_E, 
            color='blue', s=0.5, alpha=0.8, label="SST")
ax1.scatter(L23_spike_mon_PV.t/second, L23_spike_mon_PV.i + L23_N_E + L23_N_SOM, 
            color='red', s=0.5, alpha=0.8, label="PV")
ax1.set_xlim(0, 4)
ax1.set_ylabel('Neuron index')
ax1.set_title('Layer 2/3 Spike Raster Plot')
ax1.legend()

# Layer 4
ax2 = plt.subplot(6, 1, 2)
ax2.scatter(L4_spike_mon_E.t/second, L4_spike_mon_E.i, 
            color='green', s=0.5, alpha=0.6, label="E")
ax2.scatter(L4_spike_mon_SST.t/second, L4_spike_mon_SST.i + L4_N_E, 
            color='blue', s=0.5, alpha=0.8, label="SST")
ax2.scatter(L4_spike_mon_PV.t/second, L4_spike_mon_PV.i + L4_N_E + L4_N_SOM, 
            color='red', s=0.5, alpha=0.8, label="PV")
ax2.set_xlim(0, 4)
ax2.set_ylabel('Neuron index')
ax2.set_title('Layer 4 Spike Raster Plot')
ax2.legend()

# Layer 5/6
ax3 = plt.subplot(6, 1, 3)
ax3.scatter(L56_spike_mon_E.t/second, L56_spike_mon_E.i, 
            color='green', s=0.5, alpha=0.6, label="E")
ax3.scatter(L56_spike_mon_SST.t/second, L56_spike_mon_SST.i + L56_N_E, 
            color='blue', s=0.5, alpha=0.8, label="SST")
ax3.scatter(L56_spike_mon_PV.t/second, L56_spike_mon_PV.i + L56_N_E + L56_N_SOM, 
            color='red', s=0.5, alpha=0.8, label="PV")
ax3.set_xlim(0, 4)
ax3.set_ylabel('Neuron index')
ax3.set_title('Layer 5/6 Spike Raster Plot')
ax3.legend()

# 2. LFP 
# Layer 2/3
ax4 = plt.subplot(6, 1, 4)
ax4.plot(L23_time_stable, L23_lfp_stable, 'b-', linewidth=0.5)
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('LFP (norm)')
ax4.set_title('Layer 2/3 Local Field Potential')
ax4.set_xlim(3000, 4000)
ax4.grid(True, alpha=0.3)

# Layer 4
ax5 = plt.subplot(6, 1, 5)
ax5.plot(L4_time_stable, L4_lfp_stable, 'b-', linewidth=0.5)
ax5.set_xlabel('Time (ms)')
ax5.set_ylabel('LFP (norm)')
ax5.set_title('Layer 4 Local Field Potential')
ax5.set_xlim(3000, 4000)
ax5.grid(True, alpha=0.3)

# Layer 5/6
ax6 = plt.subplot(6, 1, 6)
ax6.plot(L56_time_stable, L56_lfp_stable, 'b-', linewidth=0.5)
ax6.set_xlabel('Time (ms)')
ax6.set_ylabel('LFP (norm)')
ax6.set_title('Layer 5/6 Local Field Potential')
ax6.set_xlim(3000, 4000)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#==============================================================================
# POWER SPECTRUM
#==============================================================================

fs = 10000  

freq_L23, psd_L23 = welch(L23_lfp_stable, fs=fs, nperseg=4096)
freq_L4, psd_L4 = welch(L4_lfp_stable, fs=fs, nperseg=4096)
freq_L56, psd_L56 = welch(L56_lfp_stable, fs=fs, nperseg=4096)

plt.figure(figsize=(10, 12))


plt.subplot(3, 1, 1)
plt.plot(freq_L23[:50], psd_L23[:50], 'b-', label='Layer 2/3', linewidth=2.5)
plt.ylabel('Power', fontsize=18)
plt.grid(True)
peak_idx = np.argmax(psd_L23[:50])
plt.axvline(freq_L23[peak_idx], color='r', linestyle='--')
plt.legend(fontsize=20,loc='upper center')
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(3, 1, 2)
plt.plot(freq_L4[:50], psd_L4[:50], 'g-', label='Layer 4', linewidth=2.5)
plt.ylabel('Power', fontsize=18)
plt.grid(True)
peak_idx = np.argmax(psd_L4[:50])
plt.axvline(freq_L4[peak_idx], color='r', linestyle='--')
plt.legend(fontsize=20,loc='upper center')
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(3, 1, 3)
plt.plot(freq_L56[:50], psd_L56[:50], 'purple', label='Layer 5/6', linewidth=2.5)
plt.xlabel('Frequency (Hz)', fontsize=18)
plt.ylabel('Power', fontsize=18)
plt.grid(True)
peak_idx = np.argmax(psd_L56[:50])
plt.axvline(freq_L56[peak_idx], color='r', linestyle='--')
plt.legend(fontsize=20,loc='upper center')
plt.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()

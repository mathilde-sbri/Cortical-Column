"""
Global simulation parameters and constants
"""
import brian2 as b2
from brian2 import *

# Simulation parameters
SIMULATION_TIME = 4000*ms
DT = 0.1*ms
RANDOM_SEED = 58879

# Common neuron parameters
V_RESET = -65.*mV
VT = -50.*mV
EI = -80.*mV
EE = 0.*mV
T_REF = 5*ms
TAU_W = 500*ms
NEURON_CAPACITANCE = 200*pF
NEURON_LEAK_CONDUCTANCE = 10*nS
INITIAL_VOLTAGE = -60*mV

# Leak potentials
E_LEAK_RS = -60*mV  # Regular spiking
E_LEAK_FS = -60*mV  # Fast spiking
E_LEAK_SST = -55*mV # Somatostatin
E_LEAK_VIP = -65*mV # VIP

# Conductance
Q_PV_TO_EPV = 6*nS     
Q_SOM_TO_EPV = 5*nS     
Q_E_TO_E = 1.25*nS      
Q_E_TO_PV = 3.75*nS     
Q_E_TO_SOM = 2.5*nS   
Q_E_TO_VIP = 1.5*nS
Q_VIP_TO_SOM = 5.0*nS
Q_VIP_TO_PV = 2.0*nS 

Q_EXT = 1.25*nS        


# Time constants
TAU_E = 5*ms
TAU_I = 5*ms
TAU_E_PV = 1*ms
TAU_E_SOM = 2*ms
TAU_E_VIP = 2*ms # few data but seems slow?



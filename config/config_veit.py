from brian2 import *
from math import sqrt

DOWNSCALE = 1.0
CONTRAST = 1.0
VIP_RATE_HZ = 8.0

Ne = int(4000*DOWNSCALE)
Npv = int(500*DOWNSCALE)
Nsom = int(500*DOWNSCALE)

tau_m = 5.4*ms; tau_s = 0.6*ms; t_ref = 1.2*ms; syn_delay = 1.8*ms
EL = -60*mV; VT = -50*mV; DeltaT = 2*mV; V_reset = -75*mV; Vth = -40*mV

wE = 0.48*mV
wI = -1.92*mV   # g=4
wVIP_SST = -6.4*mV

mu_bg = {'E':3.0*mV,'PV':3.0*mV,'SOM':7.0*mV}
sd_bg = {'E':2.1*mV,'PV':2.1*mV,'SOM':3.0*mV}
mu_st = {'E':3.0*mV,'PV':3.0*mV,'SOM':0.0*mV}
sd_st = {'E':2.1*mV,'PV':2.1*mV,'SOM':0.0*mV}
sd_gl = 0.25*mV

def _mu(pop): return mu_bg[pop] + CONTRAST*mu_st[pop]
def _sd(pop): return sqrt(sd_bg[pop]**2 + (CONTRAST*sd_st[pop])**2 + sd_gl**2)

eqs = """
dv/dt = ( (EL - v) + DeltaT*exp((v - VT)/DeltaT) + sE + sI + I0 )/tau_m
         + sigma_tot*xi*tau_m**-0.5 : volt (unless refractory)
sE : volt
sI : volt
dsE/dt = -sE/tau_s : volt
dsI/dt = -sI/tau_s : volt
I0 : volt
sigma_tot : volt
"""

CONFIG = {
    'simulation': {'SIMULATION_TIME': 4000*ms, 'DT': 0.025*ms, 'RANDOM_SEED': 58879},
    'models': {
        'synapse_model': 'current',           
        'equations': {'E': eqs, 'PV': eqs, 'SOM': eqs, 'VIP': eqs},
        'threshold': 'v>Vth',
        'reset': 'v=V_RESET',
        'namespace': {                        
            'EL': EL, 'VT': VT, 'DeltaT': DeltaT, 'tau_m': tau_m, 'tau_s': tau_s, 'Vth': Vth
        }
    },
    'intrinsic_params': {
        'E': {'a':0*nS,'b':0*pA,'DeltaT':DeltaT},
        'PV':{'a':0*nS,'b':0*pA,'DeltaT':DeltaT},
        'SOM':{'a':0*nS,'b':0*pA,'DeltaT':DeltaT},
        'VIP':{'a':0*nS,'b':0*pA,'DeltaT':DeltaT},
    },
    'neurons': {
        'V_RESET': V_reset,
        'VT': VT, 'VTH': Vth,
        'EE': 0.*mV, 'EI': -80.*mV,   
        'T_REF': t_ref,
        'TAU_W': 1*ms,
        'CAPACITANCE': 200*pF,
        'LEAK_CONDUCTANCE': 10*nS,
        'E_LEAK': {'E':EL,'PV':EL,'SOM':EL,'VIP':EL},
        'INITIAL_VOLTAGE': -60*mV,
    },
    'initial_conditions': {
        'DEFAULT': {'v': -60*mV, 'sE': 0*mV, 'sI': 0*mV, 'I0': 0*mV, 'sigma_tot': 0*mV, 'Vcut_offset_factor': 0},
    },
    'time_constants': {'TAU_M': tau_m, 'TAU_S': tau_s, 'DELAY': syn_delay},
    'synapses': {  
        'Q': {
            'E_TO_E': wE, 'E_TO_PV': wE, 'E_TO_SOM': wE,
            'PV_TO_EPV': wI, 'SOM_TO_EPV': wI,     
            'VIP_TO_SOM': wVIP_SST,                
            'EXT': 0.48*mV,                       
        }
    },
    'layers': {
        'L23': {
            'connection_prob': 0.034,  
            'neuron_counts': {'E': Ne, 'PV': Npv, 'SOM': Nsom, 'VIP': 0},
            'poisson_inputs': {
                'SOM_vip': {
                    'target': 'sI',
                    'rate': VIP_RATE_HZ*Hz,
                    'weight': -6.4*mV,
                    'N': int(500*DOWNSCALE)
                }
            }
        }
    },
    'inter_layer_connections': {},  
    'per_pop_drive': {
        'E': (_mu('E'), _sd('E')),
        'PV': (_mu('PV'), _sd('PV')),
        'SOM': (_mu('SOM'), _sd('SOM')),
    }
}

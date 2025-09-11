# One-layer (L23) Veit et al. 2023 configuration

import brian2 as b2
from brian2 import ms, mV, Hz, nS, pA

N_E   = 4000
N_PV  = 500
N_SOM = 500   
N_VIP = 500

TAU_M   = 5.4*ms        
TAU_S   = 0.6*ms        
T_REF   = 1.2*ms       
DELAY   = 1.8*ms       

EL   = -60.0*mV        
VT   = -50.0*mV         
VTH  = -20.0*mV        
VRES = -75.0*mV         
DELTA_T = 2.0*mV       

w_mVmsec_exc = 0.48     
g_inh        = 4.0      
w_exc = (w_mVmsec_exc/float(TAU_S/ms)) * mV  
w_inh = g_inh * w_exc                        

w_vip_mVmsec = -6.4     
w_vip_to_SST = (w_vip_mVmsec/float(TAU_S/ms)) * mV  

contrast_c = 1.0

mu_bg = {'E': 3.0*mV, 'PV': 3.0*mV, 'SOM': 7.0*mV}
sd_bg = {'E': 2.1*mV, 'PV': 2.1*mV, 'SOM': 3.0*mV}
mu_stim = {'E': 3.0*mV, 'PV': 3.0*mV, 'SOM': 0.0*mV}
sd_stim = {'E': 2.1*mV, 'PV': 2.1*mV, 'SOM': 0.0*mV}

sigma_global = 0.25*mV

vip_rate = 14*Hz

EIF_EQS = """
dv/dt = (-(v - EL) + DELTA_T*exp((v - VT)/DELTA_T) + sE - sI + mu_drive)/TAU_M
        + sigma_drive*sqrt(2/TAU_M)*xi : volt (unless refractory)
dsE/dt = -sE/TAU_S : volt
dsI/dt = -sI/TAU_S : volt
mu_drive : volt
sigma_drive : volt
"""


P_conn_same = {
    'E_E'   : 0.07,  
    'PV_E'  : 0.15,   
    'SOM_E' : 0.10,  
    'E_PV'  : 0.05,   
    'PV_PV' : 0.10,   
    'SOM_PV': 0.10,   
    'E_SOM' : 0.10,   
    'PV_SOM': 0.00,   
    'SOM_SOM': 0.00, 
    'VIP_SOM': 1.0,   
}

_LAYER_CONFIGS = {
    'L23': {
        'connection_prob': P_conn_same,
        'input_rate': None,   
        'neuron_counts': {
            'E':   N_E,
            'PV':  N_PV,
            'SOM': N_SOM,
            'VIP': N_VIP,    
        },
   
        'poisson_inputs': {}, 
    }
}

CONFIG = {
    'simulation': {
        'SIMULATION_TIME': 1000*ms,
        'DT': 0.025*ms,
        'RANDOM_SEED': 12345,
    },

    'models': {
        'synapse_model': 'current', 
        'equations': {
            'E':   EIF_EQS,
            'PV':  EIF_EQS,
            'SOM': EIF_EQS,
            'VIP': """
            dv/dt = 0*volt/second : volt
            dsE/dt = -sE/TAU_S : volt
            dsI/dt = -sI/TAU_S : volt
            mu_drive : volt
            sigma_drive : volt
            """,
        },
        'threshold': 'v > V_th',
        'reset':     'v = V_reset',
        'common_namespace': {
            'TAU_M': TAU_M,
            'TAU_S': TAU_S,
            'EL': EL,
            'VT': VT,
            'DELTA_T': DELTA_T,
            'V_th': VTH,
            'V_reset': VRES,
        },
        'namespace': {}
    },

    'intrinsic_params': {
        'E':   {},
        'PV':  {},
        'SOM': {},
        'VIP': {},
    },

    'neurons': {
        'V_RESET': VRES,
        'VT': VT,
        'EE': 0.0*mV,   
        'EI': -80.0*mV, 
        'T_REF': T_REF,
        'TAU_W': 0.0*ms,     
        'CAPACITANCE': 200*pA*ms/mV, 
        'LEAK_CONDUCTANCE': 10*nS,   
        'INITIAL_VOLTAGE': EL,
        'E_LEAK': { 'E': EL, 'PV': EL, 'SOM': EL, 'VIP': EL },
    },

    'initial_conditions': {
        'DEFAULT': {'v': EL, 'sE': 0.0*mV, 'sI': 0.0*mV, 'mu_drive': 0.0*mV, 'sigma_drive': 0.0*mV},
        'E':   {},
        'PV':  {},
        'SOM': {},
        'VIP': {'v': EL},
    },

    'time_constants': {
        'E': TAU_S,  
        'I': TAU_S,
        'DELAY': DELAY,
    },

    'synapses': {
        'Q': {
            'E_E'   :  w_exc,
            'PV_E'  :  w_inh,   
            'SOM_E' :  w_inh,
            'E_PV'  :  w_exc,
            'PV_PV' :  w_inh,
            'SOM_PV':  w_inh,
            'E_SOM' :  w_exc,
            'PV_SOM':  0.0*mV, 
            'SOM_SOM': 0.0*mV, 
            'VIP_SOM': abs(w_vip_to_SST), 
            'EXT'    :  w_exc,  
        }
    },

    'layers': _LAYER_CONFIGS,

    'inter_layer_connections': {},


    'drive': {
        'contrast': contrast_c,
        'mu_bg':   mu_bg,
        'sd_bg':   sd_bg,
        'mu_stim': mu_stim,
        'sd_stim': sd_stim,
        'sigma_global': sigma_global,
        'vip_rate': vip_rate,
    }
}

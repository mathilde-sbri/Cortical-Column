from brian2 import *
import pandas as pd
from collections import defaultdict

conn_df = pd.read_csv('scaled_matrix_0_to_0.34.csv', index_col=0, skipinitialspace=True)
conductances =  pd.read_csv('conductance2.csv', index_col=0, skipinitialspace=True)
p = 1
q = 1

_LAYER_CONFIGS = {
    'L1': {
        'connection_prob': {
     },
     'conductance': {
     },
        'input_rate': 10*Hz,
        'neuron_counts': {
            'E': 8,
            'PV': 1,
            'SOM': 1,
            'VIP': 39
        },
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 4},
            'PV': {'target': 'gE', 'weight': 'EXT', 'N': 1},
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 1},
        }
    },
    'L23': {
        'connection_prob': {

        },
        'conductance': {
     },
        'input_rate': 10*Hz,
        'neuron_counts': {'E': 3520, 'PV': 317, 'SOM': 475, 'VIP': 88},
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 60},
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 25},
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 25},
        }
    },

    'L4': {
        'connection_prob': {
            
        },
        'conductance': {
     },
        'input_rate': 10*Hz,
        'neuron_counts': {'E': 5760, 'PV': 950, 'SOM': 420, 'VIP': 70},
        'poisson_inputs': {
           'E':  {'target': 'gE',  'weight': 'EXT', 'N': 60},
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 25},
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 25},
    
        }
    },

    'L5': {
        'connection_prob': {
           
        },
        'conductance': {
     },
        'input_rate': 10*Hz,
        'neuron_counts': {'E': 1600, 'PV': 208, 'SOM': 152, 'VIP': 40},
        'poisson_inputs': {
           'E':  {'target': 'gE',  'weight': 'EXT', 'N': 60},
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 25},
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 25},
        }
    },

    'L6': {
        'connection_prob': {
           
        },
        'conductance': {
     },
        'input_rate': 10*Hz,
        'neuron_counts': {'E': 2040, 'PV': 187, 'SOM': 137, 'VIP': 36},
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 60},
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 25},
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 25},
        }
    }
}

_layer_csv = {
    
    'L23': {'E_row': 'E2/3',   'PV_row': 'i2/3Pva',  'SOM_row': 'i2/3Sst',  'VIP_row': 'i2/3Htr',
            'E_col': 'E2/3',   'PV_col': 'i2/3 Pvalb','SOM_col': 'i2/3 Sst','VIP_col': 'i2/3 Htr3a'},
    'L4' : {'E_row': 'E4',     'PV_row': 'i4Pvalb',  'SOM_row': 'i4Sst',    'VIP_row': 'i4Htr3a',
            'E_col': 'E4',     'PV_col': 'i4 Pvalb', 'SOM_col': 'i4 Sst',   'VIP_col': 'i4 Htr3a'},
    'L5' : {'E_row': 'E5',     'PV_row': 'i5Pvalb',  'SOM_row': 'i5Sst',    'VIP_row': 'i5Htr3a',
            'E_col': 'E5',     'PV_col': 'i5 Pvalb', 'SOM_col': 'i5 Sst',   'VIP_col': 'i5 Htr3a'},
    'L6' : {'E_row': 'E6',     'PV_row': 'i6Pvalb',  'SOM_row': 'i6Sst',    'VIP_row': 'i6Htr3a',
            'E_col': 'E6',     'PV_col': 'i6 Pvalb', 'SOM_col': 'i6 Sst',   'VIP_col': 'i6 Htr3a'},
}

def _prob(src_row, tgt_col):
    return p * (conn_df.loc[src_row].to_dict()[tgt_col])

def _cond(src_row, tgt_col):
    return q*conductances.loc[src_row].to_dict()[tgt_col]



_INTER_LAYER_CONNECTIONS = defaultdict(dict)
_INTER_LAYER_CONDUCTANCES = defaultdict(dict)

_layers = ['L23', 'L4', 'L5', 'L6']
_pops = ['E', 'PV', 'SOM', 'VIP']

for src in _layers:
    for dst in _layers:
        s = _layer_csv[src]
        t = _layer_csv[dst]
        for src_pop in _pops:
            for dst_pop in _pops:
                conn = f'{src_pop}_{dst_pop}'
                row  = f'{src_pop}_row'
                col  = f'{dst_pop}_col'
                if src == dst:
                    _LAYER_CONFIGS[src].setdefault('connection_prob', {})[conn] = _prob(s[row], t[col])
                    _LAYER_CONFIGS[src].setdefault('conductance', {})[conn] = _cond(s[row], t[col])
                else:
                    _INTER_LAYER_CONNECTIONS[(src, dst)][conn]  = _prob(s[row], t[col])
                    _INTER_LAYER_CONDUCTANCES[(src, dst)][conn] = _cond(s[row], t[col])


tau_e_AMPA = 5*ms
tau_i_PV   = 6*ms
tau_i_SOM  = 30*ms
tau_i_VIP  = 8*ms
v_reset = -65.*mV
vt = -50.*mV
ee = 0.*mV
ei = -80.*mV
t_ref = 5*ms

CONFIG = {
    'simulation': {
        'SIMULATION_TIME': 4000*ms,
        'DT': 0.1*ms,
        'RANDOM_SEED': 58879,
    },
    'models': {
        'equations': {
            'E': """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + gE*(Ee - v) - (gPV + gSOM + gVIP)*(v - Ei) - w + I)/C : volt (unless refractory)
        IsynE   = gE*(Ee - v) : amp
        IsynIPV = gPV*(Ei - v) : amp
        IsynISOM= gSOM*(Ei - v) : amp
        IsynIVIP= gVIP*(Ei - v) : amp
        gI = gPV + gSOM + gVIP : siemens
        IsynI   = gI*(Ei - v) : amp
        dgE/dt   = -gE/tau_e_AMPA : siemens
        dgPV/dt  = -gPV/tau_i_PV  : siemens
        dgSOM/dt = -gSOM/tau_i_SOM: siemens
        dgVIP/dt = -gVIP/tau_i_VIP: siemens
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
        tauw : second
            """,
            'PV': """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + gE*(Ee - v) - (gPV + gSOM + gVIP)*(v - Ei) - w + I)/C : volt (unless refractory)
        IsynE   = gE*(Ee - v) : amp
        IsynIPV = gPV*(Ei - v) : amp
        IsynISOM= gSOM*(Ei - v) : amp
        IsynIVIP= gVIP*(Ei - v) : amp
        gI = gPV + gSOM + gVIP : siemens
        IsynI   = gI*(Ei - v) : amp
        dgE/dt   = -gE/tau_e_AMPA : siemens
        dgPV/dt  = -gPV/tau_i_PV  : siemens
        dgSOM/dt = -gSOM/tau_i_SOM: siemens
        dgVIP/dt = -gVIP/tau_i_VIP: siemens
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
        tauw : second
            """,
            'SOM': """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + gE*(Ee - v) - (gPV + gSOM + gVIP)*(v - Ei) - w + I)/C : volt (unless refractory)
        IsynE   = gE*(Ee - v) : amp
        IsynIPV = gPV*(Ei - v) : amp
        IsynISOM= gSOM*(Ei - v) : amp
        IsynIVIP= gVIP*(Ei - v) : amp
        gI = gPV + gSOM + gVIP : siemens
        IsynI   = gI*(Ei - v) : amp
        dgE/dt   = -gE/tau_e_AMPA : siemens
        dgPV/dt  = -gPV/tau_i_PV  : siemens
        dgSOM/dt = -gSOM/tau_i_SOM: siemens
        dgVIP/dt = -gVIP/tau_i_VIP: siemens
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
        tauw : second
            """,
            'VIP': """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + gE*(Ee - v) - (gPV + gSOM + gVIP)*(v - Ei) - w + I)/C : volt (unless refractory)
        IsynE   = gE*(Ee - v) : amp
        IsynIPV = gPV*(Ei - v) : amp
        IsynISOM= gSOM*(Ei - v) : amp
        IsynIVIP= gVIP*(Ei - v) : amp
        gI = gPV + gSOM + gVIP : siemens
        IsynI   = gI*(Ei - v) : amp
        dgE/dt   = -gE/tau_e_AMPA : siemens
        dgPV/dt  = -gPV/tau_i_PV  : siemens
        dgSOM/dt = -gSOM/tau_i_SOM: siemens
        dgVIP/dt = -gVIP/tau_i_VIP: siemens
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
        tauw : second
            """,
        },
        'threshold': 'v>Vcut',
        'reset': 'v=V_reset; w+=b',
        'common_namespace' : {
            'tau_e_AMPA': tau_e_AMPA,
            'tau_i_PV': tau_i_PV,
            'tau_i_SOM': tau_i_SOM,
            'tau_i_VIP': tau_i_VIP,
            'VT': vt,
            'V_reset': v_reset,
            'Ee': ee,
            'Ei': ei,
        },
    },

    'intrinsic_params': {
        'E':   {'a': 4*nS, 'b': 130*pA, 'DeltaT': 2*mV, 'C': 97*pF, 'gL': 4.2*nS, 'tauw': 200*ms, 'EL': -66*mV},
        'PV':  {'a': 0*nS, 'b': 0*pA,   'DeltaT': 0.5*mV, 'C': 38*pF, 'gL': 3.8*nS, 'tauw': 50*ms, 'EL': -68*mV},
        'SOM': {'a': 4*nS, 'b': 25*pA,  'DeltaT': 1.5*mV, 'C': 37*pF, 'gL': 2.3*nS, 'tauw': 300*ms, 'EL': -68*mV},
        'VIP': {'a': 2*nS, 'b': 50*pA,  'DeltaT': 2*mV, 'C': 37*pF, 'gL': 4*nS, 'tauw': 150*ms, 'EL': -65*mV},
    },
    'neurons': {
        'V_RESET': -65.*mV,
        'VT': -50.*mV,
        'EE': 0.*mV,
        'EI': -80.*mV,
        'T_REF': 5*ms,
        'INITIAL_VOLTAGE': -60*mV,
    },
    'initial_conditions': {
        'DEFAULT': {'v': -60*mV, 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'E':   {'v': -60*mV, 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'PV':  {'v': -60*mV, 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'SOM': {'v': -60*mV, 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'VIP': {'v': -60*mV, 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
    },
    'time_constants': {
        'E_AMPA': tau_e_AMPA,
        'I_PV': tau_i_PV,
        'I_SOM': tau_i_SOM,
        'I_VIP': tau_i_VIP,
    },
    'synapses': {
        'Q': {
   
            'EXT': 0.55*nS,
        },
    },
       

    'layers': _LAYER_CONFIGS,
    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
    'inter_layer_conductances': _INTER_LAYER_CONDUCTANCES,
}
from brian2 import *
import pandas as pd
from collections import defaultdict

conn_df = pd.read_csv('conn_prob.csv', index_col=0, skipinitialspace=True)

conductances = pd.read_csv('conductances.csv', index_col=0, skipinitialspace=True)*0.5


_LAYER_CONFIGS = {
    'L1': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 13*Hz, 
        'neuron_counts': {
            'E': 8,
            'PV': 1,
            'SOM': 1,
            'VIP': 39
        },
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 8}, 
            'PV': {'target': 'gE', 'weight': 'EXT', 'N': 3},  
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 3},
            'VIP': {'target': 'gE', 'weight': 'EXT', 'N': 25},
        },
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (1.1, 1.35),
        }
    },
    'L23': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 10*Hz, 
        'neuron_counts': {'E': 3520, 'PV': 317, 'SOM': 475, 'VIP': 88},
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 80},  
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 35},  
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 35},
            'VIP': {'target': 'gE', 'weight': 'EXT', 'N': 20},
        },
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (0.4, 1.1),
        }
    },
    'L4': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 13*Hz, 
        'neuron_counts': {'E': 5760, 'PV': 950, 'SOM': 420, 'VIP': 70},
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 100}, 
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 40},  
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 40},
            'VIP': {'target': 'gE', 'weight': 'EXT', 'N': 30},  
        },
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.3, 0.4),
        }
    },
    'L5': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 10*Hz,  
        'neuron_counts': {'E': 1600, 'PV': 208, 'SOM': 152, 'VIP': 40},
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 80}, 
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 25},  
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 25},
            'VIP': {'target': 'gE', 'weight': 'EXT', 'N': 25},
        },
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.6, -0.3),
        }
    },
    'L6': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 12*Hz, 
        'neuron_counts': {'E': 2040, 'PV': 187, 'SOM': 137, 'VIP': 36},
        'poisson_inputs': {
            'E':  {'target': 'gE',  'weight': 'EXT', 'N': 50}, 
            'PV': {'target': 'gE',  'weight': 'EXT', 'N': 20},  
            'SOM': {'target': 'gE', 'weight': 'EXT', 'N': 20},
            'VIP': {'target': 'gE', 'weight': 'EXT', 'N': 15},
        },
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-1.0, -0.6),
        }
    }
}

_layer_csv = {
    'L1': {'VIP_row': 'i1Htr3a',
           'VIP_col': 'i1Htr3a'},
    'L23': {'E_row': 'E2/3',   'PV_row': 'i2/3Pvalb','SOM_row': 'i2/3Sst',  'VIP_row': 'i2/3Htr3a',
            'E_col': 'E2/3',   'PV_col': 'i2/3Pvalb','SOM_col': 'i2/3Sst',  'VIP_col': 'i2/3Htr3a'},
    'L4' : {'E_row': 'E4',     'PV_row': 'i4Pvalb',  'SOM_row': 'i4Sst',    'VIP_row': 'i4Htr3a',
            'E_col': 'E4',     'PV_col': 'i4Pvalb', 'SOM_col': 'i4Sst',   'VIP_col': 'i4Htr3a'},
    'L5' : {'E_row': 'E5',     'PV_row': 'i5Pvalb',  'SOM_row': 'i5Sst',    'VIP_row': 'i5Htr3a',
            'E_col': 'E5',     'PV_col': 'i5Pvalb', 'SOM_col': 'i5Sst',   'VIP_col': 'i5Htr3a'},
    'L6' : {'E_row': 'E6',     'PV_row': 'i6Pvalb',  'SOM_row': 'i6Sst',    'VIP_row': 'i6Htr3a',
            'E_col': 'E6',     'PV_col': 'i6Pvalb', 'SOM_col': 'i6Sst',   'VIP_col': 'i6Htr3a'},
}

def _prob(src_row, tgt_col):
    try:
        val = conn_df.loc[src_row].to_dict()[tgt_col]
        return  val
    except KeyError:
        print(f"WARNING: Could not find connection {src_row} -> {tgt_col}")
        return 0
    

def _cond(src_row, tgt_col):
    try:
        return conductances.loc[src_row].to_dict()[tgt_col]
    except KeyError:
        print(f"WARNING: Could not find conductance {src_row} -> {tgt_col}")
        return 0.0

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



for layer in ['L23', 'L4', 'L5', 'L6']:
    cfg = _LAYER_CONFIGS[layer]
    
    for conn_type in cfg['conductance'].keys():
        if 'PV' in conn_type.split('_')[0]: 
            cfg['conductance'][conn_type] *= 1.5
        if 'SOM' in conn_type.split('_')[0]: 
            cfg['conductance'][conn_type] *= 1.3
        if 'VIP' in conn_type.split('_')[0]: 
            cfg['conductance'][conn_type] *= 1.4

for (src, dst), conductances in _INTER_LAYER_CONDUCTANCES.items():
    for conn_type in conductances.keys():
        if 'PV' in conn_type.split('_')[0]:
            _INTER_LAYER_CONDUCTANCES[(src, dst)][conn_type] *= 1.5

for layer in ['L23', 'L4', 'L5', 'L6']:
    cfg = _LAYER_CONFIGS[layer]
    
    if 'E_E' in cfg['conductance']:
        cfg['conductance']['E_E'] *= 0.6 
    
    if 'E_PV' in cfg['conductance']:
        cfg['conductance']['E_PV'] *= 1.0





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
        'E':   {'a': 4*nS, 'b': 110*pA, 'DeltaT': 2*mV, 'C': 97*pF, 'gL': 4.2*nS, 'tauw': 200*ms, 'EL': -66*mV},
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
        'DEFAULT': {'v': '-60*mV + rand()*8*mV', 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'E':   {'v': '-60*mV + rand()*8*mV', 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'PV':  {'v': '-60*mV + rand()*8*mV', 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'SOM': {'v': '-60*mV + rand()*8*mV', 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'VIP': {'v': '-60*mV + rand()*8*mV', 'gE': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
    },
    'time_constants': {
        'E_AMPA': tau_e_AMPA,
        'I_PV': tau_i_PV,
        'I_SOM': tau_i_SOM,
        'I_VIP': tau_i_VIP,
    },
    'synapses': {
        'Q': {
            'EXT': 0.8*nS,
        },
    },
    
    'layers': _LAYER_CONFIGS,
    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
    'inter_layer_conductances': _INTER_LAYER_CONDUCTANCES,
}

from brian2 import *
import pandas as pd

conn_df = pd.read_csv('scaled_matrix_0_to_0.34.csv', index_col=0, skipinitialspace=True)

p = 1

_LAYER_CONFIGS = {
    'L1': {
        'connection_prob': {
            'E_E': p,
            'E_PV': p,
            'E_SOM': p,
            'E_VIP': p,
            'PV_E': p,
            'PV_PV': p,
            'SOM_E': p,
            'SOM_PV': p,
            'VIP_SOM': p,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 80,
            'PV': 10,
            'SOM': 10,
            'VIP': 390
        },
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.012},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.012},
        }
    },
    'L23': {
        'connection_prob': {
            'E_E'   : p*(conn_df.loc["E2/3"].to_dict()["E2/3"]),
            'E_PV'  : p*(conn_df.loc["E2/3"].to_dict()["i2/3 Pvalb"]),
            'E_SOM' : p*(conn_df.loc["E2/3"].to_dict()["i2/3 Sst"]),
            'E_VIP' : p*(conn_df.loc["E2/3"].to_dict()["i2/3 Htr3a"]),

            'PV_E'  : p*(conn_df.loc["i2/3Pva"].to_dict()["E2/3"]),
            'PV_PV' : p*(conn_df.loc["i2/3Pva"].to_dict()["i2/3 Pvalb"]),

            'SOM_E' : p*(conn_df.loc["i2/3Sst"].to_dict()["E2/3"]),
            'SOM_PV': p*(conn_df.loc["i2/3Sst"].to_dict()["i2/3 Pvalb"]),

            'VIP_SOM': p*(conn_df.loc["i2/3Htr"].to_dict()["i2/3 Sst"]),
        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 3520, 'PV': 317, 'SOM': 475, 'VIP': 88},
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.012},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.012},
        }
    },

    'L4': {
        'connection_prob': {
            'E_E'   : p*(conn_df.loc["E4"].to_dict()["E4"]),
            'E_PV'  : p*(conn_df.loc["E4"].to_dict()["i4 Pvalb"]),
            'E_SOM' : p*(conn_df.loc["E4"].to_dict()["i4 Sst"]),
            'E_VIP' : p*(conn_df.loc["E4"].to_dict()["i4 Htr3a"]),

            'PV_E'  : p*(conn_df.loc["i4Pvalb"].to_dict()["E4"]),
            'PV_PV' : p*(conn_df.loc["i4Pvalb"].to_dict()["i4 Pvalb"]),

            'SOM_E' : p*(conn_df.loc["i4Sst"].to_dict()["E4"]),
            'SOM_PV': p*(conn_df.loc["i4Sst"].to_dict()["i4 Pvalb"]),

            'VIP_SOM': p*(conn_df.loc["i4Htr3a"].to_dict()["i4 Sst"]),
        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 5760, 'PV': 950, 'SOM': 420, 'VIP': 70},
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 10*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'gE', 'rate': 10*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
        }
    },

    'L5': {
        'connection_prob': {
            'E_E'   : p*(conn_df.loc["E5"].to_dict()["E5"]),
            'E_PV'  : p*(conn_df.loc["E5"].to_dict()["i5 Pvalb"]),
            'E_SOM' : p*(conn_df.loc["E5"].to_dict()["i5 Sst"]),
            'E_VIP' : p*(conn_df.loc["E5"].to_dict()["i5 Htr3a"]),

            'PV_E'  : p*(conn_df.loc["i5Pvalb"].to_dict()["E5"]),
            'PV_PV' : p*(conn_df.loc["i5Pvalb"].to_dict()["i5 Pvalb"]),

            'SOM_E' : p*(conn_df.loc["i5Sst"].to_dict()["E5"]),
            'SOM_PV': p*(conn_df.loc["i5Sst"].to_dict()["i5 Pvalb"]),

            'VIP_SOM': p*(conn_df.loc["i5Htr3a"].to_dict()["i5 Sst"]),
        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 1600, 'PV': 208, 'SOM': 152, 'VIP': 40},
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.012},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.012},
        }
    },

    'L6': {
        'connection_prob': {
            'E_E'   : p*(conn_df.loc["E6"].to_dict()["E6"]),
            'E_PV'  : p*(conn_df.loc["E6"].to_dict()["i6 Pvalb"]),
            'E_SOM' : p*(conn_df.loc["E6"].to_dict()["i6 Sst"]),
            'E_VIP' : p*(conn_df.loc["E6"].to_dict()["i6 Htr3a"]),

            'PV_E'  : p*(conn_df.loc["i6Pvalb"].to_dict()["E6"]),
            'PV_PV' : p*(conn_df.loc["i6Pvalb"].to_dict()["i6 Pvalb"]),

            'SOM_E' : p*(conn_df.loc["i6Sst"].to_dict()["E6"]),
            'SOM_PV': p*(conn_df.loc["i6Sst"].to_dict()["i6 Pvalb"]),

            'VIP_SOM': p*(conn_df.loc["i6Htr3a"].to_dict()["i6 Sst"]),
        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 2040, 'PV': 187, 'SOM': 137, 'VIP': 36},
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
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

_INTER_LAYER_CONNECTIONS = {}
_layers = ['L23', 'L4', 'L5', 'L6']

for src in _layers:
    for dst in _layers:
        if src == dst:
            continue  
        s = _layer_csv[src]
        t = _layer_csv[dst]
        _INTER_LAYER_CONNECTIONS[(src, dst)] = {
            'E_E'   : _prob(s['E_row'] , t['E_col']),
            'E_PV'  : _prob(s['E_row'] , t['PV_col']),
            'E_SOM' : _prob(s['E_row'] , t['SOM_col']),
            'E_VIP' : _prob(s['E_row'] , t['VIP_col']),

            'PV_E'  : _prob(s['PV_row'], t['E_col']),
            'PV_PV' : _prob(s['PV_row'], t['PV_col']),

            'SOM_E' : _prob(s['SOM_row'], t['E_col']),
            'SOM_PV': _prob(s['SOM_row'], t['PV_col']),

            'VIP_SOM': _prob(s['VIP_row'], t['SOM_col']),
        }



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
        'SIMULATION_TIME': 1000*ms,
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
        'VIP': {'a': 2*nS, 'b': 50*pA,  'DeltaT': 2*mV, 'C': 1537*pF, 'gL': 4*nS, 'tauw': 150*ms, 'EL': -65*mV},
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
            'PV_PV': 6*nS,
            'PV_E': 6*nS,
            'SOM_E': 5*nS,
            'SOM_PV': 5*nS,
            'E_E': 1.75*nS,
            'E_PV': 3.75*nS,
            'E_SOM': 2.5*nS,
            'E_VIP': 1.5*nS,
            'VIP_SOM': 5.0*nS,
            'EXT': 1.25*nS,
        },
       
    },
    'layers': _LAYER_CONFIGS,
    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
}

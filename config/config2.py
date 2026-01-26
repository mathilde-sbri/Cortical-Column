from brian2 import *
import numpy as np
import pandas as pd
from collections import defaultdict


tau_e_AMPA = 5*ms
tau_e_NMDA = 100*ms
tau_i_PV   = 6*ms
tau_i_SOM  = 30*ms
tau_i_VIP  = 8*ms

v_reset = -65.*mV
vt      = -50.*mV
ee      = 0.*mV
ei      = -80.*mV
t_ref   = 5*ms

alpha_g = 1.0
beta_p  = 1.0

def g_NMDA(v_mV):
    return 1.0 / (1.0 + 0.28 * np.exp(-0.062 * v_mV))


def load_connectivity_from_csv(conn_prob_file, cond_ampa_file, cond_nmda_file):
    """
    Load connectivity matrices from CSV files.
    Returns both intra-layer configs and inter-layer connection dictionaries.
    """
    conn_prob_df = pd.read_csv(conn_prob_file, index_col=0)
    cond_ampa_df = pd.read_csv(cond_ampa_file, index_col=0)
    cond_nmda_df = pd.read_csv(cond_nmda_file, index_col=0)
    
    layers = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
    cell_types = ['E', 'PV', 'SOM', 'VIP']
    
    layer_name_map = {
        'L23': 'L23',
        'L4AB': 'L4AB', 
        'L4C': 'L4C',
        'L5': 'L5',
        'L6': 'L6'
    }
    
    layer_configs = {}
    inter_layer_connections = {}
    inter_layer_conductances = {}
    
    for layer in layers:

        connection_prob = {}
        conductance = {}
        
        for src_type in cell_types:
            for tgt_type in cell_types:
                src_col = f'{src_type}_{layer}'
                tgt_col = f'{tgt_type}_{layer}'
                
                prob_val = conn_prob_df.loc[src_col, tgt_col]
                if prob_val > 0:
                    conn_key = f'{src_type}_{tgt_type}'
                    connection_prob[conn_key] = prob_val
                
                # Conductances
                if src_type == 'E':
                    ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                    nmda_val = cond_nmda_df.loc[src_col, tgt_col]
                    
                    if ampa_val > 0:
                        cond_key = f'{src_type}_{tgt_type}_AMPA'
                        conductance[cond_key] = ampa_val
                    
                    if nmda_val > 0:
                        cond_key = f'{src_type}_{tgt_type}_NMDA'
                        conductance[cond_key] = nmda_val
                else:
                    ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                    if ampa_val > 0:
                        cond_key = f'{src_type}_{tgt_type}'
                        conductance[cond_key] = ampa_val
        
        layer_configs[layer] = {
            'connection_prob': connection_prob,
            'conductance': conductance
        }
    

    for src_layer in layers:
        for tgt_layer in layers:
            if src_layer == tgt_layer:
                continue  
            
            conn_dict = {}
            cond_dict = {}
            
            for src_type in cell_types:
                for tgt_type in cell_types:
                    src_col = f'{src_type}_{src_layer}'
                    tgt_col = f'{tgt_type}_{tgt_layer}'
                    
                    prob_val = conn_prob_df.loc[src_col, tgt_col]
                    
                    if prob_val > 0:
                        conn_key = f'{src_type}_{tgt_type}'
                        conn_dict[conn_key] = prob_val
                        
                        if src_type == 'E':
                            ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                            nmda_val = cond_nmda_df.loc[src_col, tgt_col]
                            
                            if ampa_val > 0:
                                cond_dict[f'{src_type}_{tgt_type}_AMPA'] = ampa_val
                            if nmda_val > 0:
                                cond_dict[f'{src_type}_{tgt_type}_NMDA'] = nmda_val
                        else:
                            ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                            if ampa_val > 0:
                                cond_dict[f'{src_type}_{tgt_type}'] = ampa_val
            
            if conn_dict:
                inter_layer_connections[(src_layer, tgt_layer)] = conn_dict
                inter_layer_conductances[(src_layer, tgt_layer)] = cond_dict
    
    return layer_configs, inter_layer_connections, inter_layer_conductances




csv_layer_configs, _INTER_LAYER_CONNECTIONS, _INTER_LAYER_CONDUCTANCES = load_connectivity_from_csv(
    'config/connection_probabilities2.csv',
    'config/conductances_AMPA2.csv',
    'config/conductances_NMDA2.csv'
)


_LAYER_CONFIGS = {
   

    'L23': {
        'connection_prob': csv_layer_configs['L23']['connection_prob'],
        'conductance': csv_layer_configs['L23']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 65},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 22},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 22},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 22},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 65},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 22},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 22},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 22},
        },
        'input_rate': 10*Hz,
        'neuron_counts': {'E': 3520, 'PV': 317, 'SOM': 475, 'VIP': 88},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (0.45, 1.1),
        },
    },

    'L4AB': {
        'connection_prob': csv_layer_configs['L4AB']['connection_prob'],
        'conductance': csv_layer_configs['L4AB']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 60},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 22},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 24}, 
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 24},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 60},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 22},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 24},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 18},
        },
        'input_rate': 8*Hz,
        'neuron_counts': {'E': 2720, 'PV': 408, 'SOM': 204, 'VIP': 68},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (0.14, 0.45),
        }
    },

     'L4C': {
        'connection_prob': csv_layer_configs['L4C']['connection_prob'],
        'conductance': csv_layer_configs['L4C']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 60},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 60},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
        },
        'input_rate': 10*Hz,
        'neuron_counts': {'E': 3192, 'PV': 365, 'SOM': 182, 'VIP': 61},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.14, 0.14),
        }
    },

    'L5': {
        'connection_prob': csv_layer_configs['L5']['connection_prob'],
        'conductance': csv_layer_configs['L5']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 60},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 60},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
        },
        'input_rate': 8*Hz,
        'neuron_counts': {'E': 1600, 'PV': 208, 'SOM': 152, 'VIP': 40},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.34, -0.14),
        }
    },

    'L6': {
        'connection_prob': csv_layer_configs['L6']['connection_prob'],
        'conductance': csv_layer_configs['L6']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 60},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 28},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 28},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 60},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 28},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 18},
        },
        'input_rate': 10*Hz,
        'neuron_counts': {'E': 2040, 'PV': 187, 'SOM': 137, 'VIP': 36},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.62, -0.34),
        }
    },
}


CONFIG = {
    'simulation': {
        'SIMULATION_TIME': 2000*ms,
        'DT': 0.1*ms,
        'RANDOM_SEED': 58879,
    },

    'models': {
        'equations': {
            'E': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

        dw/dt   = (a*(v - EL) - w)/tauw : amp
        taum    = C/gL : second
        I       : amp
        a       : siemens
        b       : amp
        DeltaT  : volt
        Vcut    : volt
        EL      : volt
        C       : farad
        gL      : siemens
        tauw    : second
            """,

            'PV': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

        dw/dt   = (a*(v - EL) - w)/tauw : amp
        taum    = C/gL : second
        I       : amp
        a       : siemens
        b       : amp
        DeltaT  : volt
        Vcut    : volt
        EL      : volt
        C       : farad
        gL      : siemens
        tauw    : second
            """,

            'SOM': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

        dw/dt   = (a*(v - EL) - w)/tauw : amp
        taum    = C/gL : second
        I       : amp
        a       : siemens
        b       : amp
        DeltaT  : volt
        Vcut    : volt
        EL      : volt
        C       : farad
        gL      : siemens
        tauw    : second
            """,

            'VIP': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

        dw/dt   = (a*(v - EL) - w)/tauw : amp
        taum    = C/gL : second
        I       : amp
        a       : siemens
        b       : amp
        DeltaT  : volt
        Vcut    : volt
        EL      : volt
        C       : farad
        gL      : siemens
        tauw    : second
            """,
        },

        'threshold': 'v>Vcut',
        'reset': 'v=V_reset; w+=b',

        'common_namespace': {
            'tau_e_AMPA': tau_e_AMPA,
            'tau_e_NMDA': tau_e_NMDA,
            'tau_i_PV':   tau_i_PV,
            'tau_i_SOM':  tau_i_SOM,
            'tau_i_VIP':  tau_i_VIP,
            'VT':         vt,
            'V_reset':    v_reset,
            'Ee':         ee,
            'Ei':         ei,
        },
    },

    'intrinsic_params': {
        'E':   {'a': 4*nS, 'b': 20*pA, 'DeltaT': 2*mV,
                'C': 97*pF, 'gL': 4.2*nS, 'tauw': 150*ms, 'EL': -66*mV},
        'PV':  {'a': 0*nS, 'b': 0*pA, 'DeltaT': 0.5*mV,
                'C': 38*pF, 'gL': 3.8*nS, 'tauw': 50*ms,  'EL': -68*mV},
        'SOM': {'a': 4*nS, 'b': 25*pA, 'DeltaT': 1.5*mV,
                'C': 37*pF, 'gL': 2.3*nS, 'tauw': 300*ms, 'EL': -68*mV},
        'VIP': {'a': 2*nS, 'b': 50*pA, 'DeltaT': 2*mV,
                'C': 37*pF, 'gL': 4*nS, 'tauw': 150*ms, 'EL': -65*mV},
    },

    'neurons': {
        'V_RESET': v_reset,
        'VT':      vt,
        'EE':      ee,
        'EI':      ei,
        'T_REF':   t_ref,
        'INITIAL_VOLTAGE': -60*mV,
    },

    'initial_conditions': {
        'DEFAULT': {'v': '-60*mV + rand()*8*mV',
                    'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                    'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                    'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'E':   {'v': '-60*mV + rand()*8*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'PV':  {'v': '-60*mV + rand()*8*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'SOM': {'v': '-60*mV + rand()*8*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'VIP': {'v': '-60*mV + rand()*8*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
    },

    'time_constants': {
        'E_AMPA': tau_e_AMPA,
        'E_NMDA': tau_e_NMDA,
        'I_PV':   tau_i_PV,
        'I_SOM':  tau_i_SOM,
        'I_VIP':  tau_i_VIP,
    },

    'synapses': {
        'Q': {
            'EXT_AMPA': 0.4*nS,
            'EXT_NMDA': 0.15*nS,
        },
    },

    'layers': _LAYER_CONFIGS,
    
    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
    'inter_layer_conductances': _INTER_LAYER_CONDUCTANCES,
    'electrode_positions' : [
        (0, 0, -0.94),
        (0, 0, -0.79),
        (0, 0, -0.64),
        (0, 0, -0.49),
        (0, 0, -0.34),
        (0, 0, -0.19),
        (0, 0, -0.04),
        (0, 0, 0.10),
        (0, 0, 0.26),
        (0, 0, 0.40),
        (0, 0, 0.56),
        (0, 0, 0.70),
        (0, 0, 0.86),
        (0, 0, 1.00),
        (0, 0, 1.16),
        (0, 0, 1.30),
    ],
}


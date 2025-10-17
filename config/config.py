from brian2 import *

_LAYER_CONFIGS = {
    'L1': {
        'connection_prob': {
            'E_E': 0.034,
            'E_PV': 0.034,
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.034,
            'PV_PV': 0.034,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_SOM': 0.034,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 8,
            'PV': 1,
            'SOM': 1,
            'VIP': 39
        },
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
        }
    },
    'L23': {
        'connection_prob': {
            'E_E': 0.040,
            'E_PV': 0.040,
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.040,
            'PV_PV': 0.040,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_SOM': 0.034,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 3520,
            'PV': 317,
            'SOM': 475,
            'VIP': 88
        },
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
        }
    },
    'L4': {
        'connection_prob': {
            'E_E': 0.034,
            'E_PV': 0.034,
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.034,
            'PV_PV': 0.034,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_SOM': 0.034,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 5760,
            'PV': 950,
            'SOM': 420,
            'VIP': 70
        },
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.025},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.025},
        }
    },
    'L5': {
        'connection_prob': {
            'E_E': 0.034,
            'E_PV': 0.034,
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.034,
            'PV_PV': 0.034,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_SOM': 0.034,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 1600,
            'PV': 208,
            'SOM': 152,
            'VIP': 40
        },
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
        }
    },
    'L6': {
        'connection_prob': {
            'E_E': 0.034,
            'E_PV': 0.034,
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.034,
            'PV_PV': 0.034,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_SOM': 0.034,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 2040,
            'PV': 187,
            'SOM': 137,
            'VIP': 36
        },
        'poisson_inputs': {
            'E':  {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'gE', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
        }
    }
}

_INTER_LAYER_CONNECTIONS = {
    # ('L4', 'L23'): {'E_E': 0.01, 'E_PV': 0.01},
    # ('L23', 'L5'): {'E_E': 0.01, 'E_PV': 0.01},
    # ('L5', 'L6'): {'E_E': 0.02, 'PV_E': 0.01}, 
    # ('L6', 'L4'): {'E_E': 0.01, 'PV_E': 0.01},  
}

_INTER_LAYER_CONDUCTANCES = {}

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
        'VIP': {'a': 2*nS, 'b': 50*pA,  'DeltaT': 2*mV, 'C': 72*pF, 'gL': 3.3*nS, 'tauw': 150*ms, 'EL': -65*mV},
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
            'E_E': 1.25*nS,
            'E_PV': 3.75*nS,
            'E_SOM': 2.5*nS,
            'E_VIP': 1.5*nS,
            'VIP_SOM': 5.0*nS,
            'EXT': 1.25*nS,
        },
       
    },
    'layers': _LAYER_CONFIGS,
    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
    'inter_layer_conductances': _INTER_LAYER_CONDUCTANCES,
}

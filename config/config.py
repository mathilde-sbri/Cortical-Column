
from brian2 import *


_LAYER_CONFIGS = {
    'L1': {
        'connection_prob': {
            'E_E': 0.034,
            'E_PV': 0.068,  
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.034,
            'PV_PV': 0.034,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_PV': 0.068,  
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
            'E':  {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.034},
            'PV': {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.034},
        }
    },
    'L23': {
        'connection_prob': {
            'E_E': 0.034,
            'E_PV': 0.068,
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.034,
            'PV_PV': 0.034,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_PV': 0.068,
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
            'E':  {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.034},
            'PV': {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.034},
        }
    },
    'L4': {
        'connection_prob': {
            'E_E': 0.034,
            'E_PV': 0.068,
            'E_SOM': 0.034,
            'E_VIP': 0.034,
            'PV_E': 0.034,
            'PV_PV': 0.034,
            'SOM_E': 0.034,
            'SOM_PV': 0.034,
            'VIP_PV': 0.068,
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
            'E':  {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.034},
            'PV': {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.034},
        }
    },
    'L5': {
        'connection_prob': {
            'E_E': 0.017,
            'E_PV': 0.034,
            'E_SOM': 0.017,
            'E_VIP': 0.017,
            'PV_E': 0.017,
            'PV_PV': 0.017,
            'SOM_E': 0.017,
            'SOM_PV': 0.017,
            'VIP_PV': 0.034,
            'VIP_SOM': 0.017,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 1600,
            'PV': 208,
            'SOM': 152,
            'VIP': 40
        },
        'poisson_inputs': {
            'E':  {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
        }
    },
    'L6': {
        'connection_prob': {
            'E_E': 0.017,
            'E_PV': 0.034,
            'E_SOM': 0.017,
            'E_VIP': 0.017,
            'PV_E': 0.017,
            'PV_PV': 0.017,
            'SOM_E': 0.017,
            'SOM_PV': 0.017,
            'VIP_PV': 0.034,
            'VIP_SOM': 0.017,
        },
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 2040,
            'PV': 187,
            'SOM': 137,
            'VIP': 36
        },
        'poisson_inputs': {
            'E':  {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
            'PV': {'target': 'ge', 'rate': 5*Hz, 'weight': 'EXT', 'N_fraction_of_E': 0.017},
        }
    }
}

_INTER_LAYER_CONNECTIONS = {
    ('L23', 'L4'): 0.0001,
    ('L4', 'L5'): 0.0001,
    ('L5', 'L23'): 0.0001
}

# common_namespace variables 
tau_e_e = 5*ms
tau_i = 5*ms
tau_e_pv = 1*ms
tau_e_som = 2*ms
tau_e_vip = 2*ms
v_reset = -65.*mV
vt = -50.*mV
ee = 0.*mV
ei = -80.*mV
t_ref = 5*ms
tau_w = 500*ms

CONFIG = {
    'simulation': {
        'SIMULATION_TIME': 4000*ms,
        'DT': 0.1*ms,
        'RANDOM_SEED': 58879,
    },
    'models': {
        'equations': {
            'E': """
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
            """,
            'PV': """
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
            """,
            'SOM': """
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
            """,
            'VIP': """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
        IsynE=ge*(Ee-v) : amp
        IsynI=gi*(Ei-v) : amp
        dge/dt = -ge/tau_e_vip : siemens
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
            """,
        },
        'threshold': 'v>Vcut',
        'reset': 'v=V_reset; w+=b',
        'common_namespace' : {
            'tau_e': tau_e_e,
            'tau_i': tau_i,
            'tau_e_pv': tau_e_pv,
            'tau_e_som': tau_e_som,
            'tau_e_vip': tau_e_vip,
            'tauw': tau_w,
            'VT': vt,
            'V_reset': v_reset,
            'Ee': ee,
            'Ei': ei,
        },

    },
    'intrinsic_params': {
        'E':   {'a': 4*nS, 'b': 130*pA, 'DeltaT': 2*mV},
        'PV':  {'a': 0*nS, 'b': 0*pA, 'DeltaT': 0.5*mV},
        'SOM': {'a': 4*nS, 'b': 25*pA, 'DeltaT': 1.5*mV},
        'VIP': {'a': 2*nS, 'b': 50*pA, 'DeltaT': 2*mV},
    },
    'neurons': {
        'V_RESET': -65.*mV,
        'VT': -50.*mV,
        'EE': 0.*mV,
        'EI': -80.*mV,
        'T_REF': 5*ms,
        'TAU_W': 500*ms,
        'CAPACITANCE': 200*pF,
        'LEAK_CONDUCTANCE': 10*nS,
        'INITIAL_VOLTAGE': -60*mV,
        'E_LEAK': {
            'E': -60*mV,   
            'PV': -60*mV,  
            'SOM': -55*mV, 
            'VIP': -65*mV,
        },
    },
    'initial_conditions': {
        'DEFAULT': {'v': -60*mV, 'ge': 0*nS, 'gi': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'E':   {'v': -60*mV, 'ge': 0*nS, 'gi': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'PV':  {'v': -60*mV, 'ge': 0*nS, 'gi': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'SOM': {'v': -60*mV, 'ge': 0*nS, 'gi': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'VIP': {'v': -60*mV, 'ge': 0*nS, 'gi': 0*nS, 'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
    },
    'time_constants': {
        'E': tau_e_e,
        'I': tau_i,
        'E_PV': tau_e_pv,
        'E_SOM': tau_e_som,
        'E_VIP': tau_e_vip,
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
            'VIP_PV': 2.0*nS,
            'EXT': 1.25*nS,
        }
    },
    'layers': _LAYER_CONFIGS,
    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
}


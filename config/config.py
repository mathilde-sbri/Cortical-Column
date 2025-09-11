
from brian2 import *


_LAYER_CONFIGS = {
    'L1': {
        'connection_prob': 0.034,
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
        'connection_prob': 0.034,
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
        'connection_prob': 0.034,
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
        'connection_prob': 0.017,
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
        'connection_prob': 0.017,
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
        'E': 5*ms,
        'I': 5*ms,
        'E_PV': 1*ms,
        'E_SOM': 2*ms,
        'E_VIP': 2*ms,
    },
    'synapses': {
        'Q': {
            'PV_TO_EPV': 6*nS,
            'SOM_TO_EPV': 5*nS,
            'E_TO_E': 1.25*nS,
            'E_TO_PV': 3.75*nS,
            'E_TO_SOM': 2.5*nS,
            'E_TO_VIP': 1.5*nS,
            'VIP_TO_SOM': 5.0*nS,
            'VIP_TO_PV': 2.0*nS,
            'EXT': 1.25*nS,
        }
    },
    'layers': _LAYER_CONFIGS,
    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
}

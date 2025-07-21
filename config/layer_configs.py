"""
Layer-specific configurations : connectivity, number of neurons, ..
"""
from brian2 import *

LAYER_CONFIGS = {
    'L1': {
        'connection_prob': 0.034,
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 8,
            'PV': 1,
            'SOM': 1,
            'VIP': 39  
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
        }
    }
}


INTER_LAYER_CONNECTIONS = {
    ('L23', 'L4'): 0.0001,
    ('L4', 'L5'): 0.0001,
    ('L5', 'L23'): 0.0001
}
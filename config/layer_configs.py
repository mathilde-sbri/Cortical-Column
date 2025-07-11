"""
Layer-specific configurations : connectivity, number of neurons, ..
"""
from brian2 import *

LAYER_CONFIGS = {
    'L23': {
        'connection_prob': 0.034,
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 4944,
            'PV': 260,
            'SOM': 188
        }
    },
    'L4': {
        'connection_prob': 0.034,
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 4040,
            'PV': 392,
            'SOM': 212
        }
    },
    'L56': {
        'connection_prob': 0.017,
        'input_rate': 2*Hz,
        'neuron_counts': {
            'E': 8016,
            'PV': 260,
            'SOM': 1032
        }
    }
}

INTER_LAYER_CONNECTIONS = {
    ('L23', 'L4'): 0.0001,
    ('L4', 'L56'): 0.0001,
    ('L56', 'L23'): 0.0001
}
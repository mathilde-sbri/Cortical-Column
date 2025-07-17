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
            'SOM': 188,
            'VIP': 134  #also to check. vip neurons should be higher in proportion in l2/3 because they are active in feedback processing
        }
    },
    'L4': {
        'connection_prob': 0.034,
        'input_rate': 5*Hz,
        'neuron_counts': {
            'E': 4040,
            'PV': 392,
            'SOM': 212,
            'VIP': 116
        }
    },
    'L56': {
        'connection_prob': 0.017,
        'input_rate': 2*Hz,
        'neuron_counts': {
            'E': 8016,
            'PV': 260,
            'SOM': 1032,
            'VIP': 185
        }
    }
}


INTER_LAYER_CONNECTIONS = {
    ('L23', 'L4'): 0.0001,
    ('L4', 'L56'): 0.0001,
    ('L56', 'L23'): 0.0001
}
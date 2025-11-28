
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


_conn_prob_data = """Source,i1Htr3a,E2/3,i2/3Pvalb,i2/3Sst,i2/3Htr3a,E4C,i4CPvalb,i4CSst,i4CHtr3a,E4AB,i4ABPvalb,i4ABSst,i4ABHtr3a,E5,i5Pvalb,i5Sst,i5Htr3a,E6,i6Pvalb,i6Sst,i6Htr3a
i1Htr3a,0.656,0.356,0.093,0.068,0.464,0.040,0.000,0.000,0.000,0.108,0.000,0.000,0.000,0.148,0.000,0.000,0.000,0.078,0.000,0.000,0.000
E2/3,0.000,0.360,0.310,0.310,0.310,0.000,0.002,0.002,0.002,0.002,0.006,0.006,0.006,0.350,0.250,0.182,0.160,0.200,0.000,0.000,0.000
i2/3Pvalb,0.024,0.502,0.303,0.060,0.240,0.054,0.002,0.003,0.003,0.054,0.002,0.003,0.003,0.040,0.273,0.000,0.000,0.000,0.000,0.000,0.000
i2/3Sst,0.279,0.378,0.714,0.164,0.210,0.025,0.002,0.025,0.003,0.025,0.002,0.025,0.003,0.021,0.000,0.000,0.000,0.000,0.000,0.000,0.000
i2/3Htr3a,0.000,0.187,0.150,0.512,0.028,0.025,0.002,0.003,0.003,0.025,0.002,0.003,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
E4C,0.000,0.050,0.040,0.035,0.015,0.320,0.350,0.400,0.300,0.400,0.300,0.250,0.200,0.215,0.115,0.112,0.115,0.012,0.000,0.000,0.000
i4CPvalb,0.000,0.013,0.048,0.001,0.001,0.450,0.400,0.040,0.200,0.090,0.040,0.025,0.040,0.022,0.010,0.010,0.010,0.000,0.000,0.000,0.000
i4CSst,0.060,0.013,0.023,0.001,0.001,0.600,0.450,0.050,0.180,0.150,0.080,0.000,0.045,0.007,0.010,0.000,0.010,0.000,0.000,0.000,0.000
i4CHtr3a,0.000,0.013,0.001,0.022,0.001,0.250,0.015,0.650,0.020,0.050,0.005,0.055,0.008,0.000,0.010,0.010,0.010,0.000,0.000,0.000,0.000
E4AB,0.000,0.400,0.230,0.205,0.070,0.020,0.020,0.020,0.020,0.387,0.400,0.470,0.370,0.155,0.146,0.130,0.145,0.024,0.000,0.000,0.000
i4ABPvalb,0.000,0.050,0.170,0.005,0.005,0.040,0.028,0.028,0.028,0.490,0.435,0.041,0.218,0.063,0.026,0.017,0.017,0.000,0.000,0.000,0.000
i4ABSst,0.080,0.050,0.088,0.005,0.005,0.100,0.028,0.028,0.028,0.670,0.400,0.057,0.208,0.017,0.017,0.000,0.017,0.000,0.000,0.000,0.000
i4ABHtr3a,0.000,0.050,0.005,0.078,0.005,0.000,0.010,0.018,0.010,0.268,0.016,0.745,0.024,0.000,0.017,0.017,0.017,0.000,0.000,0.000,0.000
E5,0.017,0.021,0.050,0.150,0.050,0.004,0.025,0.025,0.025,0.004,0.025,0.025,0.025,0.466,0.473,0.563,0.445,0.150,0.110,0.190,0.170
i5Pvalb,0.000,0.000,0.102,0.000,0.000,0.000,0.017,0.015,0.015,0.000,0.017,0.015,0.015,0.555,0.661,0.030,0.020,0.060,0.090,0.010,0.010
i5Sst,0.203,0.020,0.000,0.017,0.000,0.028,0.015,0.003,0.015,0.028,0.015,0.003,0.015,0.517,0.657,0.040,0.070,0.070,0.100,0.010,0.010
i5Htr3a,0.000,0.000,0.000,0.000,0.000,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.325,0.020,0.725,0.020,0.030,0.010,0.080,0.010
E6,0.000,0.000,0.000,0.000,0.000,0.150,0.090,0.090,0.040,0.090,0.055,0.055,0.040,0.112,0.010,0.010,0.010,0.426,0.445,0.560,0.425
i6Pvalb,0.000,0.100,0.000,0.000,0.000,0.050,0.030,0.000,0.000,0.050,0.030,0.000,0.000,0.100,0.030,0.030,0.030,0.750,0.850,0.080,0.160
i6Sst,0.000,0.000,0.000,0.000,0.000,0.000,0.025,0.000,0.000,0.000,0.025,0.000,0.000,0.030,0.030,0.030,0.030,0.610,0.650,0.050,0.050
i6Htr3a,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.030,0.030,0.030,0.030,0.300,0.050,0.864,0.030"""


_conductance_data = """Source,i1Htr3a,E2/3,i2/3Pvalb,i2/3Sst,i2/3Htr3a,E4C,i4CPvalb,i4CSst,i4CHtr3a,E4AB,i4ABPvalb,i4ABSst,i4ABHtr3a,E5,i5Pvalb,i5Sst,i5Htr3a,E6,i6Pvalb,i6Sst,i6Htr3a
i1Htr3a,0.1419,0.0351,0.0396,0.0462,0.0626,0.0135,0.0000,0.0000,0.0000,0.0135,0.0000,0.0000,0.0000,0.0270,0.0000,0.0000,0.0000,0.0270,0.0000,0.0000,0.0000
E2/3,0.0000,0.0176,0.0591,0.0330,0.0440,0.0055,0.0242,0.0116,0.0158,0.0055,0.0241,0.0115,0.0157,0.0545,0.0462,0.0165,0.0000,0.0000,0.0000,0.0000,0.0000
i2/3Pvalb,0.0297,0.0324,0.0561,0.0330,0.0330,0.0189,0.0281,0.0165,0.0165,0.0189,0.0280,0.0165,0.0165,0.0135,0.0642,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000
i2/3Sst,0.0378,0.0401,0.0411,0.0113,0.0428,0.0101,0.0206,0.0057,0.0214,0.0100,0.0205,0.0056,0.0214,0.0147,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000
i2/3Htr3a,0.0000,0.0189,0.0147,0.0324,0.0297,0.0095,0.0074,0.0132,0.0149,0.0094,0.0073,0.0131,0.0148,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000
E4C,0.0000,0.0323,0.0136,0.0116,0.0158,0.0096,0.0220,0.0238,0.0083,0.0523,0.0220,0.0237,0.0082,0.0313,0.0209,0.0150,0.0158,0.0180,0.0000,0.0000,0.0000
i4CPvalb,0.0000,0.0222,0.0116,0.0083,0.0165,0.0216,0.0281,0.0165,0.0165,0.0215,0.0280,0.0165,0.0165,0.0243,0.0387,0.0000,0.0165,0.0000,0.0000,0.0000,0.0000
i4CSst,0.0156,0.0118,0.0163,0.0057,0.0132,0.0095,0.0206,0.0057,0.0170,0.0094,0.0205,0.0056,0.0169,0.0195,0.0182,0.0116,0.0214,0.0000,0.0000,0.0000,0.0000
i4CHtr3a,0.0000,0.0095,0.0074,0.0132,0.0149,0.0095,0.0074,0.0132,0.0149,0.0094,0.0073,0.0131,0.0148,0.0000,0.0074,0.0132,0.0149,0.0000,0.0000,0.0000,0.0000
E4AB,0.0000,0.0522,0.0136,0.0115,0.0157,0.0095,0.0220,0.0237,0.0082,0.0096,0.0220,0.0238,0.0083,0.0112,0.0208,0.0150,0.0157,0.0180,0.0000,0.0000,0.0000
i4ABPvalb,0.0000,0.0221,0.0115,0.0082,0.0165,0.0215,0.0280,0.0165,0.0165,0.0216,0.0281,0.0165,0.0165,0.0243,0.0387,0.0000,0.0165,0.0000,0.0000,0.0000,0.0000
i4ABSst,0.0156,0.0117,0.0163,0.0056,0.0131,0.0094,0.0205,0.0056,0.0169,0.0095,0.0206,0.0057,0.0170,0.0194,0.0181,0.0115,0.0214,0.0000,0.0000,0.0000,0.0000
i4ABHtr3a,0.0000,0.0094,0.0073,0.0131,0.0148,0.0094,0.0073,0.0131,0.0148,0.0095,0.0074,0.0132,0.0149,0.0000,0.0073,0.0131,0.0148,0.0000,0.0000,0.0000,0.0000
E5,0.0249,0.0156,0.0417,0.0165,0.0315,0.0068,0.0209,0.0083,0.0158,0.0067,0.0208,0.0082,0.0157,0.0194,0.0417,0.0450,0.0440,0.0335,0.0440,0.0365,0.0440
i5Pvalb,0.0000,0.0000,0.0411,0.0000,0.0000,0.0114,0.0165,0.0135,0.0135,0.0113,0.0165,0.0135,0.0135,0.0395,0.0530,0.0000,0.0600,0.0595,0.0330,0.0030,0.0000
i5Sst,0.0246,0.0162,0.0000,0.0312,0.0000,0.0095,0.0182,0.0116,0.0214,0.0094,0.0181,0.0115,0.0214,0.0274,0.0630,0.0330,0.0428,0.0174,0.0330,0.0030,0.0428
i5Htr3a,0.0000,0.0000,0.0000,0.0000,0.0000,0.0095,0.0074,0.0132,0.0149,0.0094,0.0073,0.0131,0.0148,0.0189,0.0147,0.0363,0.0197,0.0189,0.0147,0.0163,0.0297
E6,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0065,0.0440,0.0165,0.0440,0.0436,0.0483,0.0515,0.0440
i6Pvalb,0.0000,0.0135,0.0000,0.0000,0.0000,0.0068,0.0000,0.0000,0.0000,0.0067,0.0000,0.0000,0.0000,0.0201,0.0312,0.0000,0.0330,0.0540,0.0972,0.0030,0.0330
i6Sst,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0174,0.0165,0.0330,0.0428,0.0174,0.0330,0.0330,0.0428
i6Htr3a,0.0000,0.0000,0.0000,0.0000,0.0000,0.0095,0.0074,0.0132,0.0149,0.0094,0.0073,0.0131,0.0148,0.0189,0.0147,0.0263,0.0297,0.0189,0.0147,0.0363,0.0297"""


from io import StringIO

conn_df = pd.read_csv(StringIO(_conn_prob_data), index_col=0)
cond_df = pd.read_csv(StringIO(_conductance_data), index_col=0)

INTER_LAYER_PROB_SCALE = 0.5
INTER_LAYER_COND_SCALE = 0.5


NMDA_AMPA_RATIO = {
    'recurrent': 0.65,   
    'feedforward': 0.40,    
    'feedback': 0.55,       
    'E_to_I': 0.50,      
}


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
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 8}, 
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 3},  
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 3},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 25},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 8},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 3},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 3},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 25},
        },
        'coordinates': {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': (1.1, 1.19),
        }
    },
    
    'L23': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 12*Hz, 
        'neuron_counts': {'E': 3520, 'PV': 317, 'SOM': 475, 'VIP': 88},
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 80},  
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 35},  
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 35},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 80},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 35},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 35},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
        },
        'coordinates': {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': (0.45, 1.1),
        }
    },
    
    'L4AB': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 8*Hz, 
        'neuron_counts': {'E': 2720, 'PV': 408, 'SOM': 204, 'VIP': 68},
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 100}, 
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 40},  
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 40},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 30},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 100},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 40},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 40},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 30},
        },
        'coordinates': {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': (0.14, 0.45),
        }
    },
    
    'L4C': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 8*Hz, 
        'neuron_counts': {'E': 3192, 'PV': 365, 'SOM': 182, 'VIP': 61},
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 100}, 
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 40},  
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 40},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 30},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 100},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 40},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 40},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 30},
        },
        'coordinates': {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': (-0.14, 0.14),
        }
    },
    
    'L5': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 12*Hz,  
        'neuron_counts': {'E': 1600, 'PV': 208, 'SOM': 152, 'VIP': 40},
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 80}, 
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 25},  
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 25},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 25},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 80},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 25},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 25},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 25},
        },
        'coordinates': {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': (-0.34, -0.14),
        }
    },
    
    'L6': {
        'connection_prob': {},
        'conductance': {},
        'input_rate': 15*Hz, 
        'neuron_counts': {'E': 2040, 'PV': 187, 'SOM': 137, 'VIP': 36},
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 50}, 
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},  
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 50},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},
        },
        'coordinates': {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': (-0.62, -0.34),
        }
    }
}


_layer_csv = {
    'L1':   {'VIP_row': 'i1Htr3a',   'VIP_col': 'i1Htr3a'},
    
    'L23':  {'E_row': 'E2/3',        'PV_row': 'i2/3Pvalb',  'SOM_row': 'i2/3Sst',   'VIP_row': 'i2/3Htr3a',
             'E_col': 'E2/3',        'PV_col': 'i2/3Pvalb',  'SOM_col': 'i2/3Sst',   'VIP_col': 'i2/3Htr3a'},
    
    'L4C':  {'E_row': 'E4C',         'PV_row': 'i4CPvalb',   'SOM_row': 'i4CSst',    'VIP_row': 'i4CHtr3a',
             'E_col': 'E4C',         'PV_col': 'i4CPvalb',   'SOM_col': 'i4CSst',    'VIP_col': 'i4CHtr3a'},
    
    'L4AB': {'E_row': 'E4AB',        'PV_row': 'i4ABPvalb',  'SOM_row': 'i4ABSst',   'VIP_row': 'i4ABHtr3a',
             'E_col': 'E4AB',        'PV_col': 'i4ABPvalb',  'SOM_col': 'i4ABSst',   'VIP_col': 'i4ABHtr3a'},
    
    'L5':   {'E_row': 'E5',          'PV_row': 'i5Pvalb',    'SOM_row': 'i5Sst',     'VIP_row': 'i5Htr3a',
             'E_col': 'E5',          'PV_col': 'i5Pvalb',    'SOM_col': 'i5Sst',     'VIP_col': 'i5Htr3a'},
    
    'L6':   {'E_row': 'E6',          'PV_row': 'i6Pvalb',    'SOM_row': 'i6Sst',     'VIP_row': 'i6Htr3a',
             'E_col': 'E6',          'PV_col': 'i6Pvalb',    'SOM_col': 'i6Sst',     'VIP_col': 'i6Htr3a'},
}


def _prob(src_row, tgt_col):
    try:
        val = conn_df.loc[src_row, tgt_col]
        return float(val)
    except KeyError:
        print(f"WARNING: Could not find connection {src_row} -> {tgt_col}")
        return 0.0

def _cond(src_row, tgt_col):
    try:
        val = cond_df.loc[src_row, tgt_col]
        return float(val)
    except KeyError:
        print(f"WARNING: Could not find conductance {src_row} -> {tgt_col}")
        return 0.0


_INTER_LAYER_CONNECTIONS = defaultdict(dict)
_INTER_LAYER_CONDUCTANCES = defaultdict(dict)

_layers = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
_pops = ['E', 'PV', 'SOM', 'VIP']

for src in _layers:
    for dst in _layers:
        s = _layer_csv[src]
        t = _layer_csv[dst]
        
        for src_pop in _pops:
            for dst_pop in _pops:
                row_key = f'{src_pop}_row'
                col_key = f'{dst_pop}_col'
                
                if row_key not in s or col_key not in t:
                    continue
                
                prob = _prob(s[row_key], t[col_key])
                cond = _cond(s[row_key], t[col_key])
                
                is_excitatory = (src_pop == 'E')
                
                if src == dst:
                    if is_excitatory:
                        conn_key = f'{src_pop}_{dst_pop}'
                        _LAYER_CONFIGS[src].setdefault('connection_prob', {})[conn_key] = prob
                        
                        if dst_pop == 'E':
                            nmda_ratio = NMDA_AMPA_RATIO['recurrent']
                        else:
                            nmda_ratio = NMDA_AMPA_RATIO['E_to_I']
                        
                        _LAYER_CONFIGS[src].setdefault('conductance', {})[f'{conn_key}_AMPA'] = cond / 2
                        _LAYER_CONFIGS[src].setdefault('conductance', {})[f'{conn_key}_NMDA'] = (cond / 2) * nmda_ratio
                    else:
                        conn_key = f'{src_pop}_{dst_pop}'
                        _LAYER_CONFIGS[src].setdefault('connection_prob', {})[conn_key] = prob
                        _LAYER_CONFIGS[src].setdefault('conductance', {})[conn_key] = cond / 2
                else:
                    conn_key = f'{src_pop}_{dst_pop}'
                    
                    if is_excitatory:
                        _INTER_LAYER_CONNECTIONS[(src, dst)][conn_key] = prob * INTER_LAYER_PROB_SCALE
                        

                        src_depth = {'L23': 1, 'L4AB': 2, 'L4C': 2, 'L5': 3, 'L6': 4}
                        if src_depth.get(src, 0) > src_depth.get(dst, 0):
                            nmda_ratio = NMDA_AMPA_RATIO['feedback']
                        else:
                            nmda_ratio = NMDA_AMPA_RATIO['feedforward']
                        
                        if dst_pop != 'E':
                            nmda_ratio = NMDA_AMPA_RATIO['E_to_I']
                        
                        _INTER_LAYER_CONDUCTANCES[(src, dst)][f'{conn_key}_AMPA'] = cond * INTER_LAYER_COND_SCALE
                        _INTER_LAYER_CONDUCTANCES[(src, dst)][f'{conn_key}_NMDA'] = cond * INTER_LAYER_COND_SCALE * nmda_ratio
                    else:
                        _INTER_LAYER_CONNECTIONS[(src, dst)][conn_key] = prob * INTER_LAYER_PROB_SCALE
                        _INTER_LAYER_CONDUCTANCES[(src, dst)][conn_key] = cond * INTER_LAYER_COND_SCALE



for layer in ['L23', 'L4AB', 'L4C', 'L5', 'L6']:
    cfg = _LAYER_CONFIGS[layer]
    
    for conn_type in list(cfg.get('conductance', {}).keys()):
        src_pop = conn_type.split('_')[0]
        
        if src_pop == 'PV':
            cfg['conductance'][conn_type] *= 1.2
        elif src_pop == 'SOM':
            cfg['conductance'][conn_type] *= 1.2
        elif src_pop == 'VIP':
            cfg['conductance'][conn_type] *= 1.3

for layer in ['L23', 'L4AB', 'L4C', 'L5', 'L6']:
    cfg = _LAYER_CONFIGS[layer]
    
    if 'E_E' in cfg.get('connection_prob', {}):
        cfg['connection_prob']['E_E'] *= 0.85
    if 'E_E_AMPA' in cfg.get('conductance', {}):
        cfg['conductance']['E_E_AMPA'] *= 0.85
        cfg['conductance']['E_E_NMDA'] *= 0.85

for layer in ['L23', 'L4AB', 'L4C', 'L5', 'L6']:
    cfg = _LAYER_CONFIGS[layer]
    
    if 'E_SOM_AMPA' in cfg.get('conductance', {}):
        cfg['conductance']['E_SOM_AMPA'] *= 1.8
        cfg['conductance']['E_SOM_NMDA'] *= 1.8
    if 'E_SOM' in cfg.get('connection_prob', {}):
        cfg['connection_prob']['E_SOM'] *= 1.3




_neuron_equations = {
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
gE   = gE_AMPA + gE_NMDA : siemens
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
gE   = gE_AMPA + gE_NMDA : siemens
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
gE   = gE_AMPA + gE_NMDA : siemens
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
gE   = gE_AMPA + gE_NMDA : siemens
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
}



CONFIG = {
    'simulation': {
        'SIMULATION_TIME': 4000*ms,
        'DT': 0.1*ms,
        'RANDOM_SEED': 58879,
    },
    
    'models': {
        'equations': _neuron_equations,
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
        'E':   {'a': 4*nS, 'b': 40*pA, 'DeltaT': 2*mV,
                'C': 97*pF, 'gL': 4.2*nS, 'tauw': 200*ms, 'EL': -66*mV},
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
        'DEFAULT': {
            'v': '-60*mV + rand()*8*mV',
            'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
            'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
            'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5
        },
        'E': {
            'v': '-60*mV + rand()*8*mV',
            'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
            'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
            'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5
        },
        'PV': {
            'v': '-60*mV + rand()*8*mV',
            'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
            'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
            'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5
        },
        'SOM': {
            'v': '-60*mV + rand()*8*mV',
            'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
            'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
            'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5
        },
        'VIP': {
            'v': '-60*mV + rand()*8*mV',
            'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
            'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
            'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5
        },
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
            'EXT_AMPA': 0.5*nS,
            'EXT_NMDA': 0.15*nS,
        },
    },
    
    'layers': _LAYER_CONFIGS,
    'inter_layer_connections': dict(_INTER_LAYER_CONNECTIONS),
    'inter_layer_conductances': dict(_INTER_LAYER_CONDUCTANCES),
}



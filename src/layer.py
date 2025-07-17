"""
Layer class implementation
"""
import brian2 as b2
from brian2 import *
import numpy as np
from .parameters import *
from .neuron_models import NeuronModels, NeuronParameters

class CorticalLayer:
    """
     a single cortical layer with E, PV, VIP and SOM populations
    """
    
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.neuron_groups = {}
        self.synapses = {}
        self.monitors = {}
        self.poisson_inputs = {}
        
        self._create_neuron_groups()
        self._create_poisson_inputs() 
        self._create_internal_connections()  
        self._create_monitors()  
    
    def _create_neuron_groups(self):

        common_namespace = {
            'tau_e': TAU_E, 
            'tau_i': TAU_I, 
            'tau_e_pv': TAU_E_PV,
            'tau_e_som': TAU_E_SOM,
            'tau_e_vip': TAU_E_VIP,
            'tauw': TAU_W,
            'VT': VT, 
            'V_reset': V_RESET, 
            'Ee': EE, 
            'Ei': EI
        }
        
        self.neuron_groups['E'] = NeuronGroup(
            self.config['neuron_counts']['E'],
            NeuronModels.get_excitatory_equations(),
            threshold='v>Vcut',
            reset="v=V_reset; w+=b",
            refractory=T_REF,
            namespace=common_namespace
        )
        
        self.neuron_groups['PV'] = NeuronGroup(
            self.config['neuron_counts']['PV'],
            NeuronModels.get_pv_equations(),
            threshold='v>Vcut',
            reset="v=V_reset; w+=b",
            refractory=T_REF,
            namespace=common_namespace
        )
        
        self.neuron_groups['SOM'] = NeuronGroup(
            self.config['neuron_counts']['SOM'],
            NeuronModels.get_sst_equations(),
            threshold='v>Vcut',
            reset="v=V_reset; w+=b",
            refractory=T_REF,
            namespace=common_namespace
        )

        self.neuron_groups['VIP'] = NeuronGroup(
            self.config['neuron_counts']['VIP'],
            NeuronModels.get_vip_equations(),
            threshold='v>Vcut',
            reset="v=V_reset; w+=b",
            refractory=T_REF,
            namespace=common_namespace
        )
        
        self._set_neuron_parameters()
    
    def _set_neuron_parameters(self):

        params = NeuronParameters.get_excitatory_params()
        for param, value in params.items():
            setattr(self.neuron_groups['E'], param, value)
        self.neuron_groups['E'].Vcut = VT + 5*params['DeltaT']
        self.neuron_groups['E'].v = INITIAL_VOLTAGE
        self.neuron_groups['E'].ge = 0*nS
        self.neuron_groups['E'].gi = 0*nS
        self.neuron_groups['E'].w = 0*pA
        self.neuron_groups['E'].I = 0*pA
        
        params = NeuronParameters.get_pv_params()
        for param, value in params.items():
            setattr(self.neuron_groups['PV'], param, value)
        self.neuron_groups['PV'].Vcut = VT + 5*params['DeltaT']
        self.neuron_groups['PV'].v = INITIAL_VOLTAGE
        self.neuron_groups['PV'].ge = 0*nS
        self.neuron_groups['PV'].gi = 0*nS
        self.neuron_groups['PV'].w = 0*pA
        self.neuron_groups['PV'].I = 0*pA
        
        params = NeuronParameters.get_sst_params()
        for param, value in params.items():
            setattr(self.neuron_groups['SOM'], param, value)
        self.neuron_groups['SOM'].Vcut = VT + 5*params['DeltaT']
        self.neuron_groups['SOM'].v = INITIAL_VOLTAGE
        self.neuron_groups['SOM'].ge = 0*nS
        self.neuron_groups['SOM'].gi = 0*nS
        self.neuron_groups['SOM'].w = 0*pA
        self.neuron_groups['SOM'].I = 0*pA

        params = NeuronParameters.get_vip_params()
        for param, value in params.items():
            setattr(self.neuron_groups['VIP'], param, value)
        self.neuron_groups['VIP'].Vcut = VT + 5*params['DeltaT']
        self.neuron_groups['VIP'].v = INITIAL_VOLTAGE
        self.neuron_groups['VIP'].ge = 0*nS
        self.neuron_groups['VIP'].gi = 0*nS
        self.neuron_groups['VIP'].w = 0*pA
        self.neuron_groups['VIP'].I = 0*pA

    
    def _create_monitors(self):
        for pop_name, group in self.neuron_groups.items():
            self.monitors[f'{pop_name}_state'] = StateMonitor(
                group, ['v', 'ge', 'gi'], record=True
            )
            
            self.monitors[f'{pop_name}_spikes'] = SpikeMonitor(
                group, variables='t'
            )
            
            self.monitors[f'{pop_name}_rate'] = PopulationRateMonitor(group)
    
    def _create_poisson_inputs(self):
        n_ext = int(self.config['connection_prob'] * self.config['neuron_counts']['E'])
        
        self.poisson_inputs['E'] = PoissonInput(
            self.neuron_groups['E'], 'ge', 
            N=n_ext, rate=self.config['input_rate'], weight=Q_EXT
        )
        
        self.poisson_inputs['PV'] = PoissonInput(
            self.neuron_groups['PV'], 'ge',
            N=n_ext, rate=self.config['input_rate'], weight=Q_EXT
        )
    
    def _create_internal_connections(self):
        p = self.config['connection_prob']
        
        # E to E 
        self.synapses['E_E'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['E'],
            on_pre=f'ge_post += {Q_E_TO_E/nS}*nS'
        )
        self.synapses['E_E'].connect(p=p)
        
        # E to PV 
        self.synapses['E_PV'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['PV'],
            on_pre=f'ge_post += {Q_E_TO_PV/nS}*nS'
        )
        self.synapses['E_PV'].connect(p=2*p)
        
        # E to SOM
        self.synapses['E_SOM'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['SOM'],
            on_pre=f'ge_post += {Q_E_TO_SOM/nS}*nS'
        )
        self.synapses['E_SOM'].connect(p=p)

        # E to VIP 
        self.synapses['E_VIP'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['VIP'],
            on_pre=f'ge_post += {Q_E_TO_VIP/nS}*nS'
        )
        self.synapses['E_VIP'].connect(p=p)
        
        # PV to E 
        self.synapses['PV_E'] = Synapses(
            self.neuron_groups['PV'], self.neuron_groups['E'],
            on_pre=f'gi_post += {Q_PV_TO_EPV/nS}*nS'
        )
        self.synapses['PV_E'].connect(p=p)
        
        # PV to PV 
        self.synapses['PV_PV'] = Synapses(
            self.neuron_groups['PV'], self.neuron_groups['PV'],
            on_pre=f'gi_post += {Q_PV_TO_EPV/nS}*nS'
        )
        self.synapses['PV_PV'].connect(p=p)
        
        # SOM to E 
        self.synapses['SOM_E'] = Synapses(
            self.neuron_groups['SOM'], self.neuron_groups['E'],
            on_pre=f'gi_post += {Q_SOM_TO_EPV/nS}*nS'
        )
        self.synapses['SOM_E'].connect(p=p)
        
        # SOM to PV 
        self.synapses['SOM_PV'] = Synapses(
            self.neuron_groups['SOM'], self.neuron_groups['PV'],
            on_pre=f'gi_post += {Q_SOM_TO_EPV/nS}*nS'
        )
        self.synapses['SOM_PV'].connect(p=p)

        # VIP to PV 
        self.synapses['VIP_PV'] = Synapses(
            self.neuron_groups['VIP'], self.neuron_groups['PV'],
            on_pre=f'ge_post += {Q_VIP_TO_PV/nS}*nS'
        )
        self.synapses['VIP_PV'].connect(p=2*p)
        
        # VIP to SOM
        self.synapses['VIP_SOM'] = Synapses(
            self.neuron_groups['VIP'], self.neuron_groups['SOM'],
            on_pre=f'ge_post += {Q_VIP_TO_SOM/nS}*nS'
        )
        self.synapses['VIP_SOM'].connect(p=p)
    
    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')
    
    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)
"""
Layer class implementation
"""
import brian2 as b2
from brian2 import *
import numpy as np

class CorticalLayer:
    """
     a single cortical layer with E, PV, VIP and SOM populations
    """
    
    def __init__(self, name, layer_name, config):
        self.name = name
        self.layer_name = layer_name
        self.config = config
        self.layer_config = config['layers'][layer_name]
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
            'tau_e': self.config['time_constants']['E'],
            'tau_i': self.config['time_constants']['I'],
            'tau_e_pv': self.config['time_constants']['E_PV'],
            'tau_e_som': self.config['time_constants']['E_SOM'],
            'tau_e_vip': self.config['time_constants']['E_VIP'],
            'tauw': self.config['neurons']['TAU_W'],
            'VT': self.config['neurons']['VT'],
            'V_reset': self.config['neurons']['V_RESET'],
            'Ee': self.config['neurons']['EE'],
            'Ei': self.config['neurons']['EI'],
        }
        
        eqs = self.config['models']['equations']
        threshold = self.config['models']['threshold']
        reset = self.config['models']['reset']

        self.neuron_groups['E'] = NeuronGroup(
            self.layer_config['neuron_counts']['E'],
            eqs['E'],
            threshold=threshold,
            reset=reset,
            refractory=self.config['neurons']['T_REF'],
            namespace=common_namespace
        )
        
        self.neuron_groups['PV'] = NeuronGroup(
            self.layer_config['neuron_counts']['PV'],
            eqs['PV'],
            threshold=threshold,
            reset=reset,
            refractory=self.config['neurons']['T_REF'],
            namespace=common_namespace
        )
        
        self.neuron_groups['SOM'] = NeuronGroup(
            self.layer_config['neuron_counts']['SOM'],
            eqs['SOM'],
            threshold=threshold,
            reset=reset,
            refractory=self.config['neurons']['T_REF'],
            namespace=common_namespace
        )

        self.neuron_groups['VIP'] = NeuronGroup(
            self.layer_config['neuron_counts']['VIP'],
            eqs['VIP'],
            threshold=threshold,
            reset=reset,
            refractory=self.config['neurons']['T_REF'],
            namespace=common_namespace
        )
        
        self._set_neuron_parameters()
    
    def _set_neuron_parameters(self):
        def apply_params_and_ic(pop_key):
            group = self.neuron_groups[pop_key]
            ip = self.config['intrinsic_params'][pop_key]
            group.a = ip['a']
            group.b = ip['b']
            group.DeltaT = ip['DeltaT']
            group.EL = self.config['neurons']['E_LEAK'][pop_key]
            group.C = self.config['neurons']['CAPACITANCE']
            group.gL = self.config['neurons']['LEAK_CONDUCTANCE']
            ic_all = self.config.get('initial_conditions', {})
            ic_pop = ic_all.get(pop_key, {})
            ic_default = ic_all.get('DEFAULT', {})
            vcut_factor = ic_pop.get('Vcut_offset_factor', ic_default.get('Vcut_offset_factor'))
            group.Vcut = self.config['neurons']['VT'] + vcut_factor*ip['DeltaT']
            v_default = ic_default.get('v')
            group.v = ic_pop.get('v', v_default)
            group.ge = ic_pop.get('ge', ic_default.get('ge'))
            group.gi = ic_pop.get('gi', ic_default.get('gi'))
            group.w = ic_pop.get('w', ic_default.get('w'))
            group.I = ic_pop.get('I', ic_default.get('I'))

        apply_params_and_ic('E')
        apply_params_and_ic('PV')
        apply_params_and_ic('SOM')
        apply_params_and_ic('VIP')

    
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
        poisson_cfg = self.layer_config.get('poisson_inputs', {})
        for pop_name, pconf in poisson_cfg.items():
            if pop_name not in self.neuron_groups:
                continue
            target_var = pconf.get('target', 'ge')
            if 'N' in pconf and pconf['N'] is not None:
                N = int(pconf['N'])
            else:
                frac = float(pconf.get('N_fraction_of_E', 0.0))
                N = int(frac * self.layer_config['neuron_counts']['E'])
            weight_cfg = pconf.get('weight', 'EXT')
            if isinstance(weight_cfg, str):
                weight = self.config['synapses']['Q'][weight_cfg]
            else:
                weight = weight_cfg
            rate = pconf.get('rate', self.layer_config.get('input_rate'))
            self.poisson_inputs[pop_name] = PoissonInput(
                self.neuron_groups[pop_name], target_var,
                N=N, rate=rate, weight=weight
            )
    
    def _create_internal_connections(self):
        p = self.layer_config['connection_prob']
        
        # E to E 
        self.synapses['E_E'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['E'],
            on_pre=f"ge_post += {self.config['synapses']['Q']['E_TO_E']/nS}*nS"
        )
        self.synapses['E_E'].connect(p=p)
        
        # E to PV 
        self.synapses['E_PV'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['PV'],
            on_pre=f"ge_post += {self.config['synapses']['Q']['E_TO_PV']/nS}*nS"
        )
        self.synapses['E_PV'].connect(p=2*p)
        
        # E to SOM
        self.synapses['E_SOM'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['SOM'],
            on_pre=f"ge_post += {self.config['synapses']['Q']['E_TO_SOM']/nS}*nS"
        )
        self.synapses['E_SOM'].connect(p=p)

        # E to VIP 
        self.synapses['E_VIP'] = Synapses(
            self.neuron_groups['E'], self.neuron_groups['VIP'],
            on_pre=f"ge_post += {self.config['synapses']['Q']['E_TO_VIP']/nS}*nS"
        )
        self.synapses['E_VIP'].connect(p=p)
        
        # PV to E 
        self.synapses['PV_E'] = Synapses(
            self.neuron_groups['PV'], self.neuron_groups['E'],
            on_pre=f"gi_post += {self.config['synapses']['Q']['PV_TO_EPV']/nS}*nS"
        )
        self.synapses['PV_E'].connect(p=p)
        
        # PV to PV 
        self.synapses['PV_PV'] = Synapses(
            self.neuron_groups['PV'], self.neuron_groups['PV'],
            on_pre=f"gi_post += {self.config['synapses']['Q']['PV_TO_EPV']/nS}*nS"
        )
        self.synapses['PV_PV'].connect(p=p)
        
        # SOM to E 
        self.synapses['SOM_E'] = Synapses(
            self.neuron_groups['SOM'], self.neuron_groups['E'],
            on_pre=f"gi_post += {self.config['synapses']['Q']['SOM_TO_EPV']/nS}*nS"
        )
        self.synapses['SOM_E'].connect(p=p)
        
        # SOM to PV 
        self.synapses['SOM_PV'] = Synapses(
            self.neuron_groups['SOM'], self.neuron_groups['PV'],
            on_pre=f"gi_post += {self.config['synapses']['Q']['SOM_TO_EPV']/nS}*nS"
        )
        self.synapses['SOM_PV'].connect(p=p)

        # VIP to PV 
        self.synapses['VIP_PV'] = Synapses(
            self.neuron_groups['VIP'], self.neuron_groups['PV'],
            on_pre=f"ge_post += {self.config['synapses']['Q']['VIP_TO_PV']/nS}*nS"
        )
        self.synapses['VIP_PV'].connect(p=2*p)
        
        # VIP to SOM
        self.synapses['VIP_SOM'] = Synapses(
            self.neuron_groups['VIP'], self.neuron_groups['SOM'],
            on_pre=f"ge_post += {self.config['synapses']['Q']['VIP_TO_SOM']/nS}*nS"
        )
        self.synapses['VIP_SOM'].connect(p=p)
    
    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')
    
    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)

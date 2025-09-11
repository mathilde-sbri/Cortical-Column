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
        self.syn_model = (self.config['models'].get('synapse_model', 'conductance')).lower()
        self.is_current = (self.syn_model == 'current')  
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
        
        extra_ns = self.config['models'].get('namespace', {})
        common_namespace.update(extra_ns)


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
        def set_if_exists(group, attr, value):
            if hasattr(group, attr):
                setattr(group, attr, value)

        def apply_params_and_ic(pop_key):
            group = self.neuron_groups[pop_key]
            ip = self.config['intrinsic_params'][pop_key]
            # keep your existing parameters (safe no-ops if eqs ignore them)
            set_if_exists(group, 'a', ip.get('a', 0*nS))
            set_if_exists(group, 'b', ip.get('b', 0*pA))
            set_if_exists(group, 'DeltaT', ip.get('DeltaT', self.config['neurons']['VT']-self.config['neurons']['V_RESET']))
            set_if_exists(group, 'EL', self.config['neurons']['E_LEAK'][pop_key])
            set_if_exists(group, 'C', self.config['neurons']['CAPACITANCE'])
            set_if_exists(group, 'gL', self.config['neurons']['LEAK_CONDUCTANCE'])

            # initial conditions are per-pop, fallback to DEFAULT
            ic_all = self.config.get('initial_conditions', {})
            ic_pop = ic_all.get(pop_key, {})
            ic_default = ic_all.get('DEFAULT', {})

            # option to compute Vcut for your original model; harmless if unused
            vcut_factor = ic_pop.get('Vcut_offset_factor', ic_default.get('Vcut_offset_factor'))
            if vcut_factor is not None and hasattr(group, 'Vcut'):
                group.Vcut = self.config['neurons']['VT'] + vcut_factor*ip.get('DeltaT', 0*mV)

            # set whichever state variables exist
            for k, v in dict(ic_default, **ic_pop).items():
                set_if_exists(group, k, v)

            # optional per-pop drive (e.g., Veit: I0 & sigma_tot)
            drive = self.config.get('per_pop_drive', {}).get(pop_key)
            if drive is not None:
                mu, sig = drive
                set_if_exists(group, 'I0', mu)
                set_if_exists(group, 'sigma_tot', sig)

        for pop in ['E', 'PV', 'SOM', 'VIP']:
            if pop in self.neuron_groups:
                apply_params_and_ic(pop)
    
    def _on_pre(self, weight_key, inhibitory=False):
        W = self.config['synapses']['Q'][weight_key]
        if self.is_current:
            var = 'sI' if inhibitory else 'sE'
            val = float(W / mV)  # dimensionless scalar
            return f"{var}_post += {val}*mV"
        else:
            var = 'gi' if inhibitory else 'ge'
            val = float(W / nS)  # dimensionless scalar
            return f"{var}_post += {val}*nS"

        
    def _apply_delay(self, syn):
        delay = self.config.get('time_constants', {}).get('DELAY', 0*ms)
        try:
            syn.delay = delay
        except Exception:
            pass




    
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
        E, PV, SOM, VIP = self.neuron_groups['E'], self.neuron_groups['PV'], self.neuron_groups['SOM'], self.neuron_groups['VIP']

        # E -> E 
        self.synapses['E_E'] = Synapses(E, E, on_pre=self._on_pre('E_TO_E', inhibitory=False))
        self.synapses['E_E'].connect(p=p)
        self._apply_delay(self.synapses['E_E'])

        # E -> PV
        self.synapses['E_PV'] = Synapses(E, PV, on_pre=self._on_pre('E_TO_PV', inhibitory=False))
        self.synapses['E_PV'].connect(p=2*p)
        self._apply_delay(self.synapses['E_PV'])

        # E -> SOM
        self.synapses['E_SOM'] = Synapses(E, SOM, on_pre=self._on_pre('E_TO_SOM', inhibitory=False))
        self.synapses['E_SOM'].connect(p=p)
        self._apply_delay(self.synapses['E_SOM'])

        # E -> VIP
        if len(VIP) > 0 and 'E_TO_VIP' in self.config['synapses']['Q']:
            self.synapses['E_VIP'] = Synapses(E, VIP, on_pre=self._on_pre('E_TO_VIP', inhibitory=False))
            self.synapses['E_VIP'].connect(p=p)
            self._apply_delay(self.synapses['E_VIP'])

        # PV -> E  (inhibitory)
        self.synapses['PV_E'] = Synapses(PV, E, on_pre=self._on_pre('PV_TO_EPV', inhibitory=True))
        self.synapses['PV_E'].connect(p=p)
        self._apply_delay(self.synapses['PV_E'])

        # PV -> PV
        self.synapses['PV_PV'] = Synapses(PV, PV, on_pre=self._on_pre('PV_TO_EPV', inhibitory=True))
        self.synapses['PV_PV'].connect(p=p)
        self._apply_delay(self.synapses['PV_PV'])

        # SOM -> E
        self.synapses['SOM_E'] = Synapses(SOM, E, on_pre=self._on_pre('SOM_TO_EPV', inhibitory=True))
        self.synapses['SOM_E'].connect(p=p)
        self._apply_delay(self.synapses['SOM_E'])

        # SOM -> PV
        self.synapses['SOM_PV'] = Synapses(SOM, PV, on_pre=self._on_pre('SOM_TO_EPV', inhibitory=True))
        self.synapses['SOM_PV'].connect(p=p)
        self._apply_delay(self.synapses['SOM_PV'])

        # quick solution but to change and generalise later 
        if len(VIP) > 0:
            if 'VIP_TO_PV' in self.config['synapses']['Q']:
                inh = (self.config['synapses']['Q']['VIP_TO_PV'] < 0*volt) if self.is_current else False
                self.synapses['VIP_PV'] = Synapses(VIP, PV, on_pre=self._on_pre('VIP_TO_PV', inhibitory=inh))
                self.synapses['VIP_PV'].connect(p=2*p)
                self._apply_delay(self.synapses['VIP_PV'])
            if 'VIP_TO_SOM' in self.config['synapses']['Q']:
                inh = (self.config['synapses']['Q']['VIP_TO_SOM'] < 0*volt) if self.is_current else False
                self.synapses['VIP_SOM'] = Synapses(VIP, SOM, on_pre=self._on_pre('VIP_TO_SOM', inhibitory=inh))
                self.synapses['VIP_SOM'].connect(p=p)
                self._apply_delay(self.synapses['VIP_SOM'])

    
    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')
    
    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)

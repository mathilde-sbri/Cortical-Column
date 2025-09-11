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

        models_cfg = self.config.get('models', {})
        common_namespace = dict(models_cfg.get('common_namespace', {}))
        extra_ns = models_cfg.get('namespace', {})
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

        import brian2 as b2

        def set_if_exists(group, attr, value):
            try:
                getattr(group, attr)
            except AttributeError:
                return
            setattr(group, attr, value)

        cfg = self.config
        neurons_cfg = cfg.get('neurons', {})
        intrinsic = cfg.get('intrinsic_params', {})
        ic_all = cfg.get('initial_conditions', {})
        vt = neurons_cfg.get('VT', -50.0*b2.mV)

        def apply_params_and_ic(pop_key):
            if pop_key not in self.neuron_groups:
                return
            g = self.neuron_groups[pop_key]

            ip = intrinsic.get(pop_key, {})
            set_if_exists(g, 'a',       ip.get('a', 0*b2.nS))
            set_if_exists(g, 'b',       ip.get('b', 0*b2.pA))
            set_if_exists(g, 'DeltaT',  ip.get('DeltaT', 2*b2.mV))  # safe default
            eleak_map = neurons_cfg.get('E_LEAK', {})
            set_if_exists(g, 'EL',      eleak_map.get(pop_key, neurons_cfg.get('INITIAL_VOLTAGE', -60*b2.mV)))
            set_if_exists(g, 'C',       neurons_cfg.get('CAPACITANCE', 200*b2.pF))
            set_if_exists(g, 'gL',      neurons_cfg.get('LEAK_CONDUCTANCE', 10*b2.nS))

            ic_default = ic_all.get('DEFAULT', {})
            ic_pop     = ic_all.get(pop_key, {})
            merged_ic  = dict(ic_default, **ic_pop)

            vcut_factor = merged_ic.pop('Vcut_offset_factor', None)
            if vcut_factor is not None and hasattr(g, 'Vcut'):
                dT = ip.get('DeltaT', getattr(g, 'DeltaT', 2*b2.mV))
                set_if_exists(g, 'Vcut', vt + vcut_factor * dT)

            for k, v in merged_ic.items():
                set_if_exists(g, k, v)


            drive = cfg.get('drive', None)
            if drive is not None:
                c = float(drive.get('contrast', 1.0))
                mu_bg = drive.get('mu_bg', {})
                sd_bg = drive.get('sd_bg', {})
                mu_st = drive.get('mu_stim', {})
                sd_st = drive.get('sd_stim', {})
                sg    = drive.get('sigma_global', 0*b2.mV)

                mu = mu_bg.get(pop_key, 0*b2.mV) + c * mu_st.get(pop_key, 0*b2.mV)
                sb = sd_bg.get(pop_key, 0*b2.mV)
                ss = sd_st.get(pop_key, 0*b2.mV)
                sigma = b2.sqrt(sb**2 + (c*ss)**2 + sg**2)

                set_if_exists(g, 'mu_drive',   mu)
                set_if_exists(g, 'sigma_drive', sigma)

        for pop in ['E', 'PV', 'SOM', 'VIP']:
            apply_params_and_ic(pop)

    
    def _on_pre(self, weight_key, inhibitory=False):
        W = self.config['synapses']['Q'][weight_key]
        if self.is_current:
            var = 'sI' if inhibitory else 'sE'
            val = float(W / mV)  
            return f"{var}_post += {val}*mV"
        else:
            var = 'gi' if inhibitory else 'ge'
            val = float(W / nS)  
            return f"{var}_post += {val}*nS"

        
    def _apply_delay(self, syn):
        delay = self.config.get('time_constants', {}).get('DELAY', 0*ms)
        try:
            syn.delay = delay
        except Exception:
            pass




    
    def _create_monitors(self):
        for pop_name, group in self.neuron_groups.items():
            vars_to_record = ['v', 'ge', 'gi']
            if self.is_current:           
                vars_to_record = ['v', 'sE', 'sI']
            self.monitors[f'{pop_name}_state'] = StateMonitor(
                group, vars_to_record, record=True
            )
            self.monitors[f'{pop_name}_spikes'] = SpikeMonitor(group, variables='t')
            self.monitors[f'{pop_name}_rate'] = PopulationRateMonitor(group)

    def _create_poisson_inputs(self):
        poisson_cfg = self.layer_config.get('poisson_inputs', {})
        for pop_name, pconf in poisson_cfg.items():
            if pop_name not in self.neuron_groups:
                continue
            target_var = pconf.get('target', 'ge')
            if self.is_current:
                target_var = {'ge': 'sE', 'gi': 'sI'}.get(target_var, target_var)

            if 'N' in pconf and pconf['N'] is not None:
                N = int(pconf['N'])
            else:
                frac = float(pconf.get('N_fraction_of_E', 0.0))
                N = int(frac * self.layer_config['neuron_counts']['E'])

            weight_cfg = pconf.get('weight', 'EXT')
            weight = self.config['synapses']['Q'][weight_cfg] if isinstance(weight_cfg, str) else weight_cfg
            rate = pconf.get('rate', self.layer_config.get('input_rate'))

            self.poisson_inputs[pop_name] = PoissonInput(
                self.neuron_groups[pop_name], target_var, N=N, rate=rate, weight=weight
            )

        drive = self.config.get('drive', {})
        vip_rate = drive.get('vip_rate', None)
        if vip_rate:
            Nvip = self.layer_config['neuron_counts'].get('VIP', 0)
            if Nvip > 0:
                w_vip = abs(self.config['synapses']['Q'].get('VIP_SOM', 0*mV))
                target = 'sI' if self.is_current else 'gi'
                self.poisson_inputs['VIP_to_SOM'] = PoissonInput(
                    self.neuron_groups['SOM'], target, N=Nvip, rate=vip_rate, weight=w_vip
                )

    
    def _create_internal_connections(self):
        p = self.layer_config['connection_prob']  
        connections = list(self.layer_config['connection_prob'].keys())
        for connection in connections :
            group1, group2 = connection.split("_")
            self.synapses[connection] = Synapses(self.neuron_groups[group1], self.neuron_groups[group2], on_pre=self._on_pre(connection, inhibitory= group1 != 'E'))
            self.synapses[connection].connect(p=p[connection])
            self._apply_delay(self.synapses[connection])


    
    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')
    
    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)

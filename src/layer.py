"""
Layer class implementation
"""
import brian2 as b2
from brian2 import *
import numpy as np
import cleo

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
        self._set_neuron_parameters()
        self._create_poisson_inputs()
        self._create_internal_connections()
        self._create_monitors()

        

    def _create_neuron_groups(self):
        models_cfg = self.config.get('models', {})
        common_namespace = dict(models_cfg.get('common_namespace', {}))
        extra_ns = models_cfg.get('namespace', {})
        common_namespace.update(extra_ns)

        eqs_map = self.config['models']['equations']
        threshold = self.config['models']['threshold']
        reset = self.config['models']['reset']

        for pop_name, n in self.layer_config.get('neuron_counts', {}).items():
            self.neuron_groups[pop_name] = NeuronGroup(
                int(n), eqs_map[pop_name], threshold=threshold, reset=reset,
                refractory=self.config['neurons']['T_REF'],
                namespace=common_namespace
            )
        
            cleo.coords.assign_coords_rand_rect_prism(self.neuron_groups[pop_name],xlim= self.layer_config['coordinates']['x'], ylim=self.layer_config['coordinates']['y'], zlim=self.layer_config['coordinates']['z'])

      

    def _set_neuron_parameters(self):
        def set_if_exists(group, attr, value):
            try:
                getattr(group, attr)
            except AttributeError:
                return
            setattr(group, attr, value)

        cfg = self.config
        neurons_cfg  = cfg.get('neurons', {})
        intrinsic    = cfg.get('intrinsic_params', {})
        ic_all       = cfg.get('initial_conditions', {})
        vt_default   = neurons_cfg.get('VT', -50.0*mV)

        drive = cfg.get('drive', None)
        if drive is not None:
            c      = float(drive.get('contrast', 1.0))
            mu_bg  = drive.get('mu_bg', {})
            sd_bg  = drive.get('sd_bg', {})
            mu_st  = drive.get('mu_stim', {})
            sd_st  = drive.get('sd_stim', {})
            sg     = drive.get('sigma_global', 0*mV)

        for pop_name, g in self.neuron_groups.items():
            ip = intrinsic.get(pop_name, {})
            set_if_exists(g, 'a',      ip.get('a', 0*nS))
            set_if_exists(g, 'b',      ip.get('b', 0*pA))
            set_if_exists(g, 'DeltaT', ip.get('DeltaT', 2*mV))
            set_if_exists(g, 'EL', ip.get('EL', -60*mV))
            set_if_exists(g, 'gL', ip.get('gL', 8*nS))
            set_if_exists(g, 'C', ip.get('C', 200*pF))
            set_if_exists(g, 'tauw', ip.get('tauw', 200*ms))



            ic_default = ic_all.get('DEFAULT', {})
            ic_pop     = ic_all.get(pop_name, {})
            merged_ic  = dict(ic_default, **ic_pop)

            vcut_factor = merged_ic.pop('Vcut_offset_factor', None)
            if vcut_factor is not None and hasattr(g, 'Vcut'):
                dT = ip.get('DeltaT', getattr(g, 'DeltaT', 2*mV))
                set_if_exists(g, 'Vcut', vt_default + vcut_factor * dT)

            for k, v in merged_ic.items():
                set_if_exists(g, k, v)

            if drive is not None:
                mu = mu_bg.get(pop_name, 0*mV) + c * mu_st.get(pop_name, 0*mV)
                sb = sd_bg.get(pop_name, 0*mV)
                ss = sd_st.get(pop_name, 0*mV)
                sigma = sqrt(sb**2 + (c*ss)**2 + sg**2)
                set_if_exists(g, 'mu_drive',   mu)
                set_if_exists(g, 'sigma_drive', sigma)

    def _on_pre(self, weight_key, cmap, excitatory=False):
        W = cmap
        if self.is_current: # TP CHANGE to ACCOunt FOR PV SOM CONDUCTANCES ETC
            var = 'sI' if not excitatory else 'sE'
            val = float(W / mV)
            return f"{var}_post += {val}*mV"
        else:
            pre, post = weight_key.split('_')
            var = 'g' + pre
            val = W
            return f"{var}_post += {val}*nS"


    # def _create_internal_connections(self):
    #     pmap = self.layer_config.get('connection_prob', {})
    #     cmap =self.layer_config.get('conductance', {})
    #     for connection, p in pmap.items():
    #         pre, post = connection.split('_')
    #         if pre not in self.neuron_groups or post not in self.neuron_groups:
    #             continue
    #         excitatory = (pre == 'E')
    #         syn = Synapses(
    #             self.neuron_groups[pre], self.neuron_groups[post],
    #             on_pre=self._on_pre(connection, cmap[connection], excitatory=excitatory)
    #         )
    #         syn.connect(p=float(p))
    #         self.synapses[connection] = syn

    def _create_internal_connections(self):
        pmap = self.layer_config.get('connection_prob', {})
        cmap = self.layer_config.get('conductance', {})
        
        for connection, p in pmap.items():
            pre, post = connection.split('_')
            if pre not in self.neuron_groups or post not in self.neuron_groups:
                continue
            
            excitatory = (pre == 'E')
            
            if pre == post:  
                delay_mean = 0.8*ms
                delay_std = 0.5*ms
            elif excitatory: 
                delay_mean = 0.6*ms
                delay_std = 0.4*ms
            else:  
                delay_mean = 0.5*ms
                delay_std = 0.3*ms
            
            syn = Synapses(
                self.neuron_groups[pre], 
                self.neuron_groups[post],
                on_pre=self._on_pre(connection, cmap[connection], excitatory=excitatory)
            )
            syn.connect(p=float(p))
            
            syn.delay = f'{delay_mean/ms}*ms + clip(randn()*{delay_std/ms}, -{delay_std/ms}*0.5, {delay_std/ms}*2)*ms'
            
            self.synapses[connection] = syn

    def _create_poisson_inputs(self):
        pinputs = self.layer_config.get('poisson_inputs', {})
        for pop_name, pconf in pinputs.items():
            if pop_name not in self.neuron_groups:
                continue
            target_var = pconf.get('target', 'ge')
            if self.is_current:
                target_var = {'ge': 'sE', 'gi': 'sI'}.get(target_var, target_var)

            if 'N' in pconf and pconf['N'] is not None:
                N = int(pconf['N'])
            else:
                N = float(pconf.get('N', 0.0))
          

            w_cfg = pconf.get('weight', 'EXT')
            weight = self.config['synapses']['Q'][w_cfg] if isinstance(w_cfg, str) else w_cfg
            rate = pconf.get('rate', self.layer_config.get('input_rate', 0*Hz))

            self.poisson_inputs[pop_name] = PoissonInput(
                self.neuron_groups[pop_name], target_var, N=N, rate=rate, weight=weight
            )

    def _create_monitors(self):
        for pop_name, group in self.neuron_groups.items():
            vars_to_record = ['v', 'gE', 'gI','IsynE', 'IsynIPV', 'IsynISOM', 'IsynIVIP', 'IsynI'] if not self.is_current else ['v', 'sE', 'sI']
            self.monitors[f'{pop_name}_state'] = StateMonitor(group, vars_to_record, record=True)
            self.monitors[f'{pop_name}_spikes'] = SpikeMonitor(group, variables='t')
            self.monitors[f'{pop_name}_rate']   = PopulationRateMonitor(group)

    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')

    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)

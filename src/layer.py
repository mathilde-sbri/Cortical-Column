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
            # self.neuron_groups[pop_name].EL    = self.config['intrinsic_params'][pop_name]['EL']    + 1.0*mV*randn(n)      # ±1 mV
            # self.neuron_groups[pop_name].C     = self.config['intrinsic_params'][pop_name]['C']     * (1 + 0.10*randn(n)) 
            # self.neuron_groups[pop_name].gL    = self.config['intrinsic_params'][pop_name]['gL']    * (1 + 0.10*randn(n))
            # self.neuron_groups[pop_name].DeltaT= self.config['intrinsic_params'][pop_name]['DeltaT']* (1 + 0.10*randn(n))
            # self.neuron_groups[pop_name].tauw  = self.config['intrinsic_params'][pop_name]['tauw']  * (1 + 0.10*randn(n))

            # self.neuron_groups[pop_name].gE = np.random.exponential(0.5) * nS
            # self.neuron_groups[pop_name].gPV = np.random.exponential(0.3) * nS
            # self.neuron_groups[pop_name].gSOM = np.random.exponential(0.3) * nS
            # self.neuron_groups[pop_name].gVIP = np.random.exponential(0.2) * nS
            # self.neuron_groups[pop_name].I = (200*pA) + (50*pA) * randn(n) 
            # self.neuron_groups[pop_name].v = 'V_reset + rand() * (VT - V_reset)'

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

    def _on_pre(self, weight_key, excitatory=False):
        W = self.config['synapses']['Q'][weight_key]
        if self.is_current: # TP CHANGE to ACCOunt FOR PV SOM CONDUCTANCES ETC
            var = 'sI' if not excitatory else 'sE'
            val = float(W / mV)
            return f"{var}_post += {val}*mV"
        else:
            pre, post = weight_key.split('_')
            var = 'g' + pre
            val = float(W / nS)
            return f"{var}_post += {val}*nS"

    def _apply_delay(self, syn):
        delay = self.config.get('time_constants', {}).get('DELAY', 0*ms)
        try:
            syn.delay = delay
        except Exception:
            pass

    def _create_internal_connections(self):
        pmap = self.layer_config.get('connection_prob', {})
        for connection, p in pmap.items():
            pre, post = connection.split('_')
            if pre not in self.neuron_groups or post not in self.neuron_groups:
                continue
            excitatory = (pre == 'E')
            syn = Synapses(
                self.neuron_groups[pre], self.neuron_groups[post],
                on_pre=self._on_pre(connection, excitatory=excitatory)
            )
            syn.connect(p=float(p))
            self._apply_delay(syn)
            self.synapses[connection] = syn

    def _create_poisson_inputs(self):
        pinputs = self.layer_config.get('poisson_inputs', {})
        for input_name, pconf in pinputs.items():
           
            if '_' in input_name:
                target_pop = input_name.split('_')[0]  # 'E_stim' -> 'E'
            else:
                target_pop = input_name  # 'E' -> 'E'
            
            if target_pop not in self.neuron_groups:
                continue
                
            target_var = pconf.get('target', 'gE')
            if self.is_current:
                target_var = {'gE': 'sE', 'gI': 'sI'}.get(target_var, target_var)

            if 'N' in pconf and pconf['N'] is not None:
                N = int(pconf['N'])
            else:
                frac = float(pconf.get('N_fraction_of_E', 0.0))
                NE = int(self.layer_config['neuron_counts'].get('E', 0))
                N = int(frac * NE)

            w_cfg = pconf.get('weight', 'EXT')
            weight = self.config['synapses']['Q'][w_cfg] if isinstance(w_cfg, str) else w_cfg
            rate = pconf.get('rate', self.layer_config.get('input_rate', 0*Hz))
            
            onset_time = pconf.get('onset_time', None)
            
            if onset_time is not None:
                dt = self.config['simulation']['DT']
                total_time = self.config['simulation']['SIMULATION_TIME']
                n_steps = int(total_time / dt) + 1
                rates_array = np.zeros(n_steps) * Hz
                onset_idx = int(onset_time / dt)
                rates_array[onset_idx:] = rate
                
                timed_rates = TimedArray(rates_array, dt=dt)
                poisson_group = PoissonGroup(N, rates='timed_rates(t)')
                poisson_group.namespace['timed_rates'] = timed_rates
            else:
                poisson_group = PoissonGroup(N, rates=rate)
            
            excitatory = (target_var in ['gE', 'sE'])
            if self.is_current:
                val = float(weight / mV)
                on_pre_eq = f"{target_var}_post += {val}*mV"
            else:
                val = float(weight / nS)
                on_pre_eq = f"{target_var}_post += {val}*nS"
            
            syn = Synapses(poisson_group, self.neuron_groups[target_pop], on_pre=on_pre_eq)
            syn.connect()  
            
            self.poisson_inputs[input_name] = {
                'group': poisson_group,
                'synapses': syn
            }

    def _create_monitors(self):
        for pop_name, group in self.neuron_groups.items():
            vars_to_record = ['v', 'gE', 'gI'] if not self.is_current else ['v', 'sE', 'sI']
            self.monitors[f'{pop_name}_state'] = StateMonitor(group, vars_to_record, record=True)
            self.monitors[f'{pop_name}_spikes'] = SpikeMonitor(group, variables='t')
            self.monitors[f'{pop_name}_rate']   = PopulationRateMonitor(group)

    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')

    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)

"""
Layer class implementation
"""
import brian2 as b2
from brian2 import *
import numpy as np
import cleo

"""
Layer class implementation with NMDA support
"""
import brian2 as b2
from brian2 import *
import numpy as np
import cleo


class CorticalLayer:
    
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
            
            cleo.coords.assign_coords_rand_rect_prism(
                self.neuron_groups[pop_name],
                xlim=self.layer_config['coordinates']['x'],
                ylim=self.layer_config['coordinates']['y'],
                zlim=self.layer_config['coordinates']['z']
            )

    def _set_neuron_parameters(self):
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
        vt_default = neurons_cfg.get('VT', -50.0*mV)

        drive = cfg.get('drive', None)
        if drive is not None:
            c = float(drive.get('contrast', 1.0))
            mu_bg = drive.get('mu_bg', {})
            sd_bg = drive.get('sd_bg', {})
            mu_st = drive.get('mu_stim', {})
            sd_st = drive.get('sd_stim', {})
            sg = drive.get('sigma_global', 0*mV)

        for pop_name, g in self.neuron_groups.items():
            ip = intrinsic.get(pop_name, {})
            set_if_exists(g, 'a', ip.get('a', 0*nS))
            set_if_exists(g, 'b', ip.get('b', 0*pA))
            set_if_exists(g, 'DeltaT', ip.get('DeltaT', 2*mV))
            set_if_exists(g, 'EL', ip.get('EL', -60*mV))
            set_if_exists(g, 'gL', ip.get('gL', 8*nS))
            set_if_exists(g, 'C', ip.get('C', 200*pF))
            set_if_exists(g, 'tauw', ip.get('tauw', 200*ms))

            ic_default = ic_all.get('DEFAULT', {})
            ic_pop = ic_all.get(pop_name, {})
            merged_ic = dict(ic_default, **ic_pop)

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
                set_if_exists(g, 'mu_drive', mu)
                set_if_exists(g, 'sigma_drive', sigma)

    def _get_delay_params(self, pre_pop, post_pop):
        """Get delay parameters based on connection type"""
        excitatory = (pre_pop == 'E')
        
        if pre_pop == post_pop:  # Recurrent
            delay_mean = 0.8*ms
            delay_std = 0.5*ms
        elif excitatory:  # E -> I
            delay_mean = 0.6*ms
            delay_std = 0.4*ms
        else:  # I -> anything
            delay_mean = 0.5*ms
            delay_std = 0.3*ms
            
        return delay_mean, delay_std

    def _create_internal_connections(self):
        """Create within-layer synaptic connections with AMPA+NMDA for E synapses"""
        pmap = self.layer_config.get('connection_prob', {})
        cmap = self.layer_config.get('conductance', {})
        
        for connection, p in pmap.items():
            pre, post = connection.split('_')
            if pre not in self.neuron_groups or post not in self.neuron_groups:
                continue
            
            excitatory = (pre == 'E')
            delay_mean, delay_std = self._get_delay_params(pre, post)
            
            if excitatory:
                # Look for AMPA and NMDA conductances
                ampa_key = f'{connection}_AMPA'
                nmda_key = f'{connection}_NMDA'
                
                g_ampa = cmap.get(ampa_key, 0.01)
                g_nmda = cmap.get(nmda_key, 0.005)
                
                syn = Synapses(
                    self.neuron_groups[pre],
                    self.neuron_groups[post],
                    on_pre=f'''
                    gE_AMPA_post += {g_ampa}*nS
                    gE_NMDA_post += {g_nmda}*nS
                    '''
                )
                syn.connect(p=float(p))
                syn.delay = (f'{delay_mean/ms}*ms + '
                            f'clip(randn()*{delay_std/ms}, '
                            f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
                
                self.synapses[connection] = syn
            else:
                # Inhibitory synapse - single conductance targeting specific receptor
                g_inh = cmap.get(connection, 0.02)
                
                # Determine target variable based on source population
                target_var = 'g' + pre  # gPV, gSOM, or gVIP
                
                syn = Synapses(
                    self.neuron_groups[pre],
                    self.neuron_groups[post],
                    on_pre=f'{target_var}_post += {g_inh}*nS'
                )
                syn.connect(p=float(p))
                syn.delay = (f'{delay_mean/ms}*ms + '
                            f'clip(randn()*{delay_std/ms}, '
                            f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
                
                self.synapses[connection] = syn

    def _create_poisson_inputs(self):
        """Create external Poisson inputs for all populations"""
        pinputs = self.layer_config.get('poisson_inputs', {})
        
        for pop_name, pconf in pinputs.items():
            # Handle NMDA-specific inputs (e.g., 'E_NMDA', 'PV_NMDA')
            actual_pop = pop_name.split('_')[0]
            
            if actual_pop not in self.neuron_groups:
                continue
                
            target_var = pconf.get('target', 'gE_AMPA')
            if self.is_current:
                target_var = {'gE_AMPA': 'sE', 'gE_NMDA': 'sE', 'gi': 'sI'}.get(target_var, target_var)

            if 'N' in pconf and pconf['N'] is not None:
                N = int(pconf['N'])
            else:
                N = float(pconf.get('N', 0.0))

            w_cfg = pconf.get('weight', 'EXT_AMPA')
            weight = self.config['synapses']['Q'].get(w_cfg, 0.5*nS) if isinstance(w_cfg, str) else w_cfg
            rate = pconf.get('rate', self.layer_config.get('input_rate', 0*Hz))

            self.poisson_inputs[pop_name] = PoissonInput(
                self.neuron_groups[actual_pop], target_var, N=N, rate=rate, weight=weight
            )

    def _create_monitors(self):
        """Create spike, state, and rate monitors for all populations"""
        for pop_name, group in self.neuron_groups.items():
            vars_to_record = ['v', 'gE', 'gI', 'IsynE', 'IsynIPV', 'IsynI'] if not self.is_current else ['v', 'sE', 'sI']
            
            self.monitors[f'{pop_name}_state'] = StateMonitor(group, vars_to_record, record=True)
            self.monitors[f'{pop_name}_spikes'] = SpikeMonitor(group, variables='t')
            self.monitors[f'{pop_name}_rate'] = PopulationRateMonitor(group)

    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')

    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)


def create_inter_layer_synapses(src_layer, dst_layer, config, network):
    """
    Create inter-layer synaptic connections with AMPA+NMDA for excitatory synapses.
    
    Parameters:
    -----------
    src_layer : CorticalLayer
        Source layer object
    dst_layer : CorticalLayer
        Destination layer object
    config : dict
        Configuration dictionary
    network : brian2.Network
        Network to add synapses to
        
    Returns:
    --------
    dict : Dictionary of created synapses
    """
    src_name = src_layer.layer_name
    dst_name = dst_layer.layer_name
    
    conn_dict = config.get('inter_layer_connections', {}).get((src_name, dst_name), {})
    cond_dict = config.get('inter_layer_conductances', {}).get((src_name, dst_name), {})
    
    synapses = {}
    
    for connection, p in conn_dict.items():
        if p <= 0:
            continue
            
        pre, post = connection.split('_')
        
        if pre not in src_layer.neuron_groups or post not in dst_layer.neuron_groups:
            continue
        
        excitatory = (pre == 'E')
        
        # Inter-layer delays are longer
        delay_mean = 1.5*ms if excitatory else 1.0*ms
        delay_std = 0.8*ms if excitatory else 0.5*ms
        
        if excitatory:
            ampa_key = f'{connection}_AMPA'
            nmda_key = f'{connection}_NMDA'
            
            g_ampa = cond_dict.get(ampa_key, 0.01)
            g_nmda = cond_dict.get(nmda_key, 0.005)
            
            syn = Synapses(
                src_layer.neuron_groups[pre],
                dst_layer.neuron_groups[post],
                on_pre=f'''
                gE_AMPA_post += {g_ampa}*nS
                gE_NMDA_post += {g_nmda}*nS
                '''
            )
            syn.connect(p=float(p))
            syn.delay = (f'{delay_mean/ms}*ms + '
                        f'clip(randn()*{delay_std/ms}, '
                        f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
            
            key = f'{src_name}_{dst_name}_{connection}'
            synapses[key] = syn
            network.add(syn)
        else:
            g_inh = cond_dict.get(connection, 0.02)
            target_var = 'g' + pre
            
            syn = Synapses(
                src_layer.neuron_groups[pre],
                dst_layer.neuron_groups[post],
                on_pre=f'{target_var}_post += {g_inh}*nS'
            )
            syn.connect(p=float(p))
            syn.delay = (f'{delay_mean/ms}*ms + '
                        f'clip(randn()*{delay_std/ms}, '
                        f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
            
            key = f'{src_name}_{dst_name}_{connection}'
            synapses[key] = syn
            network.add(syn)
    
    return synapses


def add_heterogeneity_to_layer(layer, config, variability=0.1):
    """
    Add heterogeneity to intrinsic parameters within a layer.
    
    Parameters:
    -----------
    layer : CorticalLayer
        Layer to add heterogeneity to
    config : dict
        Configuration dictionary
    variability : float
        Coefficient of variation for parameter heterogeneity
    """
    intrinsic = config.get('intrinsic_params', {})
    
    for pop_name, group in layer.neuron_groups.items():
        ip = intrinsic.get(pop_name, {})
        n = len(group)
        
        # Add heterogeneity to membrane properties
        for param in ['C', 'gL', 'EL', 'tauw']:
            if hasattr(group, param):
                base_val = ip.get(param, getattr(group, param))
                noise = 1 + variability * np.random.randn(n)
                noise = np.clip(noise, 0.7, 1.3)  # Limit range
                setattr(group, param, base_val * noise)
        
        # Add heterogeneity to adaptation parameters (if present)
        if hasattr(group, 'a') and pop_name != 'PV':  # PV has no adaptation
            base_a = ip.get('a', 0*nS)
            if base_a > 0:
                noise = 1 + variability * np.random.randn(n)
                noise = np.clip(noise, 0.5, 1.5)
                group.a = base_a * noise
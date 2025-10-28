"""
Column class implementation
"""
import brian2 as b2
from brian2 import *
from .layer import CorticalLayer
from cleo import ephys

class CorticalColumn:
    """
     a cortical column with multiple layers
    """
    
    def __init__(self, column_id=0, config=None):
        self.column_id = column_id
        self.layer_names = list(config['layers'].keys()) or ['L1', 'L23', 'L4', 'L5', 'L6']
        if config is None:
            raise ValueError("CorticalColumn requires a config dictionary. Pass CONFIG from config module.")
        self.config = config
        self.electrode = None
        self.layers = {}
        self.inter_layer_synapses = {}
        
        self._create_layers()
        self._create_inter_layer_connections()
        self.network = Network()
        self._assemble_network()
        self._insert_electrode()

    def _insert_electrode(self):
        array_length = 2.25 * b2.mm  # 15 intervals × 150um, from the data
        channel_count = 16

        coords = ephys.linear_shank_coords(
            array_length, 
            channel_count=16,
            start_location=(0, 0, -0.9) * b2.mm  
        )

        probe = ephys.Probe(coords, save_history=True)
        self.electrode = probe
    
    def _assemble_network(self):
        """Collect all components into the network"""
        # Add all components from each layer
        # This is necessary when working with classes, because i guess brian2's run function expects all components to be created in the same file ? not sure
        for layer in self.layers.values():
            self.network.add(*layer.neuron_groups.values())
            self.network.add(*layer.synapses.values())
            self.network.add(*layer.poisson_inputs.values())
            self.network.add(*layer.monitors.values())
        
        self.network.add(*self.inter_layer_synapses.values())
    
    def _create_layers(self):
        for layer_name in self.layer_names:
            if layer_name in self.config['layers']:
                self.layers[layer_name] = CorticalLayer(
                    f"{layer_name}_col{self.column_id}",
                    layer_name,
                    self.config
                )
    
    def _on_pre(self, weight_key, conductance, excitatory=False):
    
        pre, post = weight_key.split('_')
        var = 'g' + pre
        val = conductance
        return f"{var}_post += {val}*nS"
    


    def _create_inter_layer_connections(self):
        for (source_layer, target_layer), conns in self.config['inter_layer_connections'].items():
            if source_layer in self.layers and target_layer in self.layers:
                for conn, prob in conns.items() :
                    pre, post = conn.split('_')
                    excitatory = (pre == 'E')
                    connection_name = f"{source_layer}_{target_layer}_{conn}"
                    is_current = (self.config['models'].get('synapse_model', 'conductance').lower() == 'current')
                    W = self.config['inter_layer_conductances'][(source_layer, target_layer)][conn]
                    conductance = self.config['inter_layer_conductances'][source_layer, target_layer][conn]
                    if is_current:
                        on_pre = f"sE_post += {float(W/mV)}*mV"
                    else:
                        on_pre = self._on_pre(conn, conductance , excitatory=excitatory)
                    group1, group2 = conn.split("_")
                    syn = Synapses(self.layers[source_layer].get_neuron_group(group1),
                                self.layers[target_layer].get_neuron_group(group2),
                                on_pre=on_pre, delay=1*ms + rand()*2*ms)
                    syn.connect(p=prob)
 
                    self.inter_layer_synapses[connection_name] = syn


    def get_layer(self, layer_name):
        return self.layers.get(layer_name)
    
    def get_all_monitors(self):
        all_monitors = {}
        for layer_name, layer in self.layers.items():
            all_monitors[layer_name] = layer.monitors
        return all_monitors
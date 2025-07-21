"""
Column class implementation
"""
import brian2 as b2
from brian2 import *
from .layer import CorticalLayer
from .parameters import *
from config.layer_configs import LAYER_CONFIGS, INTER_LAYER_CONNECTIONS

class CorticalColumn:
    """
     a cortical column with multiple layers
    """
    
    def __init__(self, column_id=0, layer_names=None):
        self.column_id = column_id
        self.layer_names = layer_names or ['L1', 'L23', 'L4', 'L5', 'L6']
        self.layers = {}
        self.inter_layer_synapses = {}
        
        self._create_layers()
        self._create_inter_layer_connections()
        self.network = Network()
        self._assemble_network()
    
    def _assemble_network(self):
        """Collect all components into the network"""
        # Add all components from each layer
        # This is necessary when working with classes, because brian2's run function expects all components to be created in the same file. So I need to bypass that
        for layer in self.layers.values():
            self.network.add(*layer.neuron_groups.values())
            self.network.add(*layer.synapses.values())
            self.network.add(*layer.poisson_inputs.values())
            self.network.add(*layer.monitors.values())
        
        self.network.add(*self.inter_layer_synapses.values())
    
    def _create_layers(self):
        for layer_name in self.layer_names:
            if layer_name in LAYER_CONFIGS:
                self.layers[layer_name] = CorticalLayer(
                    f"{layer_name}_col{self.column_id}",
                    LAYER_CONFIGS[layer_name]
                )
    
    def _create_inter_layer_connections(self):
        for (source_layer, target_layer), prob in INTER_LAYER_CONNECTIONS.items():
            if source_layer in self.layers and target_layer in self.layers:
                connection_name = f"{source_layer}_{target_layer}"
                
                self.inter_layer_synapses[connection_name] = Synapses(
                    self.layers[source_layer].get_neuron_group('E'),
                    self.layers[target_layer].get_neuron_group('E'),
                    on_pre=f'ge_post += {Q_E_TO_E/nS}*nS'
                )
                self.inter_layer_synapses[connection_name].connect(p=prob)
    
    def get_layer(self, layer_name):
        return self.layers.get(layer_name)
    
    def get_all_monitors(self):
        all_monitors = {}
        for layer_name, layer in self.layers.items():
            all_monitors[layer_name] = layer.monitors
        return all_monitors
# Cortical Column Simulation 

This repository contains a modular simulation of a **cortical column** using the [Brian2](https://brian2.readthedocs.io/en/stable/) spiking neural network simulator. It models **multiple layers** of the cortex with  populations of **excitatory** and **inhibitory neurons**, including **PV**, **SOM**, and **VIP** subtypes.


---

## 📁 Project Structure

.
├── configs/
│ └── layer_configs.py # Layer-specific configurations: neuron counts, types, structure
├── src/
│ ├── column.py # Cortical column class: integrates multiple layers
│ ├── layer.py # Defines a single layer: neuron populations and local connectivity
│ ├── neuron_models.py # Neuron model definitions (APEX equations)
│ ├── parameters.py # Electrophysiological parameters: conductances, delays, etc.
├── main.py # Main script to run the simulation


---


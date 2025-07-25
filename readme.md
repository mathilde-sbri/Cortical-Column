# Cortical Column Simulation 

This repository contains a modular simulation of a **cortical column** using the [Brian2](https://brian2.readthedocs.io/en/stable/) spiking neural network simulator. It models **multiple layers** of the cortex with  populations of **excitatory** and **inhibitory neurons**, including **PV**, **SOM**, and **VIP** subtypes.


---

## Structure

- `main.py`  
  Main script to launch the simulation.

- `configs/`
  - `layer_configs.py`  
    Configuration for each cortical layer: number of neurons, layer definitions, and population types.


- `src/`
  - `column.py`  
    Full cortical column structure, with multiple layers.
  - `layer.py`  
    Code for an individual cortical layer, including neuron populations and local connectivity.
  - `neuron_models.py`  
    Contains the neuron model definitions using ADEX equations.
  - `parameters.py`  
    Electrophysiological parameters for neuron dynamics and synaptic interactions (conductances, delays, time constants).


## Getting Started

To run the simulation locally:

### 1. Clone the repository

```bash
git clone https://github.com/mathilde-sbri/Cortical-Column.git
cd Cortical-Column
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies


```bash
pip install -r requirements.txt
```

### 4. Run the simulation


```bash
python main.py
```
"""
Neuron model equations and types, maybe to merge with parameters ?
"""
import brian2 as b2
from brian2 import *
from .parameters import *

class NeuronModels:
    
    @staticmethod
    def get_excitatory_equations():
        return """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
        IsynE=ge*(Ee-v) : amp
        IsynI=gi*(Ei-v) : amp
        dge/dt = -ge/tau_e : siemens
        dgi/dt = -gi/tau_i : siemens
        dw/dt = (a*(v - EL) - w)/tauw : amp
        taum= C/gL : second
        I : amp
        a : siemens
        b : amp
        DeltaT: volt
        Vcut: volt
        EL : volt
        C : farad
        gL : siemens
        """
    
    @staticmethod
    def get_pv_equations():
        return """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
        IsynE=ge*(Ee-v) : amp
        IsynI=gi*(Ei-v) : amp
        dge/dt = -ge/tau_e_pv : siemens
        dgi/dt = -gi/tau_i : siemens
        dw/dt = (a*(v - EL) - w)/tauw : amp
        taum= C/gL : second
        I : amp
        a : siemens
        b : amp
        DeltaT: volt
        Vcut: volt
        EL : volt
        C : farad
        gL : siemens
        """
    
    @staticmethod
    def get_sst_equations():
        return """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
        IsynE=ge*(Ee-v) : amp
        IsynI=gi*(Ei-v) : amp
        dge/dt = -ge/tau_e_som : siemens
        dgi/dt = -gi/tau_i : siemens
        dw/dt = (a*(v - EL) - w)/tauw : amp
        taum= C/gL : second
        I : amp
        a : siemens
        b : amp
        DeltaT: volt
        Vcut: volt
        EL : volt
        C : farad
        gL : siemens
        """
    
    @staticmethod
    def get_vip_equations():
        return """
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
        IsynE=ge*(Ee-v) : amp
        IsynI=gi*(Ei-v) : amp
        dge/dt = -ge/tau_e_vip : siemens
        dgi/dt = -gi/tau_i : siemens
        dw/dt = (a*(v - EL) - w)/tauw : amp
        taum= C/gL : second
        I : amp
        a : siemens
        b : amp
        DeltaT: volt
        Vcut: volt
        EL : volt
        C : farad
        gL : siemens
        """

class NeuronParameters:
    
    @staticmethod
    def get_excitatory_params():
        return {
            'a': 4*nS,
            'b': 130*pA,
            'DeltaT': 2*mV,
            'EL': E_LEAK_RS,
            'C': NEURON_CAPACITANCE,
            'gL': NEURON_LEAK_CONDUCTANCE
        }
    
    @staticmethod
    def get_pv_params():
        return {
            'a': 0*nS,
            'b': 0*pA,
            'DeltaT': 0.5*mV,
            'EL': E_LEAK_FS,
            'C': NEURON_CAPACITANCE,
            'gL': NEURON_LEAK_CONDUCTANCE
        }
    
    @staticmethod
    def get_sst_params():
        return {
            'a': 4*nS,
            'b': 25*pA,
            'DeltaT': 1.5*mV,
            'EL': E_LEAK_SST,
            'C': NEURON_CAPACITANCE,
            'gL': NEURON_LEAK_CONDUCTANCE
        }
    
    @staticmethod
    # TO CHECK !!!
    def get_vip_params():
        return {
            'a': 2*nS, # or 4 ?
            'b': 50*pA,
            'DeltaT': 2*mV,
            'EL': E_LEAK_VIP,
            'C': NEURON_CAPACITANCE,
            'gL': NEURON_LEAK_CONDUCTANCE
        }
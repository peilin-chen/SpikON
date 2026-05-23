from typing import Callable
import torch
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj
import torch.nn as nn

class SLTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = None,
            detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x

class Learnable_Threshold_SLTTNeuron(SLTTNeuron):
    def __init__(self, tau: float = 2., decay_input: bool = False,
                 v_threshold: float = 1., v_reset: float = None,
                 surrogate_function=None, detach_reset: bool = False,
                 cupy_fp32_inference: bool = False,
                 min_thr: float = 1e-2, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset,
                         surrogate_function, detach_reset,
                         cupy_fp32_inference=False, 
                         **kwargs)
        
        if hasattr(self, '_memories') and 'v_threshold' in self._memories:
            self._memories.pop('v_threshold', None)

        if hasattr(self, 'v_threshold') and not isinstance(self.v_threshold, nn.Parameter):
            delattr(self, 'v_threshold')

        self.register_parameter(
            'v_threshold',
            nn.Parameter(torch.tensor(float(v_threshold)))
        )

    def forward(self, x: torch.Tensor):
        return super().forward(x)

class Learnable_Threshold_Through_Time_SLTTNeuron(SLTTNeuron):
    def __init__(self, tau: float = 2., decay_input: bool = False,
                 v_threshold: float = 1., v_reset: float = None,
                 surrogate_function=None, detach_reset: bool = False,
                 cupy_fp32_inference: bool = False,
                 T: int = 2, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset,
                         surrogate_function, detach_reset, cupy_fp32_inference=False, **kwargs)

        if hasattr(self, '_memories') and 'v_threshold' in self._memories:
            self._memories.pop('v_threshold', None)
        if hasattr(self, 'v_threshold') and not isinstance(self.v_threshold, nn.Parameter):
            delattr(self, 'v_threshold')

        self.vth_per_t = nn.Parameter(torch.full((int(T),), float(v_threshold)))
        self.T = int(T)
    
    def forward(self, x: torch.Tensor, t: int):
        self.neuronal_charge(x)

        vthr_t = self.vth_per_t[t % self.T]           
        spike  = self.surrogate_function(self.v - vthr_t)

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            self.v = self.v - spike_d * vthr_t
        else:
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

        return spike

    #def forward(self, x: torch.Tensor, t: int):
    #    self.v_threshold = self.vth_per_t[t % self.T]
    #    out = super().forward(x)
    #        
    #    return out
    
class BPTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = None,
            detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

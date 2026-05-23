import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from collections import defaultdict

__all__ = [
    'SpikingVGGBN', 'spiking_vgg11_bn'
]

import torch
cfg = {
    'VGG11': [
        [64],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512]
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}

#class NeuronT(nn.Module):
#    def __init__(self, neuron, T, **kwargs):
#        super().__init__()
#        self.T = T
#        self.neurons = nn.ModuleList([neuron(**kwargs) for _ in range(T)])
#
#    def forward(self, x, t: int):
#        return self.neurons[t](x)
#
#class VGGBlockT(nn.Module):
#    def __init__(self, in_ch, out_ch, neuron, T, BN, dropout, **kwargs):
#        super().__init__()
#        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
#        self.bn = nn.BatchNorm2d(out_ch) if BN else nn.Identity()
#        self.neuron = NeuronT(neuron, T, **kwargs)  # each timestep has its own LIF neuron
#        self.do = layer.Dropout(dropout)
#
#    def forward(self, x, t: int):
#        x = self.conv(x)
#        x = self.bn(x)
#        x = self.neuron(x, t)     # t
#        x = self.do(x)
#        return x
#
#class SpikingVGGBN(nn.Module):
#    def __init__(self, vgg_name, neuron: callable, dropout=0.0, num_classes=10, BN=False, **kwargs):
#        super(SpikingVGGBN, self).__init__()
#        #self.T = kwargs.get('T', 6)            # get T
#        self.T = kwargs.pop('T', 6)            # get T
#        self.init_channels = kwargs.get('c_in', 2)
#        self.fc_hw = kwargs.get('fc_hw', 3)
#
#        cfg_list = cfg[vgg_name]
#        layers = []
#        in_ch = self.init_channels
#        for x in cfg_list[0]:
#            if x == 'M':
#                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
#            else:
#                layers.append(VGGBlockT(in_ch, x, neuron, self.T, BN, dropout, **kwargs))
#                in_ch = x
#        self.layer1 = nn.ModuleList(layers)
#
#        def _make(vgg_cfg):
#            nonlocal in_ch
#            blocks = []
#            for x in vgg_cfg:
#                if x == 'M':
#                    blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
#                else:
#                    blocks.append(VGGBlockT(in_ch, x, neuron, self.T, BN, dropout, **kwargs))
#                    in_ch = x
#            return nn.ModuleList(blocks)
#
#        self.layer2 = _make(cfg_list[1])
#        self.layer3 = _make(cfg_list[2])
#        self.layer4 = _make(cfg_list[3])
#        self.layer5 = _make(cfg_list[4])
#
#        self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
#        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512*self.fc_hw*self.fc_hw, num_classes))
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                if m.bias is not None:
#                    nn.init.constant_(m.bias, 0)
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)
#            elif isinstance(m, nn.Linear):
#                nn.init.normal_(m.weight, 0, 0.01)
#                nn.init.constant_(m.bias, 0)
#
#    def _run_stage(self, modules: nn.ModuleList, x, t: int):
#        for m in modules:
#            # VGGBlockT needs t
#            if isinstance(m, VGGBlockT):
#                x = m(x, t)
#            else:
#                x = m(x)
#        return x
#
#    def forward(self, x, t: int):
#        x = self._run_stage(self.layer1, x, t)
#        x = self._run_stage(self.layer2, x, t)
#        x = self._run_stage(self.layer3, x, t)
#        x = self._run_stage(self.layer4, x, t)
#        x = self._run_stage(self.layer5, x, t)
#        x = self.avgpool(x)
#        x = self.classifier(x)
#        return x
    
class SpikingVGGBN_LT(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, BN=False, **kwargs):
        super(SpikingVGGBN_LT, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        self.fc_hw = kwargs.get('fc_hw', 3)
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, BN, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, BN, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, BN, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, BN, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, BN, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*self.fc_hw*self.fc_hw, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, BN, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                if BN:
                    layers.append(nn.BatchNorm2d(x))
                layers.append(neuron(**kwargs))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out

class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, gain=True, eps=1e-4):
        super(WSLinear, self).__init__(in_features, out_features, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_features, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.linear(x, self.get_weight(), self.bias)

class Scale(nn.Module):

    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x, **kwargs):
        return x * self.scale
    
class SpikingVGGWS(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, BN=False, **kwargs):
        super(SpikingVGGWS, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        self.fc_hw = kwargs.get('fc_hw', 3)
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, BN, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, BN, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, BN, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, BN, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, BN, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            WSLinear(512*self.fc_hw*self.fc_hw, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, BN, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(WSConv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                if BN:
                    layers.append(nn.BatchNorm2d(x))
                layers.append(neuron(**kwargs))
                layers.append(Scale(2.74))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out
    
class ScaledWConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ScaledWConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean)
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledWLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(ScaledWLinear, self).__init__(in_features, out_features, bias)

    def get_weight(self):
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean)
        return weight

    def forward(self, x):
        return F.linear(x, self.get_weight(), self.bias)

class TimeScaledWConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 T=6, init_gain=2.74):
        super().__init__(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias)
        self.T = int(T)
        shape = (self.T, 1)
        self.gain = nn.Parameter(torch.full(shape, float(init_gain)))

    def _centralize(self, w):
        mean = w.mean(dim=(1,2,3), keepdim=True)
        return w - mean

    def get_weight_t(self, t: int):
        w = self.weight
        w = self._centralize(w)
        gamma_t = self.gain[t]                
        return gamma_t * w

    def forward(self, x, t: int):
        return F.conv2d(x, self.get_weight_t(t), self.bias, self.stride, self.padding, self.dilation, self.groups)
    
# LTTT+SW
class VGGBlockT_LTTT_SW(nn.Module):
    def __init__(self, in_ch, out_ch, neuron, T, BN, dropout, **kwargs):
        super().__init__()
        # self.conv = ScaledWConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv = TimeScaledWConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True, T=T)
        self.bn   = nn.BatchNorm2d(out_ch) if BN else nn.Identity()
        self.neuron = neuron(T=T, **kwargs)
        #self.scale = Scale(2.74)
        self.do   = layer.Dropout(dropout)

    def forward(self, x, t: int):
        x = self.conv(x, t)
        x = self.bn(x)
        x = self.neuron(x, t) 
        #x = self.scale(x)
        x = self.do(x)
        return x

class SpikingVGG_LTTT_SW(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0,
                 num_classes=10, BN=False, **kwargs):
        super(SpikingVGG_LTTT_SW, self).__init__()
        self.whether_bias  = True
        self.init_channels = kwargs.get('c_in', 2)
        self.fc_hw         = kwargs.get('fc_hw', 3)
        self.T             = kwargs.pop('T', 6)   # get timestep

        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, BN, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, BN, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, BN, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, BN, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, BN, **kwargs)

        self.avgpool = nn.AdaptiveMaxPool2d((self.fc_hw, self.fc_hw)) # AdaptiveMaxPool2d or AdaptiveAvgPool2d
        self.classifier = nn.Sequential(nn.Flatten(),
                                        #nn.Dropout(0.3),
                                        ScaledWLinear(512*self.fc_hw*self.fc_hw, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, ScaledWLinear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def _make_layers(self, vgg_cfg, dropout, neuron_ctor, BN, **kwargs):
        layers = []
        in_ch = self.init_channels
        for x in vgg_cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #AvgPool2d or MaxPool2d
            else:
                layers.append(VGGBlockT_LTTT_SW(in_ch, x, neuron_ctor, self.T, BN, dropout, **kwargs))
                in_ch = x
        self.init_channels = in_ch
        return nn.ModuleList(layers)

    def _forward_blocklist(self, layers: nn.ModuleList, x, t: int):
        for m in layers:
            if isinstance(m, nn.MaxPool2d): #AvgPool2d or MaxPool2d
                x = m(x)
            else:
                x = m(x, t)   # VGGBlockT needs T
        return x

    def forward(self, x, t: int):
        x = self._forward_blocklist(self.layer1, x, t)
        x = self._forward_blocklist(self.layer2, x, t)
        x = self._forward_blocklist(self.layer3, x, t)
        x = self._forward_blocklist(self.layer4, x, t)
        x = self._forward_blocklist(self.layer5, x, t)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x    

# LTTT
class VGGBlockT(nn.Module):
    def __init__(self, in_ch, out_ch, neuron, T, BN, dropout, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn   = nn.BatchNorm2d(out_ch) if BN else nn.Identity()
        self.neuron = neuron(T=T, **kwargs)
        self.do   = layer.Dropout(dropout)

    def forward(self, x, t: int):
        x = self.conv(x)
        x = self.bn(x)
        x = self.neuron(x, t)     
        x = self.do(x)
        return x

class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0,
                 num_classes=10, BN=False, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias  = True
        self.init_channels = kwargs.get('c_in', 2)
        self.fc_hw         = kwargs.get('fc_hw', 3)
        self.T             = kwargs.pop('T', 6)   # get timestep

        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, BN, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, BN, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, BN, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, BN, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, BN, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(512*self.fc_hw*self.fc_hw, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def _make_layers(self, vgg_cfg, dropout, neuron_ctor, BN, **kwargs):
        layers = []
        in_ch = self.init_channels
        for x in vgg_cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(VGGBlockT(in_ch, x, neuron_ctor, self.T, BN, dropout, **kwargs))
                in_ch = x
        self.init_channels = in_ch
        return nn.ModuleList(layers)

    def _forward_blocklist(self, layers: nn.ModuleList, x, t: int):
        for m in layers:
            if isinstance(m, nn.AvgPool2d):
                x = m(x)
            else:
                x = m(x, t)   # VGGBlockT needs T
        return x

    def forward(self, x, t: int):
        x = self._forward_blocklist(self.layer1, x, t)
        x = self._forward_blocklist(self.layer2, x, t)
        x = self._forward_blocklist(self.layer3, x, t)
        x = self._forward_blocklist(self.layer4, x, t)
        x = self._forward_blocklist(self.layer5, x, t)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

class ActivationRecorder:
    def __init__(self, model, conv_types=(TimeScaledWConv2d,), fc_types=(ScaledWLinear,), 
                 store_on_cpu=True, keep_last_only=False):
        self.model = model
        self.conv_types = conv_types
        self.fc_types = fc_types
        self.store_on_cpu = store_on_cpu
        self.keep_last_only = keep_last_only

        self.current_t = None
        self.activations = defaultdict(dict)
        self.handles = []

        self._register_hooks()

    def set_t(self, t: int):
        self.current_t = int(t)

    def reset(self):
        self.activations.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self.conv_types) or isinstance(module, self.fc_types):
                h = module.register_forward_pre_hook(self._make_pre_hook(name))
                self.handles.append(h)

    def _make_pre_hook(self, layer_name):
        def _hook(module, inputs):
            if self.current_t is None:
                return
            x = inputs[0]
            if self.store_on_cpu:
                x = x.detach().to('cpu')
            else:
                x = x.detach()
            x = x.float()

            if self.keep_last_only:
                self.activations[layer_name] = {self.current_t: x}
            else:
                self.activations[layer_name][self.current_t] = x
        return _hook

    @torch.no_grad()
    def cosine_between_adjacent_ts(self, layer_name, reduce='mean'):
        if layer_name not in self.activations:
            raise KeyError(f"No activations recorded for layer: {layer_name}")

        ts = sorted(self.activations[layer_name].keys())
        sims = []
        for i in range(len(ts) - 1):
            a = self.activations[layer_name][ts[i]]
            b = self.activations[layer_name][ts[i+1]]

            a_flat = a.view(a.size(0), -1)
            b_flat = b.view(b.size(0), -1)

            per_sample = F.cosine_similarity(a_flat, b_flat, dim=1)  # [N]
            if reduce == 'mean':
                sims.append(per_sample.mean().item())
            else:
                sims.append(per_sample)  # tensor [N]
        return sims

    @torch.no_grad()
    def cosine_all_layers(self, reduce='mean'):
        out = {}
        for layer_name in self.activations.keys():
            out[layer_name] = self.cosine_between_adjacent_ts(layer_name, reduce=reduce)
        return out

    @torch.no_grad()
    def saving_between_adjacent_ts(self, layer_name, nz_eps=0.0, reduce='mean', eps=1e-12):
        if layer_name not in self.activations:
            raise KeyError(f"No activations recorded for layer: {layer_name}")

        ts = sorted(self.activations[layer_name].keys())
        savs = []
        ns = []
        deltas = []

        for i in range(len(ts) - 1):
            a = self.activations[layer_name][ts[i]]     
            b = self.activations[layer_name][ts[i+1]]
            a_flat = a.view(a.size(0), -1)
            b_flat = b.view(b.size(0), -1)

            a_nz = (a_flat.abs() > nz_eps).float()
            b_nz = (b_flat.abs() > nz_eps).float()
            m = a_nz.mean(dim=1)  # [N], ∈[0,1]
            n = b_nz.mean(dim=1)  # [N], ∈[0,1]

            c = F.cosine_similarity(a_flat, b_flat, dim=1, eps=eps)

            sqrt_mn = (m * n).sqrt()
            delta = (m + n) - 2.0 * c * sqrt_mn         # [N]
            delta = delta.clamp(0.0, 1.0)
            #saving = torch.where(n > 0, (n - delta) / (n + 1e-12), torch.zeros_like(n))
            #saving = saving.clamp(0.0, 1.0)
            saving = n - delta

            if reduce == 'mean':
                savs.append(saving.mean().item())
                ns.append(n.mean().item())
                deltas.append(delta.mean().item())
            else:
                savs.append(saving)  # [N]
                ns.append(n)
                deltas.append(delta)

        return savs, ns, deltas

    @torch.no_grad()
    def saving_all_layers(self, nz_eps=0.0, reduce='mean'):
        out1 = {}
        out2 = {}
        out3 = {}
        for layer_name in self.activations.keys():
            out1[layer_name], out2[layer_name], out3[layer_name] = self.saving_between_adjacent_ts(
                layer_name, nz_eps=nz_eps, reduce=reduce
            )
        return out1, out2, out3

    def recorded_layers(self):
        return sorted(self.activations.keys())

class SpikingVGGBN_BPTT(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingVGGBN_BPTT, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        self.fc_hw = kwargs.get('fc_hw', 3)
        
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*self.fc_hw*self.fc_hw, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(nn.BatchNorm2d(x))
                layers.append(neuron(**kwargs))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out

# BPTT+BN
def spiking_vgg11_bn_bptt(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN_BPTT('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
    
# LTTT+SW
def spiking_vgg11_lttt_sw(neuron: callable = None, num_classes=10, neuron_dropout=0.0, BN=False, **kwargs):
    return SpikingVGG_LTTT_SW('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, BN=BN, **kwargs)    

# LTTT
def spiking_vgg11_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, BN=False, **kwargs):
    return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, BN=BN, **kwargs)

# LT
def spiking_vgg11_bn_lt(neuron: callable = None, num_classes=10, neuron_dropout=0.0, BN=False, **kwargs):
    return SpikingVGGBN_LT('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, BN=BN, **kwargs)

# WS
def spiking_vgg11_ws(neuron: callable = None, num_classes=10, neuron_dropout=0.0, BN=False, **kwargs):
    return SpikingVGGWS('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, BN=BN, **kwargs)

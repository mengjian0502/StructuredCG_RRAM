"""
DNN quantization with / without merged modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import numpy as np
import torch.nn.init as init
from collections import OrderedDict


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        output_int = input.mul(scale).round_()
        output_float = output_int.div_(scale)
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class RoundUQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, nbit): 
        ctx.save_for_backward(input, alpha)

        scale = (2**nbit - 1) / alpha
        input_div = input.mul(scale)
        input_q = input_div.round().div(scale)
        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors

        lower_bound = input < 0
        upper_bound = input > alpha

        x_range = ~(lower_bound|upper_bound)

        grad_alpha = torch.sum(grad_output * torch.ge(input, alpha).float()).view(-1)
        grad_input = grad_output * x_range.float()
        return grad_input, grad_alpha, None


class WQ(nn.Module):
    def __init__(self, wbit, num_features, channel_wise=1):
        super(WQ, self).__init__()
        self.wbit = wbit
        self.num_features = num_features

        if channel_wise:
            self.register_buffer('alpha_w', torch.ones(num_features))
        else:
            self.register_buffer('alpha_w', torch.tensor(1.))

        self.channel_wise = channel_wise

    def forward(self, input):
        if self.wbit == 32:
            return input
        else:
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
            z = z_typical[f'{int(self.wbit)}bit']
            n_lv = 2 ** (self.wbit - 1) - 1

            m = input.abs().mean()
            std = input.std()
            
            self.alpha_w = 1/z[0] * std - z[1]/z[0] * m 
            input = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
            scale = n_lv / self.alpha_w

            w_float = RoundQuant.apply(input, scale)
            return w_float
    
    def extra_repr(self):
        return super(WQ, self).extra_repr() + 'wbit={}, channel_wise={}'.format(self.wbit, self.channel_wise)

class AQ(nn.Module):
    def __init__(self, abit, num_features, alpha_init):
        super(AQ, self).__init__()
        self.abit = abit
        self.alpha = nn.Parameter(torch.Tensor([alpha_init]))

    def forward(self, input):
        if input.size(1) > 3:
            if self.abit == 32:
                a_float = input
            else:
                input = torch.where(input < self.alpha, input, self.alpha)
                a_float = RoundUQ.apply(input, self.alpha, self.abit)
        else:
            a_float = input
        return a_float
    
    def extra_repr(self):
        return super(AQ, self).extra_repr() + 'abit={}'.format(self.abit)

class QConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False, 
        wbit=32, 
        abit=32, 
        channel_wise=0
    ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias
        )
        
        # precisions
        self.abit = abit
        self.wbit = wbit
        num_features = self.weight.data.size(0)

        self.WQ = WQ(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        self.AQ = AQ(abit=abit, num_features=num_features, alpha_init=10.0)
        
        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)

        out = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class QLinear(nn.Linear):
    r"""
    Fully connected layer with Quantized weight
    """
    def __init__(self, in_features, out_features, bias=True, wbit=32, abit=32, alpha_init=10.0):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # precisions
        self.wbit = wbit
        self.abit = abit
        self.alpha_init = alpha_init
        channels = self.weight.data.size(0)

        # quantizers
        # self.WQ = WQPROFIT(wbit=wbit, num_features=channels, channel_wise=0)
        # self.AQ = AQPROFIT(abit=abit)
        self.WQ = WQ(wbit=wbit, num_features=channels, channel_wise=0)
        self.AQ = AQ(abit=abit, num_features=channels, alpha_init=alpha_init)

        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)

        # import pdb;pdb.set_trace()
        # print(f"WQ.a = {self.WQ.a.data.item()}")
        # print(f"WQ.c = {self.WQ.c.data.item()}")
        # print(f"AQ.a = {self.AQ.a.data.item()}")
        # print(f"AQ.c = {self.AQ.c.data.item()}")

        out = F.linear(input_q, weight_q, self.bias)
        return out


"""
Get column-wise MAC
"""
class ColQConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False, 
        wbit=32, 
        abit=32, 
        channel_wise=0
    ):
        super(ColQConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias
        )

        self.layer_idx=0
        
        # precisions
        self.abit = abit
        self.wbit = wbit
        num_features = self.weight.data.size(0)

        # self.WQ = WQPROFIT(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        # self.AQ = AQPROFIT(abit)
        self.WQ = WQ(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        self.AQ = AQ(abit=abit, num_features=num_features, alpha_init=10.0)
        
        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)
        out_original = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # inference only
        scale_w = self.WQ.scale
        weight_int, weight_float, _ = roundquantclamp(self.weight, scale_w)
        c_out, c_in, h, w = weight_int.size()
        
        target_samples = 10000
        save_mac = False
        mac64 = torch.tensor([]).cuda()
        output = torch.zeros_like(out_original)
        for i in range(c_out):
            channel_out = torch.zeros_like(out_original)
            for k in range(h):
                for m in range(w):
                    mask = torch.zeros_like(weight_int)
                    mask[i, :, k, m] = 1.
                    weight_m = weight_int * mask     # input channel
                    partial = F.conv2d(input_q, weight_m, self.bias, self.stride, self.padding, self.dilation, self.groups)         
                    channel_out += partial
                    
                    # save mac values
                    if len(mac64) < target_samples:
                        mac_matrix = partial[:,i,:,:].reshape(-1)
                        mac64 = torch.cat((mac64, mac_matrix))  
                    else:
                        continue
            output += channel_out
        output = output.div(scale_w)
        
        if save_mac:
            # torch.save(mac64, './mac{}_{}x{}x{}x{}_nonclamp.pt'.format(c_in, c_out, c_in, h, w))
            torch.save(weight_int, './weight_int/weight_q{}_clamp.pt'.format(self.layer_idx))
        return output
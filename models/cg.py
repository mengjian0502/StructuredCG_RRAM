"""
Channel Gating Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .qmodules import WQ, AQ


class Greater_Than(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.gt(input, 0).float()

    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None 

class Conv2d_CG(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                slice_size=16, cg_groups=1, cg_threshold_init=-6.0, cg_alpha=2.0, hard_sig=False):
        super(Conv2d_CG, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.hard_sig = hard_sig
        self.iter = 1

        # base path selection
        self.slide_freq = 1e+4
        self.cg_base_group_index = 0

        self.cg_in_chunk_size = 16                  # fixed input chunk
        self.cg_groups = cg_groups
        
        # channel gating
        self.cg_alpha = cg_alpha
        self.cg_threshold = nn.Parameter(cg_threshold_init * torch.ones(1, out_channels, 1, 1))
        self.cg_in_chunk_size = in_channels // self.cg_groups

        self.cg_bn = nn.functional.instance_norm
        self.cg_gt = Greater_Than.apply
        if self.cg_in_chunk_size < 128:
            self.slice_size = slice_size
        else:
            self.slice_size = 128
        self.mask_base = torch.zeros_like(self.weight.data).cuda()
        
        self.cg_out_chunk_size = out_channels // self.cg_groups
        for i in range(self.cg_groups):
            self.mask_base[i*self.cg_out_chunk_size:(i+1)*self.cg_out_chunk_size,i*self.cg_in_chunk_size:(i+1)*self.cg_in_chunk_size,:,:] = 1
        
        self.mask_cond = 1 - self.mask_base

        # quantization
        self.weight_quant = WQ(wbit=4, num_features=out_channels, channel_wise=0)
        self.input_quant = AQ(abit=4, num_features=out_channels, alpha_init=10.0)

    def forward(self, input):
        # Quantization
        weight_q = self.weight_quant(self.weight)
        input_q = self.input_quant(input)
        if self.cg_groups > 1:
            # partial sum of the base path
            self.Yp = F.conv2d(input_q, weight_q*self.mask_base, None, self.stride, self.padding, self.dilation, self.groups)
            
            # block gating
            Yp_ = nn.functional.avg_pool3d(self.Yp, kernel_size=(self.slice_size,1,1), stride=(self.slice_size, 1, 1))
            Yp_ = Yp_.repeat_interleave(self.slice_size, dim=1)

            if self.hard_sig:
                k = 0.25
                pre_d = self.cg_alpha*(self.cg_bn(Yp_)-self.cg_threshold)
                self.d = self.cg_gt(torch.nn.functional.hardtanh(k*(pre_d+2), min_val=0., max_val=1.)-0.5)
            else:
                self.d = self.cg_gt(torch.sigmoid(self.cg_alpha*(self.cg_bn(Yp_)-self.cg_threshold))-0.5)

            # report statistics
            self.num_out = self.d.numel()
            self.num_full = self.d[self.d>0].numel()

            # group sparsity
            self.cond_group = nn.functional.avg_pool3d(self.d.data, kernel_size=(self.slice_size,1,1), stride=(self.slice_size, 1, 1))
            self.nonzero_g = self.cond_group[self.cond_group>0].numel()
            self.total_g = self.cond_group.numel()

            self.Yc = F.conv2d(input_q, weight_q * self.mask_cond, None, self.stride, self.padding, self.dilation, self.groups)
            out = self.Yp + self.Yc * self.d
        else:
            # import pdb; pdb.set_trace()
            out = F.conv2d(input_q, weight_q, None, self.stride, self.padding, self.dilation, self.groups)
        
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out

    def extra_repr(self):
        return super(Conv2d_CG, self).extra_repr() + ', slice_size={}, cg_groups={}, hard_sig={}'.format(self.slice_size, self.cg_groups, self.hard_sig)

class Linear_CG(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gamma=0.0, alpha=0.0, block_size=16):
        super(Linear_CG, self).__init__(in_features, out_features, bias)
        self.gamma = gamma
        self.alpha = alpha
        self.block_size = block_size
        self.count = -1 # disable in-module mask generation

        # quantization
        self.weight_quant = WQ(wbit=4, num_features=out_features, channel_wise=0)
        self.input_quant = AQ(abit=4, num_features=out_features, alpha_init=10.0)

    def forward(self, input):
        weight_q = self.weight_quant(self.weight)
        input_q = self.input_quant(input)

        out = F.linear(input_q, weight_q, self.bias)       
        return out 

    def extra_repr(self):
        return super(Linear_CG, self).extra_repr() + ', gamma={}, alpha={}, block_size={}'.format(
                self.gamma, self.alpha, self.block_size)
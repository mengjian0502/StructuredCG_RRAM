"""
Group Lasso Pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cg import Conv2d_CG, QConv2d_CG

def glasso_global_mp(var, dim=0, thre=0.0):
    if len(var.size()) == 4:
        var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))

    a = var.pow(2).sum(dim=dim).pow(1/2)
    b = var.abs().mean(dim=1)

    penalty_groups = a[b<thre]
    return penalty_groups.sum(), penalty_groups.numel()

def get_weight_sparsity(model, args):
    all_group = 0
    all_nonsparse = 0
    count = 0
    all_num = 0.0
    count_num_one = 0.0
    for m in model.modules():
        if isinstance(m, QConv2d_CG):
            if not count in [0] and not m.weight.size(2)==1:
                w_mean = m.weight.mean()
                w_l = m.weight

                with torch.no_grad():
                    w_l = m.weight_quant(w_l)
                
                kw = m.weight.size(2)
                count_num_layer = w_l.size(0) * w_l.size(1) * kw * kw
                all_num += count_num_layer
                count_one_layer = len(torch.nonzero(w_l.view(-1)))
                count_num_one += count_one_layer

                w_l = w_l.permute(0,2,3,1)
                num_group = w_l.size(0)*w_l.size(1)*w_l.size(2)
                w_l = w_l.contiguous().view(num_group, w_l.size(3))

                grp_values = w_l.norm(p=2, dim=1)
                non_zero_idx = torch.nonzero(grp_values) 
                num_nonzeros = len(non_zero_idx)

                all_group += num_group
                all_nonsparse += num_nonzeros

            count += 1
    overall_sparsity = 1 - count_num_one / all_num
    group_sparsity = 1 - all_nonsparse / all_group
    sparse_group = all_group - all_nonsparse
    return group_sparsity, overall_sparsity, sparse_group
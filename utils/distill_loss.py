import numpy as np
import torch

def kd_loss(feat_t, feat_s, module_list, criterion_kd, args):
    # other kd beyond KL divergence
    if args.distill == 'kd':
        loss_kd = 0
    elif args.distill == 'hint':
        f_s = module_list[1](feat_s[args.hint_layer])
        f_t = feat_t[args.hint_layer]
        loss_kd = criterion_kd(f_s, f_t)
    # elif args.distill == 'crd':
    #     f_s = feat_s[-1]
    #     f_t = feat_t[-1]
    #     loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
    elif args.distill == 'attention':
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    elif args.distill == 'nst':
        # recommend small batch size
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    elif args.distill == 'similarity':
        g_s = [feat_s[-2]]
        g_t = [feat_t[-2]]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    elif args.distill == 'rkd':
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t)
    elif args.distill == 'pkt':
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t)
    # elif args.distill == 'kdsvd':
    #     g_s = feat_s[1:-1]
    #     g_t = feat_t[1:-1]
    #     loss_group = criterion_kd(g_s, g_t)
    #     loss_kd = sum(loss_group)
    elif args.distill == 'correlation':
        f_s = module_list[1](feat_s[-1])
        f_t = module_list[2](feat_t[-1])
        loss_kd = criterion_kd(f_s, f_t)
    elif args.distill == 'vid':
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
        loss_kd = sum(loss_group)
    elif args.distill == 'abound':
        # can also add loss to this stage
        loss_kd = 0
    # elif args.distill == 'fsp':
    #     # can also add loss to this stage
    #     loss_kd = 0
    elif args.distill == 'factor':
        factor_s = module_list[1](feat_s[-2])
        factor_t = module_list[2](feat_t[-2], is_factor=True)
        loss_kd = criterion_kd(factor_s, factor_t)
    else:
        raise NotImplementedError(args.distill)
    return loss_kd
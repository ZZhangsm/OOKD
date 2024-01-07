# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F

from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM
from utils.distill_loss import kd_loss
from utils.util import AverageMeter


class Mixup(ERM):
    """Domain Mixup"""
    def __init__(self, args):
        super(Mixup, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):
        objective = 0

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches):
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)

            x = (lam * xi + (1 - lam) * xj).cuda().float()

            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi.cuda().long())
            objective += (1 - lam) * \
                         F.cross_entropy(predictions, yj.cuda().long())

        objective /= len(minibatches)

        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': objective.item()}

    def distill(self, minibatches, module_list, criterion_list, opt, args, sch):
        """One-step mixup distillation"""
        # set modules as train()
        for module in module_list:
            module.train()
        # set teacher as eval()
        module_list[-1].eval()

        if args.distill == 'abound':
            module_list[1].eval()
        elif args.distill == 'factor':
            module_list[2].eval()

        criterion_cls = criterion_list[0]
        criterion_div = criterion_list[1]
        criterion_kd = criterion_list[2]

        model_s = module_list[0]
        model_t = module_list[-1]

        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        loss = 0
        loss_cls, loss_div, loss_kd = 0, 0, 0
        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(args, minibatches):
            lam = np.random.beta(args.mixupalpha, args.mixupalpha)

            input = (lam * xi + (1 - lam) * xj).cuda().float()

            # ===================forward=====================
            preact = False
            if args.distill in ['abound']:
                preact = True
            feat_s, logit_s = model_s.featurizer(input, is_feat=True, preact=preact)
            logit_s = model_s.classifier(logit_s)
            with torch.no_grad():
                feat_t, logit_t = model_t.featurizer(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]
                logit_t = model_t.classifier(logit_t)

            loss_cls += lam * criterion_cls(logit_s, yi.cuda().long())
            loss_cls += (1 - lam) * criterion_cls(logit_s, yj.cuda().long())
            loss_div += criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            loss_kd += lam * kd_loss(feat_t, feat_s, yi.cuda(), module_list, criterion_kd, args)
            loss_kd += (1 - lam) * kd_loss(feat_t, feat_s, yj.cuda(), module_list, criterion_kd, args)

            loss += args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd

        # ===================backward=====================
        loss /= len(minibatches)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        # ===================meters========================
        # acc1, acc3 = accuracy(logit_s, target, topk=(1, 3))
        # top1.update(acc1[0], input.size(0))
        # top3.update(acc3[0], input.size(0))
        losses.update(loss.item(), input.size(0))

        return losses.avg

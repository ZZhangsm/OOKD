# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from utils.distill_loss import kd_loss
from network import common_network
from alg.algs.base import Algorithm
from utils.util import AverageMeter


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)

        self.network = nn.Sequential(
            self.featurizer, self.classifier)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        return self.network(x)

    def distill(self, minibatches, module_list, criterion_list, opt, args, sch):
        """One-step distillation"""
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
        # top1 = AverageMeter()
        # top3 = AverageMeter()


        if args.distill in ['crd']:
            input = torch.cat([data[0].cuda().float() for data in minibatches])
            target = torch.cat([data[1].cuda().long() for data in minibatches])
            index = torch.cat([data[3].cuda().long() for data in minibatches])
            contrast_idx = torch.cat([data[4].cuda().long() for data in minibatches])
        else:
            input = torch.cat([data[0].cuda().float() for data in minibatches])
            target = torch.cat([data[1].cuda().long() for data in minibatches])

        # Gaussian noise
        if args.noise:
            stdv = torch.rand(1).item()
            input = (input + torch.randn_like(input) * stdv).clamp(0., 255.)
            # input = torch.clamp(input, 0., 255.)

            # import matplotlib.pyplot as plt
            # plt.imshow(input[0].permute(1, 2, 0).cpu().numpy())
            # plt.show()


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

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if args.distill in ['crd']:
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, target, index, contrast_idx)
            loss = args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd
        elif args.distill in ['DKD']:
            loss_kd = kd_loss(logit_t, logit_s, target, module_list, criterion_kd, args)
            loss = loss_kd
        elif args.distill in ['DiffKD']:
            student_feat_refined, ddim_loss, teacher_feat_hidden, rec_loss\
                = kd_loss(feat_t, feat_s, target, module_list, criterion_kd, args)
            loss_kd = F.mse_loss(student_feat_refined, teacher_feat_hidden)
            loss = loss_kd + args.alpha * rec_loss + args.beta * ddim_loss
        else:
            loss_kd = kd_loss(feat_t, feat_s, target, module_list, criterion_kd, args)
            loss = args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd

        # ===================backward=====================
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        # ===================meters=======================
        # acc1, acc3 = accuracy(logit_s, target, topk=(1, 3))
        # top1.update(acc1[0], input.size(0))
        # top3.update(acc3[0], input.size(0))
        losses.update(loss.item(), input.size(0))
        # return {'class': loss.item()}
        # return top1.avg, losses.avg
        return losses.avg

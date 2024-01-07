# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F

from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM
from utils.distill_loss import kd_loss
from utils.util import AverageMeter


class DomainMix(ERM):
    """
    Wang, W., Liao, S., Zhao, F., Kang, C., & Shao, L. (2020).
    Domainmix: Learning generalizable person re-identification without human annotations.
    arXiv preprint arXiv:2011.11953.
    https://arxiv.org/abs/2011.11953
    """
    def __init__(self, args):
        super(DomainMix, self).__init__(args)
        self.args = args

        self.mix_type = "crossdomain"
        self.alpha = 1.0
        self.beta = 1.0
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)

    def parse_batch_train(self, minibatches):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_d = torch.cat([data[2].cuda().long() for data in minibatches])
        x, y_a, y_b, lam = self.domain_mix(all_x, all_y, all_d)

        return x, y_a, y_b, lam

    def domain_mix(self, x, y, d):
        lam = (
            self.dist_beta.rsample((1, ))
            if self.alpha > 0 else torch.tensor(1)
        ).cuda()

        # random shuffle
        perm = torch.randperm(x.size(0), dtype=torch.int64).cuda()
        if self.mix_type == "crossdomain":
            domain_list = torch.unique(d)
            if len(domain_list) > 1:
                for idx in domain_list:
                    cnt_a = torch.sum(d == idx)
                    idx_b = (d != idx).nonzero().squeeze(-1)
                    cnt_b = idx_b.shape[0]
                    perm_b = torch.ones(cnt_b).multinomial(
                        num_samples=cnt_a, replacement=bool(cnt_a > cnt_b)
                    )
                    perm[d == idx] = idx_b[perm_b]
        elif self.mix_type != "random":
            raise NotImplementedError(
                f"Chooses {'random', 'crossdomain'}, but got {self.mix_type}."
            )
        mixed_x = lam*x + (1-lam) * x[perm, :]
        y_a, y_b = y, y[perm]

        # visualize mixed_x and x
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.title('mixed_x')
        # plt.imshow(mixed_x[0].cpu().numpy().transpose(1, 2, 0))
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.title('x')
        # plt.imshow(x[0].cpu().numpy().transpose(1, 2, 0))
        # plt.axis('off')
        # plt.show()
        return mixed_x, y_a, y_b, lam


    def update(self, minibatches, opt, sch):
        x, y_a, y_b, lam = self.parse_batch_train(minibatches)
        output = self.predict(x)
        loss = lam * F.cross_entropy(output, y_a) + \
               (1 - lam) * F.cross_entropy(output, y_b)

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


        x, y_a, y_b, lam = self.parse_batch_train(minibatches)

        # ===================forward=====================
        preact = False
        if args.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s.featurizer(x, is_feat=True, preact=preact)
        logit_s = model_s.classifier(logit_s)
        with torch.no_grad():
            feat_t, logit_t = model_t.featurizer(x, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
            logit_t = model_t.classifier(logit_t)

        # cls + kl div
        output = self.predict(x)
        loss_cls = lam * criterion_cls(output, y_a) + \
               (1 - lam) * criterion_cls(output, y_b)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        loss_kd = lam * kd_loss(feat_t, feat_s, y_a, module_list, criterion_kd, args) \
            + (1 - lam) * kd_loss(feat_t, feat_s, y_b, module_list, criterion_kd, args)

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
        losses.update(loss.item(), x.size(0))
        # return {'class': loss.item()}
        # return top1.avg, losses.avg
        return losses.avg




from __future__ import print_function, division

import time
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils.util import AverageMeter


def init(model_s, model_t, init_modules, criterion, train_loaders, args):
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if args.s_net in ['resnet18', 'resnet50', 'resnet101'] and \
            args.distill == 'factor':
        lr = 0.01
    else:
        lr = args.lr
    optimizer = optim.SGD(init_modules.parameters(),
                          lr=lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, args.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        # for idx, data in enumerate(train_loader):
        #     if args.distill in ['crd_tools']:
        #         input, target, index, contrast_idx = data
        #     else:
        #         input, target, index = data

        train_minibatches_iterator = zip(*train_loaders)
        for iter_num in range(args.steps_per_epoch):
            minibatches = [(data) for data in next(train_minibatches_iterator)]
            input = torch.cat([data[0].cuda().float() for data in minibatches])
            target = torch.cat([data[1].cuda().long() for data in minibatches])
            data_time.update(time.time() - end)

            # input = input.float()
            # if torch.cuda.is_available():
            #     input = input.cuda()
            #     target = target.cuda()
            #     index = index.cuda()
            #     if args.distill in ['crd_tools']:
            #         contrast_idx = contrast_idx.cuda()

            # ============= forward ==============
            preact = (args.distill == 'abound')
            feat_s, _ =  model_s.featurizer(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t.featurizer(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if args.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif args.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif args.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplemented('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # end of epoch
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, args.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()

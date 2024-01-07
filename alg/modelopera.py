# coding=utf-8
import torch
import torch.nn.functional as F
from network import img_network


def get_fea(args):
    """
    Get feature extractor
    """
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    elif args.net.startswith('res'):
        if args.dataset == 'ColorMNIST':
            net = img_network.ResBase_cm(args.net)
        else:
            net = img_network.ResBase(args.net)
    elif args.net.startswith('Mobile'):
        net = img_network.MobileBase(args.net)
    elif args.net.startswith('wrn'):
        net = img_network.WRNBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
        # net = model.vgg
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total


def ece(network, loader):
    """
    Expected Calibration Error (ECE)
    :param network: model
    :param loader: data loader
    :return: ECE value
    """
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1).cuda()
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accs = []
    confs = []
    for data in loader:
        x = data[0].cuda().float()
        y = data[1].cuda().long()
        p = network.predict(x)

        if p.size(1) == 1:
            correct = (p.gt(0).eq(y).float()).sum().item()
        else:
            correct = (p.argmax(1).eq(y).float()).sum().item()

        # p with softmax
        confidence = F.softmax(p, dim=1).max(1)[0]
        accuracy = correct / len(x)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            mask = confidence.gt(bin_lower.item()) * confidence.le(bin_upper.item())
            if mask.any():
                accs.append(accuracy)
                confs.append(confidence[mask].mean().item())

    ece_val = 0
    for acc, conf in zip(accs, confs):
        ece_val += abs(acc - conf) * len(accs)
    return ece_val / len(accs)


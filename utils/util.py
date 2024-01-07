# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL

from alg import alg


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(filename, alg, args):
    save_dict = {
        "args": vars(args),
        # "model_dict": alg.cpu().state_dict()
        "model_dict": alg.state_dict()
    }
    torch.save(save_dict, os.path.join(args.output_dir, filename))

def load_checkpoint(path, alg):
    checkpoint_dict = torch.load(path)
    args = checkpoint_dict["args"]
    # model_dict = checkpoint_dict["model_dict"]
    alg.load_state_dict(checkpoint_dict["model_dict"])
    return args, alg


def train_valid_target_eval_names(args):
    # eval_name_dict = {'train': [], 'valid': [], 'target': []}
    eval_name_dict = {'valid': [], 'target': []}
    t = 0
    # for i in range(args.domain_num):
    #     if i not in args.test_envs:
    #         eval_name_dict['train'].append(t)
    #         t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'Mixup': ['class'],
                 'MLDG': ['total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty'],
                 'DIFEX': ['class', 'dist', 'exp', 'align', 'total'],
                 'CrossGrad': ['loss_f', 'loss_d'],
                 'DomainMix': ['class'],
                 'CutMix': ['class'],
                 }
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("==========================================\n")
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'DomainNet':
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    elif dataset == 'CelebA':
        domains = ['tr_env1', 'tr_env2', 'te_env']
    elif dataset == 'ColorMNIST':
        domains = ['+90%', '+80%', '-90%']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'Real_World'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'DomainNet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        'CelebA': ['tr_env1', 'tr_env2', 'te_env'],
        'ColorMNIST': ['+90%', '+80%', '-90%']
    }
    if dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    elif dataset == 'ColorMNIST':
        args.input_shape = (2, 28, 28,)
        args.num_classes = 2
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == 'office-home':
            args.num_classes = 65
        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'PACS':
            args.num_classes = 7
        elif args.dataset == 'VLCS':
            args.num_classes = 5
        elif args.dataset == 'DomainNet':
            args.num_classes = 345
        elif args.dataset == 'CelebA':
            args.num_classes = 2
    return args

def model_param(model):
    return sum(x.numel() for x in model.parameters())



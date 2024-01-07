import argparse
import os, sys, time
import numpy as np
import torch
import wandb
from torch import nn

from alg import modelopera, alg
from alg.opt import get_optimizerv2, get_scheduler
from datautil.getdataloader import get_img_dataloader
from utils.pretrain import init
from model.util import ConvReg, LinearEmbed, Connector, Paraphraser, Translator
from utils.util import Tee, img_param_init, print_environ, set_random_seed,\
    train_valid_target_eval_names, load_checkpoint, print_args, model_param, save_checkpoint

from distiller_zoo import DistillKL, HintLoss, Attention, NSTLoss, Similarity, ABLoss, FactorTransfer
from distiller_zoo import RKDLoss, PKT, Correlation, VIDLoss
from distiller_zoo.crd_tools.criterion import CRDLoss

def get_args():
    parser = argparse.ArgumentParser(description='Distillation')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint every N epoch')

    parser.add_argument('--root_dir', type=str, default='/home/my/path', help='root_dir')
    parser.add_argument('--dataset', type=str, default='PACS',
                        choices=['PACS', 'office-home', 'CelebA', 'dg5', 'ColorMNIST', 'DomainNet'])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0], help='target domains')
    parser.add_argument('--output_dir', type=str, default="test/", help='result output path')
    parser.add_argument('--t_path', type=str, default='/home/my/path/best_model.pkl',
                        help="the path to the pretrained teacher model")
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')

    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--algorithm', type=str, default="ERM")

    parser.add_argument('--net', type=str, default='resnet18', help="featurizer")
    parser.add_argument('--t_net', type=str, default='resnet50',
                        help="teacher featurizer: resnet18, resnet50, resnet101")
    parser.add_argument('--s_net', type=str, default='resnet18',
                        help="student featurizer: resnet18, resnet50, resnet101, Mobilenetv3_small,"
                             "Mobilenetv3_large, vgg11, wrn_16_2")
    parser.add_argument('--classifier', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--aug_policy', type=str, default="default",
                        choices=["default", "standard", "autoaugment", "randaugment"])

    parser.add_argument('--optim', type=str, default='sgd', help="optimizer: sgd, adam")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float, default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float, default=0.0003, help='for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--mixupalpha', type=float, default=0.2, help='mixup hyper-param')

    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")

    # distillation
    parser.add_argument('--distill', type=str, default='kd',
                        choices=['kd', 'hint', 'attention', 'similarity', 'correlation', 'vid', 'crd',
                                 'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    parser.add_argument('-r', '--gamma', type=float, default=0.6, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.4, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.4, help='weight balance for other losses')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=4096, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # prune
    parser.add_argument('--prune', type=str, default='none', choices=['none', 'random', 'el2n', 'grand'])
    parser.add_argument('--prune_ratio', type=float, default=0.8)
    parser.add_argument('--noise', type=int, default=0)

    args = parser.parse_args()
    args.steps_per_epoch = 100


    args.data_dir = os.path.join(args.root_dir, 'data', args.dataset) + '/'
    args.output_dir = os.path.join(args.root_dir, 'train_output', args.distill,
                                   '{}_{}_{}_{}'.format(args.dataset, args.test_envs[0], args.seed, args.output_dir))
    args = args_add(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output_dir, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args

def is_student(args, student=True):
    if student == True:
        args.net = args.s_net
        return args
    else:
        args.net = args.t_net
        return args


if __name__ == "__main__":
    # Get command-line arguments
    args = get_args()

    # Set random seed
    set_random_seed(args.seed)

    # Get data loaders
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)

    # Load model
    algorithm_class = alg.get_algorithm_class(args.algorithm)  # Get algorithm class
    model_t = algorithm_class(is_student(args, student=False)).cuda()  # Instantiate and move to GPU
    args_t, model_t = load_checkpoint(args.path, model_t)
    model_s = algorithm_class(is_student(args, student=True)).cuda()

    # Print out hyperparameters used
    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    print('teacher param:{}, student param:{}'
          .format(model_param(model_t), model_param(model_s)))

    s = ''
    acc_type_list = ['valid', 'target']
    acc_record, ece_record = {}, {}
    if args.dataset == 'ColorMNIST':
        data = torch.randn(64, 2, 28, 28).cuda()
    else:
        data = torch.randn(2, 3, 224, 224).cuda()
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t.featurizer(data, is_feat=True)
    feat_s, _ = model_s.featurizer(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    if args.distill == 'kd':
        criterion_kd = DistillKL(args.kd_T)
    elif args.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[args.hint_layer].shape, feat_t[args.hint_layer].shape).cuda()
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif args.distill == 'crd':
        args.s_dim = feat_s[-1].shape[1]
        args.t_dim = feat_t[-1].shape[1]
        # args.n_data = n_data
        args.n_data = np.sum(args.n_data)
        criterion_kd = CRDLoss(args).cuda()
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif args.distill == 'attention':
        criterion_kd = Attention()
    elif args.distill == 'nst':
        criterion_kd = NSTLoss()
    elif args.distill == 'similarity':
        criterion_kd = Similarity()
    elif args.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif args.distill == 'pkt':
        criterion_kd = PKT()
    # elif args.distill == 'kdsvd':
    #     criterion_kd = KDSVD()
    elif args.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], args.feat_dim).cuda()
        embed_t = LinearEmbed(feat_t[-1].shape[1], args.feat_dim).cuda()
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif args.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t).cuda() for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif args.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes).cuda()
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.featurizer.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loaders, args)
        # classification
        module_list.append(connector)
    elif args.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape).cuda()
        translator = Translator(s_shape, t_shape).cuda()
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loaders, args)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    # elif args.distill == 'fsp':
    #     s_shapes = [s.shape for s in feat_s[:-1]]
    #     t_shapes = [t.shape for t in feat_t[:-1]]
    #     criterion_kd = FSP(s_shapes, t_shapes)
    #     # init stage training
    #     init_trainable_list = nn.ModuleList([])
    #     init_trainable_list.append(model_s.get_feat_modules())
    #     init(model_s, model_t, init_trainable_list, criterion_kd, train_loaders, args)
    #     # classification training
    #     pass
    else:
        raise NotImplementedError(args.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    opt = get_optimizerv2(trainable_list, args)
    sch = get_scheduler(opt, args)  # Get scheduler for optimizer

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    best_valid_acc, target_acc = 0, 0
    train_minibatches_iterator = zip(*train_loaders)

    print('===========start distilling===========')
    start_t = time.time()
    for epoch in range(args.max_epoch):
        print('===========epoch %d===========' % (epoch))
        for iter_num in range(args.steps_per_epoch):
            minibatches = [(data) for data in next(train_minibatches_iterator)]

            train_loss = model_s.distill(minibatches, module_list, criterion_list, opt, args, sch)
            # wandb.log({'train_loss': train_loss})

            # Show progress bar
            if (iter_num + 1) % 10 == 0:
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f'
                                 % (epoch + 1, args.max_epoch, iter_num + 1, args.steps_per_epoch, train_loss))

        if (epoch in [int(args.max_epoch * 0.7), int(args.max_epoch * 0.9)]) and (not args.schuse):
            print('\nmanually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr'] * 0.1

        # If at last epoch or checkpoint frequency, print out loss and accuracy information
        if (epoch == (args.max_epoch - 1)) or (epoch % 10 == 0) \
                or (epoch >= args.max_epoch * 0.5 and epoch % args.checkpoint_freq == 0):
            print('\nTrain Loss ({loss:.4f})'.format(loss=train_loss))
            sys.stdout.flush()

            s = ''
            for item in acc_type_list:
                acc_record[item] = np.mean(np.array([modelopera.accuracy(
                    model_s, eval_loaders[i]) for i in eval_name_dict[item]]))
                s += (item + '_acc:%.4f,' % acc_record[item])
            # Print the accuracy values string
            print(s[:-1])

        print('\ntotal cost time: %.4f' % (time.time() - start_t))

        if acc_record['valid'] > best_valid_acc:
            best_valid_acc = acc_record['valid']
            target_acc = acc_record['target']
            save_checkpoint('best_model.pkl', model_s, args)
            print('saving the best models!')

        # if args.save_model_every_checkpoint:
        #     print('==> Saving...')
        #     save_checkpoint(f'model_epoch{epoch}.pkl', model_s, args)

        # early stop
        if epoch > 0.7 * args.max_epoch and acc_record['valid'] < best_valid_acc:
            print('early stop')
            break

    # Save the final mode
    print('saving the final models!')
    save_checkpoint('final_model.pkl', model_s, args)

    # Print the final valid and target accuracy values
    print('===========Distilling report===========')
    print('student valid acc: %.4f' % best_valid_acc)
    print('DG Distill result: %.4f' % target_acc)

    # Write the 'done' flag and the total cost time and accuracy values to a file
    with open(os.path.join(args.output_dir, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('teacher model:%s\n' % (args.t_net))
        f.write('student model:%s\n' % (args.s_net))
        f.write('distill method:%s\n' % (args.distill))
        f.write('temperature:%.4f\n' % (args.kd_T))
        f.write('gamma, alpha, beta:%.4f, %.2f, %.2f\n' % (args.gamma, args.alpha, args.beta))
        f.write('total cost time:%s\n' % (str(time.time() - start_t)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))

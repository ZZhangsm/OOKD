# coding=utf-8
import os
import sys
import time
import numpy as np
import argparse

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, \
    img_param_init, print_environ
from datautil.getdataloader import get_img_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alpha', type=float, default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int, default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,  default=16, help='batch_size')
    parser.add_argument('--beta', type=float, default=1, help='DIFEX beta')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--root_dir', type=str, default='/home/my/path', help='root_dir')
    parser.add_argument('--dataset', type=str, default='PACS',
                        choices=['PACS', 'office-home', 'CelebA', 'dg5', 'ColorMNIST', 'DomainNet'])

    parser.add_argument('--dis_hidden', type=int, default=256, help='dis hidden dimension')
    parser.add_argument('--disttype', type=str, default='2-norm',
                        choices=['1-norm', '2-norm', 'cos', 'norm-2-norm', 'norm-1-norm'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float, default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float, default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float, default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--layer', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--optim', type=str, default='sgd', help="optimizer: sgd, adam")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float, default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int, default=3, help="max iterations")
    parser.add_argument('--mixupalpha', type=float, default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float, default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float, default=1, help='MMD, CORAL hyper-param')

    parser.add_argument('--momentum', type=float, default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float, default=1 / 3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float, default=1 / 3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--split_style', type=str, default='strat', help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg", choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--aug_policy', type=str, default="default",
                        choices=["default", "standard", "autoaugment", "randaugment"])
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0], help='target domains')
    parser.add_argument('--output_dir', type=str, default="test/", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--distill', type=str, default='none')
    parser.add_argument('--prune', type=str, default='none', choices=['none', 'random', 'el2n', 'grand'])
    parser.add_argument('--prune_ratio', type=float, default=0.75)
    parser.add_argument('--noise', type=int, default=0)

    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = os.path.join(args.root_dir, 'data', args.dataset) + '/'
    args.output_dir = os.path.join(args.root_dir, 'teacher_output',
                                   '{}_{}_{}_{}'.format(args.dataset, args.test_envs[0], args.seed, args.output_dir))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(torch.cuda.device_count(), torch.cuda.current_device())
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output_dir, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args

if __name__ == '__main__':
    # Get command-line arguments
    args = get_args()
    # Set random seed
    set_random_seed(args.seed)

    # Get dictionary of algorithm-specific loss functions
    loss_list = alg_loss_dict(args)
    # Get training and evaluation dataloaders
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)  # Get dictionary of dataset names
    algorithm_class = alg.get_algorithm_class(args.algorithm)  # Get algorithm class
    algorithm = algorithm_class(args).cuda()  # Instantiate algorithm object and move to GPU
    algorithm.train()  # Put algorithm in train mode
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)  # Get scheduler for optimizer

    # Print out hyperparameters used
    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    print("Total number of model param in algorithm is ",
          sum(x.numel() for x in algorithm.parameters()))

    # If using DIFEX algorithm, train teacher network on Fourier-transformed images
    if 'DIFEX' in args.algorithm:
        ms = time.time()
        n_steps = args.max_epoch * args.steps_per_epoch
        print('start training fft teacher net')
        opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
        sch1 = get_scheduler(opt1, args)
        algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
        print('complet time:%.4f' % (time.time() - ms))

    # Initialize empty dictionary to record
    acc_record, ece_record = {}, {}
    # List of accuracy types to record
    # acc_type_list = ['train', 'valid', 'target']
    acc_type_list = ['valid', 'target']
    # Create iterator over training minibatches
    train_minibatches_iterator = zip(*train_loaders)
    # Initialize the best to 0
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    # wandb.watch(algorithm)
    start_t = time.time()
    for epoch in range(args.max_epoch):
        print('\n ===========epoch %d===========' % (epoch))
        for iter_num in range(args.steps_per_epoch):
            # Get minibatches for current step and move to GPU

            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            # If using VREx and reached anneal_iters, get new optimizer and scheduler
            if args.algorithm == 'VREx' and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            # Update algorithm with minibatches
            step_vals = algorithm.update(minibatches_device, opt, sch)

            # Show progress bar
            if (iter_num + 1) % 10 == 0:
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t'
                                 % (epoch + 1, args.max_epoch, iter_num + 1,
                                    args.steps_per_epoch))

        # If currently at 70-90% of max epochs and not using a scheduler, manually decrease learning rate
        if (epoch in [int(args.max_epoch * 0.7), int(args.max_epoch * 0.9)]) and (not args.schuse):
            print('\nmanually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr'] * 0.1

        # If at last epoch or checkpoint frequency, print out loss and accuracy information
        # if (epoch == (args.max_epoch - 1)) or (epoch % args.checkpoint_freq == 0)\
        #         or (epoch >= args.max_epoch * 0.5) or (epoch % 10 == 0):
        if (epoch == (args.max_epoch - 1))  or (epoch % 10 == 0)\
                or (epoch >= args.max_epoch * 0.5 and epoch % args.checkpoint_freq == 0):

            # Generate a string that lists the loss values
            s = ''
            for item in loss_list:
                s += (item + '_loss:%.4f,' % step_vals[item])
            print(s[:-1])

            # Calculate the accuracy for each type of data (train, valid, target)
            # and generate a string that lists the accuracy values
            s = ''
            for item in acc_type_list:
                acc_record[item] = np.mean(np.array([modelopera.accuracy(
                    algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                s += (item + '_acc:%.4f,' % acc_record[item])
            print(s[:-1])

            # If the current valid accuracy is greater than the best
            # update the best valid accuracy and target accuracy
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
                save_checkpoint('best_model.pkl', algorithm, args)
                print('saving the best models!')

            # If the save_model_every_checkpoint argument is set to True
            # save the model to a checkpoint file
            if args.save_model_every_checkpoint:
                print('==> Saving...')
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time() - start_t))
            algorithm_dict = algorithm.state_dict()

            # early stop
            if epoch > 0.7 * args.max_epoch and acc_record['valid'] < best_valid_acc:
                print('early stop')
                break

    # Save the final model
    # print('saving the final models!')
    # save_checkpoint('final_model.pkl', algorithm, args)

    s = ''
    for item in acc_type_list:
        ece_record[item] = np.mean(np.array([modelopera.ece(
            algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
        s += (item + '_ece:%.4f,' % ece_record[item])
    print(s[:-1])

    # Print the final results
    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    # Write info to a file
    with open(os.path.join(args.output_dir, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time() - start_t)))
        f.write('expected calibration error:\n' + s[:-1]+'\n')
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))

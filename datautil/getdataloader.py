# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from alg.algs.ERM import ERM
from datautil.imgdata.imgdataload import ImageDataset, ImageDatasetSample
from datautil.mydataloader import InfiniteDataLoader
from utils.prune import compute_el2n_score, compute_grand_score


def get_img_dataloader(args):
    rate = 0.2  # Set the split rate
    # Initialize empty lists to hold train and test datasets
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]  # Get the names of the image datasets to be used
    args.domain_num = len(names)  # Set the number of image domains
    args.n_data = [0] * args.domain_num  # Initialize an array to hold the number of samples in each domain
    # Loop through each image domain
    for i in range(len(names)):
        # If the current domain is a test environment
        if i in args.test_envs:
            # Append a new ImageDataset object for the test data to tedatalist
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
            # tedatalist.append(PACS(args.dataset, args.task, args.data_dir,
            #                        names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs)

            # Set the number of samples in the current domain
            # args.n_data[i] = len(tedatalist[-1])
        else:
            # Get the labels for the current domain's dataset
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(args.dataset, policy=args.aug_policy),
                                    test_envs=args.test_envs).labels
            l = len(tmpdatay)  # Get the length of the labels array
            # Get the indices for the train-test split
            if args.split_style == 'strat':
                # If using stratified sampling
                lslist = np.arange(l) # Generate a list of indices for the data
                # Create a stratified shuffle split object
                stsplit = ms.StratifiedShuffleSplit(2, test_size=rate, train_size=1-rate, random_state=args.seed)
                # Get the number of splits
                stsplit.get_n_splits(lslist, tmpdatay)
                # Get the indices for train and test data
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                # If not using stratified sampling
                indexall = np.arange(l)  # Generate a list of indices for the data
                np.random.seed(args.seed)  # Set the random seed
                np.random.shuffle(indexall)  # Shuffle the list of indices
                ted = int(l*rate) # Get the number of samples for the test set
                # Get the indices for train and test data
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            # Prune the training data if specified
            if args.prune == "random":
                # If using random pruning
                trn = int(len(indextr) * args.prune_ratio)
                np.random.shuffle(indextr)
                indextr = indextr[:trn]
            elif args.prune == "el2n":
                # If using the el2n pruning method
                trn = int(len(indextr) * args.prune_ratio)
                # Build the evaluation model
                # prune_model = resnet18(num_classes=args.num_classes).cuda()
                prune_model = ERM(args).network.cuda()
                # dataloader for the current domain using pytorch
                train_loader = ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(args.dataset),
                                           indices=indextr, test_envs=args.test_envs).dataloader()
                el2n_set = []
                prune_model.eval()
                for (data, target, domain) in train_loader:
                    logit_s = prune_model(data.cuda().float())
                    el2n = compute_el2n_score(logit_s, target.cuda().long())
                    el2n_set.append(el2n.detach().cpu().numpy())
                el2n_sort_idx = np.argsort(np.concatenate(el2n_set))
                indextr = indextr[el2n_sort_idx[:trn]]
            elif args.prune == "grand":
                # If using the grand pruning method
                trn = int(len(indextr) * args.prune_ratio)
                # Build the evaluation model
                # prune_model = resnet18(num_classes=args.num_classes).cuda()
                prune_model = ERM(args).network.cuda()
                # dataloader for the current domain using pytorch
                train_loader = ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(args.dataset),
                                           indices=indextr, test_envs=args.test_envs).dataloader()
                grand_set = []
                prune_model.eval()
                for (data, target, domain) in train_loader:
                    grand = compute_grand_score(prune_model, data.cuda().float(), target.cuda().long())
                    grand_set.append(grand)
                grand_sort_idx = np.argsort(np.concatenate(grand_set))
                # grand_sort_idx = torch.argsort(torch.cat(grand_set)).cpu().numpy()
                indextr = indextr[grand_sort_idx[:trn]]
            elif args.prune == "none":
                # If not using pruning
                indextr = indextr
            else:
                # If an invalid pruning method is specified
                raise ValueError("Invalid pruning method specified.")

            args.n_data[i] = len(indextr)  # Set the number of samples in the current domain
            # Append new ImageDataset objects to train -> trdatalist and test -> tedatalist
            if args.distill in ['crd']:
                trdatalist.append(ImageDatasetSample(args.dataset, args.task, args.data_dir,
                                               names[i], i, transform=imgutil.image_train(args.dataset, policy=args.aug_policy),
                                               indices=indextr, test_envs=args.test_envs))
                tedatalist.append(ImageDatasetSample(args.dataset, args.task, args.data_dir,
                                               names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte,
                                               test_envs=args.test_envs))
            else:
                trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                               names[i], i, transform=imgutil.image_train(args.dataset, policy=args.aug_policy),
                                               indices=indextr, test_envs=args.test_envs))
                tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                               names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte,
                                               test_envs=args.test_envs))

            # trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_train(args.dataset, args.aug_policy),
            #                                indices=indextr, test_envs=args.test_envs))
            # tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte,
            #                                test_envs=args.test_envs))



    # Create InfiniteDataLoader objects for each training dataset
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    # Create DataLoader objects for each training and test dataset
    # eval_loaders = [DataLoader(
    #     dataset=env,
    #     batch_size=64,
    #     num_workers=args.N_WORKERS,
    #     drop_last=False,
    #     shuffle=False)
    #     for env in trdatalist+tedatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]

    # Return the training and test data loaders
    return train_loaders, eval_loaders

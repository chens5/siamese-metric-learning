import argparse
import json
import itertools
import numpy as np
import random
from dataset import *

parser = argparse.ArgumentParser(description='construct datasets')
parser.add_argument('--dataset-name', type=str)
parser.add_argument('--train-sz', type=int)
parser.add_argument('--val-sz', type=int)
parser.add_argument('--nmax', type=int)
parser.add_argument('--nmin', type=int)
parser.add_argument('--dim', type=int)
parser.add_argument('--modality-1', type=str)
parser.add_argument('--modality-2', type=str)

args = parser.parse_args()
print(args)


RANDOM_2D = ['synthetic-random', 
             'circle', 
             'ncircle',
             'ncircle/large', 
             'grid', 
             'ncircle/dim6', 
             'ncircle/dim10', 
             'ncircle/dim14']

SINGLE_CELL = ['rna-atac']

MNET = ['modelnet']

if args.dataset_name in RANDOM_2D:
    if args.dataset_name == 'grid':
        n = 10
    else:
        n = 10000
    if 'ncircle' in args.dataset_name:
        # if 'dim' in args.dataset_name:
        #     dim = args.dim
        # else:
        #     dim=2
        dim = args.dim
        Ps, Qs, dists = noisy_circles(nmin=args.nmin, 
                                      nmax=args.nmax, 
                                      pairs=args.train_sz,
                                      dim=dim)
        Ps_val, Qs_val, dists_val = noisy_circles(nmin=args.nmin, 
                                                  nmax=args.nmax, 
                                                  pairs=args.val_sz,
                                                  dim=dim)
    else:
        pointset = fixed_point_set(dim=2, num=n, data_type=args.dataset_name)
        Ps, Qs, dists = build_dataset(pointset, 
                                      nmin=args.nmin, 
                                      nmax=args.nmax, 
                                      pairs=args.train_sz)
        Ps_val, Qs_val, dists_val = build_dataset(pointset, 
                                                  nmin=args.nmin, 
                                                  nmax=args.nmax, 
                                                  pairs=args.val_sz)
elif args.dataset_name in MNET:
    # Loading raw ModelNet data
    raw_data, labels, label_dict = load_hdf5_data('/data/sam/modelnet/data/modelnet40_ply_hdf5_2048/ply_data_train1.h5')
    # build train dataset
    Ps, Qs, dists = build_comprehensive_sampler(raw_data, 
                                                label_dict, 
                                                nmin=args.nmin, 
                                                nmax=args.nmax, 
                                                pairs=args.train_sz + args.val_sz)
    Ps_val, Qs_val, dists_val = Ps[args.train_sz: args.train_sz + args.val_sz], Qs[args.train_sz: args.train_sz + args.val_sz], dists[args.train_sz: args.train_sz + args.val_sz]
    # raw_data, labels, label_dict = load_hdf5_data('/data/sam/modelnet/data/ply_data_test1.h5')
    # Ps_val, Qs_val, dists_val = build_comprehensive_sampler(raw_data, 
    #                                                         label_dict, 
    #                                                         nmin=args.nmin, 
    #                                                         nmax=args.nmax, 
    #                                                         pairs=args.val_sz,
    #                                                         scale=True)
elif args.dataset_name in SINGLE_CELL:
    M1 = np.load(args.modality_1)
    M2 = np.load(args.modality_2)

    x_min = min(np.min(M1[:, 0]), np.min(M2[:, 0]))
    x_max = max(np.max(M1[:, 0]), np.max(M2[:, 0]))
    y_min = min(np.min(M1[:, 1]), np.min(M2[:, 1]))
    y_max = max(np.max(M1[:, 1]), np.max(M2[:, 1]))
    
    xs = np.random.uniform(low=x_min, high=x_max, size=10000)
    ys = np.random.uniform(low=y_min, high=y_max, size=10000)
    pointset = np.vstack((xs, ys)).T

    Ps, Qs, dists = build_dataset(pointset, 
                                    nmin=256, 
                                    nmax=257, 
                                    pairs=args.train_sz)
    Ps_val, Qs_val, dists_val = build_dataset(pointset, 
                                                nmin=256, 
                                                nmax=257, 
                                                pairs=args.val_sz)

    # Ps, Qs, dists = build_single_cell_data(M1, M2, n=256, pairs=args.train_sz + args.val_sz)
    # Ps_val, Qs_val, dists_val = Ps[args.train_sz: args.train_sz + args.val_sz], Qs[args.train_sz: args.train_sz + args.val_sz], dists[args.train_sz: args.train_sz + args.val_sz]
    
train_sf = '/data/sam/{}/data/train-nmax-{}-nmin-{}-sz-{}'.format(args.dataset_name, 
                                                                  args.nmax, 
                                                                  args.nmin, 
                                                                  args.train_sz)
val_sf = '/data/sam/{}/data/val-nmax-{}-nmin-{}-sz-{}'.format(args.dataset_name, 
                                                              args.nmax, 
                                                              args.nmin, 
                                                              args.val_sz)

np.savez(train_sf, Ps=Ps[:args.train_sz], Qs=Qs[:args.train_sz], dists=dists[:args.train_sz])
np.savez(val_sf, Ps=Ps_val, Qs=Qs_val, dists=dists_val)




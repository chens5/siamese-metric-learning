import h5py
import scipy
import numpy as np
import ot
import time
import multiprocessing as mp
from tqdm import tqdm, trange
import torch
import cvxpy as cp
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import argparse
from dataset import *
from models import *
import os
import pickle

def generate_name(h_in, h_out, g_out, h_layers, g_layers, final_mlp, activation, item_idx = -1):
    if final_mlp == False:
        name = 'in{dim}-out{out}-layers{layers}-act-{act}'.format(dim=h_in, out=h_out, layers=h_layers, act=activation)
        return name
    
    name='max-in{dim}-hout{hout}-hlayers{hlayers}-gout{gout}-glayers{glayers}-act-{act}.pt'.format(dim=h_in, hout=h_out, hlayers=h_layers, gout=g_out, glayers=g_layers, act=activation)
    
    return name

def generate_path(dataset, h_in, h_out, g_out, h_layers, g_layers, final_mlp, activation, item_idx = -1):
    if item_idx > 0 and final_mlp == False:
        path = '/data/sam/{data}/models/siamese/{idx}/tr-in{dim}-out{out}-layers{layers}-act-{act}.pt'.format(data=dataset, idx=item_idx, dim=h_in, out=h_out, layers=h_layers, act=activation)
        return path
    if item_idx> 0:
        path='/data/sam/{data}/models/mlp/{idx}/max-in{dim}-hout{hout}-hlayers{hlayers}-gout{gout}-glayers{glayers}-act-{act}.pt'.format(data=dataset, dim=h_in, hout=h_out, hlayers=h_layers,gout=g_out,glayers=g_layers, act=activation, idx=item_idx)
        return path
    if final_mlp == False:
        path = '/data/sam/{data}/models/siamese/tr-in{dim}-out{out}-layers{layers}-act-{act}.pt'.format(data=dataset, dim=h_in, out=h_out, layers=h_layers, act=activation)
        return path
    path='/data/sam/{data}/models/mlp/max-in{dim}-hout{hout}-hlayers{hlayers}-gout{gout}-glayers{glayers}-act-{act}.pt'.format(data=dataset, dim=h_in, hout=h_out, hlayers=h_layers, gout=g_out, glayers=g_layers, act=activation)
    return path

# train a single model
def train_product_net(dataloader, h_in, h_out, g_out,final_h_layers=4, h_layers=10, g_layers = 10, final_mlp = False,
                      activation='relu', lr=0.0001, iterations=200, device='cuda:2'):
    
    model = PointNetLearner(h_in, h_out, g_out, final_h_layers=final_h_layers, num_h_layers=h_layers, 
                            num_g_layers=g_layers, final_mlp=final_mlp, activation=activation)
    model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for i in trange(iterations):
        optimizer.zero_grad()
        counter=0
        total_loss=0
        for data in dataloader:
            counter += 1
            Pblock = data[0].to(device)
            Qblock = data[1].to(device)
            Pidx = data[2].to(device)
            Qidx = data[3].to(device)
            dists = data[4].to(device)
            loss = model(Pblock, Qblock, Pidx, Qidx, dists)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().detach().numpy()
            del Pblock
            del Qblock
            del Pidx
            del Qidx
            del dists
        losses.append(total_loss)
        if len(losses) >=5 and abs(losses[-1]-losses[-2]) < 0.001:
            break
        
    return model, losses
    
def main():
    parser = argparse.ArgumentParser(description='Training options for prduct network')
    parser.add_argument('--finalmlp', action='store_true')
    parser.add_argument('--embedding-sizes', type=int, nargs="+")
    parser.add_argument('--siamese-layers', type=int, nargs="+")
    parser.add_argument('--mlp-width', type=int, nargs="+")
    parser.add_argument('--mlp-layers', type=int, nargs="+")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--max-pairs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--final-h-size', type=int, nargs="+")
    parser.add_argument('--single-object-sampling', type=int, default = -1)
    parser.add_argument('--multiple-object-sampler', type=int, nargs='+')
    
    args = parser.parse_args()
    print("Include Final Architecture?", args.finalmlp)
    
    # set up dataset
    raw_data, labels, label_dict = load_hdf5_data(args.datapath)
    #Ps, Qs, dists = build_dataset(raw_data, pairs=args.max_pairs)
    if args.single_object_sampling > 0:
        print("Single object sampling with item:", args.single_object_sampling)
        sf_name = '/data/sam/modelnet/data/train-datasets/item-{idx}-pairs-{pairs}.npz'.format(pairs=args.max_pairs, idx=args.single_object_sampling)
        if os.path.exists(sf_name):
            load_dataset = np.load(sf_name, allow_pickle=True)
            Ps = load_dataset['P']
            Ps = Ps.tolist()
            Qs = load_dataset['Q']
            Qs = Qs.tolist()
            dists = load_dataset['dists']
        else:
            Ps, Qs, dists = build_single_item_dataset(raw_data, item_idx = args.single_object_sampling, pairs=args.max_pairs, train=True)
            Ps = Ps.tolist()
            Qs = Qs.tolist()
    elif len(args.multiple_object_sampler) >0:
        print("Single object sampling with item:", args.single_object_sampling)
        sf_name = '/data/sam/modelnet/data/train-datasets/item-1040-item-2047-pairs-4000.npz'
        if os.path.exists(sf_name):
            load_dataset = np.load(sf_name, allow_pickle=True)
            Ps = load_dataset['P']
            Ps = Ps.tolist()
            Qs = load_dataset['Q']
            Qs = Qs.tolist()
            dists = load_dataset['dists']
        else:
            Ps, Qs, dists = build_multiple_item_dataset(raw_data, items=args.multiple_object_sampler, max_pcd=args.max_pairs, train=True)
            Ps = Ps.tolist()
            Qs = Qs.tolist()
    else:
        sf_name = '/data/sam/modelnet/data/train-datasets/pairs-{pairs}.npz'.format(pairs=args.max_pairs)
        if os.path.exists(sf_name):
            load_dataset = np.load(sf_name, allow_pickle=True)
            Ps = load_dataset['P']
            Ps = Ps.tolist()
            Qs = load_dataset['Q']
            Qs = Qs.tolist()
            dists = load_dataset['dists']
        else:
            Ps, Qs, dists = build_comprehensive_sampler(raw_data, label_dict, pairs=args.max_pairs, save_data=True)
        
    dataloader = PointNetDataloader(Ps, Qs, dists, batch_size=args.batch_size)
    dim = raw_data.shape[2]
    print("Finished setting up dataset")
    
    # set up hyperparameters
    hyperparameters = {}
    # (siamese width, siamese depth)
    print(args.finalmlp)
    for sz in args.embedding_sizes:
        for layers in args.siamese_layers:
            for fh in args.final_h_size:
                hyperparameters[(sz, layers, fh)] = []
    if args.finalmlp==True:
        for key in hyperparameters:
            for width in args.mlp_width:
                for layers in args.mlp_layers:
                    hyperparameters[key].append((width, layers))
    print(hyperparameters)
    print("Starting training......")
    for key in hyperparameters:
        if len(hyperparameters[key]) == 0:
            g_out = 1
            g_layers=1
            hyperparameters[key].append((1, 1))
        for tup in hyperparameters[key]:
            g_out = tup[0]
            g_layers = tup[1]
            h_in = dim
            h_out = key[0]
            h_layers=key[1]
            fh = key[2]
            print("h width:", h_out, "h layers", h_layers, "final_h_size", fh, "g width", g_out, "g layers", g_layers)
            model, losses = train_product_net(dataloader, h_in, h_out, g_out, final_h_layers= fh,
                                              h_layers=h_layers, g_layers = g_layers, 
                                              final_mlp = args.finalmlp, activation=args.activation, 
                                              lr=args.lr, iterations=200, device=args.device)

            # plot losses
            print(losses[-1])
            name=generate_name(h_in, h_out, g_out, h_layers, g_layers, args.finalmlp, args.activation)
            plt.plot(np.arange(len(losses)), losses, label=name)
            plt.legend()
            # save model
            PATH = generate_path(args.dataset_name, h_in, h_out, g_out, h_layers, g_layers, args.finalmlp, args.activation, item_idx = args.single_object_sampling)
            print("saving model to:", PATH)
            torch.save(model.state_dict(), PATH)
            del model
        
        
    if args.finalmlp:
        plot_name = 'mlp.png'
        plt.savefig(plot_name)
    else:
        plot_name='siamese.png'
        plt.savefig(plot_name)
    
    return 0

if __name__=='__main__':
    main()
    

import h5py
import scipy
import numpy as np
import ot
import time
import multiprocessing as mp
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import argparse
from dataset import *
from models import *
import os
import pickle
from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, vecs, dists):
        self.vecs = vecs
        self.dists = dists

    def __len__(self):
        return len(self.dists)

    def __getitem__(self, idx):
        return self.vecs[idx], self.dists[idx]

def transform_pcd_for_mlp(Ps, Qs, max_points=1000):
    num_pcds = len(Ps)
    dim = Ps[0].shape[1]
    dataset = []
    szs = []
    #for i in range(len(Ps)):
    #    szs.append(len(Ps[i]))
    #print(np.max(szs))
    
    for i in range(num_pcds):
        vec = np.zeros((2 * max_points, dim + 1))
        P = Ps[i]
        Q = Qs[i]
        vec[:len(P), : dim] = P
        vec[:len(P), dim] = 1

        vec[:len(Q), : dim] = Q
        vec[:len(Q), dim] = 1
        dataset.append(vec)

    return dataset

def train_mlp(dataloader, in_dim, out_dim, layers, activation='relu', lr=0.0001, device='cuda:2', iterations=200):
    model = MLP(in_dim, out_dim,layers, activation=activation)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in trange(iterations):
        optimizer.zero_grad()
        counter=0
        total_loss=0
        for data in dataloader:
            counter += 1
            vec = data[0]
            vec = vec.flatten(start_dim=1)
            vec = vec.to(device)
            dists = data[1].to(device)
            loss = model(vec, dists)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().detach().numpy()
            del vec
            del dists
        losses.append(total_loss)
        if len(losses) >=5 and abs(losses[-1]-losses[-2]) < 0.001:
            break
        
    return model, losses

def main():
    parser = argparse.ArgumentParser(description='Training options for mlp')
    parser.add_argument('--hidden-dim', type=int, nargs="+")
    parser.add_argument('--layers', type=int, nargs="+")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sf-name', type=str)
    
    args = parser.parse_args()

    raw_data = np.load(args.datapath, allow_pickle=True)
    Ps = raw_data['P']   
    Qs = raw_data['Q']
    dists = raw_data['dists']

    vectorized_data = transform_pcd_for_mlp(Ps, Qs, max_points=2048)

    dataset = PairDataset(torch.tensor(vectorized_data, dtype=torch.float32), dists)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dimension = Ps[0].shape[1]
    mlp_dim = (dimension + 1)*2048*2
    print("dimension of MLP", mlp_dim)
    
    for hidden_dim in args.hidden_dim:
        for layers in args.layers:
            model, losses = train_mlp(dataloader, mlp_dim, hidden_dim, layers, activation=args.activation, lr=args.lr, device=args.device)
            print(losses[-1])
            plt.plot(np.arange(len(losses)), losses)
            # save model
            model_specifier = 'hidden-dim-{hdim}-layers-{layers}'.format(hdim=hidden_dim, layers=layers)
            PATH = args.sf_name + model_specifier
            print("saving model to:", PATH)
            torch.save(model.state_dict(), PATH)
            del model
    plt.savefig('vanilla')

if __name__=='__main__':
    main()



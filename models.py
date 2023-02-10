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
import h5py


class MLP(nn.Module):
    def __init__(self, dim,  hidden_dim, layers, activation='relu'):
        super(MLP, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        layers = []
        first_layer = nn.Linear(dim, self.hidden_dim)
        if activation =='relu':
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Sigmoid())
        layers.append(first_layer)
        for i in range(self.layers):
            if i < self.layers - 1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dim, 1))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise Exception("Activation function not implemented")        
        self.network = nn.Sequential(*layers)

    def forward(self, input, dists):
        output = self.network(input)
        return torch.mean(torch.square((output-dists)/dists))

class PointNetLearner(nn.Module):
    def __init__(self, h_in, h_out, g_out, final_h_layers = 2, num_h_layers=10, num_g_layers=10, final_mlp = False, activation='relu', aggregation='sum'):
        super(PointNetLearner, self).__init__()
        # initial Siamese embedding part
        self.h_out = h_out
        self.g_out = g_out
        self.h = []
        self.h.append(nn.Linear(h_in, h_out))
        for i in range(num_h_layers - 1):
            self.h.append(nn.Linear(h_out, h_out))
        self.h = nn.ModuleList(self.h)
        
        # final MLP
        self.gamma = []
        self.gamma.append(nn.Linear(h_out, g_out))
        for i in range(num_g_layers - 2):
            self.gamma.append(nn.Linear(g_out, g_out))
        self.gamma.append(nn.Linear(g_out, 1))
        self.gamma = nn.ModuleList(self.gamma)

        self.finalMLP = final_mlp
        if activation =='relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise Exception("Activation function not implemented")
        
        layers = []
        for i in range(final_h_layers):
            layers.append(nn.Linear(h_out, h_out))
            layers.append(nn.ReLU())
        self.final_h = nn.Sequential(*layers)
        self.aggregation = aggregation

    
    def get_embedding(self, P):
        out = P
        for layer in self.h:
            out = self.activation(layer(out))
        out = self.final_h(torch.sum(out, axis=0))
        return out 
    
    def get_full_result(self, P, Q):
        P_embd = P
        Q_embd = Q
        for layer in self.h:
            P_embd = self.activation(layer(P_embd))
            Q_embd = self.activation(layer(Q_embd))
        P_embd = torch.sum(P_embd, axis=0)
        Q_embd = torch.sum(Q_embd, axis=0)
        result = self.final_h(P_embd) + self.final_h(Q_embd)
        for layer in self.gamma:
            result = self.activation(layer(result))
        return result
    
    def reset_parameters(self):
        for layer in self.h:
            torch.nn.init.normal_(layer.weight)
        for layer in self.gamma:
            torch.nn.init.normal_(layer.weight)
        return
    
    def forward(self, Pblock , Qblock, Pidx, Qidx, dists):
        P_embd = Pblock
        Q_embd = Qblock
        for layer in self.h:
            P_embd = self.activation(layer(P_embd))
            Q_embd = self.activation(layer(Q_embd))
            final_embeddings_P = []
            final_embeddings_Q = []
        for i in range(len(Pidx)):
            P_start = Pidx[i][0]
            P_end = Pidx[i][1]

            Q_start = Qidx[i][0]
            Q_end = Qidx[i][1]
            outputP = torch.sum(P_embd[P_start:P_end], axis=0)
            outputQ = torch.sum(Q_embd[Q_start:Q_end], axis=0)
            final_embeddings_P.append(self.final_h(outputP))

            final_embeddings_Q.append(self.final_h(outputQ))
        final_embeddings_P = torch.vstack(final_embeddings_P)
        final_embeddings_Q = torch.vstack(final_embeddings_Q)

        if self.finalMLP:
            if self.aggregation == 'max':
                result = torch.maximum(final_embeddings_P, final_embeddings_Q)
            else:
                result = final_embeddings_P + final_embeddings_Q
            for layer in self.gamma:
                result = self.activation(layer(result))
            final = torch.relu(dists - result)
            return torch.mean(torch.square((result-dists)/dists) + final)
        
        #return final_embeddings_P, final_embeddings_Q
        final_embeddings_P = (1/ self.h_out) * final_embeddings_P
        final_embeddings_Q = (1/ self.h_out) * final_embeddings_Q
        d_pred = torch.square(torch.linalg.norm(final_embeddings_P - final_embeddings_Q, dim=1, ord=1))
        final = torch.relu(dists-d_pred)
        return torch.mean(torch.square((d_pred - dists)/dists) + final)

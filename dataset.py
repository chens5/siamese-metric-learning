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

def load_hdf5_data(filename):
    dataset = None
    labels = None
    with h5py.File(filename,'r') as h5f: 
        dataset = h5f['data'][:]
        labels = h5f['label'][:]
    dataset = dataset - np.expand_dims(np.mean(dataset, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(dataset ** 2, axis=1)), 0)
    dataset = dataset / dist  # scale
    print("Data shape:", dataset.shape)
    # organize into dictionary = {label: [sample index 1, sample index 2, .... ]}
    types = np.unique(labels)
    label_dict = {}
    for t in types:
        label_dict[t] = []
    for i in range(len(dataset)):
        label_dict[labels[i][0]].append(i)
    return dataset, labels, label_dict

def build_comprehensive_sampler(raw_data, label_dict, pairs=10, save_data=False):
    numsets = raw_data.shape[0]
    numpoints = raw_data.shape[1]
    dimension = raw_data.shape[2]
    all_labels = list(label_dict.keys())
    # construct p_label
    p_label = []
    
    for i in all_labels:
        p_label.append(len(label_dict[i])/numsets)
    Ps = []
    Qs = []
    dists = []
    nmin = numpoints//4
    nmax = numpoints
    jobs = []
    pool = mp.Pool(processes=20)
    for i in trange(numsets):
        pindex = i
        P = raw_data[pindex]
        psz = np.random.randint(low=nmin, high=nmax)
        P = P[np.random.randint(low=0, high=numpoints, size=psz)]
        # sample [pairs] number of labels
        classes_to_sample = np.random.choice(all_labels, size=pairs, replace=True, p=p_label)
        for cl in classes_to_sample:
            qindex = np.random.choice(label_dict[cl])
            Q = raw_data[qindex]
            qsz = np.random.randint(low=nmin, high=nmax)
            Q = Q[np.random.randint(low=0, high=numpoints, size=qsz)]
            
            # OT computation
            mat = ot.dist(P, Q, metric='euclidean')
            p = (1/psz) * np.ones(psz)
            q = (1/qsz) * np.ones(qsz)
            job = pool.apply_async(ot.emd2, args=(p, q, mat))
            jobs.append(job)
            #dist = ot.emd2(p, q, mat)
            
            # Add to dataset
            Ps.append(torch.tensor(P, dtype=torch.float32))
            Qs.append( torch.tensor(Q, dtype=torch.float32))
            #dists.append(dist)
    for job in tqdm(jobs):
        job.wait()
    dists = [job.get() for job in jobs]
    if save_data==True:
        save_name='/data/sam/modelnet/data/train-datasets/pairs-{pairs}'
        np.savez(save_name, P=Ps, Q=Qs, dists=dists)
    return Ps, Qs, dists

def build_multiple_item_dataset(raw_data, items= [1040, 2047], max_pcd = 3000, train=True):
    # randomly sample max_pcd samples from items
    numpoints = raw_data.shape[1]
    ref_pcd = []
    for idx in items:
        ref_pcd.append(raw_data[idx])
    nmin = 10
    nmax = 200
    jobs = []
    pool = mp.Pool(processes=20)
    Ps = []
    Qs = []
    for i in trange(max_pcd):
        # choose ref_P and ref_Q
        P = ref_pcd[np.random.choice([0, 1])]
        Q = ref_pcd [np.random.choice([0, 1])]
        # sample 
        psz = np.random.randint(low=nmin, high=nmax)
        qsz = np.random.randint(low=nmin, high=nmax)

        psc = np.random.uniform(low=-2.0, high=2.0)
        qsc = np.random.uniform(low=-2.0, high=2.0)
        pvec = np.tile(np.random.uniform(low=0.0, high=1.0, size=3), (psz, 1) )
        qvec = np.tile(np.random.uniform(low = 0.0, high=1.0, size=3), (qsz, 1))

        P = P[np.random.randint(low=0, high=numpoints, size=psz)] + psc*pvec
        Q = Q[np.random.randint(low=0, high=numpoints, size=qsz)] + qsc*qvec
        mat = ot.dist(P, Q, metric='euclidean')
        p = (1/len(P)) * np.ones(len(P))
        q = (1/len(Q)) * np.ones(len(Q))
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)
        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append( torch.tensor(Q, dtype=torch.float32))
    for job in tqdm(jobs):
        job.wait()
    dists = [job.get() for job in jobs]
    permutation = np.random.permutation(len(dists))
    Ps = np.array(Ps)[permutation]
    Qs = np.array(Qs)[permutation]
    dists = np.array(dists)[permutation]
    if train:
        save_name='/data/sam/modelnet/data/train-datasets/item-{idx1}-item-{idx2}-pairs-{pairs}'.format(idx1 = items[0], idx2=items[1], pairs=max_pcd)
        np.savez(save_name, P=Ps, Q=Qs, dists=dists)

    
    return Ps, Qs, dists
    

def build_single_item_dataset(raw_data,item_idx = 300, pairs=500, train=True):
    numsets = raw_data.shape[0]
    numpoints = raw_data.shape[1]
    dimension = raw_data.shape[2]
    Ps = []
    Qs = []
    dists = []
    nmin = 10
    nmax = 200
#     all_pcd = []
    ref_point_cloud = raw_data[item_idx]
#     for i in trange(pairs):
#         sz = np.random.randint(low=nmin, high=nmax)
#         pcd = ref_point_cloud[np.random.randint(low=0, high=2048, size=sz)]
#         all_pcd.append(pcd)
    jobs = []
    pool = mp.Pool(processes=20)
    for i in trange(pairs):
        #for j in range(i+1,pairs): 
        # sample random pair of point sets
        psz = np.random.randint(low=nmin, high=nmax)
        qsz = np.random.randint(low=nmin, high=nmax)
        # Scale each P and Q by some random vector
        psc = np.random.uniform(low=-2.0, high=2.0)
        qsc = np.random.uniform(low=-2.0, high=2.0)
        pvec = np.tile(np.random.uniform(low=0.0, high=1.0, size=3), (psz, 1) )
        qvec = np.tile(np.random.uniform(low = 0.0, high=1.0, size=3), (qsz, 1))
        P = ref_point_cloud[np.random.randint(low=0, high=numpoints, size=psz)] + psc*pvec
        Q = ref_point_cloud[np.random.randint(low=0, high=numpoints, size=qsz)] + qsc*qvec
        # compute OT distance
        mat = ot.dist(P, Q, metric='euclidean')
        p = (1/len(P)) * np.ones(len(P))
        q = (1/len(Q)) * np.ones(len(Q))
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)
        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append( torch.tensor(Q, dtype=torch.float32))
    for job in tqdm(jobs):
        job.wait()
    dists = [job.get() for job in jobs]
    permutation = np.random.permutation(len(dists))
    Ps = np.array(Ps)[permutation]
    Qs = np.array(Qs)[permutation]
    dists = np.array(dists)[permutation]
    if train:
        save_name='/data/sam/modelnet/data/train-datasets/item-{idx}-pairs-{pairs}'.format(pairs=pairs, idx=item_idx)
        np.savez(save_name, P=Ps, Q=Qs, dists=dists)
    else:
        save_name='/data/sam/modelnet/data/test-datasets/item-{idx}-pairs-{pairs}'.format(pairs=pairs, idx=item_idx)
        np.savez(save_name, P=Ps, Q=Qs, dists=dists)
    
    return Ps, Qs, dists
    
def build_dataset(raw_data, pairs=1000):
    numsets = raw_data.shape[0]
    numpoints = raw_data.shape[1]
    dimension = raw_data.shape[2]
    Ps = []
    Qs = []
    dists = []
    nmin = numpoints-600
    nmax = numpoints
    
    for i in trange(pairs):
        # sample random pair of point sets
        P = raw_data[np.random.randint(low=0, high=numsets)]
        Q = raw_data[np.random.randint(low=0, high=numsets)]
        
        # randomly sample points from two given point sets
        psz = np.random.randint(low=nmin, high=nmax)
        qsz = np.random.randint(low=nmin, high=nmax)
        P = P[np.random.randint(low=0, high=numpoints, size=psz)]
        Q = Q[np.random.randint(low=0, high=numpoints, size=qsz)]
        
        # compute OT distance
        mat = ot.dist(P, Q, metric='euclidean')
        p = (1/psz) * np.ones(psz)
        q = (1/qsz) * np.ones(qsz)
        dist = ot.emd2(p, q, mat)
        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append( torch.tensor(Q, dtype=torch.float32))
        dists.append(dist)
        
    return Ps, Qs, dists

class PointNetDataloader:
    def __init__(self, Ps, Qs, dists, batch_size, shuffle=False):
        self.Ps = Ps
        self.Qs = Qs
        self.total = len(Ps)
        self.dists = torch.tensor(dists)
        self.shuffle = shuffle
        self.batch_size = batch_size

        # Output tensors for each element
        self.Pblock = torch.cat(self.Ps)
        self.Qblock = torch.cat(self.Qs)
        self.Pidx = []
        pstart = 0
        self.Qidx = []
        qstart = 0
        for i in range(self.total):
            psz = len(self.Ps[i])
            qsz = len(self.Qs[i])
            pend = pstart + psz
            qend = qstart + qsz
            self.Pidx.append([pstart, pend])
            self.Qidx.append([qstart, qend])
            pstart = pend
            qstart = qend
        self.Pidx = torch.tensor(self.Pidx)
        self.Qidx = torch.tensor(self.Qidx)
        self.current_batch = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        batch_index = self.current_batch * self.batch_size
        if batch_index >= self.total:
            self.current_batch = 0
            raise StopIteration
        Pidx = self.Pidx[batch_index : batch_index + self.batch_size]
        pblock_start = Pidx[0][0]
        pblock_end = Pidx[-1][1]
        Pblock = self.Pblock[pblock_start:pblock_end]

        Qidx = self.Qidx[batch_index:batch_index + self.batch_size]
        qblock_start = Qidx[0][0]
        qblock_end = Qidx[-1][1]
        Qblock = self.Qblock[qblock_start:qblock_end]
        
        self.current_batch += 1
        return Pblock, Qblock, Pidx, Qidx, self.dists[batch_index:batch_index + self.batch_size]
    
     

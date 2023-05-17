import numpy as np
import ot
import multiprocessing as mp
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import argparse
from autoencoder_model import *
from dataset import *
import scipy.io
import geomloss
from torch.utils.tensorboard import SummaryWriter
import json
import csv


MSELOSS = nn.MSELoss(reduction='mean')
KLDIV = nn.KLDivLoss(reduction='batchmean')
SINKHORNLOSS = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
MIN = 1e-07

POINTS = ['synthetic-random', 
          'circle', 
          'grid', 
          'ncircle', 
          'ncircle/large', 
          'modelnet', 
          'modelnet/large',
          'ncircle/dim6', 
          'ncircle/dim10', 
          'ncircle/dim14',
          'rna-atac']

def construct_random_point_dataset(dimension=2, nmin=5, nmax=20, numdist=20, nprocesses = 20):
    Ps = []
    Qs = []

    jobs = []
    pool = mp.Pool(processes=20)

    for i in range(numdist):
        psz = np.random.randint(low=nmin, high=nmax)
        P = np.random.uniform(size=(psz, dimension))

        qsz = np.random.randint(low=nmin, high=nmax)
        Q = np.random.uniform(size=(qsz, dimension))

        M = ot.dist(P, Q)

        p = np.ones(psz)/psz
        q = np.ones(qsz)/qsz

        job = pool.apply_async(ot.emd2, args=(p, q, M))
        jobs.append(job)

        Ps.append(torch.tensor(P))
        Qs.append(torch.tensor(Q))
    for job in tqdm(jobs):
        job.wait()
    dists = [job.get() for job in jobs]
    return Ps, Qs, dists

def construct_image_dataset(train_data, emds, idlist1, idlist2, train=700000, test=200000):
    sources = []
    targets = []
    print("Beginning to construct dataset.....")
    for i in trange(train+test):
        idx1 = idlist1[i]
        idx2 = idlist2[i]
        source = train_data[idx1][0]/torch.sum(train_data[idx1][0])
        target = train_data[idx2][0]/torch.sum(train_data[idx2][0])
        sources.append(source)
        targets.append(target)
    train_dataset = EMDPairDataset(sources[:train], targets[:train], emds[:train]) 
    val_dataset = EMDPairDataset(sources[train:train+test], targets[train:train+test], emds[train:train+test])
    return train_dataset, val_dataset

def train_point_autoencoder(dataset : EMDPairDataset, dimension: int, initial: dict, 
                            phi: dict, device: str, lr, name: str, val_dataset=None,
                            iterations=200, num_decoding=20, batch_size=64, enc=False, lam=0.1, tr=1):
    embedding_size = phi['output']
    encoder = PointEncoder(dimension, initial, phi, activation='lrelu', max=True)
    if enc:
        model = encoder
    else:
        decoder = PointDecoder(dimension, embedding_size, num_decoding=num_decoding)
        model = AutoEncoder(encoder, decoder)
    
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir='runs/autoencoder/modelnet/large/{name}'.format(name=name))
    epoch_losses = []
    for epoch in trange(iterations):
        optimizer.zero_grad()
        epoch_loss = 0

        for i in range(len(dataset)):
            input1 = dataset[i][0].type(torch.float32).to(device)
            input2 = dataset[i][1].type(torch.float32).to(device)
            yval = dataset[i][2].type(torch.float32).to(device)
            if enc:
                feat1 = model(input1)
                feat2 = model(input2)
                sinkhorn1 = 0.0
                sinkhorn2 = 0.0
            else:
                feat1, feat2, unfeat1, unfeat2 = model(input1, input2)
                sinkhorn1 = lam * SINKHORNLOSS(input1, unfeat1)
                sinkhorn2 = lam * SINKHORNLOSS(input1, unfeat2)
            l2_diff = torch.linalg.vector_norm(feat1 - feat2)
            mse_loss = MSELOSS(l2_diff, yval)
            
            loss = (1/batch_size) * (mse_loss + sinkhorn1 + sinkhorn2).type(torch.float32)
            loss = mse_loss.type(torch.float32)
            epoch_loss += loss.detach()
            loss.backward()
            if (i != 0 and i % batch_size == 0) or i == len(dataset) - 1:
                optimizer.step()
                optimizer.zero_grad()
        
        epoch_losses.append(epoch_loss)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        # if enc:
        #     val_loss = validation_loss(val_dataset, model, device)
        # else:
        #     val_loss = validation_loss(val_dataset, model.encoder, device)
        # writer.add_scalar('Loss/val',  val_loss, epoch)
        # Break if difference between losses is small
        if torch.isnan(epoch_loss):
            print("LOSS IS NAN :( ")
            break
        if len(epoch_losses) > 10 and torch.abs(epoch_losses[-2] - epoch_losses[-1])< 0.001:
            break
        dataset.shuffle()
    return model, epoch

def train_image_autoencoder(dataloader: DataLoader, sz: int, 
                embedding_size: int, device: str, 
                lr, name:str, iterations=200, tr=1):
    
    encoder = ImageEncoder(image_size=(sz, sz), embedding_size=embedding_size)
    decoder = ImageDecoder(image_size=(sz, sz), embedding_size=embedding_size)
    model = AutoEncoder(encoder, decoder)
    model.to(device)
    model.reset_parameters()

    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir='runs/{name}'.format(name=name))
    epoch_losses = []
    for i in trange(iterations):
        optimizer.zero_grad()
        epoch_loss = 0
        for data in dataloader:

            input1 = data[0].type(torch.float32).to(device)
            input2 = data[1].type(torch.float32).to(device)
            yval = data[2].type(torch.float32).to(device)

            feat1, feat2, unfeat1, unfeat2 = model(input1, input2)
            l2_diff = torch.linalg.vector_norm(feat1 - feat2, dim=1)
            mse_loss = MSELOSS(l2_diff, yval)
            klin1 = torch.clip(input1, min=MIN, max=1)
            klin2 = torch.clip(input2, min=MIN, max=1)
            kldiv1 = KLDIV(unfeat1, klin1)
            kldiv2 = KLDIV(unfeat2, klin2)
            
            loss = (mse_loss + kldiv1 + kldiv2).type(torch.float32)

            loss.backward()
            epoch_loss += loss.detach()
            
            optimizer.step()
            optimizer.zero_grad()
        epoch_losses.append(epoch_loss)
        writer.add_scalar('Loss/train', epoch_loss, i)
        # Break if difference between losses is small
        if torch.isnan(epoch_loss):
            print("LOSS IS NAN :( ")
            break
        if len(epoch_losses) > 10 and torch.abs(epoch_losses[-2] - epoch_losses[-1])< 0.001:
            break
    writer.flush()

    writer.close()
    return model

# Validation loss is MRE loss
def validation_loss(val_dataset: EMDPairDataset, model: AutoEncoder, device: str):
    total_loss = 0
    count = 0
    for i in range(len(val_dataset)):
        input1 = val_dataset[i][0].type(torch.float32).to(device)
        #input1 = torch.unsqueeze(input1, dim=0)
        input2 = val_dataset[i][1].type(torch.float32).to(device)
        #input2 = torch.unsqueeze(input2, dim=0)
        yval = torch.tensor(val_dataset[i][2])
        # feat1, feat2, _, _ = model(input1, input2)
        feat1 = model(input1)
        feat2 = model(input2)
        l2_diff = torch.linalg.vector_norm(feat1 - feat2)
        if yval > 0.0:
            loss = torch.sum(torch.abs(l2_diff - yval)/yval)
            total_loss += loss.detach()
            count += 1
    return total_loss/count

# reads in a json parameter file
# Format: {modelname: {dimension: , initial: , phi: , rho: }}
def read_parameter_file(filename:str):
    f = open(filename)
    parameters = json.load(f)
    f.close()
    return parameters

def main():
    parser = argparse.ArgumentParser(description='Training options for prduct network')
    parser.add_argument('--parameter-file', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-iter', type=int)
    parser.add_argument('--tr', type=int, default=1)
    parser.add_argument('--enc', action='store_true')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dim', type=int)
    parser.add_argument('--decoding', type=int, default=100)
    parser.add_argument('--train-ds', type=int, nargs='+')
    parser.add_argument('--val-ds', type=int, nargs='+')
    
    args = parser.parse_args()
    parameters = read_parameter_file(args.parameter_file)

    if args.dataset_name == 'mnist':
        train_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        mat = scipy.io.loadmat(args.datapath)
        train_dataset, val_dataset = construct_image_dataset(
                                            train_data, 
                                            mat['D'][0], 
                                            mat['is'][0], 
                                            mat['it'][0],
                                            train=1000,
                                            test=10
                                            )
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
        print("starting training")
        with open('output/{}/image-autoencoder.csv'.format(args.dataset_name), 'w', newline='') as csvfile:
            fieldnames = ['embedding_size', 'validation_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for modelname in parameters:
                embedding_size = parameters[modelname]['initial']['output']
                image_size = parameters[modelname]['dimension']
                model = train_image_autoencoder(train_dataloader, 
                                    image_size,
                                    embedding_size, 
                                    args.device, 
                                    args.lr,
                                    name=modelname,
                                    iterations=args.max_iter,
                                    tr=args.tr
                                    )
                # RUN model on validation dataset and output
                val_loss = validation_loss(val_dataset, model, args.device)
                print("Embedding size:", embedding_size, "Validation loss:", val_loss.item())
                writer.writerow({'embedding_size': embedding_size, 'validation_loss': val_loss})
                # Save model
                model_name = '/data/sam/{dsname}/models/{name}.pt'.format(
                                                                            dsname=args.dataset_name,
                                                                            name=modelname
                                                                            ) 
                torch.save(model.state_dict(), f=model_name)

    if args.dataset_name in POINTS:
        train_sf = '/data/sam/{}/data/train-nmax-{}-nmin-{}-sz-{}.npz'.format(args.dataset_name, 
                                                                          args.train_ds[0], 
                                                                          args.train_ds[1], 
                                                                          args.train_ds[2])
        val_sf = '/data/sam/{}/data/val-nmax-{}-nmin-{}-sz-{}.npz'.format(args.dataset_name, 
                                                                      args.val_ds[0], 
                                                                      args.val_ds[1], 
                                                                      args.val_ds[2])
                
        train_data = np.load(train_sf, allow_pickle=True)
        Ps = train_data['Ps']
        Qs = train_data['Qs']
        dists = train_data['dists']
        val_data = np.load(val_sf, allow_pickle=True)
        Ps_val = val_data['Ps']
        Qs_val = val_data['Qs']
        dists_val = val_data['dists']
        train_dataset = EMDPairDataset(Ps, Qs, torch.tensor(dists))
        val_dataset = EMDPairDataset(Ps_val, Qs_val, dists_val)

        if args.enc:
            csv_name = 'output/{}/encoder-{}.csv'.format(args.dataset_name, args.tr)
        else:
            csv_name = 'output/{}/autoencoder-{}.csv'.format(args.dataset_name, args.tr)

        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['init_hidden', 
                          'init_output', 
                          'init_layers', 
                          'phi_hidden', 
                          'phi_output', 
                          'phi_layers', 
                          'epoch',
                          'validation_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for modelname in parameters:
                dimension = args.dim
                initial = parameters[modelname]['initial']
                phi = parameters[modelname]['phi']
                model, epoch = train_point_autoencoder(train_dataset, 
                                                dimension, 
                                                initial, 
                                                phi,
                                                val_dataset=val_dataset,
                                                device=args.device, 
                                                lr = args.lr, 
                                                name=modelname,
                                                iterations=args.max_iter,
                                                batch_size = args.batch_size, 
                                                enc = args.enc,
                                                num_decoding=args.decoding,
                                                tr=args.tr)
                if args.enc:
                    val_loss = validation_loss(val_dataset, model, args.device)
                else:
                    val_loss = validation_loss(val_dataset, model.encoder, args.device)
                print("model name:", modelname, "Validation loss:", val_loss.item())
                writer.writerow({'init_hidden': initial['hidden'], 
                                 'init_output': initial['output'],
                                 'init_layers': initial['layers'],
                                 'phi_hidden': phi['hidden'],
                                 'phi_output': phi['output'],
                                 'phi_layers': phi['layers'],
                                 'epoch': epoch,
                                 'validation_loss': val_loss.item()})
                if args.enc:
                    save_file = '/data/sam/{data}/models/encoder/{name}-lrelu.pt'.format(data=args.dataset_name, name=modelname)
                else:
                    save_file = '/data/sam/{data}/models/autoencoder/{name}-max.pt'.format(data=args.dataset_name, name=modelname)
                torch.save(model.state_dict(), f=save_file)

    return 0

if __name__=='__main__':
    main()
    

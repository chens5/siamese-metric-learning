import numpy as np
import ot
import csv
import multiprocessing as mp
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from autoencoder_model import *
import scipy.io
import geomloss
import argparse
import json
from dataset import *
#from train_autoencoder import *
from torch.utils.tensorboard import SummaryWriter

import os
#from train_autoencoder import construct_image_dataset

MSELOSS = nn.MSELoss(reduction='mean')
KLDIV = nn.KLDivLoss(reduction='batchmean')
SINKHORNLOSS = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
MIN = 1e-07

IMAGE_DATASETS=['mnist']

POINT_DATASETS = ['synthetic-uniform', 
                  'circle', 
                  'grid', 
                  'ncircle', 
                  'modelnet', 
                  'modelnet/large',
                  'ncircle/large',
                  'ncircle/dim6', 
                  'ncircle/dim10', 
                  'ncircle/dim14', 
                  'rna-atac',
                  'modelnet/w2',
                  'ncircle/w2',
                  'ncircle/dim6/w2',
                  'rna',
                  'rna/w2',
                  'rna-2k',
                  'rna-2k/w2', 
                  'rna-atac/w2']

def train_point_productnet(dataset : EMDPairDataset, dimension: int, initial: dict, 
                            phi: dict, rho: dict, device: str, lr, name: str,
                            activation='relu', mean=False, iterations=200, 
                            batch_size=64, val_dataset=None):
    embedding_size = phi['output']
    
    encoder = PointEncoder(dimension, initial, phi, bn=False, mean=mean, activation=activation)
    final = initialize_mlp(embedding_size, rho['hidden'], 1, rho['layers'], activation=activation)
    model = ProductNet(encoder, None, final, image=False)
    model.to(device)
    print(encoder)
    print(final)
    
    optimizer = Adam(model.parameters(), lr=lr)
    if mean:
        writer = SummaryWriter(log_dir='runs/modelnet/pnet/{name}-mean'.format(name=name))
    else:
        writer = SummaryWriter(log_dir='runs/modelnet/large/pnet/{name}'.format(name=name))
    epoch_losses = []
    for epoch in trange(iterations):
        optimizer.zero_grad()
        epoch_loss = 0
        for i in trange(len(dataset)):
            input1 = dataset[i][0].type(torch.float32).to(device)
            input2 = dataset[i][1].type(torch.float32).to(device)

            # n = dataset[i][0].size()[0]
            # m = dataset[i][1].size()[0]
            # vec1 = torch.unsqueeze(torch.ones(n)/n, 1)
            # vec2 = torch.unsqueeze(torch.ones(m)/m, 1)
            # vec1 = vec1.to(device)
            # input1 = torch.hstack((dataset[i][0], vec1))
            # vec2 = vec2.to(device)
            # input2 = torch.hstack((dataset[i][1], vec2))
            yval = dataset[i][2].type(torch.float32).to(device)
            # input1 = dataset[i][0]
            # input2 = dataset[i][1]
            #yval = dataset[i][2]

            pred, feat1, feat2 = model(input1, input2)
            yval = torch.unsqueeze(yval, dim=0)
            loss = 1/batch_size * MSELOSS(pred, yval)
            epoch_loss += loss.detach()
            loss.backward()
            if (i != 0 and i % batch_size == 0) or i == len(dataset) - 1:
                optimizer.step()
                optimizer.zero_grad()

        
        epoch_losses.append(epoch_loss/len(dataset))
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        # if val_dataset != None:
        #     val_loss = validation_loss(val_dataset, model, device)
        #     writer.add_scalar('Loss/val', val_loss, epoch)
        # Break if difference between losses is small
        if torch.isnan(epoch_loss):
            print("LOSS IS NAN :( ")
            break
        if len(epoch_losses) > 20 and torch.abs(epoch_losses[-2] - epoch_losses[-1])< 0.0001:
            break
        #dataset.shuffle()
    return model, epoch

def train(dataloader: DataLoader, sz: int, embedding_size: int, 
          phi_spec: dict, rho_spec: dict, device: str, name:str, module=None, 
          lr=0.001, iterations=200, tr=1, mean=False):
    '''
    Function to train the entire model 
    '''

    # Initialize h
    if module == None:
        encoder = ImageEncoder(image_size=(sz, sz), embedding_size=embedding_size)
        decoder = ImageDecoder(image_size=(sz, sz), embedding_size=embedding_size)
        h = AutoEncoder(encoder, decoder)
    else:
        h = module
    
    # Initialize phi
    phi_layer_num = phi_spec['layers']
    phi_hidden_dim = phi_spec['hidden']
    phi_out_dim = phi_spec['out']

    phi = initialize_mlp(embedding_size, phi_hidden_dim, phi_out_dim, phi_layer_num)

    # Initialize rho
    rho_layer_num = rho_spec['layers']
    rho_hidden_dim = rho_spec['hidden']

    rho = initialize_mlp(phi_out_dim, rho_hidden_dim, 1, rho_layer_num)

    # Initialize ProductNet model
    model = ProductNet(h, phi, rho, image=True)
    
    model.to(device)

    # Freeze embedding (h network) parameters
    if module:
        for param in model.initial.parameters():
            param.requires_grad = False

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    writer = SummaryWriter(log_dir='runs/image/{name}'.format(num=tr, esz=embedding_size))
    epoch_losses = []
    for i in trange(iterations):
        optimizer.zero_grad()
        epoch_loss = 0
        for data in dataloader:
            input1 = data[0].type(torch.float32).to(device)
            input2 = data[1].type(torch.float32).to(device)
            yval = data[2].type(torch.float32).to(device)

            pred, feat1, feat2 = model(input1, input2)
            nonzero = torch.nonzero(yval)
            pred = pred.squeeze()

            log_mse_loss = MSELOSS(pred[nonzero].log(), yval[nonzero].log())

            loss = log_mse_loss 

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


def validation_loss(val_dataset: EMDPairDataset, model: AutoEncoder, device: str, image=False):
    total_loss = []
    count = 0
    for i in range(len(val_dataset)):
        # n = val_dataset[i][0].size()[0]
        # m = val_dataset[i][1].size()[0]
        # vec1 = torch.unsqueeze(torch.ones(n)/n, 1)
        # vec2 = torch.unsqueeze(torch.ones(m)/m, 1)
        # vec1 = vec1.to(device)
        # input1 = torch.hstack((val_dataset[i][0].to(device), vec1))
        # vec2 = vec2.to(device)
        # input2 = torch.hstack((val_dataset[i][1].to(device), vec2))
        # input1 = input1.to(device)
        # input2 = input2.to(device)
        input1 = val_dataset[i][0].to(device)
        input2 = val_dataset[i][1].to(device)
        if image:
            input1 = torch.unsqueeze(input1, dim=0)
            input2 = torch.unsqueeze(input2, dim=0)
        yval = torch.tensor(val_dataset[i][2])
        pred, _, _ = model(input1, input2)
        if yval > 0.0:
            loss = torch.sum(torch.abs(pred - yval)/yval)
            total_loss.append(loss.detach().item())
            count += 1
    return np.mean(total_loss), np.std(total_loss)


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
    parser.add_argument('--freeze-embedding-layer', action='store_true')
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--mean', action='store_true')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dim', type=int)
    parser.add_argument('--train-ds', type=int, nargs='+')
    parser.add_argument('--val-ds', type=int, nargs='+')


    args = parser.parse_args()
    print("parsed args")

    parameters = read_parameter_file(args.parameter_file)
    print("parsed parameters")

    if args.dataset_name in IMAGE_DATASETS:

        # Initialize dataset 
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
                                            train=20000,
                                            test=2000
                                            )
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
        print("starting training")
        with open('output/image-productnet.csv', 'w', newline='') as csvfile:
            fieldnames = ['embedding_size', 
                          'phi_hidden', 
                          'phi_output', 
                          'phi_layers', 
                          'validation_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for modelname in parameters:
                embedding_size = parameters[modelname]['initial']['output']
                phi_spec = parameters[modelname]['phi']
                rho_spec = parameters[modelname]['rho']
                model = train(dataloader=train_dataloader, 
                            sz=args.image_size,
                            embedding_size=embedding_size, 
                            phi_spec=phi_spec,
                            rho_spec=rho_spec,
                            device=args.device, 
                            lr = args.lr,
                            iterations=args.max_iter,
                            tr=args.tr
                            )
                # RUN model on validation dataset and output
                val_loss = validation_loss( val_dataset, model, args.device, image=True)
                print('\n Modelname: ', modelname, "Validation loss:", val_loss)
                writer.writerow({'embedding_size': embedding_size, 
                                 'phi_hidden': phi_spec['hidden'],
                                 'phi_output': phi_spec['output'],
                                 'phi_layers': phi_spec['layers'],
                                 'rho_hidden': rho_spec['hidden'],
                                 'rho_layers': rho_spec['layers'],
                                 'validation_loss': val_loss.item()})
                # Save model
                model_name = '/data/sam/{dsname}/models/productnet/{name}.pt'.format(
                                                                            dsname=args.dataset_name,
                                                                            name=modelname
                                                                            ) 
                torch.save(model.state_dict(), f=model_name)
    elif args.dataset_name in POINT_DATASETS:
        print("Using mean agg:", args.mean)
        if not os.path.exists('/data/sam/{}/models/productnet'.format(args.dataset_name)):
            os.makedirs('/data/sam/{}/models/productnet'.format(args.dataset_name))

        train_sf = '/data/sam/{}/data/train-nmax-{}-nmin-{}-sz-{}.npz'.format(args.dataset_name, 
                                                                          args.train_ds[0], 
                                                                          args.train_ds[1], 
                                                                          args.train_ds[2])
        val_sf = '/data/sam/{}/data/val-nmax-{}-nmin-{}-sz-{}.npz'.format(args.dataset_name, 
                                                                      args.val_ds[0], 
                                                                      args.val_ds[1], 
                                                                      args.val_ds[2])
        print("validation set", val_sf)
        train_data = np.load(train_sf, allow_pickle=True)
        Ps = train_data['Ps']
        Qs = train_data['Qs']
        dists = train_data['dists']
        val_data = np.load(val_sf, allow_pickle=True)
        Ps_val = val_data['Ps']
        Qs_val = val_data['Qs']
        dists_val = val_data['dists']
        for i in trange(len(dists)):
            Ps[i] = Ps[i].type(torch.float32).to(args.device)
            Qs[i] = Qs[i].type(torch.float32).to(args.device)
        print("loaded all datasets to device:", args.device)
        train_dataset = EMDPairDataset(Ps, Qs, torch.tensor(dists))
        val_dataset = EMDPairDataset(Ps_val, Qs_val, dists_val)
        print("size of train_dataset", len(train_dataset))
        if args.mean:
            output_name = 'output/{}/productnet-{}-mean.csv'.format(args.dataset_name, args.activation)
        else:
            output_name = 'output/{}/productnet-{}-single.csv'.format(args.dataset_name, args.activation)
        with open(output_name, 'w', newline='') as csvfile:
            fieldnames = ['init_hidden', 
                          'init_output', 
                          'init_layers', 
                          'phi_hidden', 
                          'phi_output', 
                          'phi_layers', 
                          'rho_hidden',
                          'rho_layers',
                          'epochs',
                          'validation_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for modelname in tqdm(parameters):
                dimension = args.dim
                initial = parameters[modelname]['initial']
                phi = parameters[modelname]['phi']
                rho = parameters[modelname]['rho']
                model, epochs_trained = train_point_productnet(train_dataset, 
                                                dimension, 
                                                initial, 
                                                phi,
                                                rho,
                                                val_dataset=val_dataset,
                                                device=args.device, 
                                                lr = args.lr, 
                                                name=modelname,
                                                iterations=args.max_iter,
                                                batch_size = args.batch_size,
                                                mean=args.mean,
                                                activation=args.activation)
                val_loss = validation_loss(val_dataset, model, args.device, image=False)
                print("model name:", modelname, "Validation loss:", val_loss, '\n')
                writer.writerow({'init_hidden': initial['hidden'], 
                                 'init_output': initial['output'],
                                 'init_layers': initial['layers'],
                                 'phi_hidden': phi['hidden'],
                                 'phi_output': phi['output'],
                                 'phi_layers': phi['layers'],
                                 'rho_hidden':rho['hidden'],
                                 'rho_layers': rho['layers'],
                                 'epochs': epochs_trained,
                                 'validation_loss': val_loss[0]})
                if args.mean:
                    save_file = '/data/sam/{data}/models/productnet/{name}-{act}-mean.pt'.format(data=args.dataset_name,
                                                                             name=modelname,
                                                                             act=args.activation)
                else:
                    save_file = '/data/sam/{data}/models/productnet/{name}-{act}-exp.pt'.format(data=args.dataset_name,
                                                                                name=modelname,
                                                                                act=args.activation)
                print("saved model in:", save_file)
                torch.save(model.state_dict(), f=save_file)


    return 0

if __name__=='__main__':
    main()

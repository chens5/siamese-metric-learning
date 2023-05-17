from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

"""
Differentiable sinkhorn divergence.
This module provides a pytorch/autograd compatible implementation of sinkhorn
divergence, 
using the method described in :cite:`feydyInterpolatingOptimalTransport2019`
"""
from typing import Optional,  overload, Literal
from math import ceil

import torch
from torch import Tensor


@overload
def sinkhorn_internal(a: Tensor, b: Tensor, C: Tensor, 
                      epsilon: float, k: int=100, *, 
                      check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
                      return_has_converged:Literal[False] = False) -> tuple[Tensor, Tensor, Tensor]:
    ...

@overload
def sinkhorn_internal(a: Tensor, b: Tensor, C: Tensor, 
                      epsilon: float, k: int=100, *,
                      check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
                      return_has_converged:Literal[True]) -> tuple[Tensor, Tensor, Tensor, bool]:
    ...

def sinkhorn_internal(a: Tensor, b: Tensor, C: Tensor, 
                      epsilon: float, k: int=100, *, 
                      check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
                      return_has_converged:Literal[True, False] = False):
    r"""Same as sinkhorn, but returns f, g and P instead of the result
    Beware, this function does not have the shortcut differentiation
    
    (It can still be differentiated by autograd though)

    Args:
        a: (\*batch, n) vector of the first distribution
        b: (\*batch, m) vector of the second distribtion
        C: (\*batch, n, m) cost matrix
        epsilon: regularisation term for sinkhorn
        k: max number of sinkhorn iterations (default 100)
        check_convergence_interval: if int, check for convergence every 
            ``check_convergence_interval``. 
            If float, check for convergence every ``check_convergence_interval * k``. 
            If 0, never check for convergence 
            (apart from the last iteration if ``return_has_converged==True``)
            If convergence is reached early, the algorithm returns.
        cv_atol, cv_rtol: absolute and relative tolerance for the converegence criterion
        return_has_converged: whether to return a boolean indicating whether the 
            algorithm has converged. Setting this to True means that the  function
            will always check for convergence at the last iteration 
            (regardless of the value of ``check_convergence_interval``)
    Returns:
        f: Tensor (\*batch, n)
        g: Tensor (\*batch, m)
        log_P: Tensor (\*batch, n, m)
        has_converged: bool only returened if ``return_has_converged`` is True. 
            Indicates whether the algorithm has converged
    """
    *batch, n = a.shape
    *batch_, m = b.shape
    *batch__, n_, m_ = C.shape
    batch = torch.broadcast_shapes(batch, batch_, batch__)
    device = a.device
    dtype=a.dtype
    assert n == n_
    assert m == m_
    steps_at_which_to_check_for_convergence:list[bool]
    if isinstance(check_convergence_interval, float):
        check_convergence_interval = ceil(check_convergence_interval * k)
        steps_at_which_to_check_for_convergence = \
                [(step - 1) % check_convergence_interval == 0 for step in range(k)]
    elif check_convergence_interval == 0:
        steps_at_which_to_check_for_convergence = [False] * k 
    elif isinstance(check_convergence_interval, int):
        steps_at_which_to_check_for_convergence = \
                [(step - 1) % check_convergence_interval == 0 for step in range(k)]
    else: 
        raise ValueError("check_convergence_interval must be a float or an int")
    if return_has_converged:
        steps_at_which_to_check_for_convergence[-1] = True # to be able to say for sure whether
        # we have converged

    log_a = a.log()[..., :, None]
    log_b = b.log()[..., None, :]
    mC_eps = - C / epsilon

    f_eps = torch.randn((*batch, n, 1), device=device, dtype=dtype)
    #f over epsilon + log_a 
    #batch, n
    g_eps = torch.randn((*batch, 1, m), device=device, dtype=dtype)
    #g over epsilon + log_b
    #batch, m
    has_converged: bool = True
    for _,  should_check_convergence in zip(range(k), steps_at_which_to_check_for_convergence):
        if should_check_convergence:
            f_eps_old = f_eps
            g_eps_old = g_eps

        f_eps =  - torch.logsumexp(mC_eps + g_eps + log_b, dim=-1, keepdim=True)
        g_eps =  - torch.logsumexp(mC_eps + f_eps + log_a, dim=-2, keepdim=True)

        if (should_check_convergence 
            and torch.allclose(f_eps, f_eps_old, atol=cv_atol, rtol=cv_rtol) #type:ignore
            and torch.allclose(g_eps, g_eps_old, atol=cv_atol, rtol=cv_rtol)): #type:ignore
            break
    else:
        has_converged=False
    log_P = mC_eps + f_eps + g_eps + log_a + log_b
    # Note: we dont actually need save_for_backwards here
    # we could say ctx.f_eps = f_eps etc.
    # save_for_backwards does sanity checks 
    # such as checking that the saved variables do not get changed inplace
    # (if you have output them, which is not the case)
    # see https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/5
    f = epsilon * f_eps.squeeze(-1)
    g = epsilon * g_eps.squeeze(-2)
    if return_has_converged:
        return f, g, log_P, has_converged
    return f, g, log_P
    #print(res, (f.squeeze(-1) * a).sum(-1) + (g.squeeze(-2) * b).sum(-1))


class Sinkhorn(torch.autograd.Function):
    """Computes Sinkhorn divergence"""
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, C: Tensor, 
        epsilon: float, k: int=100, 
        check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
        return_has_converged:Literal[True, False] = False
                ) -> Tensor | tuple[Tensor, bool]:
        r"""Batched version of sinkhorn distance

        It is computed as in [@feydyInterpolatingOptimalTransport2019, Property 1]

        The 3 batch dims will be broadcast to each other. 
        Every steps is only broadcasted torch operations,
        so it should be reasonably fast on gpu

        Args:
            a: (\*batch, n) First distribution. 
            b: (\*batch, m) Second distribution
            C: (\*batch, n, m) Cost matrix
            epsilon: Regularization parameter
            k: max number of sinkhorn iterations (default 100)
            check_convergence_interval: if int, check for convergence every 
                ``check_convergence_interval``. 
                If float, check for convergence every ``check_convergence_interval * k``. 
                If 0, never check for convergence 
                (apart from the last iteration if ``return_has_converged==True``)
                If convergence is reached early, the algorithm returns.
            cv_atol, cv_rtol: absolute and relative tolerance for the converegence criterion
            return_has_converged: whether to return a boolean indicating whether the 
                algorithm has converged. Setting this to True means that the  function
                will always check for convergence at the last iteration 
                (regardless of the value of ``check_convergence_interval``)

        Returns:
            divergence: (\*batch) :math:`\text{divergence}[*i] = OT^\epsilon(a[*i], b[*i], C[*i])`
        """

        with torch.no_grad():
            f, g, log_P, *has_converged = sinkhorn_internal(
                    a, b, C, epsilon, k, 
                check_convergence_interval=check_convergence_interval,
                cv_atol=cv_atol, cv_rtol=cv_rtol, 
                return_has_converged=return_has_converged)#type:ignore
            res = (f * a).sum(-1) + (g * b).sum(-1)

        # Note: we dont actually need save_for_backwards here
        # we could say ctx.f_eps = f_eps etc.
        # save_for_backwards does sanity checks 
        # such as checking that the saved variables do not get changed inplace
        # (if you have output them, which is not the case)
        # see https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/5
        ctx.save_for_backward(f, g, log_P)
        ctx.epsilon = epsilon

        if has_converged: 
            return res, *has_converged
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        r"""We use the fact that the primal solution $P$
        and the dual solutions $f, g$
        are the gradients for respectively the cost matrix $C$, 
        and the input distributions $\mu_x$ and $\mu_y$
        :cite:`peyreComputationalOT2018`

        This allows us to shortcut the backward pass. 
        Note that :cite:t:`peyreComputationalOT2018` discourage doing this
        if the sinkhorn iterations do not converge
        """
        #output gradient d _ res y . so 
        # grad_output (*batch, 1) (or maybe just (*batch)
        # f_eps (*batch, n, 1)
        # g_eps (*batch, 1, m)
        # log_P (*batch, n, m)
        f, g, log_P = ctx.saved_tensors
        # epsilon = ctx.epsilon

        d_a = f * grad_output[..., None]
        d_a = d_a - d_a.mean(-1, keepdim=True) # so that the gradient
        #maintains the space of distributions
        d_b = g * grad_output[..., None]
        #d_b = d_b.squeeze(-2)
        d_b = d_b - d_b.mean(-1, keepdim=True) # idem
        d_C = log_P.exp() * grad_output[..., None, None]
        
        return d_a, d_b, d_C, *([None] * 6)
        

def sinkhorn(a: Tensor, b: Tensor, C: Tensor, 
             epsilon: float, max_iter: int=100, 
            *,  
            check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
            return_has_converged:Literal[True, False] = False
             ) -> Tensor:
    r"""Differentiable sinkhorn distance

    This is a pytorch implementation of sinkhorn, 
    batched (over ``a``, ``b`` and ``C``) 

    It is compatible with pytorch autograd gradient computations.

    See the documentation of :class:`Sinkhorn` for details.

    Args:
        a: (\*batch, n) vector of the first distribution
        b: (\*batch, m) vector of the second distribtion
        C: (\*batch, n, m) cost matrix
        epsilon: regularisation term for sinkhorn
        max_iter: max number of sinkhorn iterations (default 100)
        check_convergence_interval: if int, check for convergence every 
            ``check_convergence_interval``. 
            If float, check for convergence every ``check_convergence_interval * max_iter``. 
            If 0, never check for convergence 
            (apart from the last iteration if ``return_has_converged==True``)
            If convergence is reached early, the algorithm returns.
        cv_atol, cv_rtol: absolute and relative tolerance for the converegence criterion
        return_has_converged: whether to return a boolean indicating whether the 
            algorithm has converged. Setting this to True means that the  function
            will always check for convergence at the last iteration 
            (regardless of the value of ``check_convergence_interval``)

    Returns:
        Tensor: (\*batch). result of the sinkhorn computation
    """
    return Sinkhorn.apply(a, b, C, epsilon, max_iter, 
            check_convergence_interval, cv_atol, cv_rtol, return_has_converged)



def initialize_mlp(input_sz, hidden_sz, output_sz, layers, batch_norm=False, activation='relu'):
    if layers == 1:
        assert hidden_sz == output_sz
    if activation == 'relu':
        func = nn.ReLU
    elif activation =='lrelu':
        func = nn.LeakyReLU
    else:
        raise NameError('Not implemented')

    phi_layers= []
    phi_layers.append(nn.Linear(input_sz, hidden_sz))
    phi_layers.append(func())
    if batch_norm:
        phi_layers.append(nn.BatchNorm1d(input_sz))
    for i in range(layers - 1):
        if i < layers - 2:
            phi_layers.append(nn.Linear(hidden_sz, hidden_sz))
            phi_layers.append(func())
            if batch_norm:
                phi_layers.append(nn.BatchNorm1d(hidden_sz))
        else:
            phi_layers.append(nn.Linear(hidden_sz, output_sz))
            phi_layers.append(func())
    phi = nn.Sequential(*phi_layers)
    return phi

class ProductNet(nn.Module):
    def __init__(self, initial: nn.Module, phi: nn.Module, rho: nn.Module, image=False):
        super(ProductNet, self).__init__()
        self.initial = initial
        self.phi = phi
        self.rho = rho
        self.image = image
    
    def forward(self, input1, input2):
        embd1 = self.initial(input1)
        embd2 = self.initial(input2)
        if self.image:
            embd1 = self.phi(embd1)
            embd2 = self.phi(embd2)

        out = embd1 + embd2

        out = self.rho(out)

        return out, embd1, embd2


class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder:nn.Module):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
    
    def forward(self, input1, input2):
        embd1 = self.encoder(input1)
        embd2 = self.encoder(input2)
        
        unfeat1 = self.decoder(embd1)
        unfeat2 = self.decoder(embd2)
        return embd1, embd2, unfeat1, unfeat2

class PointEncoder(nn.Module):
    def __init__(self, dimension, mlp_params: dict, phi_params: dict, bn=False, mean=False,max=False, activation='relu'):
        super(PointEncoder, self).__init__()
        mlp_hdim = mlp_params['hidden']
        mlp_output = mlp_params['output']
        mlp_layers = mlp_params['layers']

        phi_hdim = phi_params['hidden']
        phi_layers = phi_params['layers']
        phi_output = phi_params['output']
        self.mlp = initialize_mlp(dimension, mlp_hdim, mlp_output, mlp_layers, activation=activation)
        self.phi = initialize_mlp(mlp_output, phi_hdim, phi_output, phi_layers, activation=activation)
        self.mean = mean
        self.max = max
    
    def forward(self, input):
        
        out = self.mlp(input)
        if self.mean:
            out = torch.mean(out, dim=0)
        elif self.max:
            out = torch.max(out, dim=0)[0]
        else:
            out = torch.sum(out, dim=0)
        out = self.phi(out)
        return out

class PointDecoder(nn.Module):
    def __init__(self, dimension, embedding_size, num_decoding, hdim=100, lnum=2):
        super(PointDecoder, self).__init__()
        self.num_decoding = num_decoding
        output_dim = dimension * self.num_decoding
        self.dec = initialize_mlp(embedding_size, hdim, output_dim, lnum)
        self.dimension = dimension
    
    def forward(self, input):
        out = self.dec(input)
        out = out.reshape((self.num_decoding, self.dimension))
        return out

class ImageEncoder(nn.Module):
    def __init__(self, image_size=(28, 28), embedding_size=10):
        super(ImageEncoder, self).__init__()
        self.sz = image_size[0]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(5, 5), padding='same')
        self.fc1 = nn.Linear(5*self.sz*self.sz, 100)
        self.fc2 = nn.Linear(100, embedding_size)
        self.flat = nn.Flatten()
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, input):
        out = torch.relu(self.conv1(input))
        out = torch.relu(self.conv2(out))
        out = self.flat(out)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        return out

class ImageDecoder(nn.Module):
    def __init__(self, image_size=(28, 28), embedding_size=10):
        super(ImageDecoder, self).__init__()
        self.sz = image_size[0]
        self.fc1 = nn.Linear(embedding_size, 100)
        self.fc2 = nn.Linear(100, 5 * self.sz * self.sz)
        self.conv1 = nn.Conv2d(in_channels=5,out_channels=10, kernel_size=(5, 5), padding='same')
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(3, 3), padding='same')
        self.flat = nn.Flatten()
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, input):
        batch_sz = input.size()[0]
        out = torch.relu(self.fc1(input))
        out = torch.relu(self.fc2(out))
        out=out.reshape((batch_sz, 5, self.sz, self.sz))
        out = torch.relu(self.conv1(out))
        out = self.conv2(out)
        out = self.flat(out)
        out = torch.log_softmax(out, dim=1)
        out = out.reshape((batch_sz, 1,  self.sz, self.sz))
        return out
    
class EMDPairDataset(Dataset):
    def __init__(self, sources, targets, emds):
        self.sources = sources
        self.targets = targets
        self.emds = emds
    
    def __len__(self):
        return len(self.emds)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx], self.emds[idx]
    
    def shuffle(self):
        permutation = np.random.permutation(len(self.emds))
        self.sources = self.sources[permutation]
        self.targets = self.sources[permutation]



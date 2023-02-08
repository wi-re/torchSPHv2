
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils.repeat import repeat
import torch
from torch import Tensor, nn
from torch.nn import Parameter

from cutlass import cutlass

import math
import numpy as np
from typing import Any
from typing import List, Tuple, Union
from torch_sparse import SparseTensor
from torch_scatter import scatter

def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)
     
def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
def zeros(value: Any):
    constant(value, 0.)
    
basestring = (str, bytes)
def is_list_of_strings(lst):
        if lst and isinstance(lst, list):
            return all(isinstance(elem, basestring) for elem in lst)
        else:
            return False
from cutlass import *
import scipy.optimize

MCache = None

def optimizeWeights2D(weights, basis, periodicity, nmc = 32 * 1024, targetIntegral = 1, windowFn = None, verbose = False):
    global MCache
    M = None
    numWeights = weights.shape[0] * weights.shape[1]    
    
    # print(weights.shape, numWeights)
    normalizedWeights = (weights - torch.sum(weights) / weights.numel())/torch.std(weights)
    if not MCache is None:
        cfg, M = MCache
        w,b,n,p,wfn = cfg
        if not(w == weights.shape and np.all(b == basis) and n == nmc and np.all(p ==periodicity) and wfn == windowFn):
            M = None
    # else:
        # print('no cache')
    if M is None:
        r = torch.sqrt(torch.rand(size=(nmc,1)).to(weights.device).type(torch.float32))
        theta = torch.rand(size=(nmc,1)).to(weights.device).type(torch.float32) *2 * np.pi

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        u = evalBasisFunction(weights.shape[0], x.T, which = basis[0], periodic = periodicity[0])[0,:].mT
        v = evalBasisFunction(weights.shape[1], y.T, which = basis[1], periodic = periodicity[1])[0,:].mT
        
    #     print('u', u.shape, u)
    #     print('v', v.shape, v)
        
        window = weights.new_ones(x.shape[0]) if windowFn is None else windowFn(torch.sqrt(x**2 + y**2))[:,0]
        
        
        nuv = torch.einsum('nu, nv -> nuv', u, v)
        nuv = nuv * window[:,None, None]

    #     print('nuv', nuv.shape, nuv)
        M = np.pi * torch.sum(nuv, dim = 0).flatten().detach().cpu().numpy() / nmc
#     print('M', M.shape, M)
        MCache = ((weights.shape, basis, nmc, periodicity, windowFn), M)

    
    w = normalizedWeights.flatten().detach().cpu().numpy()


    eps = 1e-2
    
    if 'chebyshev' in basis or 'fourier' in basis:        
        res = scipy.optimize.minimize(fun = lambda x: (M.dot(x) - targetIntegral)**2, \
                                      jac = lambda x: 2 * M * (M.dot(x) - targetIntegral), \
                                      hess = lambda x: 2. * np.outer(M,M), x0 = w, \
                                      method ='trust-constr', constraints = None,\
                                      options={'disp': False, 'maxiter':100})
    else:
        sumConstraint = scipy.optimize.NonlinearConstraint(fun = np.sum, lb = -eps, ub = eps)
        stdConstraint = scipy.optimize.NonlinearConstraint(fun = np.std, lb = 1 - eps, ub = 1 + eps)

        res = scipy.optimize.minimize(fun = lambda x: (M.dot(x) - targetIntegral)**2, \
                                      jac = lambda x: 2 * M * (M.dot(x) - targetIntegral), \
                                      hess = lambda x: 2. * np.outer(M,M), x0 = w, \
                                      method ='trust-constr', constraints = [sumConstraint, stdConstraint],\
                                      options={'disp': False, 'maxiter':100})
    result = torch.from_numpy(res.x.reshape(weights.shape)).type(torch.float32).to(weights.device)
    if verbose:
        print('result: ', res)
        print('initial weights:', normalizedWeights)
        print('result weights:',result)
        print('initial:', M.dot(w))
        print('integral:', M.dot(res.x))
        print('sumConstraint:', np.sum(res.x))
        print('stdConstraint:', np.std(res.x))
    return result, res.constr, res.fun, M.dot(w), M.dot(res.x)

class RbfConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        size: Union[int, List[int]] = 3,
        periodic : Union[int, List[int]] = False,
        rbf : Union[int, List[int]] = 'chebyshev',
        aggr: str = 'sum',
        dense_for_center: bool = False,
        bias: bool = False,
        initializer = torch.nn.init.xavier_normal_,
        activation = None,
        batch_size = [16,16],
        windowFn = None,
        normalizeWeights = True,
        normalizationFactor = None,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dim = dim
        self.periodic = periodic if isinstance(periodic, list) else repeat(periodic, dim)
        self.size = size if isinstance(size, list) else repeat(size, dim)
        self.rbfs = rbf if is_list_of_strings(rbf) else [rbf] * dim
        self.initializer = initializer
        self.batchSize = batch_size
        self.activation = None if activation is None else getattr(nn.functional, activation)
        self.windowFn = windowFn
        
        # print('Creating layer %d -> %d features'%( in_channels, out_channels))
        # print('For dimensionality: %d'% dim)
        # print('Parameters:')
        # print('\tRBF: ', self.rbfs)
        # print('\tSize: ', self.size)
        # print('\tPeriodic: ', self.periodic)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.K = torch.tensor(self.size).prod().item()
        if dim == 1:
            self.weight = Parameter(torch.Tensor(self.size[0], in_channels[0], out_channels))
        if dim == 2:
            self.weight = Parameter(torch.Tensor(self.size[0],self.size[1], in_channels[0], out_channels))
        if dim == 3:
            self.weight = Parameter(torch.Tensor(self.size[0],self.size[1], self.size[2], in_channels[0], out_channels))
        initializer(self.weight)
        with torch.no_grad():
            if self.rbfs[0] in ['chebyshev', 'fourier', 'gabor']:
                for i in range(self.dim):
                    if len(self.rbfs) == 1:
                        self.weight[i] *= np.exp(-i)
                    if len(self.rbfs) == 2:
                        self.weight[i,:] *= np.exp(-i)
                    if len(self.rbfs) == 3:
                        self.weight[i,:,:] *= np.exp(-i)
            if self.rbfs[1] in ['chebyshev', 'fourier', 'gabor']:
                for i in range(self.dim):
                    if len(self.rbfs) == 2:
                        self.weight[:,i] *= np.exp(-i)
                    if len(self.rbfs) == 3:
                        self.weight[:,i,:] *= np.exp(-i)
            if len(self.rbfs) > 2 and self.rbfs[2] in ['chebyshev', 'fourier', 'gabor']:
                for i in range(self.dim):
                    self.weight[:,:,i] = self.weight[:,:,i] * np.exp(-i)
            if normalizeWeights:
                if len(self.rbfs) == 2:
                    print('Starting normalization')
                    for i in range(in_channels[0]):
                        for j in range(out_channels):
                            newWeights, _, _, init, final = optimizeWeights2D(weights = self.weight[:,:,i,j].detach(),\
                                                                            basis = self.rbfs, periodicity = self.periodic, \
                                                                            nmc = 32*1024, targetIntegral = 1/in_channels[0], \
                                                                            windowFn = self.windowFn, verbose = False) 
                            self.weight[:,:,i,j] = newWeights
                            print('Normalizing [%2d x %2d]: %1.4e => %1.4e (target: %1.4e)' %(i,j, init, final, 1/in_channels[0]))

                            # self.weight[:,:,i,j] /= in_channels[0]
                    print('Done with normalization\n------------------------------------------')

        self.root_weight = dense_for_center
        if dense_for_center:
            self.lin = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer= 'uniform')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # if not isinstance(self.weight, nn.UninitializedParameter):
            # size = self.weight.size(0) * self.weight.size(1)
            # self.initializer(self.weight)
        if self.root_weight:
            self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        # print('x', x[0].shape, x)
        # print('edge_index', edge_index.shape, edge_index)
        # print('edge_attr', edge_attr.shape, edge_attr)
        # print('Size', Size)
        # if args.cutlad:
            # out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        # else:
        # out = self.propagate2(edge_index, x=x, edge_attr=edge_attr, size=size)
#         print('out: ', out.shape, out)

    
        inFeatures = x[1]
        
        edge_weights = None
        if not(self.windowFn is None):
            edge_weights = self.windowFn(torch.linalg.norm(edge_attr, axis = 1))

        convolution = cutlass.apply
            
        out = convolution(edge_index, inFeatures, edge_attr, edge_weights, self.weight, 
                                        x[0].shape[0], self.node_dim,
                                    self.size , self.rbfs, self.periodic, 
                                    self.batchSize[0],self.batchSize[1])

        # print('out', out.shape)

        # out = scatter(out, edge_index[0], dim=0, dim_size = x[0].shape[0], reduce='sum')

        # print('out', out.shape)
        x_r = x[0]
        if x_r is not None and self.root_weight:
            out = out + self.lin(x_r)

        if self.bias is not None:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.dim == 1:
            u = evalBasisFunction(self.size[0], edge_attr[:,0], which=self.rbfs[0], periodic = self.periodic[0]).T
             
            return torch.einsum('nu, uio,ni -> no',u,self.weight, x_j)
        if self.dim == 2:
            u = evalBasisFunction(self.size[0], edge_attr[:,0], which=self.rbfs[0], periodic = self.periodic[0]).T
            v = evalBasisFunction(self.size[1], edge_attr[:,1], which=self.rbfs[1], periodic = self.periodic[1]).T
            
            return torch.einsum('nu, nv, uvio,ni -> no',u,v,self.weight, x_j)
        if self.dim == 3:
            u = evalBasisFunction(self.size[0], edge_attr[:,0], which=self.rbfs[0], periodic = self.periodic[0]).T
            v = evalBasisFunction(self.size[1], edge_attr[:,1], which=self.rbfs[1], periodic = self.periodic[1]).T
            w = evalBasisFunction(self.size[2], edge_attr[:,1], which=self.rbfs[1], periodic = self.periodic[2]).T
            
            return torch.einsum('nu, nv, uvio,ni -> no',u,v,w,self.weight, x_j)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dim={self.dim})')
    
    
    def propagate2(self, edge_index: Adj, size: Size = None, **kwargs):
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)

        if decomposed_layers > 1:
            user_args = self.__user_args__
            decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
            decomp_kwargs = {
                a: kwargs[a].chunk(decomposed_layers, -1)
                for a in decomp_args
            }
            decomp_out = []

        for i in range(decomposed_layers):
            if decomposed_layers > 1:
                for arg in decomp_args:
                    kwargs[arg] = decomp_kwargs[arg][i]
            # print('self.__user_args__', self.__user_args__)
            # print('edge_index', edge_index)
            # print('size', size)
            # print('kwargs', kwargs)
            coll_dict = self.__collect__(self.__user_args__, edge_index,
                                            size, kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            for hook in self._message_forward_pre_hooks.values():
                res = hook(self, (msg_kwargs, ))
                if res is not None:
                    msg_kwargs = res[0] if isinstance(res, tuple) else res
                    
            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            
            convolution = cutlass.apply
            
            inFeatures = kwargs['x'][0]
            
            edge_weights = None
            if not(self.windowFn is None):
                edge_weights = self.windowFn(torch.linalg.norm(kwargs['edge_attr'], axis = 1))
                # print(torch.linalg.norm(kwargs['edge_attr'], axis = 1))
                # print(edge_weights.shape)
                # print(edge_weights)
                # print(inFeatures.shape)
                # inFeatures = inFeatures * window[:,None]
                # print(inFeatures.shape)


            out = convolution(edge_index, inFeatures, kwargs['edge_attr'], edge_weights, self.weight, 
                                            size[1], self.node_dim,
                                        self.size , self.rbfs, self.periodic, 
                                        self.batchSize[0],self.batchSize[1])

            for hook in self._aggregate_forward_hooks.values():
                res = hook(self, (aggr_kwargs, ), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            out = self.update(out, **update_kwargs)

            if decomposed_layers > 1:
                decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out
  
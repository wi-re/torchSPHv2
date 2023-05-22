# required for timing functions
from __future__ import print_function, division
# basic python includes 
import struct
import os
import math
import inspect
import re
import timeit
import time
from contextlib import contextmanager
from functools import partial
# numpy and some basic functions
import numpy as np
from numpy import pi, exp, sqrt
# imports required for 2d and 3d plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from mpl_toolkits.axes_grid1 import make_axes_locatable
#scipy optimization functions
from scipy import optimize
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import SR1
from scipy.integrate import dblquad
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
from numpy import sin, cos, tan, log, arcsin, arccos, sqrt
import pandas as pd
# from tqdm import trange, tqdm
import random
import warnings
import yaml


warnings.filterwarnings(action='ignore')

import time
import torch
from torch_geometric.loader import DataLoader
# from tqdm import trange, tqdm
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
import os

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
from scipy.optimize import minimize 

from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter
import matplotlib.patches as patches

import h5py
import numpy as np
import os
# from tqdm import tqdm

from typing import Union, List

import argparse


windowFn = lambda r: torch.clamp(1-r**2, min = 0)

def loadMesh(meshPath):
    meshFile = h5py.File(os.path.expanduser(meshPath), 'r')
    basis = meshFile.attrs['basis']
    n = meshFile.attrs['n']
    periodic = meshFile.attrs['periodic']
    

    areaTensor = torch.abs(torch.from_numpy(np.array(meshFile['meshAttributes']['areas'])).type(torch.float32))
    supportTensor = torch.from_numpy(np.array(meshFile['meshAttributes']['supports'])).type(torch.float32)
    centerTensor = torch.from_numpy(np.array(meshFile['meshAttributes']['centers'])).type(torch.float32)
    vertexTensor = torch.from_numpy(np.array(meshFile['meshAttributes']['vertices'])).type(torch.float32)

    neighborTensor = torch.from_numpy(np.array(meshFile['integralAttributes']['neighborCount'])).type(torch.int32)
    kernelTensor = torch.from_numpy(np.array(meshFile['integralAttributes']['splineKernel'])).type(torch.float32)
    gradientTensor = torch.from_numpy(np.array(meshFile['integralAttributes']['splineGradient'])).type(torch.float32)

    neighborhoodTensor = torch.from_numpy(np.array(meshFile['networkAttributes']['neighbors'])).type(torch.long)
    filterTensor = torch.from_numpy(np.array(meshFile['networkAttributes']['filterMatrices'])).type(torch.float32)
    integralTensor = torch.from_numpy(np.array(meshFile['networkAttributes']['filterMatrices'])).type(torch.float32)
    return (np.array(basis), n[0], periodic), \
        (areaTensor, supportTensor, centerTensor, vertexTensor),\
        (neighborTensor,kernelTensor,gradientTensor),\
        (neighborhoodTensor, filterTensor, integralTensor)

from torch_scatter import scatter

from joblib import Parallel, delayed

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

class cutlad(torch.autograd.Function):
    @staticmethod
    # @profile
    def forward(ctx, featureTensor, neighborhoodTensor, filterTensor, weightTensor):
        with record_function("cutlass forward step"): 
            ctx.save_for_backward(featureTensor, neighborhoodTensor, filterTensor, weightTensor)
            
            x_j = torch.index_select(featureTensor, 0, neighborhoodTensor[1])            
            indices = torch.arange(0,neighborhoodTensor.shape[1])
            
            batches = torch.split(indices, 32 * 1024)
            out = featureTensor.new_zeros((featureTensor.shape[0], weightTensor.shape[3]))
        
            for batch in batches:
                distributedFeatures = featureTensor[neighborhoodTensor[1,batch]]
                distributedOutputs = torch.einsum('ni, nuv, uvio -> no', distributedFeatures, filterTensor[batch], weightTensor)
                out = out + scatter(distributedOutputs, neighborhoodTensor[0,batch], dim=0, dim_size = featureTensor.shape[0], reduce='add')
            return out
    
    @staticmethod
    def backward(ctx, grad_output):
        featureTensor, neighborhoodTensor, filterTensor, weightTensor = ctx.saved_tensors
        
        x_j = torch.index_select(featureTensor, 0, neighborhoodTensor[1])    
        gradFeatures = torch.index_select(grad_output, 0, neighborhoodTensor[1])
        
        indices = torch.arange(0,neighborhoodTensor.shape[1])

        batches = torch.split(indices, 32 * 1024)

        featureGrad = None        
        weightGrad = None
        if ctx.needs_input_grad[0]:
            featureGrad = featureTensor.new_zeros(featureTensor.shape)
            transposedWeights = torch.transpose(weightTensor, 2, 3)
        if ctx.needs_input_grad[3]:        
            weightGrad = featureTensor.new_zeros(weightTensor.shape)
        
        for batch in batches:  
            if ctx.needs_input_grad[0]:          
                distributedOutputs = torch.einsum('nuv, uvio,ni -> no',
                                          filterTensor[batch], 
                                          transposedWeights, 
                                          gradFeatures[batch])
                featureGrad = featureGrad + scatter(distributedOutputs, neighborhoodTensor[1,batch], dim=0, dim_size = featureTensor.shape[0], reduce='add')

            if ctx.needs_input_grad[3]:        
                localGrad = torch.einsum('nuv, ni, no -> uvio', filterTensor[batch], x_j[batch], gradFeatures[batch])
                weightGrad = weightGrad + localGrad

        return featureGrad, None, None, weightGrad



convolutionOperator = cutlad.apply

from rbfConv import *

class meshLayer(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                dim = 2,
                size: Union[int, List[int]] = 3,
                periodic : Union[int, List[int]] = False,
                rbf : Union[int, List[int]] = 'chebyshev',
                dense_for_center: bool = False,
                bias: bool = False,
                initializer = torch.nn.init.xavier_normal_,
                activation = None,
                normalizedWeights = True,
                **kwargs):
        
        super().__init__(**kwargs)        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dim = dim
        self.periodic = periodic if isinstance(periodic, list) else repeat(periodic, dim)
        self.size = size if isinstance(size, list) else repeat(size, dim)
        self.rbfs = rbf if is_list_of_strings(rbf) else [rbf] * dim
        self.initializer = initializer
        self.activation = None if activation is None else getattr(nn.functional, activation)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # self.K = torch.tensor(self.size).prod().item()
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
         
        self.root_weight = dense_for_center
        if dense_for_center:
            self.lin = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer= 'uniform')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.weight.requires_grad = True
        if normalizedWeights:
            with torch.no_grad():
                cpuWeights = self.weight.detach().cpu()

                weightShape = self.weight.shape
                numFilters = weightShape[2] * weightShape[3]
            #             print(numFilters, weightShape)

                parRes = Parallel(n_jobs=-1)(delayed(optimizeWeights2D)(\
                                    weights = cpuWeights[:,:,idx % weightShape[2], idx // weightShape[2]],\
                                    basis = self.rbfs,
                                    periodicity = periodic,
                                    nmc = 2**16, targetIntegral = 1/self.weight.shape[2], \
            #                                 nmc = 32 * 1024, targetIntegral = 1, \
                                    windowFn = windowFn, verbose = False)\
                     for idx in range(numFilters))

                for idx in range(numFilters):
                    result, constr, fun, init, final = parRes[idx]
                    cpuWeights[:,:,idx % weightShape[2], idx // weightShape[2]] = result
                with torch.no_grad():
                    self.weight[:] = cpuWeights.to(self.weight.device)[:]
        # print(self.activation)

    def forward(self, featureTensor, neighborhoodTensor, filterTensor):
        out = convolutionOperator(featureTensor, neighborhoodTensor, filterTensor, self.weight)
        
        if featureTensor is not None and self.root_weight:
            out = out + self.lin(featureTensor)

        if self.bias is not None:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
            
        return out


class RbfNet(torch.nn.Module):
    def __init__(self, inputDimensions, outputDimensions, layerDescription, dropout = None):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        outFeatures = inputDimensions
        for layer in layerDescription['Layers']:
            inFeatures = layer['inFeatures'] if 'inFeatures' in layer else outFeatures
            outFeatures = layer['outFeatures'] if 'outFeatures' in layer else outputDimensions
            dimension = layer['dimension'] if 'dimension' in layer else 2
            size = int(layer['size'])
            rbf = layer['rbf']
            bias = layer['bias'] if 'bias' in layer else False
            centerLayer = layer['centerLayer'] if 'centerLayer' in layer else False
            periodic = layer['periodic'] if 'periodic' in layer else False
            activation = layer['activation'] if 'activation' in layer else None
#             batchSize = layer['batchSize'] if 'batchSize' in layer else [forwardBatch, backwardBatch]

            self.convs.append(meshLayer(
                in_channels = inFeatures, out_channels = outFeatures,
                dim = dimension, size = size,
                rbf = rbf, periodic = periodic,
                dense_for_center = centerLayer, bias = bias, activation = activation,
                normalizedWeights = False))
    
        self.linear = Linear(layerDescription['Layers'][-1]['outFeatures'], outputDimensions, bias=False,
                              weight_initializer= 'uniform')
            
        for layer in range(len(self.convs)):
            cpuWeights = self.convs[layer].weight.detach().cpu()

            weightShape = self.convs[layer].weight.shape
            numFilters = weightShape[2] * weightShape[3]
#             print(numFilters, weightShape)

            parRes = Parallel(n_jobs=-1)(delayed(optimizeWeights2D)(\
                                weights = cpuWeights[:,:,idx % weightShape[2], idx // weightShape[2]],\
                                basis = self.convs[layer].rbfs,
                                periodicity = self.convs[layer].periodic,
                                nmc = 32 * 1024, targetIntegral = 1/self.convs[layer].weight.shape[2], \
#                                 nmc = 32 * 1024, targetIntegral = 1, \
                                windowFn = windowFn, verbose = False)\
                 for idx in range(numFilters))

            for idx in range(numFilters):
                result, constr, fun, init, final = parRes[idx]
                cpuWeights[:,:,idx % weightShape[2], idx // weightShape[2]] = result
            with torch.no_grad():
                self.convs[layer].weight[:] = cpuWeights.to(self.convs[layer].weight.device)[:]
                
                
    def forward(self, featureTensor, neighborhoodTensor, filterTensor):
        ans = featureTensor
        
#         print(filterTensor.shape)
        for layer in self.convs:
            ans = layer(ans, neighborhoodTensor, filterTensor)
        return self.linear(ans)

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
from bqplot import *
import pandas as pd
from tqdm import trange, tqdm
import random
import warnings
import yaml
# %matplotlib notebook
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
from tqdm import tqdm

import argparse

from meshOps import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('-s','--lr_decay_step', type=int, default=1)
parser.add_argument('-r','--lr_decay_rate', type=float, default=0.99)
parser.add_argument('-l','--lr', type=float, default=1e-2)

parser.add_argument('-a', '--arch', type=str, default='16 32 32 16')
parser.add_argument('--activation', type=str, default='elu')
parser.add_argument('--initialSkip', type = int, default = 5000)
parser.add_argument('--cutoff', type=int, default=-1)
parser.add_argument('-e','--epochs', type=int, default=2)
parser.add_argument('-d','--stepSize', type = int, default = 100)


parser.add_argument('-i','--flowPath', type=str, default='~/foamData/uniformMesh.hdf5')
parser.add_argument('-m','--meshPath', type=str, default='mesh.hdf5')

parser.add_argument('-v','--verbose', type=bool, default=False)
args = parser.parse_args()

meshPath = args.meshPath
meshPath = os.path.expanduser(meshPath)

flowPath = args.flowPath
flowPath = os.path.expanduser(flowPath)
# basePath = args.basePath
# basePath = os.path.expanduser(basePath)

# meshFolder = basePath + '/constant/polyMesh/'

from pytorchSPH.solidBC import *

inFile = h5py.File(os.path.expanduser(flowPath), 'r')
frameCount = int(len(inFile.keys()) -1) # adjust for bptcls

print(frameCount)

last = list(inFile.keys())[-1]

print(inFile[last].keys())

cellPositions = np.array(inFile[last]['C'])[:,:2]
cellAreas = np.array(inFile[last]['V'])

# frames  = [int(f) for f in list(inFile.keys())][:-1]
frames  = [int(f) for f in list(inFile.keys())]
# frames = frames[::5]
# print(frames)



params, meshAttributes, integralAttributes, networkAttributes = loadMesh(meshPath)
basis,n,periodic = params
areaTensor, supportTensor, centerTensor, vertexTensor = meshAttributes
neighborTensor, kernelTensor, gradientTensor = integralAttributes
neighborhoodTensor, filterTensor, integralTensor = networkAttributes
# print(basis, n)

print(neighborhoodTensor.shape)

from joblib import Parallel, delayed
from rbfConv import *

mask_x = torch.logical_and(centerTensor[:,0] > -10, centerTensor[:,0] < 39)
mask = mask_x
# mask_y = torch.logical_and(centerTensor[:,1] > -9.5, centerTensor[:,1] < 9.5)
# mask = torch.logical_and(mask_x, mask_y).to('cuda')
# mask[:] = True

def loadFrame(frame):
#     print('loading frame %5d at %gs' %( frame, 0.01 * frame))
    s = '%05d' % frame
    grp = inFile[s]
    velocities = torch.from_numpy(np.array(grp['U'])[:,:2]).type(torch.float32)
    nut = torch.from_numpy(np.array(grp['nuTilda'])).type(torch.float32)

    step = 1

    s = '%05d' % (frame+step)
    grp = inFile[s]
    gt_velocity = torch.from_numpy(np.array(grp['U'])[:,:2]).type(torch.float32)
    gtTensor = (gt_velocity - velocities) / (step * 0.01)
    # gtTensor = gt_velocity

    def normalizeTensor(tensor):
        t = ((tensor - torch.mean(tensor,dim=0)[:,None]) ).reshape(tensor.shape)
        t = t / torch.max(torch.abs(t),dim=0)[0][:,None]
#         print(t.shape)
        return t

    featureTensor = torch.hstack((
        velocities
        , nut[:,None]\
        , areaTensor[:,None]\
        , supportTensor[:,None]\
        , normalizeTensor(kernelTensor[:,None])\
        , gradientTensor\
    ))
    return featureTensor, gtTensor

featureTensor, gtTensor = loadFrame(0)

widths = [8]
widths = args.arch.strip().split(' ')

layerDescription ='Layers:'

for i, w in enumerate(widths):
    win = featureTensor.shape[1] if i == 0 else widths[i-1]
    wout = gtTensor.shape[1] if i == len(widths) - 1 else widths[i]
    wout = widths[i]
    relu = 'placeholder' if i == len(widths) -1 else 'activation'
    layerDescription = layerDescription + f'''
    - inFeatures: {win}
      outFeatures: {wout}
      dimension: 2
      bias: False
      centerLayer: True
      periodic: False 
      size: {n}
      rbf: {basis.tolist()}
      {relu}: {args.activation}    '''
    
layerDescription = yaml.load(layerDescription, Loader = yaml.Loader)

# print(layerDescription)

model = RbfNet(featureTensor.shape[1],gtTensor.shape[1], layerDescription, dropout = 1.)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters', count_parameters(model))


lr = args.lr
# optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.95)
# optimizer.zero_grad()
# model.train()

trainingFrames = [int(f) for f in list(inFile.keys())]
trainingSet = trainingFrames[args.initialSkip::args.stepSize]
# trainingSet = [trainingSet[len(trainingSet)//2]]
print(len(trainingSet))

model = model.to('cuda')
model.train()

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.95)

# gtTensor_gpu = gtTensor.to('cuda')
# featureTensor_gpu = featureTensor.to('cuda')
neighborhoodTensor_gpu = neighborhoodTensor.to('cuda')
filterTensor_gpu = filterTensor.to('cuda')

featureTensor, gtTensor = loadFrame(1200)

output = model(featureTensor.to('cuda'), neighborhoodTensor_gpu, filterTensor_gpu)

from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
imageFolder = '/home/winchenbach/Desktop/' + timestamp
os.makedirs(imageFolder)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.95)
optimizer.zero_grad()

featureTensor, gtTensor = loadFrame(trainingSet[0])
output = model(featureTensor.to('cuda'), neighborhoodTensor_gpu, filterTensor_gpu)

# fig, axis = plt.subplots(2, 2, figsize=(9,6), sharex = True, sharey = True, squeeze = False)

# x = centerTensor[:,0]
# y = centerTensor[:,1]

# triplot = False

# def plot(axis, x,y,c, trip = True, label = None, s = 0.25):
#     if trip:
#         sc = axis.tripcolor(x[mask],y[mask],c[mask].detach().cpu().numpy())
#     else:
#         sc = axis.scatter(x[mask],y[mask],c = c[mask].detach().cpu().numpy(), s = s)
#     axis.axis('equal')
#     ax1_divider = make_axes_locatable(axis)
#     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
#     cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
#     cbar.ax.tick_params(labelsize=8) 
#     if label is not None:
#         axis.set_title(label)
#     return sc, cbar

# scFeatures, cbarFeatures = plot(axis[0,0], x, y, torch.linalg.norm(featureTensor[:,:2],axis=1), triplot, 'features',s=1.5)
# scgt, cbargt = plot(axis[1,0], x, y, torch.linalg.norm(gtTensor[:,:2],axis=1), triplot, 'gt',s=1.5)
# scpred, cbarpred = plot(axis[1,1], x, y, torch.linalg.norm(output[:,:2],axis=1), triplot, 'pred',s=1.5)
# scloss, cbarloss = plot(axis[0,1], x, y, torch.linalg.norm(output[:,:2] - gtTensor[:,:2].to('cuda'),axis=1), triplot, 'loss',s=1.5)

# fig.suptitle('Epoch %2d, Timestep %5d -> Loss %1.5e'%(0,0,0))

epochs = args.epochs
t = tqdm(range(epochs))
t2 = tqdm(range(len(trainingSet)))
losses = []
# fig.tight_layout()

frameIndex = 0

# fig.canvas.draw()
# fig.canvas.flush_events()
# imagePath = imageFolder + ('/%04d.png' % frameIndex)
# frameIndex = frameIndex + 1
# plt.savefig(imagePath, dpi = 200)

# frameIndex = 0
for e in t:   
    epochLoss = []
    t2.reset(total=len(trainingSet))
    i = 0
    for f in trainingSet:
        optimizer.zero_grad()
        featureTensor, gtTensor = loadFrame(f)
        
        output = model(featureTensor.to('cuda'), neighborhoodTensor_gpu, filterTensor_gpu)

        loss = output - gtTensor.to('cuda')
        loss = torch.sum(loss * loss, dim = 1)/2
        loss = loss[mask]


        lossTerm = torch.sum(loss)# + torch.std(loss)

        epochLoss.append([torch.mean(loss).cpu().item(),torch.sum(loss).cpu().item(),torch.max(loss).cpu().item(),torch.std(loss).cpu().item()])

    #     print(meanLoss)
        lossTerm.backward()
        optimizer.step()

        t2.set_description('%3d: Loss %1.5e %1.5e %1.5e' % (e, torch.mean(loss), torch.max(loss), torch.std(loss)))
        t2.update()
        
        
    epochLoss = np.array(epochLoss)
    t.set_description('%3d: Loss %1.5e %1.5e %1.5e' % (e, np.mean(epochLoss[:,0]), np.mean(epochLoss[:,2]), np.mean(epochLoss[:,3])))
    
    losses.append(epochLoss)
    
    if e % args.lr_decay_step == 0:
        lr = lr * args.lr_decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_decay_rate * param_group['lr']
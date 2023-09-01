# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time
import torch
from torch_geometric.loader import DataLoader
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

from rbfConv import RbfConv
# from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
# %matplotlib notebook
import copy

import time
import torch
from torch_geometric.loader import DataLoader
from tqdm.notebook import trange, tqdm
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

from rbfConv import RbfConv
from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))


import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

seed = 0


import random 
import numpy as np
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# print(torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('running on: ', device)
torch.set_num_threads(1)

from joblib import Parallel, delayed

from cutlass import *
from rbfConv import *
from tqdm.notebook import tqdm

from datautils import *
# from sphUtils import *
from lossFunctions import *



parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', type=str, default='/home/winchenbach/servus05/dev/torchSPH2/Cconv/paperData_ablation_windowFunctions/default - n=[ 4, 4] rbf=[rbf linear,rbf linear] map = cartesian window = cubicSpline d = 16 e = 25 arch 32 64 64 3 distance = 16 - 2023-07-30_00-52-51 seed 404868288')
parser.add_argument('-d','--data', type=str, default='~/servus05/dev/datasets/generative2D/test')
parser.add_argument('-v','--verbose', type= bool, default = False)

args = parser.parse_args()



from plotting import *
plt.style.use('dark_background')
# plt.style.use('default')
from tqdm.notebook import trange, tqdm
from datautils import *

# basePath = '~/servus05/dev/datasets/generative2D/train'
# basePath = '~/servus05/dev/datasets/generative2D/test'
# basePath = '/mnt/data/datasets/generative2D/train'
basePath = os.path.expanduser(args.data)

simulationFiles = [basePath + '/' + f for f in os.listdir(basePath) if f.endswith('.hdf5')]

if args.verbose:
    print('Input Testing files:')
    for i in range(len(simulationFiles)):
        print('\t[%2d] - %s' %(i, simulationFiles[i]))
    # debugPrint(simulationFiles)

from tqdm import tqdm

inputFile = args.input

with open("%s/results.json" % inputFile, "r") as read_file:
    decodedArray = json.load(read_file)    

if args.verbose:
    print('Reading from input file ', inputFile)
    print('Training contains epochs: ', decodedArray['epochData'].keys())
    print('Epochs consist of ', len(decodedArray['epochData']['001']['training']), 'iterations')
    print('Network Hyperparameters:')
    for k in decodedArray['hyperParameters']:
        print('\t%25s = %25s' % (k, decodedArray['hyperParameters'][k]))

import pandas as pd

dataSet = pd.DataFrame()
counter = 0
if args.verbose:
    print('Processing training data...')
    for epoch in tqdm(decodedArray['epochData']):
        trainingData = decodedArray['epochData'][epoch]['training']
        for it in tqdm(range(len(trainingData)), leave = False):
            data = np.array(trainingData[it])
            stepLosses = np.mean(data, axis = 1)
            dataFrame = pd.DataFrame([{
                'rbf_x' : decodedArray['hyperParameters']['rbf_x'], 
                'rbf_y' : decodedArray['hyperParameters']['rbf_y'], 
                'n'     : decodedArray['hyperParameters']['n'], 
                'm'     : decodedArray['hyperParameters']['m'],
                'window': decodedArray['hyperParameters']['windowFunction'],
                'map'   : decodedArray['hyperParameters']['coordinateMapping'],
                'seed'  : decodedArray['hyperParameters']['networkSeed'],
                'arch'  : decodedArray['hyperParameters']['arch'],
                'epoch' : int(epoch),
                'epochIteration' : it,
                'iteration': counter + 1,
                'firstStepLoss': stepLosses[0],
                'lastStepLoss': stepLosses[-1],
                'meanLoss': np.mean(stepLosses)
                                    }])
            counter = counter + 1
            dataSet = pd.concat([dataSet, dataFrame], ignore_index = True)
    print('Training data processed, writing to file "%s - training.csv"' % inputFile)
else:    
    for epoch in decodedArray['epochData']:
        trainingData = decodedArray['epochData'][epoch]['training']
        for it in range(len(trainingData)):
            data = np.array(trainingData[it])
            stepLosses = np.mean(data, axis = 1)
            dataFrame = pd.DataFrame([{
                'rbf_x' : decodedArray['hyperParameters']['rbf_x'], 
                'rbf_y' : decodedArray['hyperParameters']['rbf_y'], 
                'n'     : decodedArray['hyperParameters']['n'], 
                'm'     : decodedArray['hyperParameters']['m'],
                'window': decodedArray['hyperParameters']['windowFunction'],
                'map'   : decodedArray['hyperParameters']['coordinateMapping'],
                'seed'  : decodedArray['hyperParameters']['networkSeed'],
                'arch'  : decodedArray['hyperParameters']['arch'],
                'epoch' : int(epoch),
                'epochIteration' : it,
                'iteration': counter + 1,
                'firstStepLoss': stepLosses[0],
                'lastStepLoss': stepLosses[-1],
                'meanLoss': np.mean(stepLosses)
                                    }])
            counter = counter + 1
            dataSet = pd.concat([dataSet, dataFrame], ignore_index = True)

dataSet.to_csv('%s - training.csv' % inputFile)
from rbfConv import *
from datautils import *
from rbfNet import *


def loadRbfModel(file, frame, networkPath, epoch):
    with open(os.path.expanduser("%s/results.json" % networkPath), "r") as read_file:
        decodedArray = json.load(read_file)
        dataDict = decodedArray
    
    n = dataDict['hyperParameters']['n']
    m = dataDict['hyperParameters']['m']
    coordinateMapping = dataDict['hyperParameters']['coordinateMapping']
    windowFn = getWindowFunction(dataDict['hyperParameters']['windowFunction'])
    rbf_x = dataDict['hyperParameters']['rbf_x']
    rbf_y = dataDict['hyperParameters']['rbf_y']
    dist = dataDict['hyperParameters']['frameDistance']
    unroll = dataDict['hyperParameters']['maxRollOut']
    arch = [32, 64, 64, 2]
    arch = [16, 32, 32, 2]
    arch = dataDict['hyperParameters']['arch']
    arch = [int(a) for a in arch.split(' ') if a != '']
    epochs = dataDict['hyperParameters']['epochs']
#     print(arch)
#     print(n, m)
#     print(dataDict['hyperParameters']['windowFunction'])
#     print(rbf_x)
#     print(rbf_y)

    attributes, inputData, groundTruthData = loadFrame(file, frame, 1 + np.arange(unroll), dist)
    inputData['fluidGravity'] = inputData['fluidGravity'][:,:2]
    
    
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(attributes, inputData)
#     print(fluidFeatures.shape)
#     print(boundaryFeatures.shape)
    if 'network' not in dataDict['hyperParameters']:
        model = RbfNet(fluidFeatures.shape[1],boundaryFeatures.shape[1], layers = arch, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 32, )
    else:
        network = dataDict['hyperParameters']['network']
        model = None
        if network == 'default':
            model = RbfNet(fluidFeatures.shape[1],boundaryFeatures.shape[1], layers = arch, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 32, normalized = False )
        if network == 'denormalized':
            model = RbfNet(fluidFeatures.shape[1],boundaryFeatures.shape[1], layers = arch, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 32, normalized = False)
        if network == 'split':
            model = RbfSplitNet(fluidFeatures.shape[1],boundaryFeatures.shape[1], layers = arch, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 32, )
        if network == 'interleaved':
            model = RbfInterleaveNet(fluidFeatures.shape[1],boundaryFeatures.shape[1], layers = arch, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 32, )
        if network == 'input':
            model = RbfInputNet(fluidFeatures.shape[1],boundaryFeatures.shape[1], layers = arch, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 32, )
        if network == 'output':
            model = RbfOutputNet(fluidFeatures.shape[1],boundaryFeatures.shape[1], layers = arch, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 32, )


    model.load_state_dict(torch.load(os.path.expanduser('%s/model_%03d.torch' % (networkPath, epoch if epoch >= 0 else epochs - 1))))
    model = model.to(device)
    model.train(False)
    return model, dataDict['hyperParameters']

@torch.jit.script
def wendland(q, h):
    C = 7 / np.pi
    b1 = torch.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**2    
@torch.jit.script
def wendlandGrad(q,r,h):
    C = 7 / np.pi    
    return - r * C / h**3 * (20. * q * (1. -q)**3)[:,None]
    
def getParticleQuantities(positions, bPositions, area, boundaryArea, attributes):
    # bPositions = boundaryPositions.to(device)
    # area = inputData['fluidArea'].to(device)
    # boundaryArea = inputData['boundaryArea'].to(device)

    fi, fj = radius(positions, positions, attributes['support'], max_num_neighbors = 256, batch_x = None, batch_y = None)
    bf, bb = radius(bPositions, positions, attributes['support'], max_num_neighbors = 256, batch_x = None, batch_y = None)
    bi, bj = radius(bPositions, bPositions, attributes['support'], max_num_neighbors = 256, batch_x = None, batch_y = None)

    i, ni = torch.unique(fi, return_counts = True)
    b, nb = torch.unique(bf, return_counts = True)

    fluidNeighbors = torch.stack([fi, fj], dim = 0)
    fluidDistances = (positions[fi] - positions[fj])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)
    fluidDistances[fluidRadialDistances < 1e-5,:] = 0
    fluidDistances[fluidRadialDistances >= 1e-5,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-5,None]
    fluidRadialDistances /= attributes['support']

    boundaryNeighbors = torch.stack([bf, bb], dim = 0)
    boundaryDistances = (positions[bf] - bPositions[bb])
    boundaryRadialDistances = torch.linalg.norm(boundaryDistances,axis=1)
    boundaryDistances[boundaryRadialDistances < 1e-5,:] = 0
    boundaryDistances[boundaryRadialDistances >= 1e-5,:] /= boundaryRadialDistances[boundaryRadialDistances >= 1e-5,None]
    boundaryRadialDistances /= attributes['support']

    boundaryBoundaryNeighbors = torch.stack([bi, bj], dim = 0)
    boundaryBoundaryDistances = (bPositions[bi] - bPositions[bj])
    boundaryBoundaryRadialDistances = torch.linalg.norm(boundaryBoundaryDistances,axis=1)
    boundaryBoundaryDistances[boundaryBoundaryRadialDistances < 1e-5,:] = 0
    boundaryBoundaryDistances[boundaryBoundaryRadialDistances >= 1e-5,:] /= boundaryBoundaryRadialDistances[boundaryBoundaryRadialDistances >= 1e-5,None]
    boundaryBoundaryRadialDistances /= attributes['support']
    density = scatter(area[fj] * wendland(fluidRadialDistances, attributes['support']), fi, dim=0, dim_size = positions.shape[0], reduce='add') + \
        scatter(boundaryArea[bb] * wendland(boundaryRadialDistances, attributes['support']), bf, dim=0, dim_size = positions.shape[0], reduce='add')
#     densityLoss = (density - groundTruths[0][:,4].to(device))**2

    boundaryDensity = scatter(area[bf] * wendland(boundaryRadialDistances, attributes['support']), bb, dim=0, dim_size = bPositions.shape[0], reduce='add') + \
            scatter(boundaryArea[bj] * wendland(boundaryBoundaryRadialDistances, attributes['support']), bi, dim=0, dim_size = bPositions.shape[0], reduce='add')
    # print(boundaryDensity)
    colorField = scatter(area[fj]/density[fj] * wendland(fluidRadialDistances, attributes['support']), fi, dim=0, dim_size = positions.shape[0], reduce='add') + \
        scatter(boundaryArea[bb]/boundaryDensity[bb] * wendland(boundaryRadialDistances, attributes['support']), bf, dim=0, dim_size = positions.shape[0], reduce='add')
    boundaryColorField = scatter(area[bf]/density[bf] * wendland(boundaryRadialDistances, attributes['support']), bb, dim=0, dim_size = bPositions.shape[0], reduce='add') + \
        scatter(boundaryArea[bi]/boundaryDensity[bi] * wendland(boundaryBoundaryRadialDistances, attributes['support']), bj, dim=0, dim_size = bPositions.shape[0], reduce='add')

    fluidGrad = wendlandGrad(fluidRadialDistances, fluidDistances, attributes['support'])
    boundaryGrad = wendlandGrad(boundaryRadialDistances, boundaryDistances, attributes['support'])
    colorGrad = scatter((area[fj]/density[fj] * (colorField[fj] - colorField[fi]))[:,None] * fluidGrad, fi, dim=0, dim_size = positions.shape[0], reduce='add') + \
            scatter((boundaryArea[bb]/boundaryDensity[bb] * (boundaryColorField[bb] - colorField[bf]))[:,None] * boundaryGrad, bf, dim=0, dim_size = positions.shape[0], reduce='add')

    return density, colorField, colorGrad
def getMeshQuantities(xFluid, bPositions, area, boundaryArea, density, velocity, attributes, n = 512, supportScale = 1):
    x = torch.linspace(-1,1,n, device = xFluid.device, dtype = xFluid.dtype)
    y = torch.linspace(-1,1,n, device = xFluid.device, dtype = xFluid.dtype)
    xx,yy = torch.meshgrid(x,y, indexing='xy')

    xxf = xx.flatten()
    yyf = yy.flatten()
    positions = torch.vstack((xxf, yyf)).mT
    z = yyf
    # z = torch.linalg.norm(positions,dim=1).reshape(xx.shape)
#     bPositions = boundaryPositions.to(device)
#     area = inputData['fluidArea'].to(device)
#     boundaryArea = inputData['boundaryArea'].to(device)

    xm, xf = radius(xFluid, positions, attributes['support'] * supportScale, max_num_neighbors = 256, batch_x = None, batch_y = None)


    # fluidNeighbors = torch.stack([xf, xm], dim = 0)
    fluidDistances = (positions[xm] - xFluid[xf])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)
    fluidDistances[fluidRadialDistances < 1e-5,:] = 0
    fluidDistances[fluidRadialDistances >= 1e-5,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-5,None]
    fluidRadialDistances /= attributes['support'] * supportScale


    bm, bb = radius(bPositions, positions, attributes['support'] * supportScale, max_num_neighbors = 256, batch_x = None, batch_y = None)


    # fluidNeighbors = torch.stack([xf, xm], dim = 0)
    boundaryDistances = (positions[bm] - bPositions[bb])
    boundaryRadialDistances = torch.linalg.norm(boundaryDistances,axis=1)
    boundaryDistances[boundaryRadialDistances < 1e-5,:] = 0
    boundaryDistances[boundaryRadialDistances >= 1e-5,:] /= boundaryRadialDistances[boundaryRadialDistances >= 1e-5,None]
    boundaryRadialDistances /= attributes['support'] * supportScale
    
    meshDensity = scatter(area[xf] * wendland(fluidRadialDistances, attributes['support'] * supportScale), xm, dim=0, dim_size = positions.shape[0], reduce = 'add') + \
    scatter(boundaryArea[bb] * wendland(boundaryRadialDistances, attributes['support'] * supportScale), bm, dim=0, dim_size = positions.shape[0], reduce = 'add')
    
    meshVelocity = scatter((area[xf] / density[xf] * wendland(fluidRadialDistances, attributes['support'] * supportScale))[:,None] * velocity[xf], xm, dim=0, dim_size = positions.shape[0], reduce = 'add')

    meshDivergence = scatter(area[xf] / density[xf] * torch.einsum('nd, nd -> n', (wendlandGrad(fluidRadialDistances, fluidDistances, attributes['support'] * supportScale)), velocity[xf] - meshVelocity[xm]), xm, dim=0, dim_size = positions.shape[0], reduce = 'add')
    
    return positions.reshape((xx.shape[0],xx.shape[1],2)), meshDensity.reshape((xx.shape[0],xx.shape[1])), meshVelocity.reshape((xx.shape[0],xx.shape[1],2)), meshDivergence.reshape((xx.shape[0],xx.shape[1]))

def plotMeshData(fp, meshDensity, gtMeshDensity, meshVelocity, gtMeshVelocity, meshDivergence, gtMeshDivergence):
    fig, axis = plt.subplots(3, 5, figsize=(16,12), sharex = False, sharey = False, squeeze = False)
    meshPlot(axis[0,0], fp, meshDensity, 'Pred Density')
    meshPlot(axis[0,1], fp, meshVelocity[:,:,0], 'Pred Velocity.x')
    meshPlot(axis[0,2], fp, meshVelocity[:,:,1], 'Pred Velocity.y')
    meshPlot(axis[0,3], fp, torch.linalg.norm(meshVelocity, dim=2), '|Pred Velocity|')
    meshPlot(axis[0,4], fp, meshDivergence, 'Pred Divergence')

    meshPlot(axis[1,0], fp, gtMeshDensity, 'GT Density')
    meshPlot(axis[1,1], fp, gtMeshVelocity[:,:,0], 'GT Velocity.x')
    meshPlot(axis[1,2], fp, gtMeshVelocity[:,:,1], 'GT Velocity.y')
    meshPlot(axis[1,3], fp, torch.linalg.norm(gtMeshVelocity, dim=2), '|GT Velocity|')
    meshPlot(axis[1,4], fp, gtMeshDivergence, 'GT Divergence')

    meshPlot(axis[2,0], fp, meshDensity - gtMeshDensity, 'Diff Density')
    meshPlot(axis[2,1], fp, meshVelocity[:,:,0] - gtMeshVelocity[:,:,0], 'Diff Velocity.x')
    meshPlot(axis[2,2], fp, meshVelocity[:,:,1] - gtMeshVelocity[:,:,1], 'Diff Velocity.y')
    meshPlot(axis[2,3], fp, torch.linalg.norm(meshVelocity - gtMeshVelocity, dim=2), '|Diff Velocity|')
    meshPlot(axis[2,4], fp, meshDivergence - gtMeshDivergence, 'Diff Divergence')

    fig.tight_layout()
    

def meshPlot(ax, positions, data, title = None):
    ax.axis('equal')
#     print(positions[:,:,0].flatten().shape)
#     print(data.shape)
    im = ax.pcolormesh(positions[:,:,0].detach().cpu().numpy(), positions[:,:,1].detach().cpu().numpy(), data.detach().cpu().numpy())    
#     ax.scatter(boundaryPositions[:,0], boundaryPositions[:,1], s=1,c='white',alpha=0.5)
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax1_divider = make_axes_locatable(ax)
    cax1 = ax1_divider.append_axes("bottom", size="7%", pad="2%")
    GTcbar = fig.colorbar(im, cax=cax1,orientation='horizontal')
    GTcbar.ax.tick_params(labelsize=8) 
    if title is not None:
        ax.set_title(title)
    return im, GTcbar
def analyzeParticles(positions, gtPositions, boundaryPositions, velocities, gtVelocities, fluidArea, boundaryArea, attributes):    
    positionLoss = torch.linalg.norm(positions - gtPositions.to(device), dim=1)
    velocityLoss = torch.linalg.norm(velocities - gtVelocities.to(device), dim=1)
    density, colorField, colorGrad = getParticleQuantities(positions, boundaryPositions, fluidArea, boundaryArea, attributes)
    
    gTPositionLoss = torch.linalg.norm(gtPositions - gtPositions, dim=1)
    gtVelocityLoss = torch.linalg.norm(gtVelocities - gtVelocities, dim=1)
    gtDensity2, gtColorField, gtColorGrad = getParticleQuantities(gtPositions, boundaryPositions, fluidArea, boundaryArea, attributes)
    
    return positionLoss, velocityLoss, density, gtDensity2, colorField, gtColorField, colorGrad, gtColorGrad
    
def particleAnalysis(positions, gtPositions, boundaryPositions, velocities, gtVelocities, fluidArea, boundaryArea, attributes, plot = True):
    positionLoss, velocityLoss, density, gtDensity, colorField, gtColorField, colorGrad, gtColorGrad = analyzeParticles(positions, gtPositions, boundaryPositions, velocities, gtVelocities, fluidArea, boundaryArea, attributes)
    if plot:
        fig, axis = plt.subplots(3, 6, figsize=(12,8*1.09), sharex = False, sharey = False, squeeze = False)

        scatterPlot(axis[0,0], positions, positionLoss, boundaryPositions, 'Position Loss')
        scatterPlot(axis[0,1], positions, velocityLoss, boundaryPositions, 'Velocity Loss')
        scatterPlot(axis[0,2], positions, density, boundaryPositions, 'Density')
        scatterPlot(axis[0,3], positions, colorField, boundaryPositions, 'Color')
        scatterPlot(axis[0,4], positions, colorGrad[:,0], boundaryPositions, 'Grad.x Color')
        scatterPlot(axis[0,5], positions, colorGrad[:,1], boundaryPositions, 'Grad.y Color')

        scatterPlot(axis[1,0], gtPositions, torch.zeros_like(density), boundaryPositions, 'GT Position Loss')
        scatterPlot(axis[1,1], gtPositions, torch.zeros_like(density), boundaryPositions, 'GT Velocity Loss')
        scatterPlot(axis[1,2], gtPositions, gtDensity, boundaryPositions, 'GT Density')
        scatterPlot(axis[1,3], gtPositions, gtColorField, boundaryPositions, 'GT Color')
        scatterPlot(axis[1,4], gtPositions, gtColorGrad[:,0], boundaryPositions, 'GT Grad.x Color')
        scatterPlot(axis[1,5], gtPositions, gtColorGrad[:,1], boundaryPositions, 'GT Grad.y Color')

        scatterPlot(axis[2,0], gtPositions, positionLoss, boundaryPositions, 'Position Loss')
        scatterPlot(axis[2,1], gtPositions, velocityLoss, boundaryPositions, 'Velocity Loss')
        scatterPlot(axis[2,2], gtPositions, density - gtDensity, boundaryPositions, 'Diff Density')
        scatterPlot(axis[2,3], gtPositions, colorField - gtColorField, boundaryPositions, 'Diff Color')
        scatterPlot(axis[2,4], gtPositions, colorGrad[:,0] - gtColorGrad[:,0], boundaryPositions, 'Diff Grad.x Color')
        scatterPlot(axis[2,5], gtPositions, colorGrad[:,1] - gtColorGrad[:,1], boundaryPositions, 'Diff Grad.y Color')

        fig.tight_layout()
    return positionLoss, velocityLoss, density, gtDensity, colorField, gtColorField, colorGrad, gtColorGrad
import scipy.stats as stats

def getPSD(data):
    image = data.detach().cpu().numpy()
    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(npix) * npix

    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.01, npix//(2*np.pi)+1, 0.5)
    kbins = np.arange(0.5, npix//2+1, 1.)

    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)

    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins

def analyzeMesh(data, filterLevels = 0):
# data = torch.linalg.norm(meshVelocity,dim=1).reshape(xx.shape)
    if filterLevels > 0 :
        filtered = scipy.ndimage.gaussian_filter(data.detach().cpu().numpy(),filterLevels)
        filtered = torch.tensor(filtered, device = data.device, dtype = data.dtype)
    else:
        filtered = data
    # data = torch.linalg.norm(meshVelocity,dim=1)
    fft = torch.fft.fftshift(torch.fft.fft2(filtered))
    # fft = torch.fft.fft2(torch.linalg.norm(meshVelocity,dim=1).reshape(xx.shape))
    fftfreq = torch.fft.fftshift(torch.fft.fftfreq(filtered.shape[0],2 / filtered.shape[0]))
    # fftfreq = (torch.fft.fftfreq(xx.shape[0],2 / xx.shape[0]))
    fx, ft = torch.meshgrid(fftfreq, fftfreq, indexing = 'xy')
    # print(fx.shape)
    # print(ft.shape)
    fp = torch.vstack((fx.flatten(),ft.flatten())).mT

    return fp, fft, getPSD(filtered), filtered
def meshAnalysis(positions, data, gtData,plot = True, linThresh = 1e-2, linScale = 1):
    fp, fft, psd, filtered = analyzeMesh(data)
    fp, gtFft, gtPsd, gtFiltered = analyzeMesh(gtData)
    if plot:
        fig, axis = plt.subplots(3, 4, figsize=(16,11*1.09), sharex = False, sharey = False, squeeze = False)
        meshPlot(axis[0,0], positions, data, 'Original Prediction')
        meshPlot(axis[0,1], positions, filtered, 'Filtered Prediction')
        ax = axis[0,2]
        ax.axis('equal')
        ax.set_title('FFT Prediction')
        im = ax.pcolormesh(fp[:,0].reshape(data.shape).detach().cpu().numpy(), fp[:,1].reshape(data.shape).detach().cpu().numpy(), torch.real(fft).reshape(data.shape).detach().cpu().numpy(),
                          norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=-torch.max(torch.abs(torch.real(fft))), vmax=torch.max(torch.abs(torch.real(fft))), base=10), cmap = 'twilight')    
        ax.set_xscale('symlog', linthresh=linThresh, linscale = linScale, subs = [1, 2, 3, 4, 5, 6, 7, 8, 9])  
        ax.set_yscale('symlog', linthresh=linThresh, linscale = linScale, subs = [1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax1_divider = make_axes_locatable(ax)
        cax1 = ax1_divider.append_axes("bottom", size="5%", pad="15%")
        GTcbar = fig.colorbar(im, cax=cax1,orientation='horizontal')
        GTcbar.ax.tick_params(labelsize=8) 
        kvals,Abins = psd
        axis[0,3].set_title('PSD Prediction')
        axis[0,3].loglog(kvals, Abins)
        axis[0,3].set_xlabel("$k$")
        axis[0,3].set_ylabel("$P(k)$")
        # axis[0,0].legend()

        meshPlot(axis[1,0], positions, gtData, 'Original Groundtruth')
        meshPlot(axis[1,1], positions, gtFiltered, 'Filtered Groundtruth')
        ax = axis[1,2]
        ax.axis('equal')
        ax.set_title('FFT Groundtruth')
        im = ax.pcolormesh(fp[:,0].reshape(data.shape).detach().cpu().numpy(), fp[:,1].reshape(data.shape).detach().cpu().numpy(), torch.real(gtFft).reshape(data.shape).detach().cpu().numpy(),
                          norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=-torch.max(torch.abs(torch.real(gtFft))), vmax=torch.max(torch.abs(torch.real(gtFft))), base=10), cmap = 'twilight')     
        ax.set_xscale('symlog', linthresh=linThresh, linscale = linScale, subs = [1, 2, 3, 4, 5, 6, 7, 8, 9])  
        ax.set_yscale('symlog', linthresh=linThresh, linscale = linScale, subs = [1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax1_divider = make_axes_locatable(ax)
        cax1 = ax1_divider.append_axes("bottom", size="5%", pad="15%")
        GTcbar = fig.colorbar(im, cax=cax1,orientation='horizontal')
        GTcbar.ax.tick_params(labelsize=8) 
        kvals,Abins = gtPsd
        axis[1,3].set_title('PSD Groundtruth')
        axis[1,3].loglog(kvals, Abins)
        axis[1,3].set_xlabel("$k$")
        axis[1,3].set_ylabel("$P(k)$")

        meshPlot(axis[2,0], positions, data - gtData, 'Original Difference')
        meshPlot(axis[2,1], positions, filtered - gtFiltered, 'Filtered Difference')
        ax = axis[2,2]
        ax.axis('equal')
        ax.set_title('FFT Difference')
        diff = (torch.real(fft) - torch.real(gtFft))
        im = ax.pcolormesh(fp[:,0].reshape(data.shape).detach().cpu().numpy(), fp[:,1].reshape(data.shape).detach().cpu().numpy(), diff.reshape(data.shape).detach().cpu().numpy(),
                          norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=-torch.max(torch.abs(diff)), vmax=torch.max(torch.abs(diff)), base=10), cmap = 'twilight')     
        ax.set_xscale('symlog', linthresh=linThresh, linscale = linScale, subs = [1, 2, 3, 4, 5, 6, 7, 8, 9])  
        ax.set_yscale('symlog', linthresh=linThresh, linscale = linScale, subs = [1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax1_divider = make_axes_locatable(ax)
        cax1 = ax1_divider.append_axes("bottom", size="5%", pad="15%")
        GTcbar = fig.colorbar(im, cax=cax1,orientation='horizontal')
        GTcbar.ax.tick_params(labelsize=8) 
        kvals,Abins = gtPsd
        axis[2,3].set_title('PSD')
        axis[2,3].loglog(kvals, Abins,label = 'GT')
        kvals,Abins = psd
        axis[2,3].loglog(kvals, Abins,label = 'pred')
        axis[2,3].set_xlabel("$k$")
        axis[2,3].set_ylabel("$P(k)$")
        axis[2,3].legend()

        fig.tight_layout()
    
    return fp, fft, gtFft, psd, gtPsd
def getLosses(positions, velocities, gtPositions, gtDensity, gtVelocities, fluidArea, boundaryPositions, boundaryArea, attributes):
    positionLoss, velocityLoss, density, gtDensity, colorField, gtColorField, colorGrad, gtColorGrad = particleAnalysis(positions, gtPositions, boundaryPositions, velocities, gtVelocities, fluidArea, boundaryArea, attributes, plot = False)
    fp, meshDensity, meshVelocity, meshDivergence = getMeshQuantities(positions, boundaryPositions, fluidArea, boundaryArea, density, velocities, attributes, n = 512)
    fp, gtMeshDensity, gtMeshVelocity, gtMeshDivergence = getMeshQuantities(gtPositions, boundaryPositions, fluidArea, boundaryArea, gtDensity, gtVelocities, attributes, n = 512)
    # plotMeshData(fp, meshDensity, gtMeshDensity, meshVelocity, gtMeshVelocity, meshDivergence, gtMeshDivergence)

    data = torch.linalg.norm(meshVelocity,dim=2)
    gtData = torch.linalg.norm(gtMeshVelocity,dim=2)
    ffp, fft, gtFft, psd, gtPsd = meshAnalysis(fp, data, gtData,linThresh=1e0, linScale = 0.5, plot = False)
    kvals,Abins = psd
    kvals,gtAbins = gtPsd

    positionLossTerm = torch.mean(positionLoss)
    velocityLossTerm = torch.mean(velocityLoss)
    densityLossTerm = torch.mean(torch.abs(density - gtDensity))
    colorFieldLossTerm = torch.mean(torch.abs(colorField - gtColorField))
    colorGradLossTerm = torch.mean(torch.linalg.norm(colorGrad - gtColorGrad,dim=1))

    meshDensityLossTerm = torch.mean(torch.abs(meshDensity - gtMeshDensity))
    meshVelocityLossTerm = torch.mean(torch.abs(meshVelocity - gtMeshVelocity))
    meshDivergenceLossTerm = torch.mean(torch.abs(meshDivergence - gtMeshDivergence))

    psdLossTerm = np.sum(np.abs(np.log10(Abins) - np.log10(gtAbins)))
    
    return {\
            'position': positionLossTerm.item(), 
            'velocity': velocityLossTerm.item(), 
            'density': densityLossTerm.item(), 
            'color': colorFieldLossTerm.item(), 
            'colorGrad': colorGradLossTerm.item(), 
            'meshDensity': meshDensityLossTerm.item(), 
            'meshVelocity': meshVelocityLossTerm.item(), 
            'meshDivergence': meshDivergenceLossTerm.item(), 
            'psd': psdLossTerm.item(), 
            'kvals': kvals, 
            'Abins': Abins, 
            'gtAbins':gtAbins}
# lossDict = getLosses(positions, velocities, gtPositions, gtDensity, fluidArea, boundaryPositions, boundaryArea, attributes)

def loadFrame(filename, frame, frameOffsets = [1], frameDistance = 1, adjustForFrameDistance = True):
    if 'zst' in filename:
        return loadFrameZSTD(filename, frame, frameOffsets, frameDistance)
    inFile = h5py.File(filename)
    inGrp = inFile['simulationExport']['%05d' % frame]
#     debugPrint(inFile.attrs.keys())
    attributes = {
     'support': np.max(inGrp['fluidSupport'][:]),
     'targetNeighbors': inFile.attrs['targetNeighbors'],
     'restDensity': inFile.attrs['restDensity'],
     'dt': inGrp.attrs['dt'],
     'time': inGrp.attrs['time'],
     'radius': inFile.attrs['radius'],
     'area': inFile.attrs['radius'] **2 * np.pi,
    }
#     debugPrint(inGrp.attrs['timestep'])

    # support = inFile.attrs['support']
    # targetNeighbors = inFile.attrs['targetNeighbors']
    # restDensity = inFile.attrs['restDensity']
    # dt = inFile.attrs['initialDt']

    inputData = {
        'fluidPosition': torch.from_numpy(inGrp['fluidPosition'][:]).type(torch.float32),
        'fluidVelocity': torch.from_numpy(inGrp['fluidVelocity'][:]).type(torch.float32),
        'fluidArea' : torch.from_numpy(inGrp['fluidArea'][:]).type(torch.float32),
        'fluidDensity' : torch.from_numpy(inGrp['fluidDensity'][:]).type(torch.float32),
        'fluidSupport' : torch.from_numpy(inGrp['fluidSupport'][:]).type(torch.float32),
        'fluidGravity' : torch.from_numpy(inGrp['fluidGravity'][:]).type(torch.float32) if 'fluidGravity' not in inFile.attrs else torch.from_numpy(inFile.attrs['fluidGravity']).type(torch.float32) * torch.ones(inGrp['fluidDensity'][:].shape[0])[:,None],
        'boundaryPosition': torch.from_numpy(inFile['boundaryInformation']['boundaryPosition'][:]).type(torch.float32),
        'boundaryNormal': torch.from_numpy(inFile['boundaryInformation']['boundaryNormals'][:]).type(torch.float32),
        'boundaryArea': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).type(torch.float32),
        'boundaryVelocity': torch.from_numpy(inFile['boundaryInformation']['boundaryVelocity'][:]).type(torch.float32),
        'boundaryDensity': torch.from_numpy(inGrp['boundaryDensity'][:]).type(torch.float32)
    }
    if adjustForFrameDistance:
        if frame >= frameDistance:
            priorGrp = inFile['simulationExport']['%05d' % (frame - frameDistance)]
            priorPosition = torch.from_numpy(priorGrp['fluidPosition'][:]).type(torch.float32)
            inputData['fluidVelocity'] = (inputData['fluidPosition'] - priorPosition) / (frameDistance * attributes['dt'])
            # priorVelocity = torch.from_numpy(priorGrp['fluidVelocity'][:]).type(torch.float32)

    groundTruthData = []
    for i in frameOffsets:
        gtGrp = inFile['simulationExport']['%05d' % (frame + i * frameDistance)]
#         debugPrint((frame + i * frameDistance))
#         debugPrint(gtGrp.attrs['timestep'])
        gtData = {
            'fluidPosition'    : torch.from_numpy(gtGrp['fluidPosition'][:]).type(torch.float32),
            'fluidVelocity'    : torch.from_numpy(gtGrp['fluidVelocity'][:]).type(torch.float32),
            'fluidDensity'     : torch.from_numpy(gtGrp['fluidDensity'][:]).type(torch.float32),
    #         'fluidPressure'    : torch.from_numpy(gtGrp['fluidPressure'][:]),
    #         'boundaryDensity'  : torch.from_numpy(gtGrp['fluidDensity'][:]),
    #         'boundaryPressure' : torch.from_numpy(gtGrp['fluidPressure'][:]),
        }
        
        groundTruthData.append(torch.hstack((gtData['fluidPosition'].type(torch.float32), gtData['fluidVelocity'], gtData['fluidDensity'][:,None])))
        
    
    inFile.close()
    
    return attributes, inputData, groundTruthData
def getUnrollFrame(simulationIndex, initialFrame, unrollSteps, verbose = False):
    # if verbose:
        # print('Loading from file %s - frame %d ' %(simulationFiles[simulationIndex], initialFrame))
    attributes, inputData, groundTruths = loadFrame(simulationFiles[simulationIndex], initialFrame, 1 + np.arange(unrollSteps), hyperParams['frameDistance'], adjustForFrameDistance = False)
    # if verbose:
        # print('Constructing fluid features')
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(attributes, inputData)
    fluidFeatures = fluidFeatures.to(device)
    # model, hyperParams = loadRbfModel(simulationFiles[0], 0, subfolders[networkIndex], -1)
    predictedPositions = fluidPositions.to(device)
    predictedVelocity = inputData['fluidVelocity'].to(device)
    gravity = inputData['fluidGravity'][:,:2].to(device) 
    losses = []
    with torch.no_grad():
        if verbose:
            # print('Unrolling through network...')
            for unrollStep in tqdm(range(unrollSteps), leave = False):
                _, _, groundTruths2 = loadFrame(simulationFiles[simulationIndex], initialFrame + unrollStep * hyperParams['frameDistance'], 1 + np.arange(hyperParams['frameDistance']), 1)
                loss, predictedPositions, predictedVelocity = runNetwork(predictedPositions, predictedVelocity, attributes, hyperParams['frameDistance'], gravity, fluidFeatures, boundaryPositions.to(device), boundaryFeatures.to(device), groundTruths[unrollStep], model, None, None, True)    
                positions = predictedPositions
                velocities = predictedVelocity
                gtPositions = groundTruths[unrollStep][:,:2].to(device)
                gtVelocity = torch.vstack([g[:,2:4][None,:] for g in groundTruths2])
                gtVelocities = torch.mean(gtVelocity, axis = 0).to(device)

                gtVelocities = groundTruths[unrollStep][:,2:4].to(device)
                gtDensity = groundTruths[unrollStep][:,4].to(device)
                fluidArea = inputData['fluidArea'].to(device)
                boundaryPositions = inputData['boundaryPosition'].to(device)
                boundaryArea = inputData['boundaryArea'].to(device)

                lossDict = getLosses(positions, velocities, gtPositions, gtDensity, gtVelocities, fluidArea, boundaryPositions, boundaryArea, attributes)
                losses.append(lossDict)
            # print('Unrolling done')
        else:
            for unrollStep in range(unrollSteps):
                _, _, groundTruths2 = loadFrame(simulationFiles[simulationIndex], initialFrame + unrollStep * hyperParams['frameDistance'], 1 + np.arange(hyperParams['frameDistance']), 1)
                loss, predictedPositions, predictedVelocity = runNetwork(predictedPositions, predictedVelocity, attributes, hyperParams['frameDistance'], gravity, fluidFeatures, boundaryPositions.to(device), boundaryFeatures.to(device), groundTruths[unrollStep], model, None, None, True)    
                positions = predictedPositions
                velocities = predictedVelocity
                gtPositions = groundTruths[unrollStep][:,:2].to(device)
                gtVelocity = torch.vstack([g[:,2:4][None,:] for g in groundTruths2])
                gtVelocities = torch.mean(gtVelocity, axis = 0).to(device)

                gtVelocities = groundTruths[unrollStep][:,2:4].to(device)
                gtDensity = groundTruths[unrollStep][:,4].to(device)
                fluidArea = inputData['fluidArea'].to(device)
                boundaryPositions = inputData['boundaryPosition'].to(device)
                boundaryArea = inputData['boundaryArea'].to(device)

                lossDict = getLosses(positions, velocities, gtPositions, gtDensity, gtVelocities, fluidArea, boundaryPositions, boundaryArea, attributes)
                losses.append(lossDict)

    unrollDict = pd.DataFrame()
    if verbose:
        for il, l in tqdm(enumerate(losses), leave = False):
            dataFrame = pd.DataFrame([{
                'rbf_x' : decodedArray['hyperParameters']['rbf_x'], 
                'rbf_y' : decodedArray['hyperParameters']['rbf_y'], 
                'n'     : decodedArray['hyperParameters']['n'], 
                'm'     : decodedArray['hyperParameters']['m'],
                'window': decodedArray['hyperParameters']['windowFunction'],
                'map'   : decodedArray['hyperParameters']['coordinateMapping'],
                'seed'  : decodedArray['hyperParameters']['networkSeed'],
                'arch'  : decodedArray['hyperParameters']['arch'],
                'testFile': simulationFiles[simulationIndex].split('.')[0].split(' ')[-1],
                'initialFrame': initialFrame,
                'unrollStep': il,
                'positionError': l['position'],
                'velocityError': l['velocity'],
                'densityError': l['density'],
                'colorError': l['color'],
                'colorGradError': l['colorGrad'],
                'meshDensityError': l['meshDensity'],
                'meshVelocityError': l['meshVelocity'],
                'meshDivergenceError': l['meshDivergence'],
                'psdError': l['psd'],
                                    }])
            unrollDict = pd.concat([unrollDict, dataFrame], ignore_index = True)
    else:       
        for il, l in enumerate(losses):
            dataFrame = pd.DataFrame([{
                'rbf_x' : decodedArray['hyperParameters']['rbf_x'], 
                'rbf_y' : decodedArray['hyperParameters']['rbf_y'], 
                'n'     : decodedArray['hyperParameters']['n'], 
                'm'     : decodedArray['hyperParameters']['m'],
                'window': decodedArray['hyperParameters']['windowFunction'],
                'map'   : decodedArray['hyperParameters']['coordinateMapping'],
                'seed'  : decodedArray['hyperParameters']['networkSeed'],
                'arch'  : decodedArray['hyperParameters']['arch'],
                'testFile': simulationFiles[simulationIndex].split('.')[0].split(' ')[-1],
                'initialFrame': initialFrame,
                'unrollStep': il,
                'positionError': l['position'],
                'velocityError': l['velocity'],
                'densityError': l['density'],
                'colorError': l['color'],
                'colorGradError': l['colorGrad'],
                'meshDensityError': l['meshDensity'],
                'meshVelocityError': l['meshVelocity'],
                'meshDivergenceError': l['meshDivergence'],
                'psdError': l['psd'],
                                    }])
            unrollDict = pd.concat([unrollDict, dataFrame], ignore_index = True)
    return unrollDict


if args.verbose:
    print('Loading Neural Network from %s' % inputFile)

model, hyperParams = loadRbfModel(simulationFiles[0], 0, inputFile, -1)

overallDict = pd.DataFrame()
if args.verbose:
    print('Performing unroll testing')
    for simulationIndex in tqdm([0,1,2,4], leave = False):
        for initialFrame in tqdm([0, 512, 1024, 2175], leave = False):
            unrollDict = getUnrollFrame(simulationIndex, initialFrame, 64, verbose = True)
            overallDict = pd.concat([overallDict, unrollDict], ignore_index = True)
else:
    overallDict = pd.DataFrame()
    for simulationIndex in [0,1,2,4]:
        for initialFrame in [0, 512, 1024, 2175]:
            unrollDict = getUnrollFrame(simulationIndex, initialFrame, 64)
            overallDict = pd.concat([overallDict, unrollDict], ignore_index = True)

if args.verbose:
    print('Done.')
    print('Writing testing output to "%s - testing.csv"' % inputFile)

overallDict.to_csv('%s - testing.csv' % inputFile)
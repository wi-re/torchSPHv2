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
import pandas as pd

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
from tqdm import trange, tqdm
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

from joblib import Parallel, delayed

from cutlass import *
from rbfConv import *
from tqdm import trange, tqdm

from datautils import *
# from sphUtils import *
from lossFunctions import *

from plotting import *
# plt.style.use('dark_background')
from tqdm import trange, tqdm
import shlex as shlex


# from rbfNet import *
# from tqdm.notebook import trange, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', type=int, default=25)
parser.add_argument('-cmap','--coordinateMapping', type=str, default='cartesian')
parser.add_argument('-w','--windowFunction', type=str, default='None')
parser.add_argument('-c','--cutoff', type=int, default=127)
parser.add_argument('-b','--batch_size', type=int, default=4)
parser.add_argument('-o','--output', type = str, default = 'paperData_collisionAblationsBase3')
# parser.add_argument('-i','--input', type = str, default = '/mnt/data/datasets/generativeCollisions')
parser.add_argument('-i','--input', type = str, default = '~/dev/datasets/generativeCollisions')
parser.add_argument('--cutlassBatchSize', type=int, default=128)
parser.add_argument('-r','--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('-x','--rbf_x', type=str, default='fourier')
parser.add_argument('-y','--rbf_y', type=str, default='fourier')
parser.add_argument('-n','--n', type=int, default=4)
parser.add_argument('-m','--m', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--networkseeds', type=str, default='42')
parser.add_argument('-d','--frameDistance', type=int, default=1)
parser.add_argument('--dataDistance', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('-f','--forwardLoss', type=bool, default=False)
parser.add_argument('-v','--verbose', type=bool, default=False)
parser.add_argument('-l','--li', type=bool, default=True)
parser.add_argument('-a','--activation', type=str, default='relu')

parser.add_argument('--widths', type=str, default='32')
parser.add_argument('--depths', type=str, default='6')
parser.add_argument('--limitData', type=int, default=-1)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('-u', '--maxUnroll', type=int, default=10)
parser.add_argument('--minUnroll', type=int, default=2)
parser.add_argument('--overfit', type=bool, default=False)

parser.add_argument('-augj', '--augmentJitter', type=bool, default=False)
parser.add_argument('-j', '--jitterAmount', type=float, default=0.01)
parser.add_argument('-augr', '--augmentAngle', type=bool, default=False)
parser.add_argument('-adjust', '--adjustForFrameDistance', type = bool, default = False)
parser.add_argument('-netArch', '--network', type=str, default='default')
parser.add_argument('-norm', '--normalized', type=bool, default=False)

args = parser.parse_args()

def verbosePrint(string):
    if args.verbose:
        print(string)

verbosePrint('Collision Layout Ablation Testing')

exportLabel = 'collisionTrainingAblation - n %2d - base %8s - window %10s - widths %s - depths %s - seeds %s - angle %s - jitter %s - overfit %s' %(
    args.n,
    args.rbf_x,
    args.windowFunction,
    args.widths,
    args.depths,
    args.networkseeds,
    1 if args.augmentAngle else 0,
    1 if args.augmentJitter else 0,
    1 if args.overfit else 0
)

verbosePrint(exportLabel)


import random 
import numpy as np

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
verbosePrint('Set random number seeds to %d' % args.seed)

if args.verbose:
    print('Available cuda devices:', torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.verbose:
    print('Running on Device %s' % device)
torch.set_num_threads(1)

# from joblib import Parallel, delayed

from cutlass import *
from rbfConv import *
from datautils import *
from plotting import *

# Use dark theme
# plt.style.use('dark_background')
from tqdm import tqdm
import os

from rbfNet import *


if args.verbose:
    print('Parsing data in %s' % args.input)
basePath = '../export'
basePath = os.path.expanduser('~/dev/datasets/generativeCollisions')
basePath = os.path.expanduser(args.input)

trainingFiles = [basePath + '/train/' + f for f in os.listdir(basePath + '/train/') if f.endswith('.hdf5')]
testingFiles = [basePath + '/testing/' + f for f in os.listdir(basePath + '/testing/') if f.endswith('.hdf5')]

training = []
validation = []
testing = []

verbosePrint('Gathered %4d training files and %4d testing files' % (len(trainingFiles), len(testingFiles)))


    # for s in simulationFiles:    
#     _, train, valid, test = splitFileZSTD(s, split = True, limitRollOut = False, skip = 0, cutoff = 1800, distance = 1)
#     training.append((s,train))
#     validation.append((s,valid))
#     testing.append((s,test))
# debugPrint(training)

# simulationFiles = [basePath + '/' + f for f in os.listdir(basePath) if f.endswith('.hdf5')]

if args.limitData > 0:
    files = []
    for i in range(max(len(trainingFiles), args.limitData)):
        files.append(trainingFiles[i])
    simulationFiles = files
# simulationFiles = [simulationFiles[0]]
# if args.verbose:
    # print('Input files:')
    # for i, c in enumerate(trainingFiles):
        # print('\t', i ,c)

training = []
validation = []
testing = []

for s in trainingFiles:
    f, s, u = splitFile(s, split = False, cutoff = -args.frameDistance * args.maxUnroll - 1, skip = args.frameDistance if args.adjustForFrameDistance else 10)
    training.append((f, (s,u)))
for s in testingFiles:
    f, s, u = splitFile(s, split = False, cutoff = -args.frameDistance * args.maxUnroll - 1, skip = args.frameDistance if args.adjustForFrameDistance else 10)
    testing.append((f, (s,u)))

train_ds = datasetLoader(training)
test_ds = datasetLoader(testing)

verbosePrint(f'Processed input files with cutoff value {-args.frameDistance * args.maxUnroll - 1}, skip value {args.frameDistance if args.adjustForFrameDistance else 10} (adjustForFrameDistance = {args.adjustForFrameDistance})')
verbosePrint(f'Training dataset consists of {len(train_ds)} entries')
verbosePrint(f'Testing dataset consists of {len(test_ds)} entries')

inFile = h5py.File(trainingFiles[0], 'r')
frameCount = int(len(inFile['simulationExport'].keys()))
inFile.close()

verbosePrint(f'Frame Count per file: {frameCount}')

# data = position | velocity | gravity | features
iterator = iter(train_ds)
file, frame, r = train_ds[0]
attributes, inputData, groundTruthData = loadFrame(file, frame, 1 + np.arange(1), 1)

# dataCache = torch.zeros((len(trainingFiles) * frameCount, inputData['fluidPosition'].shape[0], 12))
dataCache = {}
verbosePrint('Generating cache of training data')
for file in (tqdm(trainingFiles) if args.verbose else trainingFiles):
    data = torch.zeros((frameCount, inputData['fluidPosition'].shape[0], 12))
    for frame in (tqdm(range(frameCount-1), leave = False) if args.verbose else range(frameCount-1)):
        attributes, inputData, groundTruthData = loadFrame(file, frame, 1 + np.arange(1), 1)
        fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(attributes, inputData)
        difference = fluidPositions - torch.tensor([0,0], dtype = fluidPositions.dtype, device = fluidPositions.device)
        distance = torch.linalg.norm(difference,axis=1) + 1e-7
        difference = difference / distance[:, None]
        fluidGravity = -0.5 * 10**2 * difference * (distance)[:,None]**2

        data[frame,:,0:2] = fluidPositions
        data[frame,:,2:4] = inputData['fluidVelocity']
        data[frame,:,4:6] = fluidGravity
        data[frame,:,6:10] = fluidFeatures
        data[frame,:,10:] = groundTruthData[0][:,:2]
    dataCache[file] = data
    
verbosePrint('Generating cache of testing data')
for file in (tqdm(testingFiles) if args.verbose else testingFiles):
    data = torch.zeros((frameCount, inputData['fluidPosition'].shape[0], 12))
    for frame in (tqdm(range(frameCount-1), leave = False) if args.verbose else range(frameCount-1)):
        attributes, inputData, groundTruthData = loadFrame(file, frame, 1 + np.arange(1), 1)
        fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(attributes, inputData)
        difference = fluidPositions - torch.tensor([0,0], dtype = fluidPositions.dtype, device = fluidPositions.device)
        distance = torch.linalg.norm(difference,axis=1) + 1e-7
        difference = difference / distance[:, None]
        fluidGravity = -0.5 * 10**2 * difference * (distance)[:,None]**2
        data[frame,:,0:2] = fluidPositions
        data[frame,:,2:4] = inputData['fluidVelocity']
        data[frame,:,4:6] = fluidGravity
        data[frame,:,6:10] = fluidFeatures
        data[frame,:,10:] = groundTruthData[0][:,:2]
    dataCache[file] = data
    

batch_size = args.batch_size

if args.verbose:
    print('Setting up data loaders')
train_ds = datasetLoader(training)
train_dataloader = DataLoader(train_ds, shuffle=True, batch_size = batch_size).batch_sampler

n = args.n
m = args.m
coordinateMapping = args.coordinateMapping
windowFn = getWindowFunction(args.windowFunction)
rbf_x = args.rbf_x
rbf_y = args.rbf_y
initialLR = args.lr
maxRollOut = 10
epochs = args.epochs
frameDistance = args.frameDistance

if args.verbose:
    print('Network Hyperparameters:')
    print('[n x m]: [%dx%d]'% (n, m))
    print('[rbf_x x rbf_y]: [%sx%s]'% (rbf_x, rbf_y))
    print('Mapping:', args.coordinateMapping)
    print('window function:', args.windowFunction)
    print('activation function:', args.activation)
    print('initial learning rate: ', initialLR)
    print('Training for %d epochs' % epochs)
    print('Rollout limit (if applicable):', maxRollOut)
    print('Training with frame offset of', frameDistance)
    # print('Network architecture', args.arch)


# widths = args.arch.strip().split(' ')
# layers = [int(s) for s in widths]
# debugPrint(layers)
# if args.verbose:
    # print('Building Network')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

boundaryFeatures = None

if args.verbose:
    print('Writing output to ./%s' % (args.output))
if not os.path.exists('./%s' % (args.output)):
    os.makedirs('./%s' % (args.output))

def processBatchCached(bdata, unrollSteps : int, dataCache : torch.Tensor, attributes, model, optimizer, dataSet, augmentAngle = False, augmentJitter = False, jitterAmount = 0.01, returnLoss = True):
    pbl = gtqdms[args.gpu + args.gpus]
    dataRows = []
    for ib, b in enumerate(bdata):
        file, frame, unroll = dataSet[b]
        batchData = []
        for iu, u in enumerate(range(unrollSteps)):
            currRow = dataCache[file][frame + u,:].unsqueeze(0)
            batchData.append(currRow)

        stacked = torch.vstack(batchData)
        angle = torch.rand(1) * 2 * np.pi
        rot = torch.tensor([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]], device = stacked.device, dtype = stacked.dtype)
        if augmentJitter:
            stacked[0,:,0:2] += torch.normal(torch.zeros_like(stacked[0,:,0:2]), torch.ones_like(stacked[0,:,0:2]) * jitterAmount * attributes['support'])
        if augmentAngle:
            for iu, u in enumerate(range(unrollSteps)):
                stacked[iu,:,0:2] = torch.matmul(rot.unsqueeze(0).repeat(stacked.shape[1],1,1), stacked[iu,:,0:2].unsqueeze(2))[:,:,0] 
                stacked[iu,:,2:4] = torch.matmul(rot.unsqueeze(0).repeat(stacked.shape[1],1,1), stacked[iu,:,2:4].unsqueeze(2))[:,:,0] 
                stacked[iu,:,4:6] = torch.matmul(rot.unsqueeze(0).repeat(stacked.shape[1],1,1), stacked[iu,:,4:6].unsqueeze(2))[:,:,0] 
                stacked[iu,:,7:9] = torch.matmul(rot.unsqueeze(0).repeat(stacked.shape[1],1,1), stacked[iu,:,7:9].unsqueeze(2))[:,:,0] 
                stacked[iu,:,10:12] = torch.matmul(rot.unsqueeze(0).repeat(stacked.shape[1],1,1), stacked[iu,:,10:12].unsqueeze(2))[:,:,0] 

        dataRows.append(stacked)
    dataRows = torch.stack(dataRows)
    dataRows = dataRows.to(device)
    fluidBatches = torch.hstack([torch.ones(dataRows.shape[2]) * i for i in range(len(bdata))]).to(device)
    
    fluidPositions = torch.clone(torch.vstack([r[0,:,0:2] for r in dataRows]))
    fluidVelocity = torch.clone(torch.vstack([r[0,:,2:4] for r in dataRows]))
    fluidGravity = torch.clone(torch.vstack([r[0,:,4:6] for r in dataRows]))
    fluidFeatures = torch.vstack([r[0,:,6:10] for r in dataRows])

    boundaryPositions = None
    boundaryFeatures = None
    boundaryBatches = None

    unrollLosses = []

    optimizer.zero_grad()
    for iu, u in enumerate(range(unrollSteps)):    
        gtFluidPositions = torch.vstack([r[iu,:,10:12] for r in dataRows])

        intermediateVelocity = fluidVelocity + frameDistance * attributes['dt'] * fluidGravity
        intermediatePositions = fluidPositions + frameDistance * attributes['dt'] * intermediateVelocity

        fluidFeatures = torch.hstack((fluidFeatures[:,0][:,None], intermediateVelocity, fluidFeatures[:,3:]))
        predictedUpdate = model(fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, attributes, fluidBatches, boundaryBatches)

        loss = torch.linalg.norm(intermediatePositions + predictedUpdate[:,:2] * attributes['dt'] - gtFluidPositions, dim = 1) / attributes['dt']
        lossTerm = torch.mean(loss.reshape(len(bdata), dataRows.shape[2]), dim = 1)
        unrollLosses.append(lossTerm)

        if iu != unrollSteps - 1:
            updatedPositions = intermediatePositions + predictedUpdate[:,:2] * attributes['dt']
            fluidVelocity = (updatedPositions - fluidPositions) / attributes['dt']
            fluidPositions = updatedPositions
            difference = fluidPositions
            distance = torch.linalg.norm(difference,axis=1) + 1e-7
            difference = difference / distance[:, None]
            fluidGravity = -0.5 * 10**2 * difference * (distance)[:,None]**2

    stackedLosses = torch.vstack(unrollLosses)
    overallLoss = torch.mean(stackedLosses)
    overallLoss.backward()
    optimizer.step()
    return stackedLosses.detach().cpu().numpy() if returnLoss else None

def trainNetwork(layers, n, m, rbf_x, rbf_y, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    boundaryFeatures = None

    model = None
    if args.network == 'default':
        model = RbfNet(
            4, boundaryFeatures.shape[1] if boundaryFeatures is not None else 0, layers = layers, 
            coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, 
            rbf_x = rbf_x, rbf_y = rbf_y, 
            batchSize = args.cutlassBatchSize, normalized = args.normalized)
    if args.network == 'split':
        model = RbfSplitNet(4, boundaryFeatures.shape[1] if boundaryFeatures is not None else 0, layers = layers, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = args.cutlassBatchSize, normalized = args.normalized)
    if args.network == 'interleaved':
        model = RbfInterleaveNet(4, boundaryFeatures.shape[1] if boundaryFeatures is not None else 0, layers = layers, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = args.cutlassBatchSize, normalized = args.normalized)
    if args.network == 'input':
        model = RbfInputNet(4, boundaryFeatures.shape[1] if boundaryFeatures is not None else 0, layers = layers, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = args.cutlassBatchSize, normalized = args.normalized)
    if args.network == 'output':
        model = RbfOutputNet(4, boundaryFeatures.shape[1] if boundaryFeatures is not None else 0, layers = layers, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = args.cutlassBatchSize, normalized = args.normalized)

    lr = initialLR
#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=initialLR, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    model = model.to(device)
    model.train()
    
    losses = []
    l2 = []

    batch = [10,20,30,40]
    if args.overfit:
        batch = [10]
    unrollSteps = 1   

    dataLoader = DataLoader(train_ds, shuffle=True, batch_size = batch_size).batch_sampler
    t = gtqdms[args.gpu + args.gpus]
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        t.reset(total=iterationsPerEpoch * epochs)
    for i in range(iterationsPerEpoch * epochs):
        if not args.overfit:
            try:
                batch = next(dataIter)
            except:
                dataIter = iter(dataLoader)
                batch = next(dataIter)
        batchLoss = processBatchCached(batch, unrollSteps, dataCache, attributes, model, optimizer, train_ds, args.augmentAngle, args.augmentJitter, args.jitterAmount, returnLoss = True)

        losses.append(batchLoss)
        l2.append(np.mean(batchLoss))

        with portalocker.Lock('README.md', flags = 0x2, timeout = None):
            t.set_description('[%2dx%2d]@%10d: Training it = %5d [%2d], batch: [%s] x %2d, lr: %.4e, batchLoss: %.5e, rollingLoss: %.5e ' % 
                          (len(layers),
                           layers[0],
                           seed,
                           i % iterationsPerEpoch, 
                           i // iterationsPerEpoch,
                           ' '.join(['%04d' % b for b in batch]), 
                           unrollSteps,
                           optimizer.param_groups[0]['lr'],
                           np.mean(batchLoss), 
                           np.mean(l2[-100:] if len(l2) > 100 else l2),
                        #    ' '.join(['%.4e' % v for v in np.mean(batchLoss, axis = 0)])
                          ))
            t.update()
            pb.update()
        if i % lrStep == 0 and i > 0:
            scheduler.step()
        if i % unrollIncrement == 0 and i > 0:
            unrollSteps = unrollSteps + 1
        if i % iterationsPerEpoch == 0:
            exportLabel2 = 'collisionTrainingAblation - n %2d - base %8s - window %10s - width %s - depth %s - seed %s - angle %s - jitter %s - overfit %s' %(
                n,
                rbf_x,
                args.windowFunction,
                layers[0],
                len(layers),
                seed,
                1 if args.augmentAngle else 0,
                1 if args.augmentJitter else 0,
                1 if args.overfit else 0
            )


            torch.save(model.state_dict(), './%s/%s_model_%03d.torch' % (args.output, exportLabel2, i//iterationsPerEpoch))
            
    processedLosses = [np.mean(l, axis = 1).tolist() for l in losses]
    longestUnroll = losses[-1].shape[0]
    unrollLengths = np.arange(longestUnroll) + 1
    filtered = []

    for i in unrollLengths:
        nStepLoss = [l[i - 1] for l in processedLosses if len(l) >= i]
        filtered.append(nStepLoss)
        
        
    return {'model' : model, 'n':n, 'm':m,'layers':layers, 'rbf_x':rbf_x, 'rbf_y':rbf_y, 'processesLosses': processedLosses, 'filtered': filtered, 'l2':l2, 'window': args.windowFunction, 'seed': seed}


from tqdm import trange, tqdm


# rbf_x = rbf_y = 'fourier'
# print(rbf_x, rbf_y)

epochs = 10
batch_size = args.batch_size
iterationsPerEpoch = 2000
totalIterations = epochs * iterationsPerEpoch
initialUnroll = 1
unrollIncrement = 2000
unrollSteps = (epochs * iterationsPerEpoch) // unrollIncrement - 1
finalUnroll = initialUnroll + unrollSteps
initialLR = 1e-3
finalLR = 1e-5
lrStep = 100
lrSteps = int(np.ceil((totalIterations - lrStep) / lrStep))
gamma = np.power(finalLR / initialLR, 1/lrSteps)

# optimizer = Adam(model.parameters(), lr=initialLR, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
if args.verbose:
    print('Training network for %2d epochs @ %4d iterations -> %6d steps' % (epochs, iterationsPerEpoch, epochs * iterationsPerEpoch))
    print('initialLR: %.4e, finalLR: %.4e, lrStep: %4d, lrSteps %4d, gamma %g' %(initialLR, finalLR, lrStep, lrSteps, gamma))
    print('initialUnroll: %2d, finalUnroll: %2d, unrollIncrement: %4d, unrollSteps %2d' %(initialUnroll, finalUnroll, unrollIncrement, (epochs * iterationsPerEpoch) // unrollIncrement - 1))


gtqdms = []
if args.verbose:
    print('Setting up tqdm progress bars')

with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(0, (epochs) * args.iterations), position = g, leave = True))
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(1, epochs + 1), position = args.gpus + g, leave = True))
# print(torch.cuda.current_device())

widths = [int(s) for s in args.widths.split(' ')]
depths = [int(s) for s in args.depths.split(' ')]
seeds = [int(s) for s in args.networkseeds.split(' ')]
layouts = []
for d in depths:
    for w in widths:
        l = [w] * d + [2]
        if l not in layouts:
#             print([w] * d + [1])
            layouts.append(l)
# print(layouts)

pb = gtqdms[args.gpu]
with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    pb.reset(total = len(layouts) * len(seeds) * iterationsPerEpoch * epochs )
    pb.set_description('[gpu %d] widths %s, depths %s, n %2d, base %s, window %s' %(args.gpu, args.widths, args.depths, args.n, args.rbf_x, args.windowFunction))
    pb.update()

networks = []
evalResult = pd.DataFrame()

for layout in layouts:
    for seed in seeds:
        network = trainNetwork(layout, args.n, args.m, args.rbf_x, args.rbf_y, seed)
        with portalocker.Lock('README.md', flags = 0x2, timeout = None):
            pb.update()
        curFrame = pd.DataFrame({
            'layout': '[%s]' % ' '.join(str(s) for s in network['layers']),
            'n': network['n'],
            'm': network['m'],
            'rbf_x': network['rbf_x'],
            'rbf_y': network['rbf_y'],
            'l2last': network['l2'][-1],
            'l2min' : np.min(network['l2'][-1]),
            'l2': [network['l2']],
            'parameters': count_parameters(network['model']),
            'width': network['layers'][0],
            'depth': len(network['layers']) - 1,
            'seed': network['seed'],
            'window': network['window']
        }, index = [0])
        evalResult = pd.concat((evalResult, curFrame), ignore_index = True)
        evalResult.to_csv('%s/%s.csv' % (args.output, exportLabel))

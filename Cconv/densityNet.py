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
import copy

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


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--coordinateMapping', type=str, default='cartesian')
parser.add_argument('--windowFunction', type=str, default='Wendland4')
parser.add_argument('--cutoff', type=int, default=1800)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--rbf_x', type=str, default='linear')
parser.add_argument('--rbf_y', type=str, default='linear')
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--m', type=int, default=9)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--frameDistance', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)


args = parser.parse_args()

import random 
import numpy as np
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
# print(torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('running on: ', device)
torch.set_num_threads(1)

from joblib import Parallel, delayed

from cutlass import *
from rbfConv import *

from datautils import *
# from sphUtils import *
from lossFunctions import *

from plotting import *
plt.style.use('dark_background')
from tqdm import trange, tqdm


def loadFrame(filename, frame, frameOffsets = [1], frameDistance = 1):
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

    support = inFile.attrs['restDensity']
    targetNeighbors = inFile.attrs['targetNeighbors']
    restDensity = inFile.attrs['restDensity']
    dt = inFile.attrs['initialDt']

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
        'boundaryVelocity': torch.from_numpy(inFile['boundaryInformation']['boundaryVelocity'][:]).type(torch.float32)
    }
    
    groundTruthData = []
    for i in frameOffsets:
        gtGrp = inFile['simulationExport']['%05d' % (frame + i * frameDistance)]
#         debugPrint((frame + i * frameDistance))
#         debugPrint(gtGrp.attrs['timestep'])
        gtData = {
            'fluidPosition'    : torch.from_numpy(gtGrp['fluidPosition'][:]),
            'fluidVelocity'    : torch.from_numpy(gtGrp['fluidVelocity'][:]),
            'fluidDensity'     : torch.from_numpy(gtGrp['fluidDensity'][:]),
    #         'fluidPressure'    : torch.from_numpy(gtGrp['fluidPressure'][:]),
    #         'boundaryDensity'  : torch.from_numpy(gtGrp['fluidDensity'][:]),
    #         'boundaryPressure' : torch.from_numpy(gtGrp['fluidPressure'][:]),
        }
        
        groundTruthData.append(torch.hstack((gtData['fluidPosition'].type(torch.float32), gtData['fluidVelocity'], gtData['fluidDensity'][:,None])))
        
    
    inFile.close()
    
    return attributes, inputData, groundTruthData

class DensityNet(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures, layers = [32,64,64,2], denseLayer = True, acitvation = 'relu',
                coordinateMapping = 'polar', n = 8, m = 8, windowFn = None, rbf_x = 'linear', rbf_y = 'linear', batchSize = 32):
        super().__init__()
#         debugPrint(layers)
        
        self.features = copy.copy(layers)
#         debugPrint(fluidFeatures)
#         debugPrint(boundaryFeatures)
        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.relu = getattr(nn.functional, 'relu')
#         debugPrint(fluidFeatures)

        self.convs.append(RbfConv(
            in_channels = fluidFeatures, out_channels = 1,
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_y],
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        self.convs.append(RbfConv(
            in_channels = boundaryFeatures, out_channels = 1,
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_y], 
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        

    def forward(self, \
                fluidPositions, boundaryPositions, \
                fluidFeatures, boundaryFeatures,\
                support, fluidBatches = None, boundaryBatches = None):
        fi, fj = radius(fluidPositions, fluidPositions, support, max_num_neighbors = 256, batch_x = fluidBatches, batch_y = fluidBatches)
        bf, bb = radius(boundaryPositions, fluidPositions, support, max_num_neighbors = 256, batch_x = boundaryBatches, batch_y = fluidBatches)
        
        i, ni = torch.unique(fi, return_counts = True)
        b, nb = torch.unique(bf, return_counts = True)
        ni[i[b]] += nb

        self.li = torch.exp(-1 / np.float32(attributes['targetNeighbors']) * ni)
        
        boundaryEdgeIndex = torch.stack([bf, bb], dim = 0)
        boundaryEdgeLengths = (boundaryPositions[boundaryEdgeIndex[1]] - fluidPositions[boundaryEdgeIndex[0]])/support
        boundaryEdgeLengths = boundaryEdgeLengths.clamp(-1,1)
            
        fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
        fluidEdgeLengths = (fluidPositions[fluidEdgeIndex[1]] - fluidPositions[fluidEdgeIndex[0]])/support
#         debugPrint(torch.min(fluidEdgeLengths))
#         debugPrint(torch.max(fluidEdgeLengths))
        fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
        
#         debugPrint(fluidFeatures)
#         debugPrint(boundaryFeatures)
        
        boundaryConvolution = self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths)
#         debugPrint(fluidPositions)
#         debugPrint(fluidFeatures)
#         debugPrint(fluidEdgeIndex)
#         debugPrint(fluidEdgeLengths)
        fluidConvolution = self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths)
#         debugPrint(fluidConvolution[:,0][:8])
#         debugPrint(boundaryConvolution[:,0][:8])
        # return fluidConvolution
        return fluidConvolution  + boundaryConvolution



#  semi implicit euler, network predicts velocity update
def integrateState(inputPositions, inputVelocities, modelOutput, dt):
    predictedVelocity = modelOutput #inputVelocities +  modelOutput 
    predictedPosition = inputPositions + attributes['dt'] * predictedVelocity
    
    return predictedPosition, predictedVelocity
# velocity loss
def computeLoss(predictedPosition, predictedVelocity, groundTruth, modelOutput):
#     debugPrint(modelOutput.shape)
#     debugPrint(groundTruth.shape)
#     return torch.sqrt((modelOutput - groundTruth[:,-1:].to(device))**2)
    return torch.abs(modelOutput - groundTruth[:,-1:].to(device))
    return torch.linalg.norm(groundTruth[:,2:] - predictedVelocity, dim = 1)


def constructFluidFeatures(inputData):
    fluidFeatures = torch.hstack(\
                (torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1), \
                 inputData['fluidVelocity'].type(torch.float32), 
                 inputData['fluidGravity'].type(torch.float32)))

    fluidFeatures = torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    fluidFeatures[:,0] *= 7 / np.pi * inputData['fluidArea']  / attributes['support']**2
#     fluidFeatures = inputData['fluidArea'].type(torch.float32).unsqueeze(dim=1)
    
    boundaryFeatures = inputData['boundaryNormal'].type(torch.float32)
    boundaryFeatures = torch.ones(inputData['boundaryNormal'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    boundaryFeatures[:,0] *=  7 / np.pi * inputData['boundaryArea']  / attributes['support']**2
    
    return inputData['fluidPosition'].type(torch.float32), inputData['boundaryPosition'].type(torch.float32), fluidFeatures, boundaryFeatures


def processBatch(e, unroll, train_ds, bdata, frameDistance):
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches, groundTruths = loadBatch(train_ds, bdata, constructFluidFeatures, unroll, frameDistance)    
    
    predictedPositions = fluidPositions.to(device)
    predictedVelocity = fluidFeatures[:,1:3].to(device)
    
    unrolledLosses = []
    bLosses = []
#     debugPrint(bdata)
    boundaryPositions = boundaryPositions.to(device)
    fluidFeatures = fluidFeatures.to(device)
    boundaryFeatures = boundaryFeatures.to(device)
    fluidBatches = fluidBatches.to(device)
    boundaryBatches = boundaryBatches.to(device)
    
#     debugPrint(bdata)
#     debugPrint(predictedPosition)
    
    ls = []
    
    for u in range(unroll):
        predictions = model(predictedPositions, boundaryPositions, fluidFeatures, boundaryFeatures, attributes['support'], fluidBatches, boundaryBatches)

        predictedPositions, predictedVelocities = integrateState(predictedPositions, predictedVelocity, predictions, attributes['dt'])
        
        fluidFeatures = torch.hstack((fluidFeatures[:,0][:,None], predictedVelocity, fluidFeatures[:,3:]))
#         fluidFeatures[:,1:3] = predictedVelocity

#         debugPrint(prediction.shape)
#         debugPrint(groundTruths[0].shape)
#         loss = model.li * (predictions - groundTruths[0][:,-1:].to(device)) ** 0.5
#         debugPrint(model.li)
#         loss = computeLoss(predictedPositions, predictedVelocities, groundTruths[u].to(device), predictions)
#         loss = model.li * torch.sqrt(computeLoss(predictedPositions, predictedVelocities, groundTruths[u].to(device), predictions))
#         loss = model.li * computeLoss(predictedPositions, predictedVelocities, groundTruths[u].to(device), predictions)
        loss = computeLoss(predictedPositions, predictedVelocities, groundTruths[u].to(device), predictions)
#         p = 8
#         debugPrint(loss[:p,0].detach().cpu().numpy())
#         debugPrint(predictions[:p,0].detach().cpu().numpy())
#         debugPrint(groundTruths[0][:,-1:][:p,0].detach().cpu().numpy())
#         print('----------------------')
        ls.append(torch.mean(loss))
        batchedLoss = []
#         debugPrint(fluidBatches)
        for i in range(len(bdata)):
            L = loss[fluidBatches == i]
#             debugPrint(L)
            Lterms = (torch.mean(L), torch.max(torch.abs(L)), torch.min(torch.abs(L)), torch.std(L))
            
            
            batchedLoss.append(torch.hstack(Lterms))
        batchedLoss = torch.vstack(batchedLoss).unsqueeze(0)
#         debugPrint(batchedLoss.shape)
        batchLoss = torch.mean(loss)# + torch.max(torch.abs(loss))
        bLosses.append(batchedLoss)
        unrolledLosses.append(batchLoss)
        
#     debugPrint(bLosses)
#     debugPrint(torch.cat(bLosses))
#     debugPrint(bLosses.shape)
#     debugPrint(bLosses)
    
#     return torch.mean(torch.hstack(ls))
    
    bLosses = torch.vstack(bLosses)
    maxLosses = torch.max(bLosses[:,:,1], dim = 0)[0]
    minLosses = torch.min(bLosses[:,:,2], dim = 0)[0]
    meanLosses = torch.mean(bLosses[:,:,0], dim = 0)
    stdLosses = torch.mean(bLosses[:,:,3], dim = 0)
    
    
    del predictedPositions, predictedVelocities, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches
    
    bLosses = bLosses.transpose(0,1)
    
    return bLosses, meanLosses, minLosses, maxLosses, stdLosses

import os

basePath = '../export'
basePath = os.path.expanduser(basePath)

simulationFiles = [basePath + '/' + f for f in os.listdir(basePath) if f.endswith('.hdf5')]
# for i, c in enumerate(simulationFiles):
#     print(i ,c)
    
simulationFiles  = [simulationFiles[0]]

training = []
validation = []
testing = []


for s in simulationFiles:    
    _, train, valid, test = splitFile(s, split = True, limitRollOut = False, skip = 0, cutoff = args.cutoff)
    training.append((s,train))
    validation.append((s,valid))
    testing.append((s,test))

batch_size = args.batch_size

train_ds = datasetLoader(training)
train_dataloader = DataLoader(train_ds, shuffle=True, batch_size = batch_size).batch_sampler

validation_ds = datasetLoader(validation)
validation_dataloader = DataLoader(validation_ds, shuffle=True, batch_size = batch_size).batch_sampler

fileName, frameIndex, maxRollout = train_ds[len(train_ds)//2]
# frameIndex = 750
attributes, inputData, groundTruthData = loadFrame(simulationFiles[0], 0, 1 + np.arange(1))
fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(inputData)

# debugPrint(fluidFeatures.shape)


n = args.n
m = args.m
coordinateMapping = args.coordinateMapping
windowFn = None
if args.windowFunction == 'Wendland4':
    windowFn = lambda r: torch.clamp(torch.pow(1. - r, 4) * (1.0 + 4.0 * r), min = 0)
rbf_x = args.rbf_x
rbf_y = args.rbf_y
initialLR = args.lr
maxRollOut = 10
epochs = args.epochs
frameDistance = args.frameDistance




model = DensityNet(fluidFeatures.shape[1], boundaryFeatures.shape[1], coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 64)



lr = initialLR
optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
model = model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# if args.gpus == 1:
    # print('Number of parameters', count_parameters(model))

optimizer.zero_grad()
model.train()

hyperParameterDict = {}
hyperParameterDict['n'] = n
hyperParameterDict['m'] = m
hyperParameterDict['coordinateMapping'] = coordinateMapping
hyperParameterDict['rbf_x'] = rbf_x
hyperParameterDict['rbf_y'] = rbf_y
hyperParameterDict['windowFunction'] = 'yes' if windowFn is not None else 'no'
hyperParameterDict['initialLR'] = initialLR
hyperParameterDict['maxRollOut'] = maxRollOut
hyperParameterDict['epochs'] = epochs
hyperParameterDict['frameDistance'] = frameDistance
hyperParameterDict['parameters'] =  count_parameters(model)

# debugPrint(hyperParameterDict)


timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
networkPrefix = 'DensityNet'

exportString = '%s - n=[%2d,%2d] rbf=[%s,%s] map = %s window = %s d = %2d e = %2d - %s' % (networkPrefix, hyperParameterDict['n'], hyperParameterDict['m'], hyperParameterDict['rbf_x'], hyperParameterDict['rbf_y'], hyperParameterDict['coordinateMapping'], hyperParameterDict['windowFunction'], hyperParameterDict['frameDistance'], hyperParameterDict['epochs'], timestamp)

# if args.gpus == 1:
#     debugPrint(hyperParameterDict)
# if args.gpus == 1:
#     debugPrint(exportString)


# exportPath = './trainingData/%s - %s.hdf5' %(self.config['export']['prefix'], timestamp)
if not os.path.exists('./trainingData/%s' % exportString):
    os.makedirs('./trainingData/%s' % exportString)
# self.outFile = h5py.File(self.exportPath,'w')
from tqdm import tqdm 

import portalocker
gtqdms = []
with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(1, epochs + 1), position = g, leave = True))
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(1, epochs + 1), position = args.gpus + g, leave = True))
# print(torch.cuda.current_device())



def processDataLoader(e, rollout, ds, dataLoader, model, optimizer, train = True, prefix = ''):
    pbl = gtqdms[args.gpu + args.gpus]
    losses = []
    batchIndices = []
    
    if train:
        model.train(True)
    else:
        model.train(False)

    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pbl.reset(total=len(dataLoader))
    
    for bdata in dataLoader:
        if train:
            optimizer.zero_grad()
        
#         debugPrint(bdata)
        batchLosses, meanLosses, minLosses, maxLosses, stdLosses = processBatch(e, rollout, ds, bdata, frameDistance)
        
#         debugPrint(batchLosses)
#         sumLosses = processBatch(bdata,  1)
        batchIndices.append(np.array(bdata))
        losses.append(batchLosses.detach().cpu().numpy())
        
        sumLosses = torch.mean(batchLosses[:,:,0])
        sumLosses.backward()
        if train:
            optimizer.step()
#         debugPrint(sumLosses)
#         debugPrint(meanLosses)
    
        lossString = np.array2string(meanLosses.detach().cpu().numpy(), formatter={'float_kind':lambda x: "%.4e" % x})
        batchString = str(np.array2string(np.array(bdata), formatter={'float_kind':lambda x: "%.2f" % x, 'int':lambda x:'%04d' % x}))
        
#         debugPrint(batchString)
        
        pbl.set_description('%24s[gpu %d]: %3d [%1d] @ %1.5e: %s -> %.4e' %(prefix, args.gpu, e, rollout, lr, batchString, sumLosses.detach().cpu().numpy()))
#         t.set_description('%3d [%5d] @ %1.5e: %d - %.4e' %(e, rollout, lr, bdata[0], sumLosses.detach().cpu().numpy()))
        pbl.update()
    bIndices  = np.hstack(batchIndices)
    losses = np.vstack(losses)

    idx = np.argsort(bIndices)
    bIndices = bIndices[idx]
    losses = losses[idx]

    epochLoss = losses
    # pbl.update()
#     epochLosses.append(epochLoss)
    return epochLoss

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True, with_flops=True) as prof:    

training = {}
# training_fwd = {}
validation = {}
testing = {}

def formatTime(x):
    seconds = np.floor(x)
    milliseconds = (x - seconds) * 1000.
    minutes = np.floor(seconds / 60)
    hours = np.floor(minutes / 60)
    
#     print('%02.0f:%02.0f:%02.0fs %4.0fms'% (hours, minutes % 60, seconds % 60, milliseconds))
    
    if hours == 0:
        if minutes == 0:
            return '        %02.0fS %4.0fms'% (seconds % 60, milliseconds)
        return '    %02.0fM %02.0fS %4.0fms'% (minutes % 60, seconds % 60, milliseconds)
    return '%02.0fH %02.0fM %02.0fS %4.0fms'% (hours, minutes % 60, seconds % 60, milliseconds)
        

# if args.gpus > 1:
#     for i in range(args.gpus):
#         tqdm(range(1,epochs+1), leave = True)
#     for i in range(args.gpus):
#         tqdm(range(1,epochs+1), leave = True)


overallStart = time.perf_counter()


pb = gtqdms[args.gpu]
with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    pb.set_description('[gpu %d]' %(args.gpu))


trainingEpochLosses = []
trainingEpochLosses2 = []
validationLosses = []

for epoch in range(args.epochs):
    trainingEpochLoss = processDataLoader(epoch,1, train_ds, train_dataloader, model, optimizer, True, prefix = 'training')
    trainingEpochLosses.append(trainingEpochLoss)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e' %(args.gpu, np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1))))

    # f_batches, train_losses_fwd = processDataLoader(epoch, 1, train_ds, 'train (foward only)', train = False)
    # with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    #     pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd)))

    validationEpochLoss = processDataLoader(epoch,1, validation_ds, validation_dataloader, model, optimizer, False, prefix = 'validation')
    validationLosses.append(validationEpochLoss)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e Validation: %1.4e' %(args.gpu, np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1)), np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1))))

    torch.save(model.state_dict(), './trainingData/%s/model_%03d.json' % (exportString, epoch))

    # s_batches, test_losses  = run(test_dataloader,  test_ds,  'test ', train = False)
    # with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    #     pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e Validation: %1.4e Testing: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd), np.mean(valid_losses), np.mean(test_losses)))

    # training[epoch] = {}
    # # training_fwd[epoch] = {}
    # # validation[epoch] = {}
    # testing[epoch] = {}

    # if(args.batch_size == 1):
    #     for i, (batch, loss) in enumerate(zip(t_batches, train_losses)):
    #         training[epoch]['%4d %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]
    # else:
    #     for i, (batch, loss) in enumerate(zip(t_batches, train_losses)):
    #         b = ['%s %s' % (ba[0], ba[1]) for ba in batch]
    #         training[epoch]['%4d'% i + ', '.join(b)] = [float(loss[0]),float(loss[1]),float(loss[2])]
    # for i, (batch, loss) in enumerate(zip(f_batches, train_losses_fwd)):
    #     training_fwd[epoch]['%4d - %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]
    # for i, (batch, loss) in enumerate(zip(v_batches, valid_losses)):
    #     validation[epoch]['%4d - %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]
    # for i, (batch, loss) in enumerate(zip(s_batches, test_losses)):
    #     testing[epoch]['%4d - %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]

    # # print(' Training     Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(train_losses), np.median(train_losses), np.max(train_losses), len(train_losses)))
    # # print('Training fwd Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(train_losses_fwd), np.median(train_losses_fwd), np.max(train_losses_fwd), len(train_losses_fwd)))
    # # print('Validation   Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(valid_losses), np.median(valid_losses), np.max(valid_losses), len(valid_losses)))
    # # print('Testing      Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(test_losses), np.median(test_losses), np.max(test_losses), len(test_losses)))

    if epoch % args.lr_decay_step_size == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_decay_factor * param_group['lr']
            
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.update()
         
train_ds[0]

trainDict = {}
for i in range(len(train_ds)):
    fileName, index, _ = train_ds[i]
    trainDict['%05d' % i] = {'file':fileName, 't':int( index)}
#     break
validationDict = {}
for i in range(len(validation_ds)):
    fileName, index, _ = validation_ds[i]
    validationDict['%05d' % i] = {'file':fileName, 't': int(index)}
#     break
    
dataSetDict = {'training' : trainDict, 'validation': validationDict}

import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

dataDict = {}

for e in range(len(validationLosses)):
    dataDict['%03d' % (e+1)] = {"validation": validationLosses[0], "training": trainingEpochLosses[0]}
#     break

modelData = {'hyperParameters' : hyperParameterDict, 'dataSet': dataSetDict, 'epochData': dataDict, 'files': simulationFiles}



encodedNumpyData = json.dumps(modelData, cls=NumpyArrayEncoder, indent=4) 
with open('./trainingData/%s/results.json' % exportString, "w") as write_file:
    json.dump(modelData, write_file, cls=NumpyArrayEncoder, indent=4) 


# plotLossesv2(trainingEpochLosses, logScale = True)
fig, axis = plotLossesv2(trainingEpochLosses, logScale = True)
fig.savefig('./trainingData/%s/training.png' % exportString, dpi = 300)

plotLossesv2(validationLosses, logScale = True)
fig.savefig('./trainingData/%s/validation.png' % exportString, dpi = 300)

import seaborn as sns
import pandas as pd

ei = -1
# epochLoss = {'mean': overallLosses[ei,:,0], 'max': overallLosses[ei,:,1], 'min': overallLosses[ei,:,1], 'stddev': overallLosses[ei,:,1]}
# epochLoss = pd.DataFrame(data = epochLoss)

overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)

fig.suptitle('Training')

plt.sca(axis[0,0])
axis[0,0].set_title('Mean Loss')
axis[0,1].set_title('Max Loss')
axis[0,2].set_title('Std dev Loss')

# debugPrint(epochLosses[0].shape)

# sns.kdeplot(epochLoss, x='mean', bw_adjust=.2, log_scale=True)

for ei in range(overallLosses.shape[0]):
#     epochLoss = {'mean': overallLosses[ei,:,0], 'max': overallLosses[ei,:,1], 'min': overallLosses[ei,:,1], 'stddev': overallLosses[ei,:,1]}
#     epochLoss = pd.DataFrame(data = epochLoss)
    plt.sca(axis[0,0])
    sns.kdeplot(overallLosses[ei,:,0], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,1])
    sns.kdeplot(overallLosses[ei,:,1], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,2])
    sns.kdeplot(overallLosses[ei,:,3], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))

# axis[0,0].legend()

fig.tight_layout()

fig.savefig('./trainingData/%s/training_kde.png' % exportString, dpi = 300)

overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in validationLosses])
fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)

fig.suptitle('Training')

plt.sca(axis[0,0])
axis[0,0].set_title('Mean Loss')
axis[0,1].set_title('Max Loss')
axis[0,2].set_title('Std dev Loss')

# debugPrint(epochLosses[0].shape)

# sns.kdeplot(epochLoss, x='mean', bw_adjust=.2, log_scale=True)

for ei in range(overallLosses.shape[0]):
#     epochLoss = {'mean': overallLosses[ei,:,0], 'max': overallLosses[ei,:,1], 'min': overallLosses[ei,:,1], 'stddev': overallLosses[ei,:,1]}
#     epochLoss = pd.DataFrame(data = epochLoss)
    plt.sca(axis[0,0])
    sns.kdeplot(overallLosses[ei,:,0], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,1])
    sns.kdeplot(overallLosses[ei,:,1], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,2])
    sns.kdeplot(overallLosses[ei,:,3], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))

# axis[0,0].legend()

fig.tight_layout()

fig.savefig('./trainingData/%s/validation_kde.png' % exportString, dpi = 300)

exit()


overallEnd = time.perf_counter()   

# print(model.state_dict())

from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

data = {
    'epochs':args.epochs,
    'batch_size':args.batch_size,
    'cutoff':args.cutoff,
    'lr':args.lr,
    'rbf':[args.rbf_x, args.rbf_y],
    'kernel_size':args.n,
    'time': timestamp,
    'compute_time':overallEnd-overallStart,
    'z_training':training, 
    'z_forward':training_fwd,
    'z_validation':validation,
    'z_testing':testing,
    'layerDescription': layerDescription,
    'arch':args.arch}

filename = 'output/%s - rbf %s x %s - epochs %4d - size %4d - batch %4d - seed %4d - arch %s' % \
        (timestamp, args.rbf_x, args.rbf_y, args.epochs, args.n, args.batch_size, args.seed, args.arch)

torch.save(model.state_dict(), filename + '.torch')

with open(filename + '.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

# import sys
# original_stdout = sys.stdout # Save a reference to the original standard output

# with open('profile.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(prof.key_averages().table(sort_by='self_cpu_time_total'))
#     sys.stdout = original_stdout # Reset the standard output to its original value


# prof.export_chrome_trace("trace.json")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import trange, tqdm
import yaml
import warnings
warnings.filterwarnings(action='once')
from datetime import datetime

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from itertools import groupby
import h5py

def getSamples(frames, maxRollOut = 8, chunked = False, trainValidationSplit = 0.8, limitRollOut = False):
    if chunked:
        validationSamples = int(frames * (1 - trainValidationSplit))
        validationSamples = validationSamples - (validationSamples % maxRollOut)
        trainingSamples = frames - validationSamples

        chunks = validationSamples // maxRollOut


    #     for i in range(32):
        marker = np.ones(frames)
        for j in range(chunks):
            while True:
                i = np.random.randint(maxRollOut, frames - maxRollOut)
                if np.any(marker[i:i+maxRollOut] == 0):
                    continue
                marker[i:i+maxRollOut] = 0
                break

        count_dups = [sum(1 for _ in group) for _, group in groupby(marker.tolist())]
        counter = np.zeros(frames, dtype=np.int32)
        cs = np.cumsum(count_dups)
        prev = 1
        k = 0
        for j in range(frames):
            if prev != marker[j]:
                k = k + 1
            counter[j] = np.clip(cs[k] - j,0, maxRollOut)
            if marker[j] == 0:
                counter[j] = -counter[j]
            prev = marker[j]

    #         markers.append(counter)

    #     markers = np.array(markers)
    else:
        validationSamples = int(frames * (1 - trainValidationSplit))
        trainingSamples = frames - validationSamples


    #     for i in range(32):
        marker = np.zeros(frames)
        marker[np.random.choice(frames, trainingSamples, replace = False)] = 1
    #         print(np.random.choice(frames, trainingSamples, replace = False))

        count_dups = [sum(1 for _ in group) for _, group in groupby(marker.tolist())]
        counter = np.zeros(frames, dtype=np.int32)
        cs = np.cumsum(count_dups)
        prev = marker[0]
        k = 0
        for j in range(frames):
            if prev != marker[j]:
                k = k + 1
            counter[j] = np.clip(cs[k] - j,0, maxRollOut)
            if marker[j] == 0:
                counter[j] = -counter[j]
            prev = marker[j]

    #         markers.append(counter)

    #     markers = np.array(markers)
    trainingFrames = np.arange(frames)[counter > 0]
    validationFrames = np.arange(frames)[counter < 0]
    
    if limitRollOut:
        maxIdx = counter.shape[0] - maxRollOut + 1
        c = counter[:maxIdx][np.abs(counter[:maxIdx]) != maxRollOut]
        c = c / np.abs(c) * 8
        counter[:maxIdx][np.abs(counter[:maxIdx]) != maxRollOut] = c
        
    
    return trainingFrames, validationFrames, counter

def splitFile(s, skip = 32, cutoff = 300, chunked = True, maxRollOut = 8, split = True, trainValidationSplit = 0.8, testSplit = 0.1, limitRollOut = False):
    inFile = h5py.File(s, 'r')
    frameCount = int(len(inFile['simulationExport'].keys()) -1) # adjust for bptcls
    inFile.close()
    if cutoff > 0:
        frameCount = min(cutoff+skip, frameCount)
    actualCount = frameCount - 1 - skip
    
    if not split:
        print(frameCount, cutoff, actualCount)
        training, _, counter = getSamples(actualCount, maxRollOut = maxRollOut, chunked = chunked, trainValidationSplit = 1.)
        return s, training + skip, counter
    
    testIndex = frameCount - 1 - int(actualCount * testSplit)
    testSamples = frameCount - 1 - testIndex
    
    # print(frameCount, cutoff, testSamples)
    testingIndices, _, testingCounter = getSamples(testSamples, maxRollOut = maxRollOut, chunked = chunked, trainValidationSplit = 1.)
    testingIndices = testingIndices + testIndex
    
    # print(frameCount, cutoff, testIndex - skip)
    trainingIndices, validationIndices, trainValidationCounter = getSamples(testIndex - skip, maxRollOut = maxRollOut, chunked = chunked, trainValidationSplit = trainValidationSplit, limitRollOut = limitRollOut)
    trainingCounter = trainValidationCounter[trainingIndices]
    validationCounter = -trainValidationCounter[validationIndices]
    
    trainingIndices = trainingIndices + skip
    validationIndices = validationIndices + skip
    
    # print(trainingIndices.shape[0])
    # print(validationIndices.shape[0])
    # print(testingIndices.shape[0])
    
    return s, (trainingIndices, trainingCounter), (validationIndices, validationCounter), (testingIndices, testingCounter)
    

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


class datasetLoader(Dataset):
    def __init__(self, data):
        self.frameCounts = [indices[0].shape[0] for s, indices in data]
        self.fileNames = [s for s, indices in data]
        
        self.indices = [indices[0] for s, indices in data]
        self.counters = [indices[1] for s, indices in data]
        
#         print(frameCounts)
        
        
    def __len__(self):
#         print('len', np.sum(self.frameCounts))
        return np.sum(self.frameCounts)
    
    def __getitem__(self, idx):
#         print(idx , ' / ', np.sum(self.frameCounts))
        cs = np.cumsum(self.frameCounts)
        p = 0
        for i in range(cs.shape[0]):
#             print(p, idx, cs[i])
            if idx < cs[i] and idx >= p:
#                 print('Found index ', idx, 'in dataset ', i)
#                 print('Loading frame ', self.indices[i][idx - p], ' from dataset ', i, ' for ', idx, p)
                return self.fileNames[i], self.indices[i][idx - p], self.counters[i][idx-p]
        

                return (i, self.indices[i][idx - p]), (i, self.indices[i][idx-p])
#                 return torch.rand(10,1), 2
            p = cs[i]
        return None, None

# from pytorchSPH.neighborhood import *
# from pytorchSPH.periodicBC import *
# from pytorchSPH.solidBC import *
# from pytorchSPH.sph import *

# from sphUtils import *

# def loadFrame(simFile, frameIdx, compute = True):
#     inFile = h5py.File(simFile)
#     grp = inFile['%04d' % frameIdx]
#     cached = {                
#                 'position' : torch.from_numpy(grp['position'][:]),
#                 # 'features' : torch.from_numpy(grp['features'][:]),
#                 'outPosition' : torch.from_numpy(grp['positionAfterShift'][:]),
#                 'velocity' : torch.from_numpy(grp['velocity'][:]),
#                 'area' : torch.from_numpy(grp['area'][:]),
#                 'density' : torch.from_numpy(grp['density'][:]),
#                 'ghostIndices' : torch.from_numpy(grp['ghostIndices'][:]),
#                 'finalPosition' : torch.from_numpy(grp['positionAfterStep'][:]),
#                 'shiftedPosition': torch.from_numpy(grp['positionAfterShift'][:]),
#                 'UID' : torch.from_numpy(grp['UID'][:]),
#                 'boundaryIntegral' : torch.from_numpy(grp['boundaryIntegral'][:]),
#                 'boundaryGradient' : torch.from_numpy(grp['boundaryGradient'][:]),
#                 'support': inFile.attrs['support'],
#                 'dt': inFile.attrs['dt'],
#                 'radius': inFile.attrs['radius'],
#                     }
    
#     config = {}
#     config['dt'] = inFile.attrs['dt']
#     config['area'] = inFile.attrs['area']
#     config['support'] = inFile.attrs['support']
#     config['radius'] = inFile.attrs['radius']
#     config['viscosityConstant'] = inFile.attrs['viscosityConstant']
#     config['boundaryViscosityConstant'] = inFile.attrs['boundaryViscosityConstant']
#     config['packing'] = inFile.attrs['packing']
#     config['spacing'] = inFile.attrs['spacing']
#     config['spacingContribution'] = inFile.attrs['spacingContribution']
#     config['precision'] = torch.float32
#     config['device'] = 'cuda'

#     config['domain'] = ast.literal_eval(inFile.attrs['domain'])
#     config['solidBoundary'] = [ast.literal_eval(v) for v in inFile.attrs['solidBoundary']]
#     config['velocitySources'] = [ast.literal_eval(v) for v in inFile.attrs['velocitySources']]
#     config['emitters'] = [ast.literal_eval(v) for v in inFile.attrs['emitters']]
#     config['dfsph'] = ast.literal_eval(inFile.attrs['dfsph'])

#     config['max_neighbors'] = 256

#     for b in config['solidBoundary']:
#         b['polygon'] = torch.tensor(b['polygon']).to(config['device']).type(config['precision'])
#     #     print(b['polygon'])
#     state = {}
#     state['fluidPosition'] = cached['position'].type(config['precision']).to(config['device'])
#     state['UID'] = cached['UID'].to(config['device'])
#     state['fluidArea'] = torch.ones(state['fluidPosition'].shape[0], dtype=config['precision'], device=config['device']) * config['area']


#     state['realParticles'] = torch.sum(cached['ghostIndices'] == -1).item()
#     state['numParticles'] = state['fluidPosition'].shape[0]
#     # state['fluidPosition'] = cached['position'].type(config['precision']).to(config['device'])

#     if compute:    
#         enforcePeriodicBC(config, state)


#         state['fluidNeighbors'], state['fluidDistances'], state['fluidRadialDistances'] = \
#             neighborSearch(state['fluidPosition'], state['fluidPosition'], config, state)

#         state['boundaryNeighbors'], state['boundaryDistances'], state['boundaryGradients'], \
#             state['boundaryIntegrals'], state['boundaryIntegralGradients'], \
#             state['boundaryFluidNeighbors'], state['boundaryFluidPositions'] = boundaryNeighborSearch(config, state)

#         state['fluidDensity'] = sphDensity(config, state)  

#         state['fluidVelocity'] = torch.from_numpy(grp['velocity'][:]).type(config['precision']).to(config['device'])

#         state['velocityAfterBC'] = torch.from_numpy(grp['velocityAfterBC'][:]).type(config['precision']).to(config['device'])
#         state['positionAfterStep'] = torch.from_numpy(grp['positionAfterStep'][:]).type(config['precision']).to(config['device'])
#         state['positionAfterShift'] = torch.from_numpy(grp['positionAfterShift'][:]).type(config['precision']).to(config['device'])

#         computeGamma(config, state)
        
#     state['time'] = frameIdx * config['dt']
#     state['timestep'] = frameIdx

#     inFile.close()
    
#     return config, state

# def prepareInput(config, state):
#     positions = state['fluidPosition']
    
#     areas = state['fluidArea']
#     velocities = state['fluidVelocity']
#     bIntegral = torch.zeros(state['fluidArea'].shape).to(config['device']).type(config['precision'])
#     bGradient = torch.zeros(state['fluidVelocity'].shape).to(config['device']).type(config['precision'])
#     # if state['boundaryNeighbors'].size() != 0:
#         # bIntegral[state['boundaryNeighbors'][0]] = state['boundaryIntegrals']
#         # bGradient[state['boundaryNeighbors'][0]] = state['boundaryIntegralGradients']
    
#     gamma = state['fluidGamma']
    
#     features = torch.hstack((areas[:,None], velocities, bIntegral[:,None], bGradient, gamma[:,None]))
    
#     return positions, features

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


class datasetLoader(Dataset):
    def __init__(self, data):
        self.frameCounts = [indices[0].shape[0] for s, indices in data]
        self.fileNames = [s for s, indices in data]
        
        self.indices = [indices[0] for s, indices in data]
        self.counters = [indices[1] for s, indices in data]
        
#         print(frameCounts)
        
        
    def __len__(self):
#         print('len', np.sum(self.frameCounts))
        return np.sum(self.frameCounts)
    
    def __getitem__(self, idx):
#         print(idx , ' / ', np.sum(self.frameCounts))
        cs = np.cumsum(self.frameCounts)
        p = 0
        for i in range(cs.shape[0]):
#             print(p, idx, cs[i])
            if idx < cs[i] and idx >= p:
#                 print('Found index ', idx, 'in dataset ', i)
#                 print('Loading frame ', self.indices[i][idx - p], ' from dataset ', i, ' for ', idx, p)
                return self.fileNames[i], self.indices[i][idx - p], self.counters[i][idx-p]
        

                return (i, self.indices[i][idx - p]), (i, self.indices[i][idx-p])
#                 return torch.rand(10,1), 2
            p = cs[i]
        return None, None



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


def loadData(dataset, index, featureFun, unroll = 1, frameDistance = 1):
    fileName, frameIndex, maxRollouts = dataset[index]

    attributes, inputData, groundTruthData = loadFrame(fileName, frameIndex, 1 + np.arange(unroll), frameDistance = frameDistance)
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = featureFun(inputData)
    
    return attributes, fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, groundTruthData


def loadBatch(train_ds, bdata, featureFun, unroll = 1, frameDistance = 1):
    fluidPositions = []
    boundaryPositions = []
    fluidFeatures = []
    boundaryFeatures = []
    fluidBatchIndices = []
    boundaryBatchIndices = []
    groundTruths = []
    for i in range(unroll):
        groundTruths.append([])
    
    for i,b in enumerate(bdata):
#         debugPrint(i)
#         debugPrint(b)
        attribute, fluidPosition, boundaryPosition, fluidFeature, boundaryFeature, groundTruth = loadData(train_ds, b, featureFun, unroll = unroll, frameDistance = frameDistance)     
#         debugPrint(groundTruth)
        fluidPositions.append(fluidPosition)
#         debugPrint(fluidPositions)
        boundaryPositions.append(boundaryPosition)
        fluidFeatures.append(fluidFeature)
        boundaryFeatures.append(boundaryFeature)
        
        batchIndex = torch.ones(fluidPosition.shape[0]) * i
        fluidBatchIndices.append(batchIndex)
        
        batchIndex = torch.ones(boundaryPosition.shape[0]) * i
        boundaryBatchIndices.append(batchIndex)
        for u in range(unroll):
            groundTruths[u].append(groundTruth[u])
        
    fluidPositions = torch.vstack(fluidPositions)
    boundaryPositions = torch.vstack(boundaryPositions)
    fluidFeatures = torch.vstack(fluidFeatures)
    boundaryFeatures = torch.vstack(boundaryFeatures)
    fluidBatchIndices = torch.hstack(fluidBatchIndices)
    boundaryBatchIndices = torch.hstack(boundaryBatchIndices)
    for u in range(unroll):
        groundTruths[u] = torch.vstack(groundTruths[u])
    
    return fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatchIndices, boundaryBatchIndices, groundTruths

import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def getWindowFunction(windowFunction):
    windowFn = None
    if windowFunction == 'cubicSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3 - 4 * torch.clamp(1/2 - r, min = 0) ** 3
    if windowFunction == 'quarticSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 4 - 5 * torch.clamp(3/5 - r, min = 0) ** 4 + 10 * torch.clamp(1/5- r, min = 0) ** 4
    if windowFunction == 'quinticSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 5 - 6 * torch.clamp(2/3 - r, min = 0) ** 5 + 15 * torch.clamp(1/3 - r, min = 0) ** 5
    if windowFunction == 'Wendland2_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3 * (1 + 3 * r)
    if windowFunction == 'Wendland4_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 5 * (1 + 5 * r + 8 * r**2)
    if windowFunction == 'Wendland6_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 7 * (1 + 7 * r + 19 * r**2 + 21 * r**3)
    if windowFunction == 'Wendland2':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 4 * (1 + 4 * r)
    if windowFunction == 'Wendland4':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 6 * (1 + 6 * r + 35/3 * r**2)
    if windowFunction == 'Wendland6':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 8 * (1 + 8 * r + 25 * r**2 + 32 * r**3)
    if windowFunction == 'Hoct4':
        def hoct4(x):
            alpha = 0.0927 # Subject to 0 = (1 − α)** nk−2 + A(γ − α)**nk−2 + B(β − α)**nk−2
            beta = 0.5 # Free parameter
            gamma = 0.75 # Free parameter
            nk = 4 # order of kernel

            A = (1 - beta**2) / (gamma ** (nk - 3) * (gamma ** 2 - beta ** 2))
            B = - (1 + A * gamma ** (nk - 1)) / (beta ** (nk - 1))
            P = -nk * (1 - alpha) ** (nk - 1) - nk * A * (gamma - alpha) ** (nk - 1) - nk * B * (beta - alpha) ** (nk - 1)
            Q = (1 - alpha) ** nk + A * (gamma - alpha) ** nk + B * (beta - alpha) ** nk - P * alpha

            termA = P * x + Q
            termB = (1 - x) ** nk + A * (gamma - x) ** nk + B * (beta - x) ** nk
            termC = (1 - x) ** nk + A * (gamma - x) ** nk
            termD = (1 - x) ** nk
            termE = 0 * x

            termA[x > alpha] = 0
            termB[x <= alpha] = 0
            termB[x > beta] = 0
            termC[x <= beta] = 0
            termC[x > gamma] = 0
            termD[x <= gamma] = 0
            termD[x > 1] = 0
            termE[x < 1] = 0

            return termA + termB + termC + termD + termE

        windowFn = lambda r: hoct4(r)
    if windowFunction == 'Spiky':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3
    if windowFunction == 'Mueller':
        windowFn = lambda r: torch.clamp(1 - r ** 2, min = 0) ** 3
    if windowFunction == 'Parabola':
        windowFn = lambda r: torch.clamp(1 - r**2, min = 0)
    if windowFunction == 'Linear':
        windowFn = lambda r: torch.clamp(1 - r, min = 0)
    return windowFn
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
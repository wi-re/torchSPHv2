# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
import torch
from torch_geometric.loader import DataLoader
import argparse
from torch_geometric.nn import radius
from torch.optim import Adam
import copy
import torch
from torch_geometric.loader import DataLoader
import argparse
from torch_geometric.nn import radius
from torch.optim import Adam
import matplotlib.pyplot as plt
import portalocker
import seaborn as sns


from cutlass import *
from rbfConv import *
from datautils import *
from plotting import *

# Use dark theme
from tqdm import tqdm
import os



class DensityNet(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures, layers = [32,64,64,2], denseLayer = True, acitvation = 'relu',
                coordinateMapping = 'polar', n = 8, m = 8, windowFn = None, rbf_x = 'linear', rbf_y = 'linear', batchSize = 32):
        super().__init__()
        self.features = copy.copy(layers)
        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.relu = getattr(nn.functional, 'relu')
        # if args.verbose: 
            # print('Creating fluid layer %d -> %d with [%dx%d] @ [%sx%s]' %(fluidFeatures, 1,n,m,rbf_x,rbf_y))
        self.convs.append(RbfConv(
            in_channels = fluidFeatures, out_channels = 1,
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_y],
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        # if args.verbose: 
            # print('Creating fluid layer %d -> %d with [%dx%d] @ [%sx%s]' %(fluidFeatures, 1,n,m,rbf_x,rbf_y))
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
                attributes, fluidBatches = None, boundaryBatches = None):
        fi, fj = radius(fluidPositions, fluidPositions, attributes['support'], max_num_neighbors = 256, batch_x = fluidBatches, batch_y = fluidBatches)
        bf, bb = radius(boundaryPositions, fluidPositions, attributes['support'], max_num_neighbors = 256, batch_x = boundaryBatches, batch_y = fluidBatches)
        
        i, ni = torch.unique(fi, return_counts = True)
        b, nb = torch.unique(bf, return_counts = True)
        ni[i[b]] += nb

        self.li = torch.exp(-1 / np.float32(attributes['targetNeighbors']) * ni)
        
        boundaryEdgeIndex = torch.stack([bf, bb], dim = 0)
        boundaryEdgeLengths = (boundaryPositions[boundaryEdgeIndex[1]] - fluidPositions[boundaryEdgeIndex[0]])/attributes['support']
        boundaryEdgeLengths = boundaryEdgeLengths.clamp(-1,1)
            
        fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
        fluidEdgeLengths = (fluidPositions[fluidEdgeIndex[1]] - fluidPositions[fluidEdgeIndex[0]])/attributes['support']
        fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
        
        boundaryConvolution = self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths)
        fluidConvolution = self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths)
        return fluidConvolution  + boundaryConvolution



#  semi implicit euler, network predicts velocity update
def integrateState(attributes, inputPositions, inputVelocities, modelOutput):
    predictedVelocity = modelOutput #inputVelocities +  modelOutput 
    predictedPosition = inputPositions + attributes['dt'] * predictedVelocity
    
    return predictedPosition, predictedVelocity
# velocity loss
def computeLoss(predictedPosition, predictedVelocity, groundTruth, modelOutput):
#     debugPrint(modelOutput.shape)
#     debugPrint(groundTruth.shape)
#     return torch.sqrt((modelOutput - groundTruth[:,-1:].to(device))**2)
    return torch.abs(modelOutput - groundTruth[:,-1:].to(modelOutput.device))
    return torch.linalg.norm(groundTruth[:,2:] - predictedVelocity, dim = 1)


def constructFluidFeatures(attributes, inputData):
    fluidFeatures = torch.hstack(\
                (torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1), \
                 inputData['fluidVelocity'].type(torch.float32), 
                 inputData['fluidGravity'].type(torch.float32)))

    fluidFeatures = torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    fluidFeatures[:,0] *= 7 / np.pi * inputData['fluidArea']  / attributes['support']**2
    
    boundaryFeatures = inputData['boundaryNormal'].type(torch.float32)
    boundaryFeatures = torch.ones(inputData['boundaryNormal'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    boundaryFeatures[:,0] *=  7 / np.pi * inputData['boundaryArea']  / attributes['support']**2
    
    return inputData['fluidPosition'].type(torch.float32), inputData['boundaryPosition'].type(torch.float32), fluidFeatures, boundaryFeatures


def processBatch(model, device, li, attributes, e, unroll, train_ds, bdata, frameDistance):
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches, groundTruths = loadBatch(train_ds, bdata, constructFluidFeatures, unroll, frameDistance)    
    

    predictedPositions = fluidPositions.to(device)
    predictedVelocity = fluidFeatures[:,1:3].to(device)
    
    bLosses = []
    boundaryPositions = boundaryPositions.to(device)
    fluidFeatures = fluidFeatures.to(device)
    boundaryFeatures = boundaryFeatures.to(device)
    fluidBatches = fluidBatches.to(device)
    boundaryBatches = boundaryBatches.to(device)
            
    for u in range(unroll):
        predictions = model(predictedPositions, boundaryPositions, fluidFeatures, boundaryFeatures, attributes, fluidBatches, boundaryBatches)

        predictedPositions, predictedVelocities = integrateState(attributes, predictedPositions, predictedVelocity, predictions)
        
        fluidFeatures = torch.hstack((fluidFeatures[:,0][:,None], predictedVelocity, fluidFeatures[:,3:]))
        if li:
            loss = model.li * computeLoss(predictedPositions, predictedVelocities, groundTruths[u].to(device), predictions)
        else:
            loss = computeLoss(predictedPositions, predictedVelocities, groundTruths[u].to(device), predictions)

        batchedLoss = []
        for i in range(len(bdata)):
            L = loss[fluidBatches == i]
            Lterms = (torch.mean(L), torch.max(torch.abs(L)), torch.min(torch.abs(L)), torch.std(L))            
            batchedLoss.append(torch.hstack(Lterms))
        batchedLoss = torch.vstack(batchedLoss).unsqueeze(0)
        bLosses.append(batchedLoss)
    
    bLosses = torch.vstack(bLosses)
    maxLosses = torch.max(bLosses[:,:,1], dim = 0)[0]
    minLosses = torch.min(bLosses[:,:,2], dim = 0)[0]
    meanLosses = torch.mean(bLosses[:,:,0], dim = 0)
    stdLosses = torch.mean(bLosses[:,:,3], dim = 0)
    
    
    del predictedPositions, predictedVelocities, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches
    
    bLosses = bLosses.transpose(0,1)
    
    return bLosses, meanLosses, minLosses, maxLosses, stdLosses
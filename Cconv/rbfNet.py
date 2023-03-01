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

class RbfNet(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures, layers = [32,64,64,2], denseLayer = True, activation = 'relu',
                coordinateMapping = 'polar', n = 8, m = 8, windowFn = None, rbf_x = 'linear', rbf_y = 'linear', batchSize = 32, ignoreCenter = True):
        super().__init__()
        self.centerIgnore = ignoreCenter
#         debugPrint(layers)
        
        self.features = copy.copy(layers)
#         debugPrint(fluidFeatures)
#         debugPrint(boundaryFeatures)
        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.relu = getattr(nn.functional, 'relu')
#         debugPrint(fluidFeatures)

        self.convs.append(RbfConv(
            in_channels = fluidFeatures, out_channels = self.features[0],
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_y],
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = activation,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        self.convs.append(RbfConv(
            in_channels = boundaryFeatures, out_channels = self.features[0],
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_y],
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = activation,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        self.fcs.append(nn.Linear(in_features=fluidFeatures,out_features= layers[0],bias=False))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)

        self.features[0] = self.features[0]
#         self.fcs.append(nn.Linear(in_features=96,out_features= 2,bias=False))
        
        for i, l in enumerate(layers[1:-1]):
#             debugPrint(layers[i])
#             debugPrint(layers[i+1])
            self.convs.append(RbfConv(
                in_channels = (3 * self.features[0]) if i == 0 else self.features[i], out_channels = layers[i+1],
                dim = 2, size = [n,m],
                rbf = [rbf_x, rbf_y],
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
            self.fcs.append(nn.Linear(in_features=3 * layers[0] if i == 0 else layers[i],out_features=layers[i+1],bias=False))
            torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
            
        self.convs.append(RbfConv(
            in_channels = self.features[-2], out_channels = self.features[-1],
                dim = 2, size = [n,m],
                rbf = [rbf_x, rbf_y],
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        self.fcs.append(nn.Linear(in_features=layers[-2],out_features=self.features[-1],bias=False))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)


    def forward(self, \
                fluidPositions, boundaryPositions, \
                fluidFeatures, boundaryFeatures,\
                attributes, fluidBatches = None, boundaryBatches = None):
        fi, fj = radius(fluidPositions, fluidPositions, attributes['support'], max_num_neighbors = 256, batch_x = fluidBatches, batch_y = fluidBatches)
        bf, bb = radius(boundaryPositions, fluidPositions, attributes['support'], max_num_neighbors = 256, batch_x = boundaryBatches, batch_y = fluidBatches)
        if self.centerIgnore:
            nequals = fi != fj

        i, ni = torch.unique(fi, return_counts = True)
        b, nb = torch.unique(bf, return_counts = True)
        ni[i[b]] += nb
        self.li = torch.exp(-1 / np.float32(attributes['targetNeighbors']) * ni)
        
        boundaryEdgeIndex = torch.stack([bf, bb], dim = 0)
        boundaryEdgeLengths = (boundaryPositions[boundaryEdgeIndex[1]] - fluidPositions[boundaryEdgeIndex[0]])/attributes['support']
        boundaryEdgeLengths = boundaryEdgeLengths.clamp(-1,1)
        if self.centerIgnore:
            fluidEdgeIndex = torch.stack([fi[nequals], fj[nequals]], dim = 0)
        else:
            fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
        fluidEdgeLengths = -(fluidPositions[fluidEdgeIndex[1]] - fluidPositions[fluidEdgeIndex[0]])/attributes['support']
        fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
            
        linearOutput = self.relu(self.fcs[0](fluidFeatures))
        boundaryConvolution = self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths)
        fluidConvolution = self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths)
        ans = torch.hstack((linearOutput, fluidConvolution, boundaryConvolution))
        
        layers = len(self.convs)
        for i in range(2,layers):
            ansConv = self.convs[i]((ans, ans), fluidEdgeIndex, fluidEdgeLengths)
            ansDense = self.fcs[i - 1](ans)
            if self.features[i-2] == self.features[i-1] and ans.shape == ansConv.shape:
                ans = ansConv + ansDense + ans
            else:
                ans = ansConv + ansDense
            if i != layers - 1:
                ans = self.relu(ans)
        return ans
            


#  semi implicit euler, network predicts velocity update
def integrateState(attributes, inputPositions, inputVelocities, modelOutput, frameDistance):
    # predictedVelocity = modelOutput #inputVelocities +  modelOutput 
    # predictedPosition = inputPositions + frameDistance * attributes['dt'] * predictedVelocity
    predictedVelocity = modelOutput / (frameDistance * attributes['dt']) #inputVelocities +  modelOutput 
    predictedPosition = inputPositions + modelOutput
    
    return predictedPosition, predictedVelocity
# velocity loss
def computeLoss(predictedPosition, predictedVelocity, groundTruth, modelOutput):
#     debugPrint(modelOutput.shape)
#     debugPrint(groundTruth.shape)
#     return torch.sqrt((modelOutput - groundTruth[:,-1:].to(device))**2)
    # return torch.abs(modelOutput - groundTruth[:,-1:].to(modelOutput.device))
    # return torch.linalg.norm(groundTruth[:,2:4] - predictedVelocity, dim = 1) 
    # debugPrint(groundTruth.shape)
    # debugPrint(predictedPosition.shape)
    # debugPrint(predictedVelocity.shape)
    posLoss = torch.sqrt(torch.linalg.norm(groundTruth[:,:2] - predictedPosition, dim = 1))
    return posLoss
    velLoss = torch.sqrt(torch.linalg.norm(groundTruth[:,2:4] - predictedVelocity, dim = 1))
    return posLoss + velLoss
    # return torch.sqrt(torch.linalg.norm(groundTruth[:,2:4] - modelOutput, dim = 1))

def constructFluidFeatures(attributes, inputData):
    fluidFeatures = torch.hstack(\
                (torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1), \
                 inputData['fluidVelocity'].type(torch.float32), 
                 torch.zeros(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)))
                #  inputData['fluidGravity'].type(torch.float32)))

                #  torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)))

    # fluidFeatures = torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    # fluidFeatures[:,0] *= 7 / np.pi * inputData['fluidArea']  / attributes['support']**2
    
    boundaryFeatures = torch.hstack((inputData['boundaryNormal'].type(torch.float32), torch.zeros(inputData['boundaryNormal'].shape[0]).type(torch.float32).unsqueeze(dim=1)))
    # boundaryFeatures = torch.ones(inputData['boundaryNormal'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    # boundaryFeatures[:,0] *=  7 / np.pi * inputData['boundaryArea']  / attributes['support']**2
    
    return inputData['fluidPosition'].type(torch.float32), inputData['boundaryPosition'].type(torch.float32), fluidFeatures, boundaryFeatures


def processBatch(model, device, li, attributes, e, unroll, train_ds, bdata, frameDistance):
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches, groundTruths = \
        loadBatch(train_ds, bdata, constructFluidFeatures, unroll, frameDistance)    
    

    predictedPositions = fluidPositions.to(device)
    predictedVelocity = fluidFeatures[:,1:3].to(device)
    
    bLosses = []
    boundaryPositions = boundaryPositions.to(device)
    fluidFeatures = fluidFeatures.to(device)
    boundaryFeatures = boundaryFeatures.to(device)
    fluidBatches = fluidBatches.to(device)
    boundaryBatches = boundaryBatches.to(device)
            
    gravity = torch.zeros_like(predictedVelocity)
    gravity[:,1] = -9.81
        
    for u in range(unroll):
# Heun's method:
        vel2 = predictedVelocity + frameDistance * attributes['dt'] * gravity
        pos2 = predictedPositions + frameDistance * attributes['dt'] * (predictedVelocity + vel2) / 2
# semi implicit euler
        d = (frameDistance) * ((frameDistance) + 1) / 2
        vel2 = predictedVelocity + frameDistance * attributes['dt'] * gravity
        pos2 = predictedPositions + frameDistance * attributes['dt'] * predictedVelocity + d * attributes['dt']**2 * gravity

        fluidFeatures = torch.hstack((fluidFeatures[:,0][:,None], vel2, fluidFeatures[:,3:]))

        predictions = model(pos2, boundaryPositions, fluidFeatures, boundaryFeatures, attributes, fluidBatches, boundaryBatches)

        predictedVelocity = (pos2 + predictions - predictedPositions) / (frameDistance * attributes['dt'])
        # predictedPositions = pos2 + predictions
        # predictedVelocity = vel2 + predictions[:,2:]
        predictedPositions = pos2 + predictions[:,:2]


        # predictions = model(predictedPositions, boundaryPositions, fluidFeatures, boundaryFeatures, attributes, fluidBatches, boundaryBatches)
        # predictedPositions, predictedVelocities = integrateState(attributes, predictedPositions, predictedVelocities, predictions, frameDistance)        
        # fluidFeatures = torch.hstack((fluidFeatures[:,0][:,None], predictedVelocities, fluidFeatures[:,3:]))
                
        if li:
            loss = model.li * computeLoss(predictedPositions, predictedVelocity, groundTruths[u].to(device), predictions)
        else:
            loss = computeLoss(predictedPositions, predictedVelocity, groundTruths[u].to(device), predictions)

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
    
    
    del predictedPositions, predictedVelocity, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches
    
    bLosses = bLosses.transpose(0,1)
    
    return bLosses, meanLosses, minLosses, maxLosses, stdLosses



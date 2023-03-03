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
            bias = True,
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        self.convs.append(RbfConv(
            in_channels = boundaryFeatures, out_channels = self.features[0],
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_y],
            bias = True,
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        self.fcs.append(nn.Linear(in_features=fluidFeatures,out_features= layers[0],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)

        self.features[0] = self.features[0]
#         self.fcs.append(nn.Linear(in_features=96,out_features= 2,bias=False))
        
        for i, l in enumerate(layers[1:-1]):
#             debugPrint(layers[i])
#             debugPrint(layers[i+1])
            self.convs.append(RbfConv(
                in_channels = (3 * self.features[0]) if i == 0 else self.features[i], out_channels = layers[i+1],
                dim = 2, size = [n,m],
                rbf = [rbf_x, rbf_y],
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
            self.fcs.append(nn.Linear(in_features=3 * layers[0] if i == 0 else layers[i],out_features=layers[i+1],bias=True))
            torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
            torch.nn.init.zeros_(self.fcs[-1].bias)
            
        self.convs.append(RbfConv(
            in_channels = self.features[-2], out_channels = self.features[-1],
                dim = 2, size = [n,m],
                rbf = [rbf_x, rbf_y],
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        self.fcs.append(nn.Linear(in_features=layers[-2],out_features=self.features[-1],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)


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
        
        self.ni = ni
        self.nb = nb
#         print('ni:', torch.min(ni).detach().cpu().numpy(), torch.median(ni).detach().cpu().numpy(), torch.max(ni).detach().cpu().numpy())
#         print('nb:', torch.min(nb).detach().cpu().numpy(), torch.median(nb).detach().cpu().numpy(), torch.max(nb).detach().cpu().numpy())
        
        ni[i[b]] += nb
        self.li = torch.exp(-1 / 16 * ni)
        
        boundaryEdgeIndex = torch.stack([bf, bb], dim = 0)
        boundaryEdgeLengths = (boundaryPositions[boundaryEdgeIndex[1]] - fluidPositions[boundaryEdgeIndex[0]])/attributes['support']
        boundaryEdgeLengths = boundaryEdgeLengths.clamp(-1,1)
        if self.centerIgnore:
            fluidEdgeIndex = torch.stack([fi[nequals], fj[nequals]], dim = 0)
        else:
            fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
        fluidEdgeLengths = -(fluidPositions[fluidEdgeIndex[1]] - fluidPositions[fluidEdgeIndex[0]])/attributes['support']
        fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
#         if verbose:
#             print('prepared network with inputs:')
#             print('fluidPositions', fluidPositions[:4])
#             print('fluidFeatures', fluidFeatures[:4])
#             print('boundaryPositions', boundaryPositions[:4])
#             print('boundaryFeatures', boundaryFeatures[:4])
#             print('fluid neighbors:', fluidEdgeIndex.shape)
#             print('fluid neighbor distances', fluidEdgeLengths[:4])
#             print('boundary neighbors:', boundaryEdgeIndex.shape)
#             print('boundary neighbor distances', boundaryEdgeLengths[:4])
#             print('num neighbors', ni[:4])
#             print('li', self.li[:4])
            
            
        linearOutput = (self.fcs[0](fluidFeatures))
        boundaryConvolution = (self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths))
        fluidConvolution = (self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths))
        ans = torch.hstack((linearOutput, fluidConvolution, boundaryConvolution))
        if verbose:
#             print('linear output', linearOutput[:4])
#             print('boundary convolution output', boundaryConvolution[:4])
#             print('fluid convolution output', fluidConvolution[:4])
            print('first layer output', ans[:4])
        
        layers = len(self.convs)
        for i in range(2,layers):
            
            ansc = self.relu(ans)
            
            ansConv = self.convs[i]((ansc, ansc), fluidEdgeIndex, fluidEdgeLengths)
            ansDense = self.fcs[i - 1](ansc)
            
            
            if self.features[i-2] == self.features[i-1] and ans.shape == ansConv.shape:
                ans = ansConv + ansDense + ans
            else:
                ans = ansConv + ansDense
#             if verbose:
#                 print('\tlayer', i)
#                 print('\tlinear output', ansDense[:4])
#                 print('\tfluid convolution output', ansConv[:4])
#                 print('\tlayer output', ans[:4])
                
#             if i != layers - 1:
#                 ans = self.relu(ans)
            if verbose:
                print('\tlayer output after activation', ans[:4])
        return ans / 128
            


#  semi implicit euler, network predicts velocity update
def integrateState(attributes, inputPositions, inputVelocities, modelOutput, frameDistance):
    # predictedVelocity = modelOutput #inputVelocities +  modelOutput 
    # predictedPosition = inputPositions + frameDistance * attributes['dt'] * predictedVelocity
    predictedVelocity = modelOutput / (frameDistance * attributes['dt']) #inputVelocities +  modelOutput 
    predictedPosition = inputPositions + modelOutput
    
    return predictedPosition, predictedVelocity
# velocity loss
verbose = False
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
    if verbose:
        print('computing Loss with')
        print('predictedPositions', predictedPosition[:4])
        print('predictedVelocity', predictedVelocity[:4])
        print('groundTruth', groundTruth[:4])
        print('modelOutput', modelOutput[:4])
        print('resulting loss', posLoss[:4])
    return posLoss
    velLoss = torch.sqrt(torch.linalg.norm(groundTruth[:,2:4] - predictedVelocity, dim = 1))
    return posLoss + velLoss
    # return torch.sqrt(torch.linalg.norm(groundTruth[:,2:4] - modelOutput, dim = 1))

def runNetwork(initialPosition, initialVelocity, attributes, frameDistance, gravity, fluidFeatures, boundaryPositions, boundaryFeatures, groundTruth, model,fluidBatches, boundaryBatches, li):
    # if verbose:
    #     print('running network with')
    #     print('initialPosition', initialPosition[:4])
    #     print('initialVelocity', initialVelocity[:4])
    #     print('dt', dt)
    #     print('frameDistance', frameDistance)        
    #     print('gravity', gravity[:4])
    #     print('fluidFeatures', fluidFeatures[:4])
    #     print('boundaryPositions', boundaryPositions[:4])
    #     print('boundaryFeatures', boundaryFeatures[:4])
    #     print('fluidBatches', fluidBatches)
    #     print('boundaryBatches', boundaryBatches)
    #     print('li', li)
# Heun's method:
    # vel2 = initialVelocity + frameDistance * attributes['dt'] * gravity
    # pos2 = initialPosition + frameDistance * attributes['dt'] * (initialVelocity + vel2) / 2
# semi implicit euler
    d = (frameDistance) * ((frameDistance) + 1) / 2
    vel2 = initialVelocity + frameDistance * attributes['dt'] * gravity
    pos2 = initialPosition + frameDistance * attributes['dt'] * initialVelocity + d * attributes['dt']**2 * gravity
        
    fluidFeatures = torch.hstack((fluidFeatures[:,0][:,None], vel2, fluidFeatures[:,3:]))
    # if verbose:
    #     print('calling network with' )
    #     print('d', d)
    #     print('vel2', vel2[:4])
    #     print('pos2', pos2[:4])
    #     print('fluidFeatures', fluidFeatures[:4])
    predictions = model(pos2, boundaryPositions, fluidFeatures, boundaryFeatures, attributes, fluidBatches, boundaryBatches)

    predictedVelocity = (pos2 + predictions[:,:2] - initialPosition) / (frameDistance * attributes['dt'])
    predictedPositions = pos2 + predictions[:,:2]

    if li:
        loss =  model.li * computeLoss(predictedPositions, predictedVelocity, groundTruth.to(pos2.device), predictions)
    else:
        loss =   computeLoss(predictedPositions, predictedVelocity, groundTruth.to(pos2.device), predictions)

    return loss, predictedPositions, predictedVelocity
    
def loadData(dataset, index, featureFun, unroll = 1, frameDistance = 1):
    with record_function("load data - hdf5"): 
        fileName, frameIndex, maxRollouts = dataset[index]

        attributes, inputData, groundTruthData = loadFrame(fileName, frameIndex, 1 + np.arange(unroll), frameDistance = frameDistance)
        attributes['support'] = 4. * attributes['support']
        fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = featureFun(attributes, inputData)

        return attributes, fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, inputData['fluidGravity'], groundTruthData


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
    with record_function("process batch"): 
        fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidGravity, fluidBatches, boundaryBatches, groundTruths = \
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
        gravity = fluidGravity[:,:2].to(device)
        
    #     gravity[:,1] = -9.81

        for u in range(unroll):
            with record_function("prcess batch[unroll]"): 
    #         loss, predictedPositions, predictedVelocity = runNetwork(fluidPositions.to(device), inputData['fluidVelocity'].to(device), attributes['dt'], frameDistance, gravity, fluidFeatures, boundaryPositions.to(device), boundaryFeatures.to(device), groundTruths[0], model, None, None, True)
                loss, predictedPositions, predictedVelocity = runNetwork(predictedPositions, predictedVelocity, attributes, frameDistance, gravity, fluidFeatures, boundaryPositions, boundaryFeatures, groundTruths[u], model, fluidBatches, boundaryBatches, li)

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



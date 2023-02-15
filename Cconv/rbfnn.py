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


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=16)
parser.add_argument('--cutoff', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--rbf_x', type=str, default='rbf gaussian')
parser.add_argument('--rbf_y', type=str, default='rbf gaussian')
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--cutlad', type=bool, default=False)
parser.add_argument('--forwardBatch', type=int, default=16)
parser.add_argument('--backwardBatch', type=int, default=16)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--arch', type=str, default='16 32 32 2')
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

def processBatch(train_ds, bdata, unroll):
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches, groundTruths = loadBatch(train_ds, bdata, unroll)    
    
    predictedPositions = fluidPositions.to(device)
    predictedVelocity = fluidFeatures[:,1:3].to(device)
    
    unrolledLosses = []
    bLosses = []
#     debugPrint(bdata)
    boundaryPositions = boundaryPositions.to(device)
    fluidFeatures = fluidFeatures.to(device)
    boundaryFeatures = boundaryFeatures.to(device)
    
#     debugPrint(bdata)
#     debugPrint(predictedPosition)
    
    ls = []
    
    for u in range(1):
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
    _, train, valid, test = splitFile(s, split = True, limitRollOut = False, skip = 0, cutoff = 800)
    training.append((s,train))
    validation.append((s,valid))
    testing.append((s,test))

batch_size = 4

train_ds = datasetLoader(training)
train_dataloader = DataLoader(train_ds, shuffle=True, batch_size = batch_size).batch_sampler

validation_ds = datasetLoader(validation)
validation_dataloader = DataLoader(validation_ds, shuffle=True, batch_size = batch_size).batch_sampler

fileName, frameIndex, maxRollout = train_ds[len(train_ds)//2]
# frameIndex = 750
attributes, inputData, groundTruthData = loadFrame(simulationFiles[0], 400, 1 + np.arange(1))
fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(inputData)

debugPrint(fluidFeatures.shape)


n = 8
m = 8
coordinateMapping = 'polar'
windowFn = lambda r: torch.clamp(torch.pow(1. - r, 4) * (1.0 + 4.0 * r), min = 0)
rbf_x = 'linear'
rbf_y = 'linear'
initialLR = 1e-2
maxRollOut = 10
epochs = 25
frameDistance = 0




model = DensityNet(fluidFeatures.shape[1], boundaryFeatures.shape[1], coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = 64)



lr = initialLR
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if args.gpus == 1:
    print('Number of parameters', count_parameters(model))

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

debugPrint(hyperParameterDict)
debugPrint(exportString)
exit() 


# exportPath = './trainingData/%s - %s.hdf5' %(self.config['export']['prefix'], timestamp)
if not os.path.exists('./trainingData/%s' % exportString):
    os.makedirs('./trainingData/%s' % exportString)
# self.outFile = h5py.File(self.exportPath,'w')



train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True).batch_sampler
train_dataloader_fwd = DataLoader(train_ds, batch_size=1, shuffle=False).batch_sampler
valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=False).batch_sampler
test_dataloader  = DataLoader(test_ds,  batch_size=1, shuffle=False).batch_sampler

epochs = args.epochs
i = 0

import portalocker
gtqdms = []
with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(1, epochs + 1), position = g, leave = True))
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(1, epochs + 1), position = args.gpus + g, leave = True))
# print(torch.cuda.current_device())
def run(dataloader, dataset, description, train = False):
    # gtqdms[args.gpus + args.gpu] = tqdm(dataloader, leave = False, position = args.gpu + args.gpus)
    pb = gtqdms[args.gpu + args.gpus]
    losses = []
    batchHistory = []
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.reset(total=len(dataloader))
    i = 0
    for bdata in dataloader:
        # batchHistory.append(bdata)
        positions, features, persistent_output, ghostIndices, batches, persistent_batches, gt, support, batchIndices = prepareData(bdata, dataset, device)
        batchHistory.append(batchIndices)
        t_start = time.perf_counter()
        if train:
            model.train()
        else:
            model.train(mode=False)


        with record_function("forward"):
            optimizer.zero_grad()
            out = model(positions, features, persistent_output,
                    ghostIndices, support, batches, persistent_batches)
            # loss = (gt - out)**2 / 2
            # loss = torch.mean(loss)

            diff = out - gt

            l1 = torch.mean(torch.abs(diff))
            l2 = torch.mean(diff * diff)
            linfty = torch.max(torch.abs(diff))

            # print(l1,l2,linfty)
            losses.append([
                l1.detach().cpu().numpy(),
                l2.detach().cpu().numpy(),
                linfty.detach().cpu().numpy()]
                )
            loss = l1 + l2 + linfty

            string_ints = ["%4d" %int for int in bdata]
            str_of_ints = ",".join(string_ints)
            with portalocker.Lock('README.md', flags = 0x2, timeout = None):
                pb.set_description("[gpu %d @ %18s | %18s] %s: batch [%s] -> Loss: %1.4e | %1.4e | %1.4e" % (args.gpu, args.rbf_x, args.rbf_y, description, str_of_ints, l1,l2,linfty))
                # pb.set_description("[gpu %d @ %18s | %d] %s: batch [%s] -> Loss: %1.4e | %1.4e | %1.4e" % (args.gpu, args.arch, args.n, description, str_of_ints, l1,l2,linfty))
                # pb.refresh()
                i = i + 1
                # if i != len(dataloader) -1:
                pb.update()

        with record_function("synchronize before backward"):    
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if train:
            with record_function("backward"):
                loss.backward()

            with record_function("optimizer step"):
                optimizer.step()

        # del loss, out, gt, p_batches, batches, ghosts, output, features, positions

        with record_function("synchronize step"): 
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            
        t_end = time.perf_counter()
    # print('Average Loss for %s: %1.4e [%5d frames]'%(description, np.mean(np.array(losses)),len(dataset)))

    return batchHistory, np.array(losses)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True, with_flops=True) as prof:    

training = {}
training_fwd = {}
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

for epoch in range(args.epochs):
    t_batches, train_losses = run(train_dataloader, train_ds, 'train', train = True)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e' %(args.gpu, np.mean(train_losses)))

    f_batches, train_losses_fwd = run(train_dataloader_fwd, train_ds, 'train (foward only)', train = False)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd)))

    v_batches, valid_losses = run(valid_dataloader, valid_ds, 'valid', train = False)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e Validation: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd), np.mean(valid_losses)))

    s_batches, test_losses  = run(test_dataloader,  test_ds,  'test ', train = False)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e Validation: %1.4e Testing: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd), np.mean(valid_losses), np.mean(test_losses)))

    training[epoch] = {}
    training_fwd[epoch] = {}
    validation[epoch] = {}
    testing[epoch] = {}

    if(args.batch_size == 1):
        for i, (batch, loss) in enumerate(zip(t_batches, train_losses)):
            training[epoch]['%4d %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]
    else:
        for i, (batch, loss) in enumerate(zip(t_batches, train_losses)):
            b = ['%s %s' % (ba[0], ba[1]) for ba in batch]
            training[epoch]['%4d'% i + ', '.join(b)] = [float(loss[0]),float(loss[1]),float(loss[2])]
    for i, (batch, loss) in enumerate(zip(f_batches, train_losses_fwd)):
        training_fwd[epoch]['%4d - %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]
    for i, (batch, loss) in enumerate(zip(v_batches, valid_losses)):
        validation[epoch]['%4d - %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]
    for i, (batch, loss) in enumerate(zip(s_batches, test_losses)):
        testing[epoch]['%4d - %s %s' %(i, batch[0][0], int(batch[0][1]))] = [float(loss[0]),float(loss[1]),float(loss[2])]

    # print(' Training     Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(train_losses), np.median(train_losses), np.max(train_losses), len(train_losses)))
    # print('Training fwd Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(train_losses_fwd), np.median(train_losses_fwd), np.max(train_losses_fwd), len(train_losses_fwd)))
    # print('Validation   Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(valid_losses), np.median(valid_losses), np.max(valid_losses), len(valid_losses)))
    # print('Testing      Loss: [%1.4e - %1.4e - %1.4e] for %4d timesteps' % (np.min(test_losses), np.median(test_losses), np.max(test_losses), len(test_losses)))

    if epoch % args.lr_decay_step_size == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_decay_factor * param_group['lr']
            
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.update()
         
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

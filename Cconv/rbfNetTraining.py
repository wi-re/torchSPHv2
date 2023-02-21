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


parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', type=int, default=25)
parser.add_argument('-cmap','--coordinateMapping', type=str, default='preserving')
parser.add_argument('-w','--windowFunction', type=str, default='Wendland2')
parser.add_argument('-c','--cutoff', type=int, default=1800)
parser.add_argument('-b','--batch_size', type=int, default=8)
parser.add_argument('--cutlassBatchSize', type=int, default=128)
parser.add_argument('-r','--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('-x','--rbf_x', type=str, default='linear')
parser.add_argument('-y','--rbf_y', type=str, default='linear')
parser.add_argument('-n','--n', type=int, default=5)
parser.add_argument('-m','--m', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('-d','--frameDistance', type=int, default=1)
parser.add_argument('--dataDistance', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('-f','--forwardLoss', type=bool, default=False)
parser.add_argument('-v','--verbose', type=bool, default=False)
parser.add_argument('-l','--li', type=bool, default=True)
parser.add_argument('-a','--activation', type=str, default='relu')
parser.add_argument('--arch', type=str, default='32 64 64 2')
parser.add_argument('--limitData', type=int, default=-1)

args = parser.parse_args()

if args.verbose:
    print('Setting all rng seeds to %d' % args.seed)
import random 
import numpy as np

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

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
plt.style.use('dark_background')
from tqdm import tqdm
import os

from rbfNet import *


if args.verbose:
    print('Parsing data in ../export')
basePath = '../export'
basePath = os.path.expanduser(basePath)

simulationFiles = [basePath + '/' + f for f in os.listdir(basePath) if f.endswith('.hdf5')]

if args.limitData > 0:
    files = []
    for i in range(args.limitData):
        files.append(simulationFiles[i])
    simulationFiles = files
# simulationFiles = [simulationFiles[0]]
if args.verbose:
    print('Input files:')
    for i, c in enumerate(simulationFiles):
        print('\t', i ,c)

training = []
validation = []
testing = []


for s in simulationFiles:    
    _, train, valid, test = splitFile(s, split = True, limitRollOut = False, skip = 0, cutoff = args.cutoff, distance = args.dataDistance)
    training.append((s,train))
    validation.append((s,valid))
    testing.append((s,test))

if args.verbose:
    print('Processed data into datasets:')
    debugPrint(training)
    debugPrint(validation)
    debugPrint(testing)

batch_size = args.batch_size

if args.verbose:
    print('Setting up data loaders')
train_ds = datasetLoader(training)
train_dataloader = DataLoader(train_ds, shuffle=True, batch_size = batch_size).batch_sampler

validation_ds = datasetLoader(validation)
validation_dataloader = DataLoader(validation_ds, shuffle=True, batch_size = batch_size).batch_sampler

if args.verbose:
    print('Setting up network parameters:')
fileName, frameIndex, maxRollout = train_ds[len(train_ds)//2]
attributes, inputData, groundTruthData = loadFrame(simulationFiles[0], 0, 1 + np.arange(1))
fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(attributes, inputData)

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
    print('Rollout limit (if applicable):', maxRollout)
    print('Training with frame offset of', frameDistance)
    print('Network architecture', args.arch)


widths = args.arch.strip().split(' ')
layers = [int(s) for s in widths]
# debugPrint(layers)
if args.verbose:
    print('Building Network')
model = RbfNet(fluidFeatures.shape[1], boundaryFeatures.shape[1], layers = layers, coordinateMapping = coordinateMapping, n = n, m = m, windowFn = windowFn, rbf_x = rbf_x, rbf_y = rbf_y, batchSize = args.cutlassBatchSize)



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
hyperParameterDict['windowFunction'] =  args.windowFunction
hyperParameterDict['liLoss'] = 'yes' if args.li else 'no'
hyperParameterDict['initialLR'] = initialLR
hyperParameterDict['maxRollOut'] = maxRollOut
hyperParameterDict['epochs'] = epochs
hyperParameterDict['frameDistance'] = frameDistance
hyperParameterDict['dataDistance'] = args.dataDistance
hyperParameterDict['parameters'] =  count_parameters(model)
hyperParameterDict['cutoff'] =  args.cutoff
hyperParameterDict['dataLimit'] =  args.limitData 
hyperParameterDict['arch'] =  args.arch
lr = initialLR


timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
networkPrefix = 'RbfNet'

exportString = '%s - n=[%2d,%2d] rbf=[%s,%s] map = %s window = %s d = %2d e = %2d arch %s distance = %2d - %s' % (networkPrefix, hyperParameterDict['n'], hyperParameterDict['m'], hyperParameterDict['rbf_x'], hyperParameterDict['rbf_y'], hyperParameterDict['coordinateMapping'], args.windowFunction, hyperParameterDict['frameDistance'], hyperParameterDict['epochs'], args.arch, frameDistance, timestamp)

# if args.gpus == 1:
#     debugPrint(hyperParameterDict)
# if args.gpus == 1:
#     debugPrint(exportString)
if args.verbose:
    print('Writing output to ./trainingData/%s' % exportString)

# exportPath = './trainingData/%s - %s.hdf5' %(self.config['export']['prefix'], timestamp)
if not os.path.exists('./trainingData/%s' % exportString):
    os.makedirs('./trainingData/%s' % exportString)
# self.outFile = h5py.File(self.exportPath,'w')

gtqdms = []
if args.verbose:
    print('Setting up tqdm progress bars')

with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(0, (epochs) * (len(train_dataloader) + len(validation_dataloader))), position = g, leave = True))
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
        batchLosses, meanLosses, minLosses, maxLosses, stdLosses = processBatch(model, device, args.li, attributes, e, rollout, ds, bdata, frameDistance)
        batchIndices.append(np.array(bdata))
        losses.append(batchLosses.detach().cpu().numpy())
        
        sumLosses = torch.mean(batchLosses[:,:,0]) #+ torch.mean(batchLosses[:,:,1])
        sumLosses.backward()
        if train:
            optimizer.step()
        lossString = np.array2string(torch.mean(batchLosses[:,:,0],dim=0).detach().cpu().numpy(), formatter={'float_kind':lambda x: "%.4e" % x})
        batchString = str(np.array2string(np.array(bdata), formatter={'float_kind':lambda x: "%.2f" % x, 'int':lambda x:'%04d' % x}))
        
        with portalocker.Lock('README.md', flags = 0x2, timeout = None):
            pbl.set_description('%24s[gpu %d]: %3d [%1d] @ %1.5e: :  %s -> %.4e' %(prefix, args.gpu, e, rollout, lr, lossString, sumLosses.detach().cpu().numpy()))
            pbl.update()
            if prefix == 'training':
                pb.set_description('[gpu %d] Learning: %1.4e Validation: %1.4e' %(args.gpu, np.mean(np.mean(np.vstack(losses)[:,:,0], axis = 1)), validationLoss))
            if prefix == 'validation':
                pb.set_description('[gpu %d] Learning: %1.4e Validation: %1.4e' %(args.gpu, trainLoss, np.mean(np.mean(np.vstack(losses)[:,:,0], axis = 1))))
            pb.update()
    bIndices  = np.hstack(batchIndices)
    losses = np.vstack(losses)

    idx = np.argsort(bIndices)
    bIndices = bIndices[idx]
    losses = losses[idx]

    epochLoss = losses
    return epochLoss

training = {}
# training_fwd = {}
validation = {}
testing = {}

pb = gtqdms[args.gpu]
with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    pb.set_description('[gpu %d]' %(args.gpu))

trainLoss = 0
validationLoss = 0

trainingEpochLosses = []
trainingEpochLosses2 = []
validationLosses = []

unroll = 2

if args.verbose:
    print('Start of training')
for epoch in range(args.epochs):
    trainingEpochLoss = processDataLoader(epoch,unroll, train_ds, train_dataloader, model, optimizer, True, prefix = 'training')
    trainingEpochLosses.append(trainingEpochLoss)
    # with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        # pb.set_description('[gpu %d] Learning: %1.4e' %(args.gpu, np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1))))
    trainLoss = np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1))
    if args.forwardLoss:
        trainingEpochLoss2 = processDataLoader(epoch,unroll, train_ds, train_dataloader, model, optimizer, False, prefix = 'forward')
        trainingEpochLosses2.append(trainingEpochLoss2)
        # with portalocker.Lock('README.md', flags = 0x2, timeout = None):
            # pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e' %(args.gpu, np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1)),np.mean(np.mean(trainingEpochLoss2[:,:,0], axis = 1))))
        validationEpochLoss = processDataLoader(epoch,unroll, validation_ds, validation_dataloader, model, optimizer, False, prefix = 'validation')
        validationLosses.append(validationEpochLoss)
        validationLoss = np.mean(np.mean(validationEpochLoss[:,:,0], axis = 1))
        # with portalocker.Lock('README.md', flags = 0x2, timeout = None):
            # pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e Validation: %1.4e' %(args.gpu, np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1)), np.mean(np.mean(trainingEpochLoss2[:,:,0], axis = 1)), np.mean(np.mean(validationEpochLoss[:,:,0], axis = 1))))
    else:
        validationEpochLoss = processDataLoader(epoch,unroll, validation_ds, validation_dataloader, model, optimizer, False, prefix = 'validation')
        validationLosses.append(validationEpochLoss)
        validationLoss = np.mean(np.mean(validationEpochLoss[:,:,0], axis = 1))
        # with portalocker.Lock('README.md', flags = 0x2, timeout = None):
            # pb.set_description('[gpu %d] Learning: %1.4e Validation: %1.4e' %(args.gpu, np.mean(np.mean(trainingEpochLoss[:,:,0], axis = 1)), np.mean(np.mean(validationEpochLoss[:,:,0], axis = 1))))

    torch.save(model.state_dict(), './trainingData/%s/model_%03d.torch' % (exportString, epoch))
    if epoch % args.lr_decay_step_size == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_decay_factor * param_group['lr']
        lr = lr * args.lr_decay_factor
            
    # with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        # pb.update()
if args.verbose:
    print('End of training')
    
if args.verbose:
    print('Preparing training and validation data dicts')
trainDict = {}
for i in range(len(train_ds)):
    fileName, index, _ = train_ds[i]
    trainDict['%05d' % i] = {'file':fileName, 't':int( index)}
validationDict = {}
for i in range(len(validation_ds)):
    fileName, index, _ = validation_ds[i]
    validationDict['%05d' % i] = {'file':fileName, 't': int(index)}    
dataSetDict = {'training' : trainDict, 'validation': validationDict}

if args.verbose:
    print('Preparing training and validation loss dicts')
dataDict = {}
for e in range(len(validationLosses)):
    if args.forwardLoss:
        dataDict['%03d' % (e+1)] = {"validation": validationLosses[e], "training": trainingEpochLosses[e], "forward": trainingEpochLosses2[e]}
    else:
        dataDict['%03d' % (e+1)] = {"validation": validationLosses[e], "training": trainingEpochLosses[e]}
modelData = {'hyperParameters' : hyperParameterDict, 'dataSet': dataSetDict, 'epochData': dataDict, 'files': simulationFiles}

if args.verbose:
    print('Writing out result data to ./trainingData/%s/results.json' % exportString)
encodedNumpyData = json.dumps(modelData, cls=NumpyArrayEncoder, indent=4) 
with open('./trainingData/%s/results.json' % exportString, "w") as write_file:
    json.dump(modelData, write_file, cls=NumpyArrayEncoder, indent=4) 

if args.verbose:
    print('Plotting training losses (large plot)')
fig, axis = plotLossesv2(trainingEpochLosses, logScale = True)
fig.savefig('./trainingData/%s/training.png' % exportString, dpi = 300)

if args.forwardLoss:
    if args.verbose:
        print('Plotting forward losses (large plot)')
    fig, axis = plotLossesv2(trainingEpochLosses2, logScale = True)
    fig.savefig('./trainingData/%s/forward.png' % exportString, dpi = 300)

if args.verbose:
    print('Plotting validation losses (large plot)')
plotLossesv2(validationLosses, logScale = True)
fig.savefig('./trainingData/%s/validation.png' % exportString, dpi = 300)

ei = -1
if args.verbose:
    print('Plotting training losses (kde plot)')

overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)

fig.suptitle('Training')

plt.sca(axis[0,0])
axis[0,0].set_title('Mean Loss')
axis[0,1].set_title('Max Loss')
axis[0,2].set_title('Std dev Loss')


for ei in range(overallLosses.shape[0]):
    plt.sca(axis[0,0])
    sns.kdeplot(overallLosses[ei,:,0], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,1])
    sns.kdeplot(overallLosses[ei,:,1], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,2])
    sns.kdeplot(overallLosses[ei,:,3], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))

fig.tight_layout()
fig.savefig('./trainingData/%s/training_kde.png' % exportString, dpi = 300)


if args.forwardLoss:
    if args.verbose:
        print('Plotting forward losses (kde plot)')

    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in validationLosses])
    fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)

    fig.suptitle('Forward')

    plt.sca(axis[0,0])
    axis[0,0].set_title('Mean Loss')
    axis[0,1].set_title('Max Loss')
    axis[0,2].set_title('Std dev Loss')


    for ei in range(overallLosses.shape[0]):
        plt.sca(axis[0,0])
        sns.kdeplot(overallLosses[ei,:,0], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
        plt.sca(axis[0,1])
        sns.kdeplot(overallLosses[ei,:,1], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
        plt.sca(axis[0,2])
        sns.kdeplot(overallLosses[ei,:,3], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))

    fig.tight_layout()
    fig.savefig('./trainingData/%s/forward_kde.png' % exportString, dpi = 300)

    if args.verbose:
        print('Plotting validation losses (kde plot)')

if args.verbose:
    print('Plotting validation losses (kde plot)')

overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in validationLosses])
fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)



fig.suptitle('Validation')

plt.sca(axis[0,0])
axis[0,0].set_title('Mean Loss')
axis[0,1].set_title('Max Loss')
axis[0,2].set_title('Std dev Loss')


for ei in range(overallLosses.shape[0]):
    plt.sca(axis[0,0])
    sns.kdeplot(overallLosses[ei,:,0], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,1])
    sns.kdeplot(overallLosses[ei,:,1], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))
    plt.sca(axis[0,2])
    sns.kdeplot(overallLosses[ei,:,3], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % ei, c = cm.viridis(ei / ( overallLosses.shape[0] - 1)))


fig.tight_layout()
fig.savefig('./trainingData/%s/validation_kde.png' % exportString, dpi = 300)

exit()

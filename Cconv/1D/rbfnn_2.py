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
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--cutoff', type=int, default=2048)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--rbf_x', type=str, default='rbf linear')
parser.add_argument('--rbf_y', type=str, default='rbf linear')
parser.add_argument('--n', type=int, default=9)
parser.add_argument('--cutlad', type=bool, default=False)
parser.add_argument('--forwardBatch', type=int, default=32)
parser.add_argument('--backwardBatch', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--arch', type=str, default='16 32 32 2')
parser.add_argument('--base_path', type=str, default='./export')
parser.add_argument('--initialSkip', type = int, default = 500)
parser.add_argument('--max_rollout', type = int, default = 8)
parser.add_argument('--splitChunked', type=bool, default = False)
parser.add_argument('--trainValidSplit', type=float, default=0.8)
parser.add_argument('--testSplit', type=float, default=0.1)
parser.add_argument('--limit_rollout', type=bool, default = False)
parser.add_argument('--learnShift', type=bool, default = False)
parser.add_argument('--shiftSteps', type=int, default = 2)
parser.add_argument('--skipConnection', type=bool, default = False)
parser.add_argument('--dropOut', type=float, default=0)
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

class RbfNet(torch.nn.Module):
    def __init__(self, inputDimensions, outputDimensions, layerDescription, dropout = None):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        outFeatures = inputDimensions
        for layer in layerDescription['Layers']:
            inFeatures = layer['inFeatures'] if 'inFeatures' in layer else outFeatures
            outFeatures = layer['outFeatures'] if 'outFeatures' in layer else outputDimensions
            dimension = layer['dimension'] if 'dimension' in layer else 2
            size = layer['size'] if 'size' in layer else args.n
            rbf = layer['rbf'] if 'rbf' in layer else [args.rbf_x, args.rbf_y]
            bias = layer['bias'] if 'bias' in layer else False
            centerLayer = layer['centerLayer'] if 'centerLayer' in layer else False
            periodic = layer['periodic'] if 'periodic' in layer else False
            activation = layer['activation'] if 'activation' in layer else None
            batchSize = layer['batchSize'] if 'batchSize' in layer else [args.forwardBatch, args.backwardBatch]

            self.convs.append(RbfConv(
                in_channels = inFeatures, out_channels = outFeatures,
                dim = dimension, size = size,
                rbf = rbf, periodic = periodic,
                dense_for_center = centerLayer, bias = bias, activation = activation,
                batch_size = batchSize))

    def forward(self, positions, features, output, ghostIndices, support, batches = None, persistent_batches = None):
        if batches == None:
            ghosts = ghostIndices[ghostIndices != -1]

            row, col = radius(output, positions, support, max_num_neighbors = 256)
            edge_index = torch.stack([row, col], dim = 0)
            print(self.dropout)
            if dropout is not None:
                rperm = torch.randperm(row.shape[0] * (1-self.dropout))
                print(rperm)
                row = row[rperm]
                col = col[rperm]
                print(row, col)
                idx = torch.argsort(row)
                print(idx)
                row = row[idx]
                col = col[idx]
                print(row, col)
                edge_index = torch.stack([row, col], dim = 0)

            pseudo = (output[edge_index[1]] - positions[edge_index[0]])
            pseudo = pseudo / support
            pseudo = pseudo.clamp(-1,1)

            ans = features

            for layer in self.convs:
                ans = (ans, ans[ghostIndices == -1])

                ansc = layer(ans, edge_index, pseudo)
                ghostFeatures = torch.index_select(ansc, 0, ghosts)
                ans = torch.concat([ansc, ghostFeatures], axis =0)

                if args.skipConnection:
                    ans = torch.concat([ans, features], axis = 1)

            return ans
        else:
            ghosts = ghostIndices[ghostIndices != -1]
            row, col = radius(output, positions, support, max_num_neighbors = 256, batch_y = batches, batch_x = persistent_batches)
            edge_index = torch.stack([row, col], dim = 0)

            if self.dropout is not None:
                rperm = torch.randperm(int(row.shape[0] * (1-self.dropout)))
                row = row[rperm]
                col = col[rperm]
                idx = torch.argsort(row)
                row = row[idx]
                col = col[idx]
                edge_index = torch.stack([row, col], dim = 0)

            pseudo = (output[edge_index[1]] - positions[edge_index[0]])
            pseudo = pseudo / support
            pseudo = pseudo.clamp(-1,1)
            ans = features
            for layer in self.convs:
                ans = (ans, ans[ghostIndices == -1])

                ansc = layer(ans, edge_index, pseudo)
                ghostFeatures = torch.index_select(ansc, 0, ghosts)
                ans = torch.concat([ansc, ghostFeatures], axis =0)
            return ans

layerDescription = yaml.load('''
Layers:
    - inFeatures: 1
      outFeatures: 1
      dimension: 2
      bias: False
      centerLayer: False
      periodic: False
''', Loader = yaml.Loader)

layerDescription ='Layers:'

import os

basePath = os.path.expanduser(args.base_path)
simulationFiles = [basePath + '/' + f for f in os.listdir(basePath) if f.endswith('.hdf5')]
# for i, c in enumerate(simulationFiles):
    # print(i ,c)


from datautils import *
from sphUtils import *
from lossFunctions import *

training = []
validation = []
testing = []

# for i in tqdm(range(16)):
#     print(i)
# exit()

for s in simulationFiles:    
    _, train, valid, test = splitFile(s, split = True, limitRollOut = args.limit_rollout, maxRollOut=args.max_rollout, skip=args.initialSkip, trainValidationSplit = args.trainValidSplit, testSplit = args.testSplit, cutoff = args.cutoff)
    training.append((s,train))
    validation.append((s,valid))
    testing.append((s,test))


train_ds = datasetLoader(training)
valid_ds = datasetLoader(validation)
test_ds = datasetLoader(testing)


# positions, features, persistent_output, ghostIndices, batches, persistent_batches, gt, support, indices
# _,featureVec,_,_,_,_,_,_,_ = prepareData([0], train_ds, 'cpu')
# inputFeatures = featureVec.shape[1]

config, state = loadFrame(simulationFiles[0],2)
positions, features = prepareInput(config,state)

inputFeatures = features.shape[1]

del positions, features, config, state

# print(inputFeatures)
# exit()

widths = args.arch.strip().split(' ')
for i, w in enumerate(widths):
    win = inputFeatures if i == 0 else widths[i-1]
    wout = 2 if i == len(widths) - 1 else widths[i]
    relu = 'placeholder' if i == len(widths) -1 else 'activation'
    layerDescription = layerDescription + f'''
    - inFeatures: {win}
      outFeatures: {wout}
      dimension: 2
      bias: False
      centerLayer: True
      periodic: False 
      {relu}: relu    '''

layerDescription = yaml.load(layerDescription, Loader = yaml.Loader)

model = RbfNet(1,1, layerDescription, dropout = args.dropOut if args.dropOut > 0 else None)

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if args.gpus == 1:
    print('Number of parameters', count_parameters(model))

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


def getPrediction(config, state, model):
    positions, features = prepareInput(config,state)

    batches = torch.zeros(state['fluidArea'].shape).to(config['device']).type(torch.int64)
    ghosts = state['ghostIndices']
    realParticles = ghosts == -1
    outputPositions = positions[realParticles]

    prediction = model(positions, features, outputPositions, ghosts, config['support'], batches, batches[realParticles])
    # print(prediction.shape)
    # syncQuantity(prediction, config, state)

    prediction = prediction / 20
    
    prediction = torch.clamp(prediction * config['support'], min = -0.4 * config['support'], max = 0.4 * config['support'])
    
    
    return prediction

def run(dataloader, dataset, description, epoch, train = False):
    # gtqdms[args.gpus + args.gpu] = tqdm(dataloader, leave = False, position = args.gpu + args.gpus)
    pb = gtqdms[args.gpus + args.gpu]
    losses = []
    batchHistory = []
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.reset(total=len(dataloader))
    i = 0
    for bdata in dataloader:
        batchHistory.append(bdata)
        fileName, frameIndex, maxRollout = dataset[bdata[0]]
        config, state = loadFrame(fileName,frameIndex)
        rollout = np.random.randint(1, min(epoch//2 + 2, maxRollout + 1))
        # rollout = maxRollout
        # print(rollout)
        # print(fileName, frameIndex, maxRollout)

        t_start = time.perf_counter()
        if train:
            model.train()
        else:
            model.train(mode=False)


        with record_function("forward"):
            realLosses = []

            optimizer.zero_grad()
            for i in range(rollout):
                prediction = getPrediction(config, state, model)
                # prediction, loss = getLoss(prediction, config, state)
                # losses.append(loss)

                state['fluidPosition'] += state['fluidVelocity'] * config['dt'] + prediction
                state['fluidVelocity'] = state['fluidVelocity'] + prediction / config['dt']

                state['time'] += config['dt']
                state['timestep'] += 1
                if args.learnShift:
                    realLoss = getRealLoss(fileName, state['timestep'], config, state, 'fluidPosition')
                else:
                    realLoss = getRealLoss(fileName, state['timestep'] - 1, config, state, 'positionAfterStep')

                realLosses.append(realLoss)
                # print(realLoss)
                diff = realLoss.detach()
                l1 = torch.mean(diff)
                l2 = torch.max(diff)
                linfty = torch.std(diff)
                # l1 = torch.mean(torch.abs(diff))
                # l2 = torch.mean(diff * diff)
                # linfty = torch.max(torch.abs(diff))
                with portalocker.Lock('README.md', flags = 0x2, timeout = None):
                    bstate = '[%4d : %2d/%2d]' % (bdata[0], i, rollout)
                    # print("[gpu %d @ %18s | %18s] %s: batch [%s] -> Loss: %1.4e | %1.4e | %1.4e" % (args.gpu, args.rbf_x, args.rbf_y, description, str_of_ints, l1,l2,linfty))
                    pb.set_description("[gpu %d @ %18s | %18s] %s: batch [%s] -> Loss: %1.4e | %1.4e | %1.4e" % \
                                (args.gpu, args.rbf_x, args.rbf_y, description, bstate, l1,l2,linfty))
                    # pb.set_description("[gpu %d @ %18s | %d] %s: batch [%s] -> Loss: %1.4e | %1.4e | %1.4e" % (args.gpu, args.arch, args.n, description, str_of_ints, l1,l2,linfty))
                    # pb.refresh()
                    # i = i + 1
                    # if i != len(dataloader) -1:
                    # pb.update()

                state['fluidPosition'] = state['fluidPosition'].detach()
                if i != rollout -1:
                    advanceSimulation(prediction, config, state, shiftSteps = args.shiftSteps if args.learnShift else 0)

            # losses.append(loss)

            diff = torch.sum(torch.stack(realLosses), dim = 0)
            # sumLosses = torch.mean(sumLosses)
            # print(sumLosses)
            # exit()

            # out = model(positions, features, persistent_output,
                    # ghostIndices, support, batches, persistent_batches)
            # loss = (gt - out)**2 / 2
            # loss = torch.mean(loss)

            # diff = out - gt

            # l1 = torch.mean(torch.abs(diff))
            # l2 = torch.mean(diff * diff)
            # linfty = torch.max(torch.abs(diff))
            l1 = torch.mean(diff)
            l2 = torch.max(diff)
            linfty = torch.std(diff)

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
                bstate = '[%4d : %2d/%2d]' % (bdata[0], rollout, rollout)
                # print("[gpu %d @ %18s | %18s] %s: batch [%s] -> Loss: %1.4e | %1.4e | %1.4e" % (args.gpu, args.rbf_x, args.rbf_y, description, str_of_ints, l1,l2,linfty))
                pb.set_description("[gpu %d @ %18s | %18s] %s: batch [%s] -> Loss: %1.4e | %1.4e | %1.4e" % (args.gpu, args.rbf_x, args.rbf_y, description, bstate, l1,l2,linfty))
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
        # exit()

            
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

from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if not os.path.exists('./output'):
    os.makedirs('./output')

pb = gtqdms[args.gpu]
with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    pb.set_description('[gpu %d]' %(args.gpu))

for epoch in range(args.epochs):
    t_batches, train_losses = run(train_dataloader, train_ds, 'train', epoch, train = True)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e' %(args.gpu, np.mean(train_losses)))

    f_batches, train_losses_fwd = run(train_dataloader_fwd, train_ds, 'train (foward only)', epoch, train = False)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd)))

    v_batches, valid_losses = run(valid_dataloader, valid_ds, 'valid', epoch, train = False)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e Validation: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd), np.mean(valid_losses)))

    s_batches, test_losses  = run(test_dataloader,  test_ds,  'test ', epoch, train = False)
    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d] Learning: %1.4e Training: %1.4e Validation: %1.4e Testing: %1.4e' %(args.gpu, np.mean(train_losses), np.mean(train_losses_fwd), np.mean(valid_losses), np.mean(test_losses)))

    filename = 'output/%s - epoch %02d of %02d' % \
            (timestamp, epoch, args.epochs)

    torch.save(model.state_dict(), filename + '.torch')

    # if epoch == args.epochs -1:
    if True:
        training[epoch] = {}
        training_fwd[epoch] = {}
        validation[epoch] = {}
        testing[epoch] = {}

        if(args.batch_size == 1):
            for i, (batch, loss) in enumerate(zip(t_batches, train_losses)):
                training[epoch]['%4d %4d ' %(i, batch[0])] = [float(loss[0]),float(loss[1]),float(loss[2])]
        else:
            for i, (batch, loss) in enumerate(zip(t_batches, train_losses)):
                b = ['%s %s' % (ba[0], ba[1]) for ba in batch]
                training[epoch]['%4d'% i + ', '.join(b)] = [float(loss[0]),float(loss[1]),float(loss[2])]
        for i, (batch, loss) in enumerate(zip(f_batches, train_losses_fwd)):
            training_fwd[epoch]['%4d - %4d' %(i, batch[0])] = [float(loss[0]),float(loss[1]),float(loss[2])]
        for i, (batch, loss) in enumerate(zip(v_batches, valid_losses)):
            validation[epoch]['%4d - %4d' %(i, batch[0])] = [float(loss[0]),float(loss[1]),float(loss[2])]
        for i, (batch, loss) in enumerate(zip(s_batches, test_losses)):
            testing[epoch]['%4d - %4d' %(i, batch[0])] = [float(loss[0]),float(loss[1]),float(loss[2])]

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
    'arch':args.arch,
    'args':str(args)}

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

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
parser.add_argument('-w','--window_width', type=int, default=17)
parser.add_argument('-s','--smoothing', type=bool, default=True)
parser.add_argument('-v','--verbose', type=bool, default=False)
parser.add_argument('-x','--which_x', type=str, default='n')
parser.add_argument('-y','--which_y', type=str, default='m')
parser.add_argument('-i','--inputFolder', type=str, default='./trainingData')

args = parser.parse_args()


from plotting import *
plt.style.use('dark_background')
# plt.style.use('default')
from tqdm import trange, tqdm

# from joblib import Parallel, delayed

from cutlass import *
from rbfConv import *
from datautils import *
from plotting import *

# Use dark theme
plt.style.use('dark_background')
from tqdm import tqdm
import os

from densityNet import *

from datautils import *

basePath = '../export'
basePath = os.path.expanduser(basePath)

simulationFiles = [basePath + '/' + f for f in os.listdir(basePath) if f.endswith('.hdf5')]
debugPrint(simulationFiles)

subfolders = [ f.path for f in os.scandir(args.inputFolder) if f.is_dir() ]
# subfolders = [ f.path for f in os.scandir('./trainingDataBasisFunctions8x8') if f.is_dir() ]

which_x = args.which_x
which_y = args.which_y

dataDict = {}

subfolders[0]
for s in subfolders:
    with open("%s/results.json" % s, "r") as read_file:
        decodedArray = json.load(read_file)
        dataDict[s] = decodedArray
#         print(decodedArray['hyperParameters'])   

trainingLosses = {}
validationLosses = {}
for s in subfolders:
    trainingEpochLosses = [np.asarray(dataDict[s]['epochData'][k]['training']) for k in dataDict[s]['epochData'].keys()]
    validationEpochLosses = [np.asarray(dataDict[s]['epochData'][k]['validation']) for k in dataDict[s]['epochData'].keys()]
    trainingLosses[s] = trainingEpochLosses
    validationLosses[s] = validationEpochLosses

def plotKDEs(trainingEpochLosses, label = None):
    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
    fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)
    if label is not None:
        fig.suptitle(label)

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
# fig.savefig('./trainingData/%s/training_kde.png' % exportString, dpi = 300)

import seaborn as sns

xvars = []
yvars = []
for s in subfolders:
    xv = dataDict[s]['hyperParameters'][which_x]
    if xv not in xvars:
        xvars.append(xv)
    yv = dataDict[s]['hyperParameters'][which_y]
    if yv not in yvars:
        yvars.append(yv)
debugPrint(xvars)
debugPrint(yvars)
xvars.sort()
yvars.sort()

trainingEpochLosses = trainingLosses[subfolders[0]]
fig, axis = plt.subplots(len(yvars), 3, figsize=(16,3 * len(yvars)), sharex = True, sharey = False, squeeze = False)

# for y in yvars:
fig.suptitle('Training')

losses = {}
for s in subfolders:
    trainingEpochLosses = trainingLosses[s]
    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
    losses[s] = overallLosses

epoch = -1

def plot_func(epoch):    
    plt.sca(axis[0,0])
    for i in range(len(yvars)):
        for j in range(3):        
            axis[i,j].cla()
#             plt.sca(axis[i,j])
        axis[i,0].set_title('Mean Loss')
        axis[i,1].set_title('Max Loss')
        axis[i,2].set_title('Std dev Loss')

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for s in subfolders:
        c = xvars.index(dataDict[s]['hyperParameters'][which_x])
        r = yvars.index(dataDict[s]['hyperParameters'][which_y])
        plt.sca(axis[r,0])
        sns.kdeplot(losses[s][epoch,:,0], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % epoch, c = cols[c % len(cols)])
        plt.sca(axis[r,1])
        sns.kdeplot(losses[s][epoch,:,1], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % epoch, c = cols[c % len(cols)])
        plt.sca(axis[r,2])
        sns.kdeplot(losses[s][epoch,:,3], bw_adjust=.2, log_scale=True, label = '[%s = %s x %s = %s]' % (which_x, dataDict[s]['hyperParameters'][which_x], which_y, dataDict[s]['hyperParameters'][which_y]), c = cols[c % len(cols)])
        axis[r,2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for c in range(len(yvars)):
        handles, labels = axis[c,2].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axis[c,2].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.canvas.draw_idle()

epochs = dataDict[subfolders[0]]['hyperParameters']['epochs']

plot_func(epochs-1)
# interact(plot_func, epoch = widgets.IntSlider(value=epochs -1, min=0, max=epochs -1, step=1))

fig.tight_layout()
fig.savefig('%s/training_kde.png' % args.inputFolder, dpi = 300)

trainingEpochLosses = trainingLosses[subfolders[0]]
fig, axis = plt.subplots(len(yvars), 3, figsize=(16,3 * len(yvars)), sharex = True, sharey = False, squeeze = False)

# for y in yvars:
fig.suptitle('Validation')

losses = {}
for s in subfolders:
    trainingEpochLosses = validationLosses[s]
    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
    losses[s] = overallLosses

epoch = -1

def plot_func(epoch):    
    plt.sca(axis[0,0])
    for i in range(len(yvars)):
        for j in range(3):        
            axis[i,j].cla()
#             plt.sca(axis[i,j])
        axis[i,0].set_title('Mean Loss')
        axis[i,1].set_title('Max Loss')
        axis[i,2].set_title('Std dev Loss')

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for s in subfolders:
        c = xvars.index(dataDict[s]['hyperParameters'][which_x])
        r = yvars.index(dataDict[s]['hyperParameters'][which_y])
        plt.sca(axis[r,0])
        sns.kdeplot(losses[s][epoch,:,0], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % epoch, c = cols[c % len(cols)])
        plt.sca(axis[r,1])
        sns.kdeplot(losses[s][epoch,:,1], bw_adjust=.2, log_scale=True, label = 'epoch: %2d' % epoch, c = cols[c % len(cols)])
        plt.sca(axis[r,2])
        sns.kdeplot(losses[s][epoch,:,3], bw_adjust=.2, log_scale=True, label = '[%s = %s x %s = %s]' % (which_x, dataDict[s]['hyperParameters'][which_x], which_y, dataDict[s]['hyperParameters'][which_y]), c = cols[c % len(cols)])
        axis[r,2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for c in range(len(yvars)):
        handles, labels = axis[c,2].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axis[c,2].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.canvas.draw_idle()

epochs = dataDict[subfolders[0]]['hyperParameters']['epochs']
plot_func(epochs-1)
# interact(plot_func, epoch = widgets.IntSlider(value=epochs -1, min=0, max=epochs -1, step=1))

fig.tight_layout()
fig.savefig('%s/validation_kde.png' % args.inputFolder, dpi = 300)


import matplotlib.patheffects as path_effects

def plotToAxis(axis, mat, label = None):
    if label is not None:
        axis.set_title(label)
    im = axis.imshow(mat, vmin=mat[np.logical_not(np.isnan(mat))].min(), vmax=mat[np.logical_not(np.isnan(mat))].max())
    for (j,i),label in np.ndenumerate(mat):
        if not np.isnan(label):
            st = '%1.4e' % label
            txtLabels.append(axis.text(i,j,st,ha='center',va='center', color = 'white', fontsize=8, rotation = 0))       
            txtLabels[-1].set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

    axis.set_yticks(np.arange(len(xvars)))
    axis.set_yticklabels(xvars, minor = False, rotation =0, fontsize = 12)
    axis.set_ylabel('%s' % which_x, fontsize = 12)
    axis.set_xlabel('%s' % which_y, fontsize = 12)
    axis.set_xticks(np.arange(len(yvars)))
    axis.set_xticklabels(yvars, minor = False, rotation =90, fontsize = 12)
    
    return im
    

fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)

epoch = -1

meanLoss = np.ones((len(xvars),len(yvars))) * np.nan
maxLoss = np.ones((len(xvars),len(yvars))) * np.nan
stdDevLoss = np.ones((len(xvars),len(yvars))) * np.nan


for s in subfolders:
    trainingEpochLosses = trainingLosses[s]
    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
    
    xv = dataDict[s]['hyperParameters'][which_x]
    yv = dataDict[s]['hyperParameters'][which_y]
    xi = xvars.index(xv)
    yi = yvars.index(yv)
    
    meanLoss[xi,yi] = np.mean(overallLosses[epoch,:,0])
    maxLoss[xi,yi] = np.mean(overallLosses[epoch,:,1])
    stdDevLoss[xi,yi] = np.mean(overallLosses[epoch,:,3])

txtLabels = []


fig.suptitle('Training')
    
imMean = plotToAxis(axis[0,0], meanLoss, label = 'Mean Loss')
imMax= plotToAxis(axis[0,1], maxLoss, label = 'Max Loss')
imStd = plotToAxis(axis[0,2], stdDevLoss, label = 'Stddev Loss')

fig.tight_layout()

def plot_func(epoch):
    global txtLabels
    meanLoss = np.ones((len(xvars),len(yvars))) * np.nan
    maxLoss = np.ones((len(xvars),len(yvars))) * np.nan
    stdDevLoss = np.ones((len(xvars),len(yvars))) * np.nan


    for s in subfolders:
        trainingEpochLosses = trainingLosses[s]
        overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])

        xv = dataDict[s]['hyperParameters'][which_x]
        yv = dataDict[s]['hyperParameters'][which_y]
        xi = xvars.index(xv)
        yi = yvars.index(yv)

        meanLoss[xi,yi] = np.mean(overallLosses[epoch,:,0])
        maxLoss[xi,yi] = np.mean(overallLosses[epoch,:,1])
        stdDevLoss[xi,yi] = np.mean(overallLosses[epoch,:,3])
    
#     debugPrint(meanLoss.min())
#     debugPrint(meanLoss.max())
    
    imMean.set_clim(vmin=meanLoss[np.logical_not(np.isnan(meanLoss))].min(), vmax=meanLoss[np.logical_not(np.isnan(meanLoss))].max())
    imMax.set_clim(vmin=maxLoss[np.logical_not(np.isnan(meanLoss))].min(), vmax=maxLoss[np.logical_not(np.isnan(meanLoss))].max())
    imStd.set_clim(vmin=stdDevLoss[np.logical_not(np.isnan(meanLoss))].min(), vmax=stdDevLoss[np.logical_not(np.isnan(meanLoss))].max())
    
    imMean.set_data(meanLoss)
    imMax.set_data(maxLoss)
    imStd.set_data(stdDevLoss)
    
    for lbl in txtLabels:
        lbl.remove()
    txtLabels = []
    fig.suptitle('Training Epoch %d'%epoch)
    
    for (j,i),label in np.ndenumerate(meanLoss):
        if not np.isnan(label):
            st = '%1.4e' % label
            txtLabels.append(axis[0,0].text(i,j,st,ha='center',va='center', color = 'white', fontsize=4, rotation = 0))       
            txtLabels[-1].set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
    for (j,i),label in np.ndenumerate(maxLoss):
        if not np.isnan(label):
            st = '%1.4e' % label
            txtLabels.append(axis[0,1].text(i,j,st,ha='center',va='center', color = 'white', fontsize=4, rotation = 0))       
            txtLabels[-1].set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
    for (j,i),label in np.ndenumerate(stdDevLoss):
        if not np.isnan(label):
            st = '%1.4e' % label
            txtLabels.append(axis[0,2].text(i,j,st,ha='center',va='center', color = 'white', fontsize=4, rotation = 0))       
            txtLabels[-1].set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
            
    fig.canvas.draw_idle()

epochs = dataDict[subfolders[0]]['hyperParameters']['epochs']
# interact(plot_func, epoch = widgets.IntSlider(value=epochs -1, min=0, max=epochs -1, step=1))

plot_func(epochs-1)
fig.savefig('%s/training_matrix.png' % args.inputFolder, dpi = 300)

fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)

epoch = -1

meanLoss = np.ones((len(xvars),len(yvars))) * np.nan
maxLoss = np.ones((len(xvars),len(yvars))) * np.nan
stdDevLoss = np.ones((len(xvars),len(yvars))) * np.nan


for s in subfolders:
    trainingEpochLosses = validationLosses[s]
    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
    
    xv = dataDict[s]['hyperParameters'][which_x]
    yv = dataDict[s]['hyperParameters'][which_y]
    xi = xvars.index(xv)
    yi = yvars.index(yv)
    
    meanLoss[xi,yi] = np.mean(overallLosses[epoch,:,0])
    maxLoss[xi,yi] = np.mean(overallLosses[epoch,:,1])
    stdDevLoss[xi,yi] = np.mean(overallLosses[epoch,:,3])

txtLabels = []


fig.suptitle('Validation')
    
imMean = plotToAxis(axis[0,0], meanLoss, label = 'Mean Loss')
imMax= plotToAxis(axis[0,1], maxLoss, label = 'Max Loss')
imStd = plotToAxis(axis[0,2], stdDevLoss, label = 'Stddev Loss')

fig.tight_layout()

def plot_func(epoch):
    global txtLabels
    meanLoss = np.ones((len(xvars),len(yvars))) * np.nan
    maxLoss = np.ones((len(xvars),len(yvars))) * np.nan
    stdDevLoss = np.ones((len(xvars),len(yvars))) * np.nan


    for s in subfolders:
        trainingEpochLosses = validationLosses[s]
        overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])

        xv = dataDict[s]['hyperParameters'][which_x]
        yv = dataDict[s]['hyperParameters'][which_y]
        xi = xvars.index(xv)
        yi = yvars.index(yv)

        meanLoss[xi,yi] = np.mean(overallLosses[epoch,:,0])
        maxLoss[xi,yi] = np.mean(overallLosses[epoch,:,1])
        stdDevLoss[xi,yi] = np.mean(overallLosses[epoch,:,3])
    
#     debugPrint(meanLoss.min())
#     debugPrint(meanLoss.max())
    
    imMean.set_clim(vmin=meanLoss[np.logical_not(np.isnan(meanLoss))].min(), vmax=meanLoss[np.logical_not(np.isnan(meanLoss))].max())
    imMax.set_clim(vmin=maxLoss[np.logical_not(np.isnan(meanLoss))].min(), vmax=maxLoss[np.logical_not(np.isnan(meanLoss))].max())
    imStd.set_clim(vmin=stdDevLoss[np.logical_not(np.isnan(meanLoss))].min(), vmax=stdDevLoss[np.logical_not(np.isnan(meanLoss))].max())
    
    imMean.set_data(meanLoss)
    imMax.set_data(maxLoss)
    imStd.set_data(stdDevLoss)
    
    for lbl in txtLabels:
        lbl.remove()
    txtLabels = []
    fig.suptitle('Validation Epoch %d'%epoch)
    
    for (j,i),label in np.ndenumerate(meanLoss):
        if not np.isnan(label):
            st = '%1.4e' % label
            txtLabels.append(axis[0,0].text(i,j,st,ha='center',va='center', color = 'white', fontsize=4, rotation = 0))       
            txtLabels[-1].set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
    for (j,i),label in np.ndenumerate(maxLoss):
        if not np.isnan(label):
            st = '%1.4e' % label
            txtLabels.append(axis[0,1].text(i,j,st,ha='center',va='center', color = 'white', fontsize=4, rotation = 0))       
            txtLabels[-1].set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
    for (j,i),label in np.ndenumerate(stdDevLoss):
        if not np.isnan(label):
            st = '%1.4e' % label
            txtLabels.append(axis[0,2].text(i,j,st,ha='center',va='center', color = 'white', fontsize=4, rotation = 0))       
            txtLabels[-1].set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
            
    fig.canvas.draw_idle()

epochs = dataDict[subfolders[0]]['hyperParameters']['epochs']
# interact(plot_func, epoch = widgets.IntSlider(value=epochs -1, min=0, max=epochs -1, step=1))


plot_func(epochs-1)
fig.savefig('%s/validation_matrix.png' % args.inputFolder, dpi = 300)

from densityNet import *
simulationFiles = dataDict[subfolders[0]]['files']
# cutoff = dataDict[subfolders[0]]['hyperParameters']['cutoff']
cutoff = 1800
epochs = dataDict[subfolders[0]]['hyperParameters']['epochs']
try:
    inFile = h5py.File(simulationFiles[0], 'r')
    frameCount = int(len(inFile['simulationExport'].keys()) -1) # adjust for bptcls
    inFile.close()
    debugPrint(frameCount)
    if cutoff < 0:
        cutoff = frameCount - 100
except Exception:
    print('data not available defaulting to 1800')

networks = len(subfolders)
epochs = dataDict[subfolders[0]]['hyperParameters']['epochs']
simulationFiles = dataDict[subfolders[0]]['files']

frameCounts = []

for s in simulationFiles:    
    try:
        inFile = h5py.File(s, 'r')
        frameCount = int(len(inFile['simulationExport'].keys()) -1) # adjust for bptcls
        inFile.close()
        frameCounts.append(frameCount)
    except Exception:
        frameCounts.append(cutoff)

debugPrint(frameCounts)
simulationData = []
for s in simulationFiles:    
    try:
        inFile = h5py.File(s, 'r')
        frameCount = int(len(inFile['simulationExport'].keys()) -1) # adjust for bptcls
        inFile.close()
        simulationData.append(np.zeros((len(subfolders), frameCount, epochs, 4))* np.nan) 
    except Exception:
        simulationData.append(np.zeros((len(subfolders), cutoff, epochs, 4))* np.nan) 
# debugPrint(simulationData)

dataDict[subfolders[0]]['dataSet']['training']['00000']

# ['training']).shape

for si, s in tqdm(enumerate(subfolders)):
    trainingDataSet = dataDict[s]['dataSet']['training']
    validationDataSet = dataDict[s]['dataSet']['validation']
    
#     debugPrint(len(trainingDataSet))
    trainingEpochLosses = trainingLosses[s]
    validationEpochLosses = validationLosses[s]
    
    overallTrainingLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in trainingEpochLosses])
    overallValidationLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in validationEpochLosses])

#     debugPrint(overallTrainingLosses.shape)
#     debugPrint(overallValidationLosses.shape)
    
    for i in tqdm(range(overallTrainingLosses.shape[1]), leave = False):
        ds = trainingDataSet['%05d' % i]
        file = ds['file']
        frame = ds['t']
        simulationData[simulationFiles.index(file)][si,frame,:] = overallTrainingLosses[:,i,:]
        
    for i in tqdm(range(overallValidationLosses.shape[1]), leave = False):
        ds = validationDataSet['%05d' % i]
        file = ds['file']
        frame = ds['t']
        simulationData[simulationFiles.index(file)][si,frame,:] = overallValidationLosses[:,i,:]
    
    

fig, axis = plt.subplots(len(yvars), 1, figsize=(24,3 * len(yvars)), sharex = True, sharey = False, squeeze = False)

epoch = -1
plotMinMax = False
smoothing = args.smoothing
window_width = args.window_width

def plotLoss(epoch):
    for i in range(len(yvars)):
        axis[i,0].cla()
        axis[i,0].set_yscale('log')
        axis[i,0].set_ylabel('Loss')

    axis[len(yvars) - 1,0].set_xlabel('index')


    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for n in range(len(subfolders)):
        for i in range(len(simulationData)):
            begin = 0 if i == 0 else begin + l
            end = begin + np.sum(np.logical_not(np.isnan(simulationData[i][n,:,epoch,0])))
            l = simulationData[i].shape[1]
            
            r = yvars.index(dataDict[subfolders[n]]['hyperParameters'][which_y])
            c = xvars.index(dataDict[subfolders[n]]['hyperParameters'][which_x])

            d = simulationData[i][n,:,epoch,0]
            if smoothing:
                data = d[np.logical_not(np.isnan(d))]
                cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
                data = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
            else:
                data = d[np.logical_not(np.isnan(d))]
            
            axis[r,0].plot(np.arange(begin,end) if not smoothing else np.arange(begin + window_width // 2, end + window_width//2 - window_width + 1), data, c = cols[c % len(cols)], label = '[%s = %s x %s = %s]' % (which_x, dataDict[subfolders[n]]['hyperParameters'][which_x], which_y, dataDict[subfolders[n]]['hyperParameters'][which_y]) if i == 0 else None)
            axis[r,0].axvline(begin + l)
            axis[r,0].axvline(begin)
            if plotMinMax:
                d = simulationData[i][n,:,epoch,1]
                if smoothing:
                    data = d[np.logical_not(np.isnan(d))]
                    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
                    data = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
                else:
                    data = d[np.logical_not(np.isnan(d))]
                axis[r,0].plot(np.arange(begin,end) if not smoothing else np.arange(begin + window_width // 2, end + window_width//2 - window_width + 1), data, c = cols[c % len(cols)], ls = '--', alpha = 0.5)
                axis[r,0].axvline(begin + l)
                d = simulationData[i][n,:,epoch,2]
                if smoothing:
                    data = d[np.logical_not(np.isnan(d))]
                    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
                    data = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
                else:
                    data = d[np.logical_not(np.isnan(d))]
                axis[r,0].plot(np.arange(begin,end) if not smoothing else np.arange(begin + window_width // 2, end + window_width//2 - window_width + 1), data, c = cols[c % len(cols)], ls = '--', alpha = 0.5)
                axis[r,0].axvline(begin + l)

    for c in range(len(yvars)):
        axis[c,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        handles, labels = axis[c,0].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axis[c,0].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
#         axis[c,0].grid(which='both', ls = ':', alpha = 0.5, axis = 'y')
    fig.canvas.draw()
    fig.canvas.flush_events()

plotLoss(epochs - 1)
    
# interact(plotLoss, epoch = widgets.IntSlider(value=epochs - 1, min=0, max=epochs -1, step=1) )
fig.suptitle('Loss plot @ epoch %2d' % (epochs - 1))

fig.tight_layout()

fig.savefig('%s/loss.png' % args.inputFolder, dpi = 300)
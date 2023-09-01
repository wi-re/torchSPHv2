# sph related imports
from sph import *
from perlin import *
# neural network rlated imports
from torch.optim import Adam
from rbfConv import *
from torch_geometric.loader import DataLoader
from trainingHelper import *
# plotting/UI related imports
from plotting import *
import matplotlib as mpl
# plt.style.use('dark_background')
cmap = mpl.colormaps['viridis']
from tqdm import trange, tqdm
from IPython.display import display, Latex
from datetime import datetime
from rbfNet import *
from tqdm import tqdm
import h5py
import matplotlib.colors as colors
# %matplotlib notebook
trainingFiles = ['./output/' + f for f in os.listdir('./output/') if f.endswith('.hdf5')]
testingFiles = ['./outputTest/' + f for f in os.listdir('./outputTest/') if f.endswith('.hdf5')]
# print(testingFiles)
# Load generic settings
inFile = h5py.File(trainingFiles[0],'r')
minDomain = inFile.attrs['minDomain']
maxDomain = inFile.attrs['maxDomain']

baseArea = inFile.attrs['baseArea']
particleRadius = inFile.attrs['particleRadius']
particleSupport = inFile.attrs['particleSupport']

xsphConstant = inFile.attrs['xsphConstant']
diffusionAlpha = inFile.attrs['diffusionAlpha']
diffusionBeta = inFile.attrs['diffusionBeta']
kappa = inFile.attrs['kappa']
restDensity = inFile.attrs['restDensity']
c0 = inFile.attrs['c0']
dt = inFile.attrs['dt']

numParticles = inFile.attrs['numParticles']
timesteps = inFile.attrs['timesteps']

generator = inFile.attrs['generator']
inFile.close()

# Load generator settings
settings = {}
for f in trainingFiles:
    inFile = h5py.File(f,'r')
    generatorSettings = {}
    for k in inFile['generatorSettings'].attrs.keys():
        generatorSettings[k] = inFile['generatorSettings'].attrs[k]
#     print(generatorSettings)
    setup = {}
    setup['generatorSettings'] = generatorSettings
    setup['minDomain'] = inFile.attrs['minDomain']
    setup['maxDomain'] = inFile.attrs['maxDomain']

    setup['baseArea'] = inFile.attrs['baseArea']
    setup['particleRadius'] = inFile.attrs['particleRadius']
    setup['particleSupport'] = inFile.attrs['particleSupport']

    setup['xsphConstant'] = inFile.attrs['xsphConstant']
    setup['diffusionAlpha'] = inFile.attrs['diffusionAlpha']
    setup['diffusionBeta'] = inFile.attrs['diffusionBeta']
    setup['kappa'] = inFile.attrs['kappa']
    setup['restDensity'] = inFile.attrs['restDensity']
    setup['c0'] = inFile.attrs['c0']
    setup['dt'] = inFile.attrs['dt']

    setup['numParticles'] = inFile.attrs['numParticles']
    setup['timesteps'] = inFile.attrs['timesteps']

    setup['generator'] = inFile.attrs['generator']
    settings[f] = setup
    inFile.close()
    
for f in testingFiles:
    inFile = h5py.File(f,'r')
    generatorSettings = {}
    for k in inFile['generatorSettings'].attrs.keys():
        generatorSettings[k] = inFile['generatorSettings'].attrs[k]
#     print(generatorSettings)
    setup = {}
    setup['generatorSettings'] = generatorSettings
    setup['minDomain'] = inFile.attrs['minDomain']
    setup['maxDomain'] = inFile.attrs['maxDomain']

    setup['baseArea'] = inFile.attrs['baseArea']
    setup['particleRadius'] = inFile.attrs['particleRadius']
    setup['particleSupport'] = inFile.attrs['particleSupport']

    setup['xsphConstant'] = inFile.attrs['xsphConstant']
    setup['diffusionAlpha'] = inFile.attrs['diffusionAlpha']
    setup['diffusionBeta'] = inFile.attrs['diffusionBeta']
    setup['kappa'] = inFile.attrs['kappa']
    setup['restDensity'] = inFile.attrs['restDensity']
    setup['c0'] = inFile.attrs['c0']
    setup['dt'] = inFile.attrs['dt']

    setup['numParticles'] = inFile.attrs['numParticles']
    setup['timesteps'] = inFile.attrs['timesteps']

    setup['generator'] = inFile.attrs['generator']
    settings[f] = setup
    inFile.close()
def loadFile(t, plot = False):
    inFile = h5py.File(t,'r')
    fluidPositions = np.array(inFile['simulationData']['fluidPosition'])
    
    fluidVelocities = np.array(inFile['simulationData']['fluidVelocities'])
    fluidDensity = np.array(inFile['simulationData']['fluidDensity'])
    fluidPressure = np.array(inFile['simulationData']['fluidPressure'])
    fluidAreas = np.array(inFile['simulationData']['fluidAreas'])
    dudt = np.array(inFile['simulationData']['dudt'])
    if plot:
        fig, axis = plt.subplots(1, 5, figsize=(16,6), sharex = False, sharey = False, squeeze = False)

        def plot(fig, axis, mat, title, cmap = 'viridis'):
            im = axis.imshow(mat, extent = [0,numParticles,dt * timesteps,0], cmap = cmap)
            axis.axis('auto')
            ax1_divider = make_axes_locatable(axis)
            cax1 = ax1_divider.append_axes("bottom", size="2%", pad="6%")
            cb1 = fig.colorbar(im, cax=cax1,orientation='horizontal')
            cb1.ax.tick_params(labelsize=8) 
            axis.set_title(title)
        plot(fig,axis[0,0], fluidPositions, 'position')
        plot(fig,axis[0,1], fluidDensity, 'density')
        plot(fig,axis[0,2], fluidPressure, 'pressure')
        plot(fig,axis[0,3], fluidVelocities, 'velocity', 'RdBu')
        plot(fig,axis[0,4], dudt, 'dudt', 'RdBu')

        fig.suptitle(t)
        fig.tight_layout()
    inFile.close()
    return {'positions': torch.tensor(fluidPositions).type(torch.float32), 'density': torch.tensor(fluidDensity).type(torch.float32), 'pressure':torch.tensor(fluidPressure).type(torch.float32), 'area': torch.tensor(fluidAreas).type(torch.float32), 'velocity': torch.tensor(fluidVelocities).type(torch.float32), 'dudt' : torch.tensor(dudt).type(torch.float32)}
data = loadFile(trainingFiles[0], False)


testingFiles = trainingFiles[-4:]
trainingFiles = trainingFiles[:-4]

trainingData = {}
for f in tqdm(trainingFiles, leave = False):
    trainingData[f] = loadFile(f, False)
testingData = {}
for f in tqdm(testingFiles, leave = False):
    testingData[f] = loadFile(f, False)

# inFile.close()

# print(trainingData[trainingFiles[0]]['positions'][:1024,:])
import matplotlib.colors as colors

offset = 16
def getStackedUpdates(positions, velocities, accelerations, offset):
    dx = (velocities + accelerations)
    x = dx.mT
    cumsum = torch.cumsum(x, axis = 1)
    s = torch.sum(x, axis = 1, keepdims=True)
    r2lcumsum = x + s - cumsum
    stacked = torch.hstack((r2lcumsum[:,:-offset] - r2lcumsum[:,offset:], r2lcumsum[:,-offset:]))
    return stacked.mT


for f in tqdm(trainingFiles, leave = False):
    trainingData[f]['stacked'] = getStackedUpdates(trainingData[f]['positions'], trainingData[f]['velocity'], trainingData[f]['dudt'], offset - 1)
for f in tqdm(testingFiles, leave = False):
    testingData[f]['stacked'] = getStackedUpdates(testingData[f]['positions'], testingData[f]['velocity'], testingData[f]['dudt'], offset - 1)
#     trainingData[f]['averagedVelocity'] = torch.zeros


normalized = False # rbf normalization
batchSize = 4
maxUnrollsteps = 1
# offset = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# device = 'cpu'
dataSet = []
for f in trainingFiles:
    nSteps = settings[f]['timesteps'] - maxUnrollsteps * (offset - 1)
    for j in range(offset - 1, nSteps):
        dataSet.append((f, j))
# print('Dataset contains %d samples [%d files @ %d timesteps]' % (len(dataSet), len(trainingFiles), len(dataSet) / len(trainingFiles)))

from util import *

def loadTestcase(testingData, settings, f, frames, device, groundTruthFn, featureFn, offset):
    positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = getTestcase(testingData, settings, f, frames, device, offset)

    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

    x = x[:,None].to(device)    
    groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
    distance = (distance * direction)[:,None].to(device)
    features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)
#     print(groundTruth)
    return positions, velocities, areas, density, dudts, features, i, j, distance, groundTruth, x, u

def plotBatch(trainingData, settings, dataSet, bdata, device, offset, model = None):
    positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

    x = x[:,None].to(device)    
    groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
    distance = (distance * direction)[:,None].to(device)
    features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)

#     optimizer.zero_grad()
#     prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
#     lossTerm = lossFn(prediction, groundTruth)
#     loss = torch.mean(lossTerm)
    
    fig, axis = plt.subplot_mosaic('''AF
    BC
    DE''', figsize=(12,8), sharey = False, sharex = False)
    
    positions = torch.vstack(positions).mT.detach().cpu().numpy()
    vel = u.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    area = v.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    dudt = dudt.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    density = rho.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    inVel = inVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    outVel = outVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    gt = groundTruth.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    ft = features[:,0].reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    
    axis['A'].set_title('Position')
    axis['A'].plot(positions)
    axis['B'].set_title('Density')
    axis['B'].plot(positions, density)
    axis['C'].set_title('Difference')
    axis['C'].plot(positions, gt - ft)
    axis['D'].set_title('Instantenous Velocity')
    axis['D'].plot(positions, vel)
    axis['E'].set_title('Ground Truth')
    axis['E'].plot(positions, gt)
    axis['F'].set_title('Features[:,0]')
    axis['F'].plot(positions, ft)
    
    fig.tight_layout()
    
def plotTrainedBatch(trainingData, settings, dataSet, bdata, device, offset, modelState, groundTruthFn, featureFn, lossFn):
    positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

    x = x[:,None].to(device)    
    groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
    distance = (distance * direction)[:,None].to(device)
    features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)
    
    with torch.no_grad():
        prediction = modelState['model'](features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
        lossTerm = lossFn(prediction, groundTruth)
        loss = torch.mean(lossTerm)
    
    fig, axis = plt.subplot_mosaic('''ABC
    DEF''', figsize=(16,5), sharey = False, sharex = True)
    
    positions = torch.vstack(positions).mT.detach().cpu().numpy()
    vel = u.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    area = v.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    dudt = dudt.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    density = rho.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    inVel = inVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    outVel = outVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    gt = groundTruth.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    ft = features[:,0].reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    loss = lossTerm.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    pred = prediction.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    
    axis['A'].set_title('Density')
    axis['A'].plot(positions, density)
    axis['E'].set_title('Ground Truth - Features[:,0]')
    axis['E'].plot(positions, gt - ft)
    axis['B'].set_title('Ground Truth')
    axis['B'].plot(positions, gt)
    axis['D'].set_title('Features[:,0]')
    axis['D'].plot(positions, ft)
    axis['C'].set_title('Prediction')
    axis['C'].plot(positions, pred)
    axis['F'].set_title('Loss')
    axis['F'].plot(positions, loss)
    
    fig.tight_layout()
    
# plotBatch(trainingData, settings, dataSet, bdata, device, offset)

def buildMLP(layers, inputFeatures = 1):
    modules = []
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i]))
            torch.nn.init.uniform_(modules[-1].weight,-0.05, 0.05)
    #         torch.nn.init.zeros_(modules[-1].weight)
            torch.nn.init.zeros_(modules[-1].bias)
            modules.append(nn.BatchNorm1d(layers[i]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layers[-2],layers[-1]))
    else:
        modules.append(nn.Linear(inputFeatures,layers[-1]))        
    torch.nn.init.uniform_(modules[-1].weight,-0.05, 0.05)
    torch.nn.init.zeros_(modules[-1].bias)
    return nn.Sequential(*modules)


      
class mlpNetwork(nn.Module):
    def __init__(self,
                 feedThroughVertexFeatures = 0,
                 feedThroughEdgeFeatures = 0,
                 vertexHiddenLayout = [8,8,8], 
                 edgeHiddenLayout = [8,8,8],
                 vertexFeatures = [8,8,4,1],
                 messageFeatures = None,
                 edgeFeatures = None,
                 vertexMode = 'MessagePassing',
                 edgeMode = 'CConv',
                 edgeMLP = False,
                 inputEncode = False,
                 outputEncode = False,
                 inputEdgeEncode = False,
                 seed = None,
                 verbose = False):
        super(mlpNetwork, self).__init__()        
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0,2**30)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
#         self.layout = layers
        self.inputVertexFeatures = vertexFeatures[0]
        self.outputVertexFeatures = vertexFeatures[-1]
        self.inputEdgeFeatures = edgeFeatures[0] if edgeFeatures is not None else 1
        self.edgeMLP = edgeMLP
#         self.outputEdgeFeatures = vertexFeatures[-1]
        self.vertexMode = vertexMode
        self.edgeMode = edgeMode
        self.vertexHiddenLayout = vertexHiddenLayout
        self.edgeHiddenLayout = edgeHiddenLayout
        self.inputEncode = inputEncode
        self.outputEncode = outputEncode
        self.inputEdgeEncode = inputEdgeEncode
        self.feedThroughVertexFeatures = feedThroughVertexFeatures
        self.feedThroughEdgeFeatures = feedThroughEdgeFeatures
        
        self.vertexFeatures = vertexFeatures
#         self.vertexFeatures = [e + feedThroughVertexFeatures for e in self.vertexFeatures]
        self.edgeFeatures = [1] + [1] * (len(vertexFeatures)) if edgeFeatures is None else edgeFeatures
        self.messageFeatures = self.edgeFeatures[1:] if messageFeatures is None else messageFeatures
#         self.edgeFeatures = [e + feedThroughEdgeFeatures for e in self.edgeFeatures]
        self.verbose = verbose
    
        if verbose:
            print('Building MLP MessagePassing network with %2d input features and %2d output features' % (vertexFeatures[0], vertexFeatures[-1]))
            print(f"Vertex Mode: {vertexMode} with vertex Features: [{' '.join([str(v) for v in vertexFeatures])}]")
            print(f"Processed Vertex Layout [{' '.join([str(v) for v in self.vertexFeatures])}]")
            print(f"Edge   Mode: {edgeMode} with edge Features  : [{' '.join([str(e) for e in edgeFeatures]) if edgeFeatures is not None else 'N/A'}]")
            print(f"Processed edge   Layout [{' '.join([str(v) for v in self.edgeFeatures])}]")
            print(f"Input Encoder: {'true' if inputEncode else 'false'}")
            print(f"Output Encoder: {'true' if outputEncode else 'false'}")
            print(f"Input Edge Encoder: {'true' if inputEdgeEncode else 'false'}")           
            print(f"Feed through features for vertices {feedThroughVertexFeatures} and edges {feedThroughEdgeFeatures}")
        
        
        if self.inputEncode:
            encoderLayout = vertexHiddenLayout + [self.vertexFeatures[1]]
            if verbose:
                print(f"Building inputEncoder from {self.vertexFeatures[0]} to {self.vertexFeatures[1]} with hidden layout {[{' '.join([str(v) for v in vertexHiddenLayout])}]}")
            self.vertexFeatures = self.vertexFeatures[1:]
            if verbose:
                print(f"New Vertex Layout: [{' '.join([str(v) for v in self.vertexFeatures])}]")
            self.inputEncoder = buildMLP(encoderLayout, inputFeatures = self.vertexFeatures[0] + feedThroughVertexFeatures, gain = 1)
            
        if self.outputEncode:
            encoderLayout = vertexHiddenLayout + [self.vertexFeatures[-1]]
            if verbose:
                print(f"Building outputEncoder from {self.vertexFeatures[-2] + feedThroughVertexFeatures} to {self.vertexFeatures[-1]} with hidden layout {[{' '.join([str(v) for v in vertexHiddenLayout])}]}")
            self.vertexFeatures = self.vertexFeatures[:-1]
            if verbose:
                print(f"New Vertex Layout: [{' '.join([str(v) for v in self.vertexFeatures])}]")
            self.outputEncoder = buildMLP(encoderLayout, inputFeatures = self.vertexFeatures[-1] + feedThroughVertexFeatures, gain = 1)
        
        if self.inputEdgeEncode:
            if vertexMode == 'PointNet':
                raise Exception(f"Vertex Mode 'PointNet' does not support edge features, please disable inputEdgeEncode")
            encoderLayout = edgeHiddenLayout + [self.edgeFeatures[1]]
            if verbose:
                print(f"Building inputEdgeEncoder from {1 + feedThroughEdgeFeatures} to {self.edgeFeatures[0]} with hidden layout {[{' '.join([str(v) for v in edgeHiddenLayout])}]}")
            self.edgeFeatures = self.edgeFeatures[1:]
            if verbose:
                print(f"New Edge Layout: [{' '.join([str(v) for v in self.edgeFeatures])}]")
            self.inputEdgeEncoder = buildMLP(encoderLayout, inputFeatures = 1 + feedThroughEdgeFeatures, gain = 1)
            
        self.messageFeatures = self.vertexFeatures[1:] if messageFeatures is None else messageFeatures
        
        if verbose:
            print(f"Final vertex Layout: [{' '.join([str(v) for v in self.vertexFeatures])}]")
            print(f"Final edge   Layout: [{' '.join([str(v) for v in self.edgeFeatures])}]")
            
            print('Starting construction of Message Passing Steps: ')
            for i in range(len(self.vertexFeatures)-1):
                if i != len(self.vertexFeatures) - 2:
                    print(f'\tLayer {i}: vertex Features: {self.vertexFeatures[i]} -> {self.vertexFeatures[i+1]}, edge Features: {self.edgeFeatures[i]} -> {self.edgeFeatures[i+1]}')
                else:                    
                    print(f'\tLayer {i}: vertex Features: {self.vertexFeatures[i]} -> {self.vertexFeatures[i+1]}')
        
        self.vertexModules = nn.ModuleList([])
        self.edgeModules = nn.ModuleList([])
        self.edgeMLPModules = nn.ModuleList([])
        if edgeMLP:            
            for i in range(len(self.edgeFeatures) - 1):
                layerInput = self.edgeFeatures[i] + feedThroughEdgeFeatures
                layerOutput = self.edgeFeatures[i+1]
                mlp = buildMLP(layers  = edgeHiddenLayout + [layerOutput], inputFeatures = layerInput, gain = 1)
                if verbose:
                    print('Processing Layer %2d' % i)
                    print(f"\tEdge Input: {layerInput}, edge Output: {layerOutput}")
#                     print(f"\tEdge Features: {self.edgeFeatures[i]}")
                self.edgeMLPModules.append(mlp)
        
        if vertexMode == 'PointNet':
            pass
        elif edgeMode == 'CConv':
            for i in range(len(self.vertexFeatures) - 1):
                if verbose:
                    print('Processing Layer %2d' % i)
                    print(f"\tVertex Input: {self.vertexFeatures[i]}, vertex Output: {self.vertexFeatures[i+1]}")
                    print(f"\tEdge Features: {self.edgeFeatures[i]}")
                layerInput = self.vertexFeatures[i] + feedThroughVertexFeatures
                layerOutput = self.messageFeatures[i] if self.vertexMode == 'MP-PDE' else self.vertexFeatures[i+1] #self.vertexFeatures[i+1] if messageFeatures is None else self.messageFeatures[i]
                mlp = buildMLP(layers  = edgeHiddenLayout + [layerInput * layerOutput], inputFeatures = self.edgeFeatures[i] + feedThroughEdgeFeatures)
                self.edgeModules.append(mlp)
        elif edgeMode == 'MP-PDE':
            for i in range(len(self.vertexFeatures) - 1):
                layerInput = self.vertexFeatures[i]*2 + feedThroughVertexFeatures + self.edgeFeatures[i] + feedThroughEdgeFeatures
                layerOutput =  self.vertexFeatures[i+1] if self.vertexMode == 'MessagePassing' else self.messageFeatures[i]
                if verbose:
                    print('Processing Layer %2d' % i)
                    print(f"\tLayer Input: {layerInput}, layer Output: {layerOutput}")
#                     print(f"\tEdge Features: {self.edgeFeatures[i]}")
                mlp = buildMLP(layers  = edgeHiddenLayout + [layerOutput], inputFeatures = layerInput)
                self.edgeModules.append(mlp)        
            pass
        else:
            raise Exception(f'Edge Mode {edgeMode} is not supported')
            
        if vertexMode == 'MessagePassing':
            pass
        elif vertexMode == 'PointNet':    
            for i in range(len(self.vertexFeatures) - 1):
                layerInput = self.vertexFeatures[i] + feedThroughVertexFeatures
                layerOutput = self.vertexFeatures[i+1]
                mlp = buildMLP(layers  = vertexHiddenLayout + [layerOutput], inputFeatures = layerInput, gain = 1)
                if verbose:
                    print('Processing Layer %2d' % i)
                    print(f"\tVertex Input: {layerInput}, vertex Output: {layerOutput}")
#                     print(f"\tEdge Features: {self.edgeFeatures[i]}")
                self.vertexModules.append(mlp)
        elif vertexMode == 'MP-PDE':
            for i in range(len(self.vertexFeatures) - 1):
                layerInput = self.vertexFeatures[i] + feedThroughVertexFeatures + (self.vertexFeatures[i] if messageFeatures is None else self.messageFeatures[i])
                layerOutput = self.vertexFeatures[i+1]
                mlp = buildMLP(layers  = vertexHiddenLayout + [layerOutput], inputFeatures = layerInput, gain = 1)
                if verbose:
                    print('Processing Layer %2d' % i)
                    print(f"\tVertex Input: {layerInput} ({self.vertexFeatures[i]} + {feedThroughVertexFeatures} + {(self.vertexFeatures[i] if messageFeatures is None else self.messageFeatures[i])}), vertex Output: {layerOutput}")
                self.vertexModules.append(mlp)            
        else :
            raise Exception(f'Vertex Mode {vertexMode} is not supported')
    def verbosePrint(self, s):
        if self.verbose:
            print(s)
        
    
    def forward(self, i, j, inputVertexFeatures, inputEdgeFeatures, feedThroughVertexFeatures = None, feedThroughEdgeFeatures = None ): 
        if self.inputVertexFeatures != inputVertexFeatures.shape[1]:
            raise Exception(f'Input shape {inputVertexFeatures.shape} does not match expected input feature size {self.inputFeatures}') 
        if self.inputEdgeFeatures != inputEdgeFeatures.shape[1] and self.edgeMode != 'PointNet':
            raise Exception(f'Input shape {inputEdgeFeatures.shape} does not match expected input feature size {self.inputFeatures}')
        if self.feedThroughVertexFeatures != 0 and feedThroughVertexFeatures == None:
            raise Exception(f'Expected {self.feedThroughVertexFeatures} feed through vertex features but got None')
        if self.feedThroughEdgeFeatures != 0 and feedThroughEdgeFeatures == None:
            raise Exception(f'Expected {self.feedThroughEdgeFeatures} feed through edge features but got None')
        if self.feedThroughVertexFeatures != 0 and feedThroughVertexFeatures.shape[1] != self.feedThroughVertexFeatures:
            raise Exception(f'Expected {self.feedThroughVertexFeatures} feed through vertex features but got input of shape {feedThroughVertexFeatures.shape}')
        if self.feedThroughEdgeFeatures != 0 and feedThroughEdgeFeatures.shape[1] != self.feedThroughEdgeFeatures:
            raise Exception(f'Expected {self.feedThroughEdgeFeatures} feed through edge features but got input of shape {feedThroughVertexFeatures.shape}')
        if self.feedThroughVertexFeatures == 0 and feedThroughVertexFeatures != None:
            raise Exception(f'Expected {self.feedThroughVertexFeatures} feed through vertex features but got input of shape {feedThroughVertexFeatures.shape}')
        if self.feedThroughEdgeFeatures == 0 and feedThroughEdgeFeatures != None:
            raise Exception(f'Expected {self.feedThroughEdgeFeatures} feed through edge features but got input of shape {feedThroughEdgeFeatures.shape}')
            
        
        self.verbosePrint('Running forward pass through NN:')
        self.verbosePrint(f"\tInput Vertex Features shape: {inputVertexFeatures.shape}")
        self.verbosePrint(f"\tInput Edge   Features shape: {inputEdgeFeatures.shape}")
        self.verbosePrint(f"\tNeighborhood shape: {i.shape} and {j.shape}")
        self.verbosePrint(f"\tFeed through vertex feature shape: {'N/A' if feedThroughVertexFeatures is None else feedThroughVertexFeatures.shape}")
        self.verbosePrint(f"\tFeed through edge   feature shape: {'N/A' if feedThroughEdgeFeatures is None else feedThroughEdgeFeatures.shape}")
            
        edgeFeatures = torch.clone(inputEdgeFeatures)
        vertexFeatures = torch.clone(inputVertexFeatures)
        
        if self.inputEncode:
            self.verbosePrint(f'Running input vertex encoder ({inputVertexFeatures.shape[1]} -> {self.vertexFeatures[0]})')
            vertexFeatures = self.inputEncoder(torch.hstack((vertexFeatures, feedThroughVertexFeatures)) if feedThroughVertexFeatures is not None else vertexFeatures)
        if self.inputEdgeEncode:
            self.verbosePrint(f'Running input edge   encoder ({edgeFeatures.shape[1]} -> {self.edgeFeatures[0]})')
            edgeFeatures = self.inputEdgeEncoder(torch.hstack((edgeFeatures, feedThroughEdgeFeatures)) if feedThroughEdgeFeatures is not None else edgeFeatures)
        
        if self.edgeMode == 'CConv':
            for l, layer in enumerate(self.edgeModules):
                self.verbosePrint('Processing Layer %2d' % l)
                self.verbosePrint(f"\tVertex Input: {self.vertexFeatures[l]}, vertex Output: {self.vertexFeatures[l+1]}")
                self.verbosePrint(f"\tEdge Features: {self.edgeFeatures[l]}")
                if feedThroughEdgeFeatures is not None:
                    filterOutput = layer(torch.hstack((edgeFeatures,feedThroughEdgeFeatures))).reshape(i.shape[0], self.vertexFeatures[l] + self.feedThroughVertexFeatures, self.messageFeatures[l] if self.vertexMode == 'MP-PDE' else self.vertexFeatures[l+1])
                else:
                    filterOutput = layer(edgeFeatures).reshape(i.shape[0], self.vertexFeatures[l] + self.feedThroughVertexFeatures, self.messageFeatures[l] if self.vertexMode == 'MP-PDE' else self.vertexFeatures[l+1])
                if feedThroughVertexFeatures is not None:
                    vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
                    
                self.verbosePrint(f"\tProcessed edge mlp, input shape {edgeFeatures.shape} -> output shape {filterOutput.shape}")
                self.verbosePrint(f"\tVertex Feature Shape {vertexFeatures.shape}")
                message = torch.sum(filterOutput * vertexFeatures[j].unsqueeze(2), dim = 1)
                self.verbosePrint(f"\tMessage shape: {message.shape}")
                aggrMessage = scatter_sum(message, i, dim = 0, dim_size = vertexFeatures.shape[0])
                self.verbosePrint(f"\tAggregated message shape: {aggrMessage.shape}")

                if self.vertexMode == 'MessagePassing':
#                     if feedThroughVertexFeatures is None:
                    vertexFeatures = aggrMessage
#                     elif feedThroughVertexFeatures is not None and self.outputEncode:
#                         vertexFeatures = torch.hstack((aggrMessage, feedThroughVertexFeatures))
#                     elif feedThroughVertexFeatures is not None and not self.outputEncode and l == len(self.edgeModules) - 1:
#                         vertexFeatures = aggrMessage
                    if feedThroughVertexFeatures is not None and self.outputEncode and l == len(self.edgeModules)-1:
                        vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
                elif self.vertexMode == 'MP-PDE':
#                     if feedThroughVertexFeatures is None:
#                     if l == 0:
#                     if feedThroughVertexFeatures is not None:
#                         vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
#                     print('vertexFeatures:', vertexFeatures.shape)
#                     print('feedThroughVertexFeatures:', feedThroughVertexFeatures.shape)
#                     print('aggrMessage:', aggrMessage.shape)
            
                    vertexFeatures = self.vertexModules[l](torch.hstack((vertexFeatures, aggrMessage)))
#                     elif feedThroughVertexFeatures is not None:
#                         vertexFeatures = self.vertexModules[l](torch.hstack((vertexFeatures, aggrMessage, feedThroughVertexFeatures)))
#                     if self.outputEncode and l == len(self.vertexModules) - 1 and feedThroughVertexFeatures is not None:
#                         vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
                    if feedThroughVertexFeatures is not None and self.outputEncode and l == len(self.edgeModules)-1:
                        vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
#                         vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
#                     elif feedThroughVertexFeatures is not None and not self.outputEncode and l == len(self.edgeModules) - 1:
#                         vertexFeatures = self.vertexModules[l](torch.hstack((vertexFeatures, aggrMessage)))
                else:
                    raise Exception(f'Vertex Mode {self.vertexMode} x Edge mode {self.edgeMode} combination is not supported')     
                if self.edgeMLP and l != len(self.edgeModules)-1:
#                     print(l, len(self.vertexModules), len(self.edgeMLPModules))
                    self.verbosePrint(f"\tRunning edge MLP {edgeFeatures.shape[1] + feedThroughEdgeFeatures.shape[1] if feedThroughEdgeFeatures is not None else edgeFeatures.shape[1]} -> {self.edgeFeatures[l+1]}")
                    edgeFeatures = self.edgeMLPModules[l](torch.hstack((edgeFeatures, feedThroughEdgeFeatures)) if feedThroughEdgeFeatures is not None else edgeFeatures)
                    
                self.verbosePrint(f"\tNew Vertex feature shape: {vertexFeatures.shape}")     
        elif self.edgeMode == 'MP-PDE':
            for l, layer in enumerate(self.edgeModules):
                self.verbosePrint('Processing Layer %2d' % l)
                self.verbosePrint(f"\tVertex Input: {self.vertexFeatures[l]}, vertex Output: {self.vertexFeatures[l+1]}")
                self.verbosePrint(f"\tEdge Features: {self.edgeFeatures[l]}")
                if feedThroughEdgeFeatures is not None:
                    self.verbosePrint(f"\tGathered Edge Features: {edgeFeatures.shape[1]} + {feedThroughEdgeFeatures.shape[1]} via feed through")
                    eFeatures = torch.hstack((edgeFeatures,feedThroughEdgeFeatures))
                else:
                    self.verbosePrint(f"\tGathered Edge Features: {edgeFeatures.shape[1]}")
                    eFeatures = edgeFeatures
                if feedThroughVertexFeatures is not None:
                    self.verbosePrint(f"\tGathered Vertex Features: {vertexFeatures.shape[1]} + {feedThroughVertexFeatures.shape[1]} via feed through")
                    vFeatures = torch.hstack((vertexFeatures[i], vertexFeatures[j],feedThroughVertexFeatures[j]))
                else:
                    self.verbosePrint(f"\tGathered Vertex Features: {vertexFeatures.shape[1]}")
                    vFeatures = torch.hstack((vertexFeatures[i], vertexFeatures[j]))
#                 print(vFeatures.shape, eFeatures.shape)
                features = torch.hstack((eFeatures, vFeatures))         
                self.verbosePrint(f"\tProcessed edge mlp, input shape {features.shape}")           
                message = layer(features)
#                 message = torch.sum(filterOutput * vertexFeatures[j].unsqueeze(2), dim = 1)
                self.verbosePrint(f"\tMessage shape: {message.shape}")
                aggrMessage = scatter_sum(message, i, dim = 0, dim_size = vertexFeatures.shape[0])
                self.verbosePrint(f"\tAggregated message shape: {aggrMessage.shape}")

                if self.vertexMode == 'MessagePassing':
#                     if feedThroughVertexFeatures is None:
                    vertexFeatures = aggrMessage
#                     elif feedThroughVertexFeatures is not None and self.outputEncode:
#                         vertexFeatures = torch.hstack((aggrMessage, feedThroughVertexFeatures))
#                     elif feedThroughVertexFeatures is not None and not self.outputEncode and l == len(self.edgeModules) - 1:
#                         vertexFeatures = aggrMessage
                    if feedThroughVertexFeatures is not None and self.outputEncode and l == len(self.edgeModules)-1:
                        vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
#                     print(vertexFeatures.shape)
                elif self.vertexMode == 'MP-PDE':
            
#                     if feedThroughVertexFeatures is None:
#                     if l == 0:
                    if feedThroughVertexFeatures is not None:
                        vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
                    vertexFeatures = self.vertexModules[l](torch.hstack((vertexFeatures, aggrMessage)))
#                     elif feedThroughVertexFeatures is not None:
#                         vertexFeatures = self.vertexModules[l](torch.hstack((vertexFeatures, aggrMessage, feedThroughVertexFeatures)))
#                     if self.outputEncode and l == len(self.vertexModules) - 1 and feedThroughVertexFeatures is not None:
#                         vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
                    if (feedThroughVertexFeatures is not None and self.outputEncode and l == len(self.vertexModules)-1):
                        vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
#                         vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
#                     elif feedThroughVertexFeatures is not None and not self.outputEncode and l == len(self.edgeModules) - 1:
#                         vertexFeatures = self.vertexModules[l](torch.hstack((vertexFeatures, aggrMessage)))
                else:
                    raise Exception(f'Vertex Mode {self.vertexMode} x Edge mode {self.edgeMode} combination is not supported')     
                if self.edgeMLP and l != len(self.vertexModules)-1:
                    self.verbosePrint(f"\tRunning edge MLP {edgeFeatures.shape[1] + feedThroughEdgeFeatures.shape[1] if feedThroughEdgeFeatures is not None else edgeFeatures.shape[1]} -> {self.edgeFeatures[l+1]}")
                    edgeFeatures = self.edgeMLPModules[l](torch.hstack((edgeFeatures, feedThroughEdgeFeatures)) if feedThroughEdgeFeatures is not None else edgeFeatures)
                self.verbosePrint(f"\tNew Vertex feature shape: {vertexFeatures.shape}")      
        elif self.edgeMode == 'PointNet':
            for l, layer in enumerate(self.vertexModules):
                self.verbosePrint('Processing Layer %2d' % l)
                self.verbosePrint(f"\tVertex Input: {self.vertexFeatures[l]}, vertex Output: {self.vertexFeatures[l+1]}")
                if feedThroughVertexFeatures is None:
                    vertexFeatures = self.vertexModules[l](vertexFeatures)
                elif feedThroughVertexFeatures is not None:
                    vertexFeatures = self.vertexModules[l](torch.hstack((vertexFeatures, feedThroughVertexFeatures)))
                if self.outputEncode and l == len(self.vertexModules) - 1:
                    vertexFeatures = torch.hstack((vertexFeatures, feedThroughVertexFeatures))
        else:
            raise Exception(f'Vertex Mode {self.vertexMode} x Edge mode {self.edgeMode} combination is not supported')
            
        if self.outputEncode:
            self.verbosePrint(f'Running output vertex decoder ({self.vertexFeatures[-1] + self.feedThroughVertexFeatures} -> {self.outputVertexFeatures})')
            vertexFeatures = self.outputEncoder(vertexFeatures)
        return vertexFeatures# / particleSupport
    
    def printProcess(self): 
        self.verbosePrint('Running forward pass through NN:')
        self.verbosePrint(f"Input Vertex Features shape: {self.inputVertexFeatures}")
        self.verbosePrint(f"Input Edge   Features shape: {self.inputEdgeFeatures}")
        
        self.verbosePrint(f"Feed through vertex feature shape: {'N/A' if self.feedThroughVertexFeatures == 0 else self.feedThroughVertexFeatures}")
        self.verbosePrint(f"Feed through edge   feature shape: {'N/A' if self.feedThroughEdgeFeatures == 0 else self.feedThroughEdgeFeatures}")
        
        print('---------------------------------------\n')
        
        stepCounter = 0
        feedThroughVertexText = f"feedThrough @ {self.feedThroughVertexFeatures} ..." if self.feedThroughVertexFeatures != 0 else ''
        feedThroughEdgeText = f"feedThrough @ {self.feedThroughEdgeFeatures} ..." if self.feedThroughEdgeFeatures != 0 else ''
        
        print(f'[{stepCounter:3}] - Pre-Processing')
        stepCounter = stepCounter + 1
        if self.inputEncode:
            inputShape = self.inputVertexFeatures if self.feedThroughVertexFeatures == 0 else self.inputVertexFeatures + self.feedThroughVertexFeatures
            outputShape = self.vertexFeatures[0]
            print(f'[{stepCounter:3}] - Vertex OP - {"Input encoder":24}: [inputVertexFeatures @ {inputShape} ... {feedThroughVertexText}] -> [{outputShape}]')
            stepCounter = stepCounter + 1
        if self.inputEdgeEncode:
            inputShape = self.inputEdgeFeatures if self.feedThroughEdgeFeatures == 0 else self.inputEdgeFeatures + self.feedThroughEdgeFeatures
            outputShape = self.edgeFeatures[0]
            print(f'[{stepCounter:3}] - Edge   OP - {"Input encoder":24}: [  inputEdgeFeatures @ {inputShape} ... {feedThroughEdgeText}] -> [{outputShape}]')
            stepCounter = stepCounter + 1
#         return
        
        if self.edgeMode == 'CConv':     
            print(f'[{stepCounter:3}] - Layer Processing')
            stepCounter = stepCounter + 1       
            for l, layer in enumerate(self.edgeModules):
                print(f'[{stepCounter:3}] - Layer {l:2} - \tEdge OP (Message Passing): [edgeFeatures @ {self.edgeFeatures[l]} ... {feedThroughEdgeText}] @ {self.edgeFeatures[l] + self.feedThroughEdgeFeatures} -> [message @ {self.vertexFeatures[l] + self.feedThroughVertexFeatures, self.vertexFeatures[l+1]}]')
                print(f'[{stepCounter:3}] - Layer {l:2} - \tReshape OP: [message @ {self.vertexFeatures[l] + self.feedThroughVertexFeatures, self.vertexFeatures[l+1]}] -> M @ {self.vertexFeatures[l] + self.feedThroughVertexFeatures} x {self.vertexFeatures[l+1]}')
                print(f'[{stepCounter:3}] - Layer {l:2} - \tMessage aggregation : M @ {self.vertexFeatures[l] + self.feedThroughVertexFeatures} x {self.vertexFeatures[l+1]} x [vertexFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}]')
                if self.vertexMode == 'MessagePassing':
                    if self.feedThroughVertexFeatures != 0 and self.outputEncode and l == len(self.edgeModules)-1:
                        print(f'[{stepCounter:3}] - Layer {l:2} - \tOutput   Gathering (last step)-> Vertex Features: [vertexFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}]')
                        stepCounter = stepCounter + 1
                elif self.vertexMode == 'MP-PDE':
                    if l == 0:
                        if feedThroughVertexFeatures is not None:
                            print(f'[{stepCounter:3}] - Layer {l:2} - \tInput expansion: [vertexFeatures @ {self.vertexFeatures[0]}] -> [vertexFeatures @ {self.vertexFeatures[0]} ... {feedThroughVertexText}]  ')           
                    print(f'[{stepCounter:3}] - Layer {l:2} - \tVertex MLP (MP-PDE)      : [messageFeatures @ {self.messageFeatures[l]} ... vertexFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}] -> [vertexFeatures @ {self.vertexFeatures[l + 1]}]')
                    stepCounter = stepCounter + 1
                    if self.feedThroughVertexFeatures != 0 and self.outputEncode and l == len(self.edgeModules)-1:
                        print(f'[{stepCounter:3}] - Layer {l:2} - \tOutput   Gathering (last step)-> Vertex Features: [vertexFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}]')
                        stepCounter = stepCounter + 1
                else:
                    raise Exception(f'Vertex Mode {self.vertexMode} x Edge mode {self.edgeMode} combination is not supported')     
                if self.edgeMLP and l != len(self.edgeModules)-1:
                    print(f'[{stepCounter:3}] - Layer {l:2} - \tEdge MLP (GNS)           : [edgeFeatures @ {self.edgeFeatures[l]} ... {feedThroughEdgeText}] @ {self.edgeFeatures[l] + self.feedThroughEdgeFeatures} -> [edgeFeatures @ {self.edgeFeatures[l+1]}]')
                    stepCounter = stepCounter + 1
                    
#                 self.verbosePrint(f"\tNew Vertex feature shape: {vertexFeatures.shape}")     
        elif self.edgeMode == 'MP-PDE':
            print(f'[{stepCounter:3}] - Layer Processing')
            stepCounter = stepCounter + 1
            for l, layer in enumerate(self.edgeModules):
                print(f'[{stepCounter:3}] - Layer {l:2} - \tEdge OP (Message Passing): [edgeFeatures @ {self.edgeFeatures[l]} ... {feedThroughEdgeText} vertexFeatures[i] @ {self.vertexFeatures[l]} vertexFeatures[j] @ {self.vertexFeatures[l]} ... {feedThroughVertexText}] @ {self.edgeFeatures[l] + self.vertexFeatures[l] * 2 + self.feedThroughEdgeFeatures + self.feedThroughVertexFeatures} -> [message @ {self.messageFeatures[l]}]')
                stepCounter = stepCounter + 1
                if self.vertexMode == 'MessagePassing':
                    if self.feedThroughVertexFeatures != 0 and self.outputEncode and l == len(self.edgeModules)-1:
                        print(f'[{stepCounter:3}] - Layer {l:2} - \tOutput   Gathering (last step)-> Vertex Features: [vertexFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}]')
                        stepCounter = stepCounter + 1
                elif self.vertexMode == 'MP-PDE':
                    print(f'[{stepCounter:3}] - Layer {l:2} - \tVertex MLP (MP-PDE)      : [messageFeatures @ {self.messageFeatures[l]} ... vertexFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}] -> [vertexFeatures @ {self.vertexFeatures[l + 1]}]')
                    stepCounter = stepCounter + 1
                    if self.feedThroughVertexFeatures != 0 and self.outputEncode and l == len(self.edgeModules)-1:
                        print(f'[{stepCounter:3}] - Layer {l:2} - \tOutput   Gathering (last step)-> Vertex Features: [vertexFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}]')
                        stepCounter = stepCounter + 1
                else:
                    raise Exception(f'Vertex Mode {self.vertexMode} x Edge mode {self.edgeMode} combination is not supported')     
                if self.edgeMLP and l != len(self.vertexModules)-1:
                    print(f'[{stepCounter:3}] - Layer {l:2} - \tEdge MLP (GNS)           : [edgeFeatures @ {self.edgeFeatures[l]} ... {feedThroughEdgeText}] @ {self.edgeFeatures[l] + self.feedThroughEdgeFeatures} -> [edgeFeatures @ {self.edgeFeatures[l+1]}]')
                    stepCounter = stepCounter + 1
        elif self.edgeMode == 'PointNet':
            print(f'[{stepCounter:3}] - Layer Processing')
            for l, layer in enumerate(self.vertexModules):
                print(f'[{stepCounter:3}] - Layer {l:2} - \tVertex MLP (MP-PDE)      : [messageFeatures @ {self.vertexFeatures[l]} ... {feedThroughVertexText}] -> [vertexFeatures @ {self.vertexFeatures[l + 1]}]')
                stepCounter = stepCounter + 1
        else:
            raise Exception(f'Vertex Mode {self.vertexMode} x Edge mode {self.edgeMode} combination is not supported')
            
        print(f'[{stepCounter:3}] - Post-Processing')
        stepCounter = stepCounter + 1
        if self.outputEncode:
#             self.verbosePrint(f'Running output vertex decoder ({self.vertexFeatures[-1] + self.feedThroughVertexFeatures} -> {self.outputVertexFeatures})')
#             vertexFeatures = self.outputEncoder(vertexFeatures)
            print(f'[{stepCounter:3}] - Vertex OP - {"Output decoder":24}: [vertexFeatures @ {self.vertexFeatures[-1]} ... {feedThroughVertexText}] -> [{self.outputVertexFeatures}]')
#         return vertexFeatures / particleSupport
# inputFeatures = features.shape[1]

# MLPmodel = mlpNetwork(
#      vertexFeatures = [inputFeatures,1,4,4,8], vertexHiddenLayout = [4,4,4], feedThroughVertexFeatures = 1,
#      edgeFeatures   = [1,4,4,4,4,4],             edgeHiddenLayout   = [4,4,4], feedThroughEdgeFeatures   = 1,
# #      messageFeatures = [4,8,16,32],
#      edgeMLP = False,
# #      edgeMLP = False,
#      vertexMode = 'MessagePassing',
# #      vertexMode = 'MP-PDE',
    
# #      vertexMode = 'PointNet',
#      edgeMode = 'CConv',
# #      edgeMode = 'MP-PDE',
# #      edgeMode = 'PointNet',
#      inputEncode = True,
# #      inputEncode = False,
# #      outputEncode = True,
#      outputEncode = True,
# #      inputEdgeEncode = True,
#      inputEdgeEncode = True,
#      verbose = True
# ).to(device)
# optimizerMLP = Adam(MLPmodel.parameters(), lr=lr, weight_decay=0)

# featuresOut = MLPmodel(i, j,
#                        inputVertexFeatures = features, 
#                        inputEdgeFeatures = distance, 
# #                        feedThroughEdgeFeatures = None, 
# #                        feedThroughVertexFeatures = None,
#                        feedThroughEdgeFeatures = (u[j] - u[i])[:,None], 
#                        feedThroughVertexFeatures = x
#                       )
# print('Output of GraphNetwork has shape ', featuresOut.shape)

# MLPmodel.printProcess()

def buildMessagePassingNetwork(inputFeatures, vertexFeatures = None, vertexMLPLayout = [4,4,4], feedThroughEdgeFeatures = 0, edgeMLP = False, edgeFeatures = None, inputEncode = False, outputDecode = False, seed = None):
#     print([inputFeatures] + layout + [1])
    GraphNet = mlpNetwork(
     vertexFeatures = [inputFeatures,1] if vertexFeatures is None else [inputFeatures] + vertexFeatures + [1], vertexHiddenLayout = vertexMLPLayout, feedThroughVertexFeatures = 0, feedThroughEdgeFeatures = feedThroughEdgeFeatures,
    edgeFeatures = None if not edgeMLP else edgeFeatures,
        edgeHiddenLayout = vertexMLPLayout,
     edgeMLP = edgeMLP,
     vertexMode = 'MessagePassing',
     edgeMode = 'CConv',
     verbose = False,
        inputEncode = inputEncode,
        outputEncode = outputDecode,
        seed = seed
    ).to(device)
    return GraphNet

def buildPointNet(inputFeatures, vertexFeatures = None, vertexMLPLayout = [4,4,4], feedThroughVertexFeatures = 0, inputEncode = False, outputDecode = False, seed = None):
    GraphNet = mlpNetwork(
     vertexFeatures = [inputFeatures,1] if vertexFeatures is None else [inputFeatures] + vertexFeatures + [1], vertexHiddenLayout = vertexMLPLayout, feedThroughVertexFeatures = feedThroughVertexFeatures, feedThroughEdgeFeatures = 0,
     edgeFeatures = None,
    edgeHiddenLayout = vertexMLPLayout,
     edgeMLP = False,
     vertexMode = 'PointNet',
     edgeMode = 'PointNet',
     verbose = False,
        inputEncode = inputEncode,
        outputEncode = outputDecode,
        seed = seed
    ).to(device)
    return GraphNet

def buildGNS(inputFeatures, vertexFeatures = None, vertexMLPLayout = [], messageFeatures = None, feedThroughEdgeFeatures = 0, edgeMLP = False, edgeFeatures = None,feedThroughVertexFeatures = 0, inputEncode = False, outputDecode = False, seed = None):
#     print([inputFeatures] + layout + [1])
    GraphNet = mlpNetwork(
     vertexFeatures = [inputFeatures,1] if vertexFeatures is None else [inputFeatures] + vertexFeatures + [1], vertexHiddenLayout = vertexMLPLayout, feedThroughVertexFeatures = feedThroughVertexFeatures, feedThroughEdgeFeatures = feedThroughEdgeFeatures,
    edgeFeatures = None if not edgeMLP else edgeFeatures,
        edgeHiddenLayout = vertexMLPLayout,
    messageFeatures  = messageFeatures,
     edgeMLP = edgeMLP,
     vertexMode = 'MP-PDE',
     edgeMode = 'MP-PDE',
     verbose = False,
        inputEncode = inputEncode,
        outputEncode = outputDecode,
        seed = seed,
    ).to(device)
    return GraphNet

def getGroundTruthKernel(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction):
    return scatter_sum(torch.hstack(areas)[j] * kernel(torch.abs(distance), particleSupport), i, dim = 0, dim_size = torch.hstack(areas).shape[0])
def getGroundTruthKernelGradient(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction):
    return scatter_sum(torch.hstack(areas)[j] * kernelGradient(torch.abs(distance), direction, particleSupport), i, dim = 0, dim_size = torch.hstack(areas).shape[0])
def getGroundTruthPhysics(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction):
    return torch.hstack(outVel)
def getFeaturesKernel(positions, velocities, areas, densities, dudts, inVel, outVel):
    return torch.ones_like(areas )[:,None]
def getFeaturesPhysics(positions, velocities, areas, densities, dudts, inVel, outVel):
    return torch.vstack((inVel, densities,torch.ones_like(areas))).mT

torch.manual_seed('1337')

dataLoader = DataLoader(dataSet, shuffle=True, batch_size = batchSize).batch_sampler
dataIter = iter(dataLoader)

def lossFunction(prediction, groundTruth):
    return (prediction - groundTruth)**2 # MSE

# testData = {}
# for i in range(len(testingFiles)):
#     testData[testingFiles[i].split('/')[-1].split('.')[0]] = loadTestcase(testingData, settings, testingFiles[i], [0], device, getGroundTruth, getFeatures)

# Hyperparameters for the NN
lr = 1e-2 # Learning rate
iterations = 1000 # update iterations per epoch
epochs = 5 # total number of epochs, LR is halved every epoch
# n = 17 # number of weights per continuous convolution
# basis = 'fourier' # basis for the convolution, set to linear for CConv

# window = 'cubicSpline'
# windowNorm = 'integral'
# window = None

layers = [1]
initialLR = 1e-2
particleData = trainingData
# groundTruthFn = getGroundTruth
# featureFn = getFeatures
lossFn = lossFunction

inputFeatures = 1
seeds = np.random.randint(0, 2**30, 4)


testCase = 'kernel'

if testCase == 'kernel':
    groundTruthFn = getGroundTruthKernel
    featureFn = getFeaturesKernel
elif testCase == 'kernelGradient':
    groundTruthFn = getGroundTruthKernelGradient   
    featureFn = getFeaturesKernel 
elif testCase == 'physicsUpdate':
    groundTruthFn = getGroundTruthPhysics
    featureFn = getFeaturesPhysics
    
testData = {}
for i in range(len(testingFiles)):
#     testData[testingFiles[i].split('/')[-1].split('.')[0]] = loadTestcase(testingData, settings, testingFiles[i], [0, 128, 256, 1024], device, getGroundTruth, getFeatures, offset)
    testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset)

bdata = [0]
positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
i, j, distance, direction = batchedNeighborsearch(positions, setup)
x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

x = x[:,None].to(device)    
groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
distance = (distance * direction)[:,None].to(device)
features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)
inputFeatures = features.shape[1]

features = torch.normal(torch.zeros_like(features), torch.ones_like(features))
features = torch.ones_like(features)

def buildMLP(layers, inputFeatures = 1, gain = 1/np.sqrt(34)):
    modules = []
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i]))
#             torch.nn.init.uniform_(modules[-1].weight,-0.5, 0.5)
            torch.nn.init.xavier_normal_(modules[-1].weight,1)
    #         torch.nn.init.zeros_(modules[-1].weight)
            torch.nn.init.zeros_(modules[-1].bias)
            modules.append(nn.BatchNorm1d(layers[i]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layers[-2],layers[-1]))
    else:
        modules.append(nn.Linear(inputFeatures,layers[-1]))        
    torch.nn.init.xavier_normal_(modules[-1].weight,gain)
    torch.nn.init.zeros_(modules[-1].bias)
    return nn.Sequential(*modules)
# outFeatures = []
# width = 4
# depth = 4
# nx = 128
# for ii in range(nx):    
#     MLPmodel = buildMessagePassingNetwork(inputFeatures, feedThroughEdgeFeatures = 0, vertexFeatures = [8,8], vertexMLPLayout = [width] * depth, edgeMLP = False)#, edgeFeatures = [1,2,2])
# #     MLPmodel = buildPointNet(inputFeatures, feedThroughVertexFeatures = 0, vertexFeatures = None, vertexMLPLayout = [width] * depth)
# #     MLPmodel = buildGNS(inputFeatures, feedThroughVertexFeatures = 0, feedThroughEdgeFeatures = 0, vertexFeatures = None, vertexMLPLayout = [width] * depth, edgeMLP = False, edgeFeatures = [1] + [], messageFeatures = [8])
#     featuresOut = MLPmodel(i, j,
#                            inputVertexFeatures = features, 
#                            inputEdgeFeatures = distance, 
#                            feedThroughVertexFeatures = None,
#                            feedThroughEdgeFeatures = None, #(u[j] - u[i])[:,None], 
#                           )
#     outFeatures.append(featuresOut.detach().numpy())
# print('Output of GraphNetwork has shape ', featuresOut.shape, 'parameters:', count_parameters(MLPmodel))
# MLPmodel.printProcess()

# fig, axis = plt.subplots(1, 3, figsize=(16,5), sharex = False, sharey = False, squeeze = False)

# axis[0,0].plot(x.repeat(1,nx).mT, np.array(outFeatures)[:,:,0])
# axis[0,0].plot(x, groundTruth)
# axis[0,1].plot(x, features.detach().numpy())
# axis[0,1].plot(x, featuresOut.detach().numpy())
# axis[0,2].hist(np.array(outFeatures).flatten(),bins = 128)

# print('inputfeatures : mean %g, std %g, min %g, max %g' % (np.mean(features.detach().numpy()), np.std(features.detach().numpy()), np.min(features.detach().numpy()), np.max(features.detach().numpy())))
# print('outputFeatures: mean %g, std %g, min %g, max %g' % (np.mean(outFeatures), np.std(outFeatures), np.min(outFeatures), np.max(outFeatures)))


import pandas as pd
seed = 1

import sklearn
import pandas
def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def getTestingLossFrameMLP(net, testData, plot = False):
#     label = '%s x %2d @ %s' % (modelState['basis'], modelState['n'], str(modelState['layers']))
    testPredictions = {}
    testGTs = {}
    testPositions = {}
    with torch.no_grad():
        for i, k in  enumerate(testData.keys()):
            gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
            prediction = net(testData[k][6], testData[k][7], testData[k][5], testData[k][8], testData[k][-2] if net.feedThroughVertexFeatures != 0 else None, (testData[k][-1][testData[k][7]] - testData[k][-1][testData[k][6]])[:,None] if net.feedThroughEdgeFeatures != 0 else None).reshape(len(testData[k][0]), testData[k][0][0].shape[0]).detach().cpu().numpy()
            testPredictions[k] = prediction
            testGTs[k] = gt.detach().cpu().numpy()
            testPositions[k] = testData[k][0]
    if plot:
        fig, axis = plt.subplots(len(testPredictions.keys()),3, figsize=(16,8), sharex = True, sharey = 'col', squeeze = False)

        for s, k in  enumerate(testData.keys()):
            norm = mpl.colors.Normalize(vmin=0, vmax=len(testPredictions[k]) - 1)
            for i, (xs, rhos) in enumerate(zip(testPositions[k], testGTs[k])):
                c = cmap(norm(i))
                axis[s,0].plot(xs.cpu().numpy(), rhos, ls = '-', c = c)
            for i, (xs, rhos) in enumerate(zip(testPositions[k], testPredictions[k])):
                c = cmap(norm(i))
                axis[s,1].plot(xs.cpu().numpy(), rhos, ls = '-', c = c)
            for i, (xs, pred, gt) in enumerate(zip(testPositions[k], testPredictions[k], testGTs[k])):
                c = cmap(norm(i))
                axis[s,2].plot(xs.cpu().numpy(), pred- gt, ls = '-', c = c)
        axis[0,0].set_title('GT')
        axis[0,1].set_title('Pred')
        axis[0,2].set_title('Loss')
        fig.tight_layout()
        # axis[0,0].plot()
    lossDict = []
    for s, k in  enumerate(testData.keys()):
        lossTerms = []
        for i, (xs, pred, gt) in enumerate(zip(testPositions[k], testPredictions[k], testGTs[k])):
            loss = (pred - gt)**2
            r2 = sklearn.metrics.r2_score(gt, pred)
            l2 = sklearn.metrics.mean_squared_error(gt, pred)

            maxSignal = np.max(np.abs(gt))
    #         mse = np.mean((pred - gt)**2)
            psnr = 20 * math.log10(maxSignal / np.sqrt(l2))
    #         print(20 * math.log10(maxSignal / np.sqrt(mse)))
    #         print(maxSignal, mse)
    #         male = np.mean(np.abs(np.log(np.abs(pred) / np.abs(gt))[gt != 0]))
    #         rmsle = np.sqrt(np.mean(np.log(np.abs(pred) / np.abs(gt))[gt != 0]**2)
            minVal = np.min(loss)
            maxVal = np.max(loss)
            std = np.std(loss)
            q1, q3 = np.percentile(loss, [25, 75])
    #         print('r2', r2, 'l2', l2, 'psnr', l2, 'min', minVal, 'max', maxVal, 'q1', q1, 'q3', q3, 'std', std)
            lossTerms.append({
                    'Vertex': net.vertexMode, 
                    'Edge': net.edgeMode, 
                    'inputEncode': net.inputEncode,
                    'outputDecode': net.outputEncode,
                    'inputEdgeEncode': net.inputEdgeEncode,
                    'vertexFeatures': f"[{' '.join([str(s) for s in net.vertexFeatures])}]",
                    'vertexHiddenLayout': f"[{' '.join([str(s) for s in net.vertexHiddenLayout])}]",
                    'feedThroughVertexFeatures': net.feedThroughVertexFeatures,
                    'edgeMLP': net.edgeMLP,
                    'edgeFeatures': f"[{' '.join([str(s) for s in net.edgeFeatures])}]",
                    'edgeHiddenLayout': f"[{' '.join([str(s) for s in net.edgeHiddenLayout])}]",
                    'feedThroughEdgeFeatures': net.feedThroughEdgeFeatures,
                    'messagelayout': f"[{' '.join([str(s) for s in net.messageFeatures])}]",
                    'parameters': count_parameters(net),
                    'seed': net.seed,
                    'file':k, 'entry':str(i),'r2':r2,'l2':l2,'psnr':psnr, 'min':minVal, 'max':maxVal, 'q1':q1, 'q3':q3, 'std':std})
    #         print(r2, l2, psnr)
    #         print(male, rmsle)
    #         break
        lossDict += lossTerms
    #     break
    return pandas.DataFrame(data = lossDict)
def trainMLP(particleData, testData, settings, dataSet, trainingFiles, offset, model, epochs = 5, iterations = 1000, testInterval = 10, initialLR = 1e-2, groundTruthFn = None, featureFn = None, lossFn = None, device = 'cpu'):   
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
    
#     windowFn = getWindowFunction(window, norm = windowNorm) if window is not None else None
#     model = mlpNetwork(inputFeatures = fluidFeatures, mlpLayout = mlpLayout, layers = layers).to(device) 
#     model = RbfNet(fluidFeatures = fluidFeatures, 
#                    layers = layers, 
#                    denseLayer = True, activation = 'ReLU', coordinateMapping = 'cartesian', 
#                    n = n, windowFn = windowFn, rbf = basis, batchSize = 32, ignoreCenter = True, normalized = False).to(device)   
    lr = initialLR
#     with torch.no_grad():
#         if weightOverride is not None:
#             model.convs[0].weight[:,0,0] = torch.tensor(weightOverride).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)
    losses = []
    lossValues = []
    testLosses = {}
    testPredictions = {}
    for e in (pb := tqdm(range(epochs), leave = False)):
        for b in (pbl := tqdm(range(iterations), leave=False)):
            
            try:
                bdata = next(dataIter)
                if len(bdata) < batchSize :
                    raise Exception('batch too short')
            except:
                dataIter = iter(dataLoader)
                bdata = next(dataIter)

            positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
            i, j, distance, direction = batchedNeighborsearch(positions, setup)
            x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

            x = x[:,None].to(device)    
            groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
            distance = (distance * direction)[:,None].to(device)
            features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)

            optimizer.zero_grad()
            prediction = model(i,j, features.to(device), distance.to(device), x if model.feedThroughVertexFeatures != 0 else None, (u[j] - u[i])[:,None] if model.feedThroughEdgeFeatures!=0 else None)[:,0]
            lossTerm = lossFn(prediction, groundTruth)
            loss = torch.mean(lossTerm)

            loss.backward()
            optimizer.step()
            losses.append(lossTerm.detach().cpu())
            lossValues.append(loss.detach().cpu().item())

            lossString = np.array2string(torch.mean(lossTerm.reshape(batchSize, positions[0].shape[0]),dim=1).detach().cpu().numpy(), formatter={'float_kind':lambda x: "%.4e" % x})
            batchString = str(np.array2string(np.array(bdata), formatter={'float_kind':lambda x: "%.2f" % x, 'int':lambda x:'%6d' % x}))

            pbl.set_description('%s:  %s -> %.4e' %(batchString, lossString, loss.detach().cpu().numpy()))
            pb.set_description('epoch %2dd, lr %6.4g: loss %.4e [rolling %.4e]' % (e, lr, np.mean(lossValues), np.mean(lossValues[-100:] if len(lossValues) > 100 else lossValues)))
            
            it = e * iterations + b
            if it % testInterval == 0:
                with torch.no_grad():
                    testLossDict = {}
                    testPreds = {}
                    for i, k in  enumerate(testData.keys()):
                        gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
                        prediction = model(testData[k][6], testData[k][7], testData[k][5], testData[k][8], testData[k][-2] if model.feedThroughVertexFeatures != 0 else None, (testData[k][-1][testData[k][7]] - testData[k][-1][testData[k][6]])[:,None] if model.feedThroughEdgeFeatures != 0 else None).reshape(len(testData[k][0]), testData[k][0][0].shape[0])
            
#                         prediction = model(testData[k][6], testData[k][7], testData[k][5], testData[k][8]).reshape(len(testData[k][0]), testData[k][0][0].shape[0])
                        arr = []
                        for i, (xs, pred, gt) in enumerate(zip(testData[k][0], prediction, gt)):
                             arr.append(lossFn(pred, gt).detach().cpu().numpy())
                        testLossDict[k] = arr
                        testPreds[k] = prediction.detach().cpu()
#                         print(testLossDict[k])
                    testLosses[it] = testLossDict
                    testPredictions[it] = testPreds
#             break
#         break
        
        lr = lr * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
            
    return {'model': model, 'optimizer': optimizer, 'finalLR': lr, 'losses': losses, 'testLosses': testLosses, 'testPredictions': testPredictions, 'seed': model.seed, 'epochs': epochs, 'iterations': iterations}



class RbfNet(torch.nn.Module):
    def __init__(self, fluidFeatures, layers = [32,64,64,2], denseLayer = True, activation = 'relu',
                coordinateMapping = 'cartesian', n = 8, windowFn = None, rbf = 'linear',batchSize = 32, ignoreCenter = True, normalized = False):
        super().__init__()
        self.centerIgnore = ignoreCenter
        self.features = copy.copy(layers)
        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.relu = getattr(nn.functional, 'relu')
        self.layers = layers
        self.normalized = normalized
        if len(layers) == 1:
            self.convs.append(RbfConv(
                in_channels = fluidFeatures, out_channels = self.features[0],
                dim = 1, size = [n],
                rbf = rbf,
                bias = False,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))

            self.centerIgnore = False
            return

        self.convs.append(RbfConv(
            in_channels = fluidFeatures, out_channels = self.features[0],
            dim = 1, size = [n],
            rbf = rbf,
            bias = True,
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))
                
        self.fcs.append(nn.Linear(in_features=fluidFeatures,out_features= layers[0],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)

        self.features[0] = self.features[0]
        for i, l in enumerate(layers[1:-1]):
            self.convs.append(RbfConv(
                in_channels = (2 * self.features[0]) if i == 0 else self.features[i], out_channels = layers[i+1],
                dim = 1, size = [n],
                rbf = rbf,
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))
            self.fcs.append(nn.Linear(in_features=2 * layers[0] if i == 0 else layers[i],out_features=layers[i+1],bias=True))
            torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
            torch.nn.init.zeros_(self.fcs[-1].bias)
            
        self.convs.append(RbfConv(
            in_channels = self.features[-2] if  len(layers) > 2 else self.features[-2] * 2, out_channels = self.features[-1],
                dim = 1, size = [n],
                rbf = rbf,
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))
        self.fcs.append(nn.Linear(in_features=self.features[-2] if  len(layers) > 2 else self.features[-2] * 2,out_features=self.features[-1],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)


    def forward(self, \
                fluidFeatures, \
                fi, fj, distances):
        if self.centerIgnore:
            nequals = fi != fj

        i, ni = torch.unique(fi, return_counts = True)
        self.ni = ni
        self.li = torch.exp(-1 / 16 * ni)

        if self.centerIgnore:
            fluidEdgeIndex = torch.stack([fi[nequals], fj[nequals]], dim = 0)
        else:
            fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
            
        if self.centerIgnore:
            fluidEdgeLengths = distances[nequals]
        else:
            fluidEdgeLengths = distances
        fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
            
        fluidConvolution = (self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths)) / particleSupport
#         fluidConvolution = scatter_sum(baseArea * fluidFeatures[fluidEdgeIndex[1]] * kernelGradient(torch.abs(fluidEdgeLengths), torch.sign(fluidEdgeLengths), particleSupport), fluidEdgeIndex[0], dim = 0, dim_size = fluidFeatures.shape[0])
        
        if len(self.layers) == 1:
            return fluidConvolution 
        linearOutput = (self.fcs[0](fluidFeatures))
        ans = torch.hstack((linearOutput, fluidConvolution))
        if verbose:
            print('first layer output', ans[:4])
        
        layers = len(self.convs)
        for i in range(1,layers):
            
            ansc = self.relu(ans)
            
            ansConv = self.convs[i]((ansc, ansc), fluidEdgeIndex, fluidEdgeLengths)
            ansDense = self.fcs[i - 0](ansc)
            
            
            if self.features[i-1] == self.features[i-0] and ans.shape == ansConv.shape:
                ans = ansConv + ansDense + ans
            else:
                ans = ansConv + ansDense
            if verbose:
                print('\tlayer output after activation', ans[:4])
        return ans
    
    
def trainModel(particleData, testData, settings, dataSet, trainingFiles, offset, seed, fluidFeatures = 1, n = 16, basis = 'linear', layers = [1], window = None, windowNorm = None, epochs = 5, iterations = 1000, testInterval = 10, initialLR = 1e-2, groundTruthFn = None, featureFn = None, lossFn = None, device = 'cpu', weightOverride = None):   
#     random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    windowFn = getWindowFunction(window, norm = windowNorm) if window is not None else None
    model = RbfNet(fluidFeatures = fluidFeatures, 
                   layers = layers, 
                   denseLayer = True, activation = 'ReLU', coordinateMapping = 'cartesian', 
                   n = n, windowFn = windowFn, rbf = basis, batchSize = 32, ignoreCenter = True, normalized = False).to(device)   
    lr = initialLR
    with torch.no_grad():
        if weightOverride is not None:
            model.convs[0].weight[:,0,0] = torch.tensor(weightOverride).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)
    losses = []
    lossValues = []
    testLosses = {}
    testPredictions = {}
    for e in (pb := tqdm(range(epochs), leave = False)):
        for b in (pbl := tqdm(range(iterations), leave=False)):
            
            try:
                bdata = next(dataIter)
                if len(bdata) < batchSize :
                    raise Exception('batch too short')
            except:
                dataIter = iter(dataLoader)
                bdata = next(dataIter)

            positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
            i, j, distance, direction = batchedNeighborsearch(positions, setup)
            x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

            x = x[:,None].to(device)    
            groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
            distance = (distance * direction)[:,None].to(device)
            features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)

            optimizer.zero_grad()
            prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
            lossTerm = lossFn(prediction, groundTruth)
            loss = torch.mean(lossTerm)

            loss.backward()
            optimizer.step()
            losses.append(lossTerm.detach().cpu())
            lossValues.append(loss.detach().cpu().item())

            lossString = np.array2string(torch.mean(lossTerm.reshape(batchSize, positions[0].shape[0]),dim=1).detach().cpu().numpy(), formatter={'float_kind':lambda x: "%.4e" % x})
            batchString = str(np.array2string(np.array(bdata), formatter={'float_kind':lambda x: "%.2f" % x, 'int':lambda x:'%6d' % x}))

            pbl.set_description('%s:  %s -> %.4e' %(batchString, lossString, loss.detach().cpu().numpy()))
            pb.set_description('epoch %2dd, lr %6.4g: loss %.4e [rolling %.4e]' % (e, lr, np.mean(lossValues), np.mean(lossValues[-100:] if len(lossValues) > 100 else lossValues)))
            
            it = e * iterations + b
            if it % testInterval == 0:
                with torch.no_grad():
                    testLossDict = {}
                    testPreds = {}
                    for i, k in  enumerate(testData.keys()):
                        gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
                        prediction = model(testData[k][5], testData[k][6], testData[k][7], testData[k][8]).reshape(len(testData[k][0]), testData[k][0][0].shape[0])
                        arr = []
                        for i, (xs, pred, gt) in enumerate(zip(testData[k][0], prediction, gt)):
                             arr.append(lossFn(pred, gt).detach().cpu().numpy())
                        testLossDict[k] = arr
                        testPreds[k] = prediction.detach().cpu()
#                         print(testLossDict[k])
                    testLosses[it] = testLossDict
                    testPredictions[it] = testPreds
#             break
#         break
        
        lr = lr * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
            
    return {'model': model, 'optimizer': optimizer, 'finalLR': lr, 'losses': losses, 'testLosses': testLosses, 'testPredictions': testPredictions, 'seed': seed,
            'window': window, 'windowNorm': windowNorm, 'n':n, 'basis':basis, 'layers':layers, 'epochs': epochs, 'iterations': iterations, 'fluidFeatures': fluidFeatures          
           }

import sklearn
import pandas
def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def getTestingLossFrame(modelState, testData, plot = False):
    label = '%s x %2d @ %s' % (modelState['basis'], modelState['n'], str(modelState['layers']))
    testPredictions = {}
    testGTs = {}
    testPositions = {}
    with torch.no_grad():
        for i, k in  enumerate(testData.keys()):
            gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
            prediction = modelState['model'](testData[k][5], testData[k][6], testData[k][7], testData[k][8]).reshape(len(testData[k][0]), testData[k][0][0].shape[0]).detach().cpu().numpy()
            testPredictions[k] = prediction
            testGTs[k] = gt.detach().cpu().numpy()
            testPositions[k] = testData[k][0]
    if plot:
        fig, axis = plt.subplots(len(testPredictions.keys()),3, figsize=(16,8), sharex = True, sharey = 'col', squeeze = False)

        for s, k in  enumerate(testData.keys()):
            norm = mpl.colors.Normalize(vmin=0, vmax=len(testPredictions[k]) - 1)
            for i, (xs, rhos) in enumerate(zip(testPositions[k], testGTs[k])):
                c = cmap(norm(i))
                axis[s,0].plot(xs.cpu().numpy(), rhos, ls = '-', c = c)
            for i, (xs, rhos) in enumerate(zip(testPositions[k], testPredictions[k])):
                c = cmap(norm(i))
                axis[s,1].plot(xs.cpu().numpy(), rhos, ls = '-', c = c)
            for i, (xs, pred, gt) in enumerate(zip(testPositions[k], testPredictions[k], testGTs[k])):
                c = cmap(norm(i))
                axis[s,2].plot(xs.cpu().numpy(), pred- gt, ls = '-', c = c)
        axis[0,0].set_title('GT')
        axis[0,1].set_title('Pred')
        axis[0,2].set_title('Loss')
        fig.tight_layout()
        # axis[0,0].plot()
    lossDict = []
    for s, k in  enumerate(testData.keys()):
        lossTerms = []
        for i, (xs, pred, gt) in enumerate(zip(testPositions[k], testPredictions[k], testGTs[k])):
            loss = (pred - gt)**2
            r2 = sklearn.metrics.r2_score(gt, pred)
            l2 = sklearn.metrics.mean_squared_error(gt, pred)

            maxSignal = np.max(np.abs(gt))
    #         mse = np.mean((pred - gt)**2)
            psnr = 20 * math.log10(maxSignal / np.sqrt(l2))
    #         print(20 * math.log10(maxSignal / np.sqrt(mse)))
    #         print(maxSignal, mse)
    #         male = np.mean(np.abs(np.log(np.abs(pred) / np.abs(gt))[gt != 0]))
    #         rmsle = np.sqrt(np.mean(np.log(np.abs(pred) / np.abs(gt))[gt != 0]**2)
            minVal = np.min(loss)
            maxVal = np.max(loss)
            std = np.std(loss)
            q1, q3 = np.percentile(loss, [25, 75])
    #         print('r2', r2, 'l2', l2, 'psnr', l2, 'min', minVal, 'max', maxVal, 'q1', q1, 'q3', q3, 'std', std)
            lossTerms.append({'label': label, 'seed':modelState['seed'], 'window':modelState['window'], 'file':k, 'basis':modelState['basis'], 'n':modelState['n'],'entry':str(i), 'params': count_parameters(modelState['model']), 'depth':len(modelState['layers']), 'width':np.max(modelState['layers']),'r2':r2,'l2':l2,'psnr':psnr, 'min':minVal, 'max':maxVal, 'q1':q1, 'q3':q3, 'std':std})
    #         print(r2, l2, psnr)
    #         print(male, rmsle)
    #         break
        lossDict += lossTerms
    #     break
    return pandas.DataFrame(data = lossDict)

def getGroundTruthKernel(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction):
    return scatter_sum(torch.hstack(areas)[j] * kernel(torch.abs(distance), particleSupport), i, dim = 0, dim_size = torch.hstack(areas).shape[0])
def getGroundTruthKernelGradient(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction):
    return scatter_sum(torch.hstack(areas)[j] * kernelGradient(torch.abs(distance), direction, particleSupport), i, dim = 0, dim_size = torch.hstack(areas).shape[0])
def getGroundTruthPhysics(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction):
    return torch.hstack(outVel)
def getFeaturesKernel(positions, velocities, areas, densities, dudts, inVel, outVel):
    return torch.ones_like(areas )[:,None]
def getFeaturesPhysics(positions, velocities, areas, densities, dudts, inVel, outVel):
    return torch.vstack((inVel, densities,torch.ones_like(areas))).mT

torch.manual_seed('1337')

dataLoader = DataLoader(dataSet, shuffle=True, batch_size = batchSize).batch_sampler
dataIter = iter(dataLoader)

def lossFunction(prediction, groundTruth):
    return (prediction - groundTruth)**2 # MSE

# testData = {}
# for i in range(len(testingFiles)):
#     testData[testingFiles[i].split('/')[-1].split('.')[0]] = loadTestcase(testingData, settings, testingFiles[i], [0], device, getGroundTruth, getFeatures)

# Hyperparameters for the NN
lr = 1e-2 # Learning rate
iterations = 1000 # update iterations per epoch
epochs = 5 # total number of epochs, LR is halved every epoch
# n = 17 # number of weights per continuous convolution
# basis = 'fourier' # basis for the convolution, set to linear for CConv

# window = 'cubicSpline'
# windowNorm = 'integral'
# window = None

layers = [1]
initialLR = 1e-2
particleData = trainingData
# groundTruthFn = getGroundTruth
# featureFn = getFeatures
lossFn = lossFunction

inputFeatures = 1


testCase = 'kernel'

if testCase == 'kernel':
    groundTruthFn = getGroundTruthKernel
    featureFn = getFeaturesKernel
elif testCase == 'kernelGradient':
    groundTruthFn = getGroundTruthKernelGradient   
    featureFn = getFeaturesKernel 
elif testCase == 'physicsUpdate':
    groundTruthFn = getGroundTruthPhysics
    featureFn = getFeaturesPhysics
    
testData = {}
for i in range(len(testingFiles)):
#     testData[testingFiles[i].split('/')[-1].split('.')[0]] = loadTestcase(testingData, settings, testingFiles[i], [0, 128, 256, 1024], device, getGroundTruth, getFeatures, offset)
    testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset)


basis = 'MP-PDE'
widths = [1,4,16,64]
depths = [0,1,2,3,4,5,6,7,8]
messages = [8]
def runAblationStudyMLPOneLayer(basis, testCase, widths, depths, messages):
    global testData
    if testCase == 'kernel':
        groundTruthFn = getGroundTruthKernel
        featureFn = getFeaturesKernel
    elif testCase == 'kernelGradient':
        groundTruthFn = getGroundTruthKernelGradient   
        featureFn = getFeaturesKernel 
    elif testCase == 'physicsUpdate':
        groundTruthFn = getGroundTruthPhysics
        featureFn = getFeaturesPhysics

    testData = {}
    for i in range(len(testingFiles)):
        testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset)
        
    layouts = []
    for d in depths:
        for w in widths:
            l = [w] * d
            if l not in layouts:
                layouts.append(l)
#     print(len(messages) * len(seeds) * len(layouts))

    dataset = pandas.DataFrame()
    for l in tqdm(layouts, leave = False):
        for m in tqdm(messages, leave = False):
            for s in tqdm(seeds, leave = False):
                if basis == 'PointNet':
                    net = buildPointNet(inputFeatures, feedThroughVertexFeatures = 1, vertexFeatures = None, vertexMLPLayout = l, seed = s)
                if basis == 'DPCConv':
                    net = buildMessagePassingNetwork(inputFeatures, feedThroughEdgeFeatures = 0, vertexFeatures = None, vertexMLPLayout = l, edgeMLP = False, edgeFeatures = [1], seed = s)
                if basis == 'GNS':
                    net = buildGNS(inputFeatures, feedThroughVertexFeatures = 0, feedThroughEdgeFeatures = 0, vertexFeatures = None, vertexMLPLayout = l, edgeMLP = False, edgeFeatures = [1] + [], messageFeatures = [m], seed = s)
                if basis == 'MP-PDE':
                    net = buildGNS(inputFeatures, feedThroughVertexFeatures = 1, feedThroughEdgeFeatures = 1, vertexFeatures = None, vertexMLPLayout = l, edgeMLP = False, edgeFeatures = [1] + [], messageFeatures = [m], seed = s)

                modelstate = trainMLP(particleData, testData, settings, dataSet, trainingFiles, offset, net,
                    epochs = 5, iterations = 1000, initialLR = 1e-3, device = device, testInterval = 100,
                    groundTruthFn = groundTruthFn, featureFn = featureFn, lossFn = lossFunction)  
                df = getTestingLossFrameMLP(net, testData)

                dataset = pandas.concat([dataset, df])


                dataset.to_csv('ablationStudy_%s_%s ws %s ds %s seeds %s messsages %s.csv' % (testCase, basis, '[' + ' '.join([str(d) for d in widths]) + ']', '[' + ' '.join([str(d) for d in depths]) + ']', '[' + ' '.join([str(d) for d in seeds]) + ']','[' + ' '.join([str(d) for d in messages]) + ']'))

def runAblationStudyMLP(basis, testCase, hiddenLayout, widths, depths, messages):
    global testData
    # basis = 'MP-PDE'
    # testCase = 'physicsUpdate'
    if testCase == 'kernel':
        groundTruthFn = getGroundTruthKernel
        featureFn = getFeaturesKernel
    elif testCase == 'kernelGradient':
        groundTruthFn = getGroundTruthKernelGradient   
        featureFn = getFeaturesKernel 
    elif testCase == 'physicsUpdate':
        groundTruthFn = getGroundTruthPhysics
        featureFn = getFeaturesPhysics

    testData = {}
    for i in range(len(testingFiles)):
        testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset)

    layouts = []
    for d in depths:
        for w in widths:
            l = [w] * d
            if l not in layouts:
                layouts.append(l)
    #     print(len(messages) * len(seeds) * len(layouts))

    # print(layouts)
    l = layouts[1]
    l = layouts[-1]
    # hiddenLayout = [32] * 3
    m = [8] * (1 + len(l))
    # print(m)
    # s = 12345


    dataset = pandas.DataFrame()
    for l in tqdm(layouts, leave = False):
        for m in tqdm(messages, leave = False):
            for s in tqdm(seeds, leave = False):
                if basis == 'PointNet':
                    net = buildPointNet(3 if testCase == 'physicsUpdat' else 1, feedThroughVertexFeatures = 1, vertexFeatures = l, vertexMLPLayout = hiddenLayout, seed = s)
                if basis == 'DPCConv':
                    net = buildMessagePassingNetwork(3 if testCase == 'physicsUpdat' else 1, feedThroughEdgeFeatures = 0, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = False, edgeFeatures = [1], seed = s)
                if basis == 'GNS':
                    net = buildGNS(3 if testCase == 'physicsUpdat' else 1, feedThroughVertexFeatures = 0, feedThroughEdgeFeatures = 0, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = True, edgeFeatures = [1] + l, messageFeatures = [m], seed = s)
                if basis == 'MP-PDE':
                    net = buildGNS(3 if testCase == 'physicsUpdat' else 1, feedThroughVertexFeatures = 1, feedThroughEdgeFeatures = 1, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = True, edgeFeatures = [1] + l, messageFeatures = [m], seed = s)

                modelstate = trainMLP(particleData, testData, settings, dataSet, trainingFiles, offset, net,
                    epochs = 5, iterations = 1000, initialLR = 1e-3, device = device, testInterval = 100,
                    groundTruthFn = groundTruthFn, featureFn = featureFn, lossFn = lossFunction)  
                df = getTestingLossFrameMLP(net, testData)

                dataset = pandas.concat([dataset, df])
                
                dataset.to_csv('ablationStudy_%s_%s ws %s ds %s seeds %s messsages %s.csv' % (testCase, basis, '[' + ' '.join([str(d) for d in widths]) + ']', '[' + ' '.join([str(d) for d in depths]) + ']', '[' + ' '.join([str(d) for d in seeds]) + ']','[' + ' '.join([str(d) for d in messages]) + ']'))



basis = 'rbf square'
window =  None

def trainRBFNetwork(basis, testCase, ns, widths, depths, window):
    global testData
    if testCase == 'kernel':
        groundTruthFn = getGroundTruthKernel
        featureFn = getFeaturesKernel
    elif testCase == 'kernelGradient':
        groundTruthFn = getGroundTruthKernelGradient   
        featureFn = getFeaturesKernel 
    elif testCase == 'physicsUpdate':
        groundTruthFn = getGroundTruthPhysics
        featureFn = getFeaturesPhysics

    testData = {}
    for i in range(len(testingFiles)):
        testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset)
        
    layouts = []
    for d in depths:
        for w in widths:
            l = [w] * d + [1]
            if l not in layouts:
                layouts.append(l)
    dataset = pandas.DataFrame()

#     for basis in tqdm(bases, leave = False):
    dataset = pandas.DataFrame()
    for n in (tns:= tqdm(ns, leave = False)):
        tns.set_description('ns [%s], current: %d' %(' '.join([str(s) for s in ns]), n))
        for l in (tnl:=tqdm(layouts, leave = False)):
            tnl.set_description('layout %s' % (' '.join([str(s) for s in l])))
            for s in (ts:=tqdm(seeds, leave = False)):
                # ts.set_description('Seed %d')
                ts.set_description('seeds [%s], current: %d' %(' '.join([str(s) for s in seeds]), s))
                trainedModel = trainModel(particleData, testData, settings, dataSet, trainingFiles, fluidFeatures = 3 if testCase == 'physicsUpdate' else 1, offset = offset,
                                  n = n, basis = basis, layers = l, seed = s,
                                 window = window, windowNorm = 'integral',
                                 epochs = 5, iterations = 1000, initialLR = 1e-3, device = device, testInterval = 1000,
                                 groundTruthFn = groundTruthFn, featureFn = featureFn, lossFn = lossFunction)
        #         models.append(trainedModel)
                df = getTestingLossFrame(trainedModel, testData, plot = False)
                dataset = pandas.concat([dataset, df])
                dataset.to_csv('ablationStudy_%s_%s window %s ns %s ws %s ds %s seeds %s.csv' % (testCase, basis, 'None' if window is None else window, '[' + ' '.join([str(d) for d in ns]) + ']', '[' + ' '.join([str(d) for d in widths]) + ']', '[' + ' '.join([str(d) for d in depths]) + ']', '[' + ' '.join([str(d) for d in seeds]) + ']'))

from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('-s','--numSeeds', type=int, default=4)
parser.add_argument('-t','--testCase', type=str, default= 'kernel')
parser.add_argument('-b','--basis', type=str, default= 'rbf square')
parser.add_argument('-m','--mode', type=str, default= 'networkLayout')
parser.add_argument('-l','--hiddenLayout', type=str, default='32 32 32 32')
parser.add_argument('-w','--widths', type = str, default = '1 2 4 8 16')
parser.add_argument('-d','--depths', type = str, default = '0 1 2 3 4 5')
parser.add_argument('-n','--ns', type = str, default = '1 2 3 4 5 6 7 8 16 32')
parser.add_argument('-msg','--mesages', type = str, default = '8')




args = parser.parse_args()

widths = [int(x) for x in args.widths.split(' ') if x]
depths = [int(x) for x in args.depths.split(' ') if x]
ns = [int(x) for x in args.ns.split(' ') if x]
messages = [int(x) for x in args.mesages.split(' ') if x]

seeds = np.random.randint(0, 2**30, args.numSeeds)

if 'mlp' in args.basis:
    basis = [x for x in args.basis.split(' ') if x][-1]
    print(basis)

    if args.mode == 'networkLayout':
        layout = [int(x) for x in args.hiddenLayout.split(' ') if x]
        print('Training rbf network with:')
        print('Basis function:',basis)
        print('Test case:', args.testCase)
        print('hidden Layout: [%s]'%(' '.join([str(x) for x in layout])))
        print('widths: [%s]'%(' '.join([str(x) for x in widths])))
        print('depths: [%s]'%(' '.join([str(x) for x in depths])))
        print('messages: [%s]'%(' '.join([str(x) for x in messages])))
        
        runAblationStudyMLP(basis, args.testCase, layout, widths, depths, messages)
    else:
        print('Training mlp network with:')
        print('Basis function:',basis)
        print('Test case:', args.testCase)
        print('widths: [%s]'%(' '.join([str(x) for x in widths])))
        print('depths: [%s]'%(' '.join([str(x) for x in depths])))
        print('messages: [%s]'%(' '.join([str(x) for x in messages])))

        runAblationStudyMLPOneLayer(basis, args.testCase, widths, depths, messages)
else:
    print('Training rbf network with:')
    print('Basis function:',args.basis)
    print('Test case:', args.testCase)
    print('ns: [%s]'%(' '.join([str(x) for x in ns])))
    print('widths: [%s]'%(' '.join([str(x) for x in widths])))
    print('depths: [%s]'%(' '.join([str(x) for x in depths])))
    trainRBFNetwork(args.basis, args.testCase, ns, widths, depths, None)
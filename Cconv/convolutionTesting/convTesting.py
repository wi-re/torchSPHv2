import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))

import torch
from torch.profiler import record_function
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import numpy as np


# Timing function for performance evaluation
import time
class catchtime:    
    def __init__(self, arg = 'Unnamed Context'):
#         print('__init__ called with', arg)
        self.context = arg
        
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.time
        self.readout = f'{self.context} took: {1000 * self.time:.3f} ms'
        print(self.readout)

# Math/parallelization library includes
import numpy as np
import torch

# Imports for neighborhood searches later on
from torch_geometric.nn import radius
from torch_scatter import scatter


# Plotting includes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.tri as tri
import random

from sklearn.metrics import r2_score
from tqdm.notebook import tqdm


from scipy.interpolate import RegularGridInterpolator  
import numpy as np


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant, rng = np.random.default_rng(seed=42)
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*rng.random((res[0]+1, res[1]+1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant, seed = 1337
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    rng = np.random.default_rng(seed=seed)
    
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant, rng
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise

def generate1DPeriodicNoise(numSamples = 1024, r = 0.75, freq = 4, octaves = 2, persistence = 0.75, lacunarity = 2, plot = False, seed = 1337):
    n = 1024
    # freq = 4
    # octaves = 2
    # persistence = 0.75
    # lacunarity = 2
    noise = generate_fractal_noise_2d([n, n], [freq,freq], octaves, persistence = persistence, lacunarity = lacunarity, seed = seed)

    interp = RegularGridInterpolator((np.linspace(-1,1,n), np.linspace(-1,1,n)), noise)


#     r = 0.75
    # numSamples = 128
    thetas = np.linspace(0, 2 * np.pi, numSamples)

    x = np.cos(thetas) * r
    y = np.sin(thetas) * r

    sampled = interp(np.vstack((x,y)).transpose())
    if plot:
        fig, axis = plt.subplots(1, 2, figsize=(12,6), sharex = False, sharey = False, squeeze = False)

        circle1 = plt.Circle((0, 0), r, color='white', ls = '--', fill = False)
        axis[0,0].imshow(noise, extent =(-1,1,-1,1))
        axis[0,0].add_patch(circle1)
        axis[0,1].plot(thetas, sampled)

        fig.tight_layout()
    return sampled


def generate1DNoise(numSamples = 1024, r = 1 / (2 * np.pi), freq = 4, octaves = 2, persistence = 0.75, lacunarity = 2, plot = False, seed = 1337):
    n = 1024
    # freq = 4
    # octaves = 2
    # persistence = 0.75
    # lacunarity = 2
    noise = generate_fractal_noise_2d([n, n], [freq, freq], octaves, persistence = persistence, lacunarity = lacunarity, seed = seed)

    interp = RegularGridInterpolator((np.linspace(-1,1,n), np.linspace(-1,1,n)), noise)


#     r = 0.75
    # numSamples = 128
    thetas = np.linspace(0, 2 * np.pi, numSamples)

    x = np.linspace(-1,1,numSamples)
    y = np.zeros(numSamples)

    sampled = interp(np.vstack((x,y)).transpose())
    if plot:
        fig, axis = plt.subplots(1, 2, figsize=(12,6), sharex = False, sharey = False, squeeze = False)

        circle1 = plt.Circle((0, 0), r, color='white', ls = '--', fill = False)
        axis[0,0].imshow(noise, extent =(-1,1,-1,1))
        axis[0,0].add_patch(circle1)
        axis[0,1].plot(thetas, sampled)

        fig.tight_layout()
    return sampled


from sampling import *

import torch.optim as optim

def evalBasis(n,x,which, periodic = False, normalized = False):
    fx = evalBasisFunction(n, x , which = which, periodic=periodic)
    if normalized:
        fx = fx / torch.sum(fx, axis=0)
    return fx

def trainNetwork(dataset, target, lr=1e-1, n = 4, batch_size = 4, basis = 'linear', normalizedBasis = False, window = None, iterations = 2**12, groundTruthNoise = False, groundTruthNoiseType = 'normal', groundTruthNoiseVariance = 0.25, testSamples = 255):
    groundTruth = evalGroundTruth(dataset.shape[0], dataset, target, noise = groundTruthNoise, noiseType = groundTruthNoiseType, noiseVar = groundTruthNoiseVariance)
    windowFunctionFn = getWindowFunction(window) if window is not None else None
    windowFunction = lambda x : windowFunctionFn(torch.abs(x)) if window is not None else None
    
    train_dataloader, train_iter = generateLoaders(torch.hstack((dataset,groundTruth)), batch_size = batch_size, shuffleDataset = True, shuffled = False, shuffleSeed = None)
    def sampleDataLoader():
        try:
            bdata = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            bdata = next(train_iter)
        return bdata

    weights = (torch.rand(n) * 2 - 1) * 0.5
    weights.requires_grad = True
    optimizer = optim.SGD([weights], lr=lr, momentum=0.9)

    weightList = []
    gradList = []
    lossList = []
    r2List = []
    weightList.append(torch.clone(weights.detach()))
    
    xTest =  torch.linspace(-1,1,testSamples)
    fxTest = evalBasis(n, xTest, basis, periodic = False, normalized = normalizedBasis) if window is None else evalBasis(n, xTest, basis, periodic = False, normalized = normalizedBasis) * windowFunction(xTest)[None,:]
    yTest = target(xTest)
    
    test_losses = []
    test_evals = []
    test_r2s = []

    for i in (t := tqdm(range(iterations), leave = False)):
        testEval = torch.sum(weights[:,None].detach() * fxTest, axis = 0)
        test_losses.append(torch.mean((testEval - yTest)**2))
        test_r2s.append(r2_score(yTest, testEval.numpy()))
        test_evals.append(testEval)
                             
        optimizer.zero_grad()

        sampled = sampleDataLoader()
        x = sampled[:,:sampled.shape[1]//2].flatten()
        gt = sampled[:,sampled.shape[1]//2:].flatten()
        
        fx = evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis) * (windowFunction(x)[None,:] if windowFunction is not None else 1)
        y_pred = torch.sum(weights[:,None] * fx, axis=0)
        loss = torch.mean((y_pred - gt)**2)
        r2List.append(r2_score(gt.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))

        loss.backward()
        lossList.append(torch.clone(loss.detach()))

        gradList.append(torch.clone(weights.grad.detach()))
        optimizer.step()
        weightList.append(torch.clone(weights))
        
        t.set_description('Iteration %5d, Loss: %6.4e / %6.4e, R2: %6.4e / %6.4e, Update: %6.4e' %(i, 
                            lossList[-1], test_losses[-1].detach().cpu().numpy(), 
                            r2List[-1], test_r2s[-1], torch.linalg.norm(weightList[-1] - weightList[-2])))
    return weightList, lossList, r2List, gradList, yTest, test_evals, test_r2s, test_losses

import copy
import torch.nn as nn
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
def trainMLP(model, dataset, target, lr=1e-1, batch_size = 4, iterations = 2**12, groundTruthNoise = False, groundTruthNoiseType = 'normal', groundTruthNoiseVariance = 0.25, testSamples = 255):
    groundTruth = evalGroundTruth(dataset.shape[0], dataset, target, noise = groundTruthNoise, noiseType = groundTruthNoiseType, noiseVar = groundTruthNoiseVariance)

    train_dataloader, train_iter = generateLoaders(torch.hstack((dataset,groundTruth)), batch_size = 4, shuffleDataset = True, shuffled = False, shuffleSeed = None)
    def sampleDataLoader():
        try:
            bdata = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            bdata = next(train_iter)
        return bdata

    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    weightList = []
    gradList = []
    lossList = []
    r2List = []
    weightList.append(copy.deepcopy(model.state_dict()))

    xTest = torch.linspace(-1,1,testSamples)
    yTest = target(xTest)

    test_losses = []
    test_evals = []
    test_r2s = []

    for i in (t := tqdm(range(iterations), leave = False)):
        testEval = model(xTest[:,None]).flatten().detach()


        test_losses.append(torch.mean((testEval - yTest)**2))
        test_r2s.append(r2_score(yTest, testEval.numpy()))
        test_evals.append(testEval)

        optimizer.zero_grad()

        sampled = sampleDataLoader()
        x = sampled[:,:sampled.shape[1]//2].flatten()[:,None].type(torch.float32)
        gt = sampled[:,sampled.shape[1]//2:].flatten()


        y_pred = model(x).flatten()
        loss = torch.mean((y_pred - gt)**2)

        r2List.append(r2_score(gt.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))

        loss.backward()
        lossList.append(torch.clone(loss.detach()))

    #     gradList.append(torch.clone(weights.grad.detach()))
        optimizer.step()
    #     weightList.append(torch.clone(weights))

        weightList.append(copy.deepcopy(model.state_dict()))

        weightsCurrent = torch.hstack([weightList[-1][w].flatten() for w in weightList[-1]])
        weightsPrior = torch.hstack([weightList[-2][w].flatten() for w in weightList[-2]])

        t.set_description('Iteration %5d, Loss: %6.4e / %6.4e, R2: %6.4e / %6.4e, Update: %6.4e' %(i, 
                            lossList[-1], test_losses[-1].detach().cpu().numpy(), 
                            r2List[-1], test_r2s[-1], torch.linalg.norm(weightsCurrent - weightsPrior)))

    return weightList, lossList, r2List, gradList, yTest, test_evals, test_r2s, test_losses

import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def buildMLP(width = 1, depth = 0):
    model = nn.Sequential()
    model.add_module('inDense', nn.Linear(1,width))
    model.add_module('inNorm', nn.BatchNorm1d(num_features=4))
    model.add_module('inAct', nn.ReLU())
    for i in range(depth - 1):
        model.add_module('dense %d' % i , nn.Linear(1,width))
        model.add_module('norm %d' % i , nn.BatchNorm1d(num_features=4))
        model.add_module('act %d' % i, nn.ReLU())
    model.add_module('out', nn.Linear(width,1))
    return model
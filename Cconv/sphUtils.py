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

from pytorchSPH.config import *
from pytorchSPH.plotting import *
from pytorchSPH.compressible import compressibleSimulation
from pytorchSPH.incompressible import incompressibleSimulation
from pytorchSPH.scenarios import *
from pytorchSPH.solidBC import *

import h5py

import ast
from pytorchSPH.neighborhood import *
from pytorchSPH.sph import *

def computeGamma(config, state):
    if'velocitySources' in config:
        state['fluidGamma'] = torch.zeros(state['fluidArea'].shape, device=config['device'], dtype=config['precision'])
        for source in config['velocitySources']:
            xmask = torch.logical_and(state['fluidPosition'][:,0] >= source['min'][0], state['fluidPosition'][:,0] <= source['max'][0])
            ymask = torch.logical_and(state['fluidPosition'][:,1] >= source['min'][1], state['fluidPosition'][:,1] <= source['max'][1])

            mask = torch.logical_and(xmask, ymask)

            active = torch.any(mask)
            mu = 3.5
            xr = (state['fluidPosition'][:,0] - source['min'][0]) / (source['max'][0] - source['min'][0])

            if source['min'][0] < 0:
                xr = 1 - xr

            gamma = (torch.exp(torch.pow(torch.clamp(xr,min = 0, max = 1), mu)) - 1) / (np.exp(1) - 1)
#             print(gamma)
            state['fluidGamma'] = torch.max(gamma, state['fluidGamma'])


def advanceSimulation(prediction, config, state, shiftSteps = 2):    
    enforcePeriodicBC(config, state)
    
    def shiftNTimes(n, config, state):
        for i in range(n):
            state['fluidNeighbors'], state['fluidDistances'], state['fluidRadialDistances'] = \
                neighborSearch(state['fluidPosition'], state['fluidPosition'], config, state)

            state['boundaryNeighbors'], state['boundaryDistances'], state['boundaryGradients'], \
                state['boundaryIntegrals'], state['boundaryIntegralGradients'], \
                state['boundaryFluidNeighbors'], state['boundaryFluidPositions'] = boundaryNeighborSearch(config, state)

            state['fluidDensity'] = sphDensity(config, state)  
            solveShifting(config, state)
            state['fluidPosition'] += state['fluidUpdate']

            enforcePeriodicBC(config, state)

    shiftNTimes(shiftSteps, config, state)
    computeGamma(config, state)
    state['fluidNeighbors'], state['fluidDistances'], state['fluidRadialDistances'] = \
        neighborSearch(state['fluidPosition'], state['fluidPosition'], config, state)

    state['boundaryNeighbors'], state['boundaryDistances'], state['boundaryGradients'], \
        state['boundaryIntegrals'], state['boundaryIntegralGradients'], \
        state['boundaryFluidNeighbors'], state['boundaryFluidPositions'] = boundaryNeighborSearch(config, state)

    state['fluidDensity'] = sphDensity(config, state)  
    
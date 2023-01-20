import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter
from torch.profiler import profile, record_function, ProfilerActivity

from ..kernels import kernel, kernelGradient
from ..module import Module
from ..parameter import Parameter


# @torch.jit.script
@torch.jit.script
def computeDensity(radialDistances, areas, neighbors, support):
    with record_function("sph - density 2"): 
        rho =  scatter(kernel(radialDistances, support) * areas[neighbors[0]], neighbors[1], dim=0, dim_size=areas.shape[0], reduce="add")
        return rho


class densityModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']

    def evaluate(self, simulationState, simulation):
        fluidRadialDistances = simulationState['fluidRadialDistances']
        fluidArea = simulationState['fluidArea']
        fluidNeighbors = simulationState['fluidNeighbors']
        particleSupport = self.support

        return computeDensity(fluidRadialDistances, fluidArea, fluidNeighbors, particleSupport)

def testFunctionality(sphSimulation):
    density = densityModule()
    density.initialize(sphSimulation.config, sphSimulation)
            
    sphSimulation.sphDensity.evaluate(sphSimulation.simulationState, sphSimulation)
    density.evaluate(sphSimulation.simulationState, sphSimulation)
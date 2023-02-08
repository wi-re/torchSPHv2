# Helpful statement for debugging, prints the thing entered as x and the output, i.e.,
# debugPrint(1+1) will output '1+1 [int] = 2'
import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import os
import os, sys
# sys.path.append(os.path.join('~/dev/pytorchSPH/', "lib"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm.notebook import trange, tqdm
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

import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

import torch
# import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

from src.simulationBase import SPHSimulation
from src.kernels import kernel, kernelGradient, spikyGrad, wendland, wendlandGrad, cohesionKernel, getKernelFunctions
from src.util import *
from src.module import Module
from src.parameter import Parameter

# import modules to build simulation with
from src.modules.density import densityModule
from src.modules.neighborSearch import neighborSearchModule
from src.modules.akinciTension import akinciTensionModule
from src.modules.sdfBoundary import sdfBoundaryModule, sdPolyDerAndIntegral
from src.modules.akinciBoundary import akinciBoundaryModule
from src.modules.solidBoundary import solidBoundaryModule
from src.modules.periodicBC import periodicBCModule
from src.modules.velocityBC import velocityBCModule
from src.modules.implicitShifting import implicitIterativeShiftModule
from src.modules.gravity import gravityModule
from src.modules.xsph import xsphModule
from src.modules.dfsph import dfsphModule
from src.modules.adaptiveDT import adaptiveTimeSteppingModule
from src.modules.laminar import laminarViscosityModule
from src.modules.diffusion import diffusionModule

class dfsphSimulation(SPHSimulation):    
    def __init__(self, config = tomli.loads('')):
        super().__init__(config)
        
        self.modules = []
        self.moduleParameters = []
        
        if self.verbose: print('Processing modules')
        self.neighborSearch = neighborSearchModule()
        self.sphDensity = densityModule()
        self.periodicBC = periodicBCModule()
        self.DFSPH = dfsphModule()
        self.XSPH = xsphModule()
        self.velocityBC = velocityBCModule()
#         self.shiftModule = implicitIterativeShiftModule()
        self.gravityModule = gravityModule()
        self.adaptiveDT = adaptiveTimeSteppingModule()
        self.surfaceTension = akinciTensionModule()
        
        self.modules.append(self.neighborSearch)
        self.modules.append(self.sphDensity)
        self.modules.append(self.periodicBC)
        self.modules.append(self.velocityBC)
        self.modules.append(self.DFSPH)
        if self.config['diffusion']['velocityScheme'] == 'xsph':
            self.velocityDiffusionModule = xsphModule()
            self.modules.append(self.velocityDiffusionModule)
            
        if self.config['diffusion']['velocityScheme'] == 'deltaSPH':
            self.velocityDiffusionModule = diffusionModule()
            self.modules.append(self.velocityDiffusionModule)
        self.laminarViscosityModule = laminarViscosityModule()
        self.modules.append(self.laminarViscosityModule)


#         self.modules.append(self.shiftModule)
        self.modules.append(self.gravityModule)
        self.modules.append(self.adaptiveDT)
        self.modules.append(self.surfaceTension)    
        if self.config['simulation']['boundaryScheme'] == 'solid': 
            self.boundaryModule = solidBoundaryModule() 
            self.modules.append(self.boundaryModule)  
        if self.config['simulation']['boundaryScheme'] == 'SDF': 
            self.boundaryModule = sdfBoundaryModule() 
            self.modules.append(self.boundaryModule)  
        if self.config['simulation']['boundaryScheme'] == 'Akinci': 
            self.boundaryModule = akinciBoundaryModule() 
            self.modules.append(self.boundaryModule)  
        
        if self.verbose: print('Processing module parameters')
        for module in self.modules:    
            moduleParams =  module.getParameters()
            if moduleParams is not None:
                for param in moduleParams:
                    param.parseConfig(self.config)
                self.moduleParameters = self.moduleParameters + moduleParams
                
    def initializeSimulation(self):
        super().initializeSimulation()
        
        
    def timestep(self):
        step = ' 1 - Enforcing periodic boundary conditions'
        if self.verbose: print(step)
        with record_function(step):
            self.periodicBC.enforcePeriodicBC(self.simulationState, self)
            
        step = ' 2 - Fluid neighborhood search'
        if self.verbose: print(step)
        with record_function(step):
            self.simulationState['fluidNeighbors'], self.simulationState['fluidDistances'], self.simulationState['fluidRadialDistances'] = self.neighborSearch.search(self.simulationState, self)
            
        step = ' 3 - Boundary neighborhood search'
        if self.verbose: print(step)
        with record_function(step):
            self.boundaryModule.boundaryFilterNeighborhoods(self.simulationState, self)
            self.boundaryModule.boundaryNeighborhoodSearch(self.simulationState, self)

        step = ' 4 - Fluid - Fluid density evaluation'
        if self.verbose: print(step)
        with record_function(step):
            self.sphDensity.evaluate(self.simulationState, self)    
            self.sync(self.simulationState['fluidDensity'])
            # self.periodicBC.syncQuantity(self.simulationState['fluidDensity'], self.simulationState, self)
        
        step = ' 5 - Fluid - Boundary density evaluation'
        if self.verbose: print(step)
        with record_function(step):
            self.boundaryModule.evalBoundaryDensity(self.simulationState, self) 
            self.sync(self.simulationState['fluidDensity'])       
            # self.periodicBC.syncQuantity(self.simulationState['fluidDensity'], self.simulationState, self)
            
            
        step = ' 6 - Initializing acceleration'
        if self.verbose: print(step)
        with record_function(step):
            self.simulationState['fluidAcceleration'] = torch.zeros_like(self.simulationState['fluidVelocity'])   
            
        step = ' 7 - External force evaluation'
        if self.verbose: print(step)
        with record_function(step):
            self.gravityModule.evaluate(self.simulationState, self)
            self.sync(self.simulationState['fluidAcceleration'])
            # self.periodicBC.syncQuantity(self.simulationState['fluidAcceleration'], self.simulationState, self)
        
        step = ' 8 - Divergence free solver step'
        if self.verbose: print(step)
        with record_function(step):
            if self.config['dfsph']['divergenceSolver']:
                self.simulationState['divergenceIterations'] = self.DFSPH.divergenceSolver(self.simulationState, self)
                # self.sync(self.simulationState['fluidPredAccel'])
                # self.periodicBC.syncQuantity(self.simulationState['fluidPredAccel'], self.simulationState, self)
                self.simulationState['fluidAcceleration'] += self.simulationState['fluidPredAccel']

        # step = ' 9 - Surface tension force evaluation'
        # if self.verbose: print(step)
        # with record_function(step):
        #     self.surfaceTension.computeNormals(self.simulationState, self)
        #     self.sync(self.surfaceTension.normals)
        #     # self.periodicBC.syncQuantity(self.simulationState['fluidNormals'], self.simulationState, self)
        #     self.surfaceTension.cohesionForce(self.simulationState, self)
        #     self.surfaceTension.curvatureForce(self.simulationState, self)
        #     self.sync(self.simulationState['fluidAcceleration'])
        #     # self.periodicBC.syncQuantity(self.simulationState['fluidAcceleration'], self.simulationState, self)
            
        step = '10 - Incompressible solver step'
        if self.verbose: print(step)
        with record_function(step):
            self.simulationState['densityIterations'] = self.DFSPH.incompressibleSolver(self.simulationState, self)
            # self.sync(self.simulationState['fluidPredAccel'])
            # self.periodicBC.syncQuantity(self.simulationState['fluidPredAccel'], self.simulationState, self)
            self.simulationState['fluidAcceleration'] += self.simulationState['fluidPredAccel']
            # self.sync(self.simulationState['fluidAcceleration'])
            # self.periodicBC.syncQuantity(self.simulationState['fluidAcceleration'], self.simulationState, self)
        
        # step = '11 - Velocity update step'
        # if self.verbose: print(step)
        # with record_function(step):
            # self.simulationState['fluidVelocity'] += self.simulationState['dt'] * self.simulationState['fluidAcceleration']
            # self.periodicBC.syncQuantity(self.simulationState['fluidVelocity'], self.simulationState, self)
           
        step = '11 - velocity diffusion'
        if self.verbose: print(step)
        with record_function(step):     
            self.velocityDiffusionModule.evaluate(self.simulationState, self)    
        step = '12 - laminar viscosity'
        if self.verbose: print(step)
        with record_function(step):       
            self.laminarViscosityModule.computeLaminarViscosity(self.simulationState, self)   

        # step = '12 - XSPH diffusion evaluation'
        # if self.verbose: print(step)
        # with record_function(step):
        #     xsphFluidCorrection = self.XSPH.fluidTerm(self.simulationState, self)
        #     self.periodicBC.syncQuantity(xsphFluidCorrection, self.simulationState, self)
        #     self.simulationState['fluidVelocity'] += xsphFluidCorrection
        
#         step = ' 1 - Boundary friction evaluation'
#         if self.verbose: print(step)
#         with record_function(step):
#         self.boundaryModule.evalBoundaryFriction(self.simulationState, self)
#         xsphBoundaryCorrection = self.XSPH.boundaryTerm(self.simulationState, self)
#         self.periodicBC.syncQuantity(xsphBoundaryCorrection, self.simulationState, self)
#         self.simulationState['fluidVelocity'] += xsphBoundaryCorrection
        
        step = '13 - Velocity source contribution'
        if self.verbose: print(step)
        with record_function(step):
            self.velocityBC.enforce(self.simulationState, self)
            self.sync(self.simulationState['fluidVelocity'])
        
        # step = '14 - Position update step'
        # if self.verbose: print(step)
        # with record_function(step):
            # self.simulationState['fluidPosition'] += self.simulationState['fluidVelocity'] * self.simulationState['dt']
            
#         step = ' 1 - Shifting positions'
#         if self.verbose: print(step)
#         with record_function(step):
#         self.shiftModule.applyShifting(sphSimulation.simulationState, sphSimulation)
#         self.periodicBC.syncQuantity(self.simulationState['fluidUpdate'], self.simulationState, self)
#         self.simulationState['fluidPosition'] += self.simulationState['fluidUpdate']
        return self.simulationState['fluidAcceleration'], self.simulationState['fluidVelocity'], self.simulationState['dpdt'] if self.config['simulation']['densityScheme'] == 'continuum' else None

        step = '15 - Bookkeeping'
        if self.verbose: print(step)
        with record_function(step):
            self.simulationState['time'] += self.simulationState['dt']
            self.simulationState['timestep'] += 1

            self.simulationState['dt'] = self.adaptiveDT.updateTimestep(self.simulationState, self)
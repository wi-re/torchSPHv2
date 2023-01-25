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
from ..util import *

@torch.jit.script
def precomputeMLS(queryPositions, fluidPosition, fluidArea, fluidDensity, support):
    with record_function("MLS - precomputeMLS"): 
        # queryPositions = simulationState['fluidPosition']
        # queryPosition = pb
        # support = simulation.config['particle']['support'] * 2

        neighbors = radius(fluidPosition, queryPositions, support, max_num_neighbors = 256)
        i = neighbors[0]
        j = neighbors[1]
#         neighbors = torch.stack([i, j], dim = 0)

    #     debugPrint(neighbors)
        # debugPrint(torch.min(neighbors[0]))
        # debugPrint(torch.max(neighbors[0]))
        # debugPrint(torch.min(neighbors[1]))
        # debugPrint(torch.max(neighbors[1]))

        distances = (fluidPosition[j] - queryPositions[i])
        radialDistances = torch.linalg.norm(distances,dim=1)

        distances[radialDistances < 1e-5,:] = 0
        distances[radialDistances >= 1e-5,:] /= radialDistances[radialDistances >= 1e-5,None]
        radialDistances /= support

        kernel = kernel(radialDistances, support)

        bij = fluidPosition[j] - queryPositions[i]
        bij = torch.hstack((bij.new_ones((bij.shape[0]))[:,None], bij))
    #     debugPrint(bij)

        Mpartial = 2 * torch.einsum('nu, nv -> nuv', bij, bij) * \
                ((fluidArea / fluidDensity)[j] * kernel)[:,None,None]

        M = scatter(Mpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
        Minv = torch.linalg.pinv(M)
    #     debugPrint(Minv)

        e1 = torch.tensor([1,0,0], dtype=Minv.dtype, device=Minv.device)
        Me1 = torch.matmul(Minv,e1)
    #     debugPrint(Me1)


        pGpartial = torch.einsum('nd, nd -> n', Me1[i], bij) * \
            kernel * ((fluidArea / fluidDensity)[j])

#         pG = scatter(pGpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
    #     debugPrint(pG)

        return pGpartial, neighbors

@torch.jit.script
def computeFluidAcceleration(fluidArea, fluidDensity, fluidRestDensity, fluidPressure2, fluidNeighbors, fluidDistances, fluidRadialDistances, support):
    with record_function("DFSPH - accel (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - accel (fluid) [gradient]"): 
            grad = kernelGradient(fluidRadialDistances, fluidDistances, support)

        with record_function("DFSPH - accel (fluid) [factor]"): 
            fac = -(fluidArea * fluidRestDensity)[j]
            p = fluidPressure2 / (fluidDensity * fluidRestDensity)**2
            pi = p[i]
            pj = p[j]
        with record_function("DFSPH - accel (fluid) [mul]"): 
            term = (fac * (pi + pj))[:,None] * grad
        with record_function("DFSPH - accel (fluid) [scatter]"): 
            fluidAccelTerm = scatter_sum(term, i, dim=0, dim_size=fluidArea.shape[0])
        return fluidAccelTerm
@torch.jit.script
def computeBoundaryAccelTerm(fluidArea, fluidDensity, fluidRestDensity, fluidPressure2, pgPartial, ghostToFluidNeighbors2, fluidToGhostNeighbors, ghostParticleBodyAssociation, ghostParticleGradientIntegral, boundaryCounter : int, mlsPressure : bool = True, computeBodyForces : bool = True):
    with record_function("DFSPH - accel (boundary)"): 
        i = fluidToGhostNeighbors[0]
        b = ghostParticleBodyAssociation

    #         ghostParticlePosition = simulationState['ghostParticlePosition']

    #                     simulationState['boundaryPressure'] = mlsInterpolation(simulationState, simulation, simulationState['ghostParticlePosition'], self.support * 2) 
        with record_function("DFSPH - accel (boundary)[pressure]"): 
            if mlsPressure:
                boundaryPressure = scatter(pgPartial * fluidPressure2[ghostToFluidNeighbors2[1]], ghostToFluidNeighbors2[0], dim=0, dim_size = ghostParticleBodyAssociation.shape[0], reduce='add')
            else:
                boundaryPressure = fluidPressure2[i]

        with record_function("DFSPH - accel (boundary)[factor]"): 
            fac = - fluidRestDensity[i]
            pi = fluidPressure2[i] / (fluidDensity[i] * fluidRestDensity[i])**2
            pb = boundaryPressure / (1. * fluidRestDensity[i])**2
            grad = ghostParticleGradientIntegral

        with record_function("DFSPH - accel (boundary)[scatter]"): 
            boundaryAccelTerm = scatter_sum((fac * (pi + pb))[:,None] * grad, i, dim = 0, dim_size = fluidArea.shape[0])

    #         simulationState['boundaryAccelTerm'] = boundaryAccelTerm

    #         boundaryAccelTerm2 = scatter_sum((fac * (pb + pb))[:,None] * grad, i, dim = 0, dim_size = simulationState['numParticles'], reduce="add")
        with record_function("DFSPH - accel (boundary)[body]"): 
            if computeBodyForces:
                force = -boundaryAccelTerm * (fluidArea * fluidRestDensity)[:,None]
                boundaryPressureForce = scatter_sum(force[i], b, dim = 0, dim_size = boundaryCounter)
            else:
                boundaryPressureForce = torch.zeros((boundaryCounter,2), dtype = fluidPressure2.dtype, device = fluidArea.device)

        return boundaryPressure, boundaryAccelTerm, boundaryPressureForce
@torch.jit.script
def computeUpdatedPressureFluidSum(fluidActualArea, fluidPredAccel, fluidNeighbors, fluidRadialDistances, fluidDistances, support, dt):
    with record_function("DFSPH - pressure (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - pressure (fluid) [gradient]"): 
            grad = kernelGradient(fluidRadialDistances, fluidDistances, support)

        with record_function("DFSPH - pressure (fluid) [factor]"): 
            fac = dt**2 * fluidActualArea[j]
            aij = fluidPredAccel[i] - fluidPredAccel[j]
        with record_function("DFSPH - pressure (fluid) [scatter]"): 
            kernelSum = scatter_sum(torch.einsum('nd, nd -> n', fac[:,None] * aij, grad), i, dim=0, dim_size=fluidActualArea.shape[0])

        return kernelSum
@torch.jit.script
def computeUpdatedPressureBoundarySum(fluidToGhostNeighbors: Optional[torch.Tensor], ghostParticleGradientIntegral : Optional[torch.Tensor], fluidPredAccel, dt : float):
    with record_function("DFSPH - pressure (boundary)"): 
        if fluidToGhostNeighbors is not None and ghostParticleGradientIntegral is not None:
            with record_function("DFSPH - update pressure kernel sum (boundary) [scatter]"): 
                boundaryTerm = scatter_sum(ghostParticleGradientIntegral, fluidToGhostNeighbors[0], dim=0, dim_size=fluidPredAccel.shape[0])

            with record_function("DFSPH - update pressure kernel sum (boundary) [einsum]"): 
                kernelSum = dt**2 * torch.einsum('nd, nd -> n', fluidPredAccel, boundaryTerm)

            return kernelSum
        else:
            return torch.zeros(fluidPredAccel.shape[0], dtype = fluidPredAccel.dtype, device = fluidPredAccel.device)
@torch.jit.script
def computeAlphaFluidTerm(fluidArea, fluidRestDensity, fluidActualArea, fluidNeighbors, fluidRadialDistances, fluidDistances, support):
    with record_function("DFSPH - alpha (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - alpha (fluid) [gradient]"): 
            grad = kernelGradient(fluidRadialDistances, fluidDistances, support)
            grad2 = torch.einsum('nd, nd -> n', grad, grad)

        with record_function("DFSPH - alpha (fluid) [term]"): 
            term1 = fluidActualArea[j][:,None] * grad
            term2 = (fluidActualArea**2 / (fluidArea * fluidRestDensity))[j] * grad2

        with record_function("DFSPH - alpha (fluid) [scatter]"): 
            kSum1 = scatter_sum(term1, i, dim=0, dim_size=fluidArea.shape[0])
            kSum2 = scatter_sum(term2, i, dim=0, dim_size=fluidArea.shape[0])
            
        return kSum1, kSum2
@torch.jit.script
def computeAlphaFinal(kSum1, kSum2, dt, fluidArea, fluidActualArea, fluidRestDensity):
    with record_function("DFSPH - alpha (final)"): 
        fac = - dt **2 * fluidActualArea
        mass = fluidArea * fluidRestDensity

        return fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2

@torch.jit.script
def computeSourceTermFluid(fluidActualArea, fluidPredictedVelocity, fluidNeighbors, fluidRadialDistances, fluidDistances, support, dt : float):
    with record_function("DFSPH - source (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - source (fluid) [term]"): 
            fac = - dt * fluidActualArea[j]
            vij = fluidPredictedVelocity[i] - fluidPredictedVelocity[j]
        with record_function("DFSPH - source (fluid) [gradient]"): 
            grad = kernelGradient(fluidRadialDistances, fluidDistances, support)
            prod = torch.einsum('nd, nd -> n', vij, grad)

        with record_function("DFSPH - source (fluid) [scatter]"): 
            source = scatter_sum(fac * prod, i, dim=0, dim_size=fluidActualArea.shape[0])

        return source
class dfsphModule(Module):
    def getParameters(self):
        return [
            Parameter('dfsph', 'minDensitySolverIterations', 'int', 2, required = False, export = True, hint = ''),
            Parameter('dfsph', 'minDivergenceSolverIterations', 'int', 2, required = False, export = True, hint = ''),
            Parameter('dfsph', 'maxDensitySolverIterations', 'int', 256, required = False, export = True, hint = ''),
            Parameter('dfsph', 'maxDivergenceSolverIterations', 'int', 8, required = False, export = True, hint = ''),
            Parameter('dfsph', 'densityThreshold', 'float', 1e-4, required = False, export = True, hint = ''),
            Parameter('dfsph', 'divergenceThreshold', 'float', 1e-2, required = False, export = True, hint = ''),
            Parameter('dfsph', 'divergenceSolver', 'bool', False, required = False, export = True, hint = ''),
            Parameter('dfsph', 'relaxedJacobiOmega', 'float', 0.5, required = False, export = True, hint = '')
        ]
        
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        # self.kernel, self.gradientKernel = getKernelFunctions(simulationConfig['kernel']['defaultKernel'])
        
        
        self.minDensitySolverIterations = simulationConfig['dfsph']['minDensitySolverIterations']
        self.minDivergenceSolverIterations = simulationConfig['dfsph']['minDivergenceSolverIterations']
        self.maxDensitySolverIterations = simulationConfig['dfsph']['maxDensitySolverIterations']
        self.maxDivergenceSolverIterations = simulationConfig['dfsph']['maxDivergenceSolverIterations']
        self.densityThreshold = simulationConfig['dfsph']['densityThreshold']
        self.divergenceThreshold = simulationConfig['dfsph']['divergenceThreshold']
#         self.divergenceSolver - simulationConfig['dfsph']['divergenceSolver']
        self.relaxedJacobiOmega = simulationConfig['dfsph']['relaxedJacobiOmega']
    
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0
        
        
    def computeAlpha(self, simulationState, simulation, density = True):
        with record_function("DFSPH - alpha"): 
            kSum1, kSum2 = computeAlphaFluidTerm(simulationState['fluidArea'], simulationState['fluidRestDensity'], simulationState['fluidActualArea'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)

            if density and 'fluidToGhostNeighbors' in simulationState and simulationState['fluidToGhostNeighbors'] != None:
                kSum1 += scatter(simulationState['ghostParticleGradientIntegral'], simulationState['fluidToGhostNeighbors'][0], dim=0, dim_size=simulationState['numParticles'],reduce='add')

            return computeAlphaFinal(kSum1, kSum2, simulationState['dt'], simulationState['fluidArea'], simulationState['fluidActualArea'], simulationState['fluidRestDensity'])
        

        
    def computeSourceTerm(self, simulationState, simulation, density = True):
        with record_function("DFSPH - source"): 
            source = computeSourceTermFluid(simulationState['fluidActualArea'], simulationState['fluidPredictedVelocity'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])
            
            if density and 'fluidToGhostNeighbors' in simulationState and simulationState['fluidToGhostNeighbors'] != None:
                boundaryTerm = scatter(simulationState['ghostParticleGradientIntegral'], simulationState['fluidToGhostNeighbors'][0], dim=0, dim_size=simulationState['numParticles'],reduce='add')
                
                source = source - simulationState['dt'] * torch.einsum('nd, nd -> n',  simulationState['fluidPredictedVelocity'],  boundaryTerm)

            return 1. - simulationState['fluidDensity'] + source if density else source            
        
        
    def computeUpdatedPressure(self, simulationState, simulation, density = True):
        with record_function("DFSPH - pressure"): 
            kernelSum = computeUpdatedPressureFluidSum(simulationState['fluidActualArea'], simulationState['fluidPredAccel'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])
            if 'fluidToGhostNeighbors' in simulationState:
                kernelSum += computeUpdatedPressureBoundarySum(simulationState['fluidToGhostNeighbors'], simulationState['ghostParticleGradientIntegral'], simulationState['fluidPredAccel'], simulationState['dt'])

            residual = kernelSum - simulationState['fluidSourceTerm']

            pressure = simulationState['fluidPressure'] - self.relaxedJacobiOmega * residual / simulationState['fluidAlpha']
            pressure = torch.clamp(pressure, min = 0.) if density else pressure
            if density and self.backgroundPressure:
                pressure = torch.clamp(pressure, min = (5**2) * simulationState['fluidRestDensity'])


            return pressure, residual

        
        
    def computeAcceleration(self, simulationState, simulation, density = True):
        with record_function("DFSPH - accel"):
            fluidAccelTerm = computeFluidAcceleration(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], simulationState['fluidNeighbors'], simulationState['fluidDistances'], simulationState['fluidRadialDistances'], self.support)
            
            if density and 'fluidToGhostNeighbors' in simulationState and simulationState['fluidToGhostNeighbors'] != None:
                simulationState['boundaryPressure'], boundaryAccelTerm, simulationState['boundaryPressureForce'] = \
                    computeBoundaryAccelTerm(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], simulationState['pgPartial'], \
                                             simulationState['ghostToFluidNeighbors2'], simulationState['fluidToGhostNeighbors'], simulationState['ghostParticleBodyAssociation'], simulationState['ghostParticleGradientIntegral'], \
                                             self.boundaryCounter, mlsPressure = True, computeBodyForces = True)


                return fluidAccelTerm + boundaryAccelTerm

            return fluidAccelTerm


    def densitySolve(self, simulationState, simulation):
        with record_function("DFSPH - densitySolve"): 
            errors = []
            i = 0
            error = 0.
            minIters = self.minDensitySolverIterations
#             minIters = 32
            if 'densityErrors' in simulationState:
                minIters = max(minIters, len(simulationState['densityErrors'])*0.75)

            while((i < minIters or \
                    error > self.densityThreshold) and \
                    i <= self.maxDensitySolverIterations):
                
                with record_function("DFSPH - densitySolve (iteration)"): 
                    with record_function("DFSPH - densitySolve (computeAccel)"): 
                        simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, True)
                        simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
                        simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                    with record_function("DFSPH - densitySolve (updatePressure)"): 
                        simulationState['fluidPressure2'], simulationState['residual'] = self.computeUpdatedPressure(simulationState, simulation, True)                    
                        simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
                        error = torch.mean(torch.clamp(simulationState['residual'], min = -self.densityThreshold))# * simulationState['fluidArea'])
                        
                    errors.append((error).item())
                    i = i + 1
            simulationState['densityErrors'] = errors
            simulationState['densitySolverPressure'] = simulationState['fluidPressure']
            return errors

    def divergenceSolve(self, simulationState, simulation):
        with record_function("DFSPH - divergenceSolve"): 
            errors = []
            i = 0
            error = 0.
            while((i < self.minDivergenceSolverIterations or error > self.divergenceThreshold) and i <= self.maxDivergenceSolverIterations):
                
                with record_function("DFSPH - divergenceSolve (iteration)"): 
                    with record_function("DFSPH - divergenceSolve (computeAccel)"): 
                        simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, False)
                        simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
                        simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                    with record_function("DFSPH - divergenceSolve (updatePressure)"): 
                        simulationState['fluidPressure2'], simulationState['residual'] = self.computeUpdatedPressure(simulationState, simulation, False)                    
                        simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
                        error = torch.mean(torch.clamp(simulationState['residual'], min = -self.divergenceThreshold))# * simulationState['fluidArea'])
                        
                    errors.append((error).item())
                    i = i + 1
            simulationState['divergenceErrors'] = errors
            simulationState['divergenceSolverPressure'] = simulationState['fluidPressure']
            return errors


    def DFSPH(self, simulationState, simulation, density = True): 
        with record_function("DFSPH - solver"): 
            with record_function("DFSPH - predict velocity"): 
                simulationState['fluidPredAccel'] = torch.zeros(simulationState['fluidPosition'].shape, dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)
                simulationState['fluidPredictedVelocity'] = simulationState['fluidVelocity'] + simulationState['dt'] * simulationState['fluidAcceleration']
                simulationState['fluidActualArea'] = simulationState['fluidArea'] / simulationState['fluidDensity']

            with record_function("DFSPH - compute alpha"): 
                if density:
                    simulationState['pgPartial'], simulationState['ghostToFluidNeighbors2'] = precomputeMLS(simulationState['ghostParticlePosition'], simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2)

                simulationState['fluidAlpha'] = self.computeAlpha(simulationState, simulation, density)
                simulation.periodicBC.syncQuantity(simulationState['fluidAlpha'], simulationState, simulation)

            with record_function("DFSPH - compute source"):
                simulationState['fluidSourceTerm'] = self.computeSourceTerm(simulationState, simulation, density)
                simulation.periodicBC.syncQuantity(simulationState['fluidSourceTerm'], simulationState, simulation)
                
            with record_function("DFSPH - initialize pressure"):
                simulationState['fluidPressure2'] =  torch.zeros(simulationState['numParticles'], dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)

                if 'densitySolverPressure' in simulationState and density:
                    simulationState['fluidPressure2'] =  simulationState['densitySolverPressure'] * 0.5
                else:
                    simulationState['fluidPressure2'] = torch.zeros(simulationState['numParticles'], dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)

                simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
                totalArea = torch.sum(simulationState['fluidArea'])


            with record_function("DFSPH - solver step"):
                if density:
                    errors = self.densitySolve(simulationState, simulation)
                else:
                    errors = self.divergenceSolve(simulationState, simulation)

                
            with record_function("DFSPH - compute accel"):
                simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, density)
                simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
                simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                simulationState['fluidPredictedVelocity'] += simulationState['dt'] * simulationState['fluidPredAccel']

            return errors
        
    def incompressibleSolver(self, simulationState, simulation):
        with record_function("DFSPH - incompressibleSolver"): 
            return self.DFSPH(simulationState, simulation, True)
    def divergenceSolver(self, simulationState, simulation):
        with record_function("DFSPH - divergenceSolver"): 
            return self.DFSPH(simulationState, simulation, False)
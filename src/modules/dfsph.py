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

from ..kernels import kernel, spikyGrad, kernelGradient
from ..module import Module
from ..parameter import Parameter
from ..util import *

@torch.jit.script
def prepareMLSBoundaries(boundaryPositions, boundarySupports, neighbors, boundaryRadialDistances, fluidPosition, fluidActualArea, support):

    # boundaryPositions = state['akinciBoundary']['positions']
    # boundarySupports = state['akinciBoundary']['boundarySupport']

    bb = neighbors[0]
    bf = neighbors[1] #state['akinciBoundary']['boundaryToFluidNeighbors']
    # boundaryRadialDistances = state['akinciBoundary']['boundaryToFluidNeighborRadialDistances']

    k = kernel(boundaryRadialDistances, support)* fluidActualArea[bf]

    nominator = scatter((k)[:,None] * fluidPosition[bf], bb, dim=0, dim_size = boundaryPositions.shape[0], reduce = 'add')
    denominator = scatter((k), bb, dim=0, dim_size = boundaryPositions.shape[0], reduce = 'add')
    d = torch.clone(boundaryPositions)
    d[denominator > 1e-9] = nominator[denominator > 1e-9] / denominator[denominator > 1e-9,None]
    # debugPrint(state['fluidPosition'][bf] - d[bb])

    xbar =  fluidPosition[bf] - d[bb]

    prod = torch.einsum('nu, nv -> nuv', xbar, xbar) * k[:,None,None]

    Mpartial = scatter(prod, bb, dim = 0, dim_size = boundaryPositions.shape[0], reduce = 'add')

    M1 = torch.linalg.pinv(Mpartial)

    vec = xbar * k[:,None]
    bbar = torch.hstack((torch.ones_like(boundarySupports).unsqueeze(1), boundaryPositions - d))
    
    return M1, vec, bbar

@torch.jit.script
def evalMLSBoundaries(M1, vec, bbar, neighbors, boundaryRadialDistances, fluidPosition, fluidActualArea, fluidPressure, support):
    bb = neighbors[0]
    bf = neighbors[1] 
    k = kernel(boundaryRadialDistances, support)* fluidActualArea[bf]
    
    vecSum = scatter(vec * fluidPressure[bf,None], bb, dim = 0, dim_size = M1.shape[0], reduce = 'add')
    alphaP = scatter(fluidPressure[bf] * k, bb, dim = 0, dim_size = M1.shape[0], reduce = 'add')
    alphaS = scatter( k, bb, dim = 0, dim_size = M1.shape[0], reduce = 'add')

    alpha = alphaP
    alphaP[alphaS > 1e-6] = alphaP[alphaS > 1e-6] / alphaS[alphaS > 1e-6]

    c = torch.hstack((alpha.unsqueeze(1), torch.matmul(M1, vecSum.unsqueeze(2))[:,:,0]))
    pb = torch.einsum('nu, nu -> n', bbar, c)
    return pb


@torch.jit.script
def precomputeMLS(queryPositions, fluidPosition, fluidArea, fluidDensity, support, neighbors, radialDistances):
    with record_function("MLS - precomputeMLS"): 
        # queryPositions = simulationState['fluidPosition']
        # queryPosition = pb

#         i = neighbors[0]
#         j = neighbors[1]
# #         neighbors = torch.stack([i, j], dim = 0)

#     #     debugPrint(neighbors)
#         # debugPrint(torch.min(neighbors[0]))
#         # debugPrint(torch.max(neighbors[0]))
#         # debugPrint(torch.min(neighbors[1]))
#         # debugPrint(torch.max(neighbors[1]))

#         distances = (fluidPosition[j] - queryPositions[i])
#         radialDistances = torch.linalg.norm(distances,dim=1)

#         distances[radialDistances < 1e-5,:] = 0
#         distances[radialDistances >= 1e-5,:] /= radialDistances[radialDistances >= 1e-5,None]
#         radialDistances /= support
        i = neighbors[0]
        j = neighbors[1]

        kernel = kernel(radialDistances, support)

        bij = fluidPosition[j] - queryPositions[i]
        bij = torch.hstack((bij.new_ones((bij.shape[0]))[:,None], bij))
    #     debugPrint(bij)

        Mpartial = 2 * torch.einsum('nu, nv -> nuv', bij, bij) * \
                ((fluidArea / fluidDensity)[j] * kernel)[:,None,None]
        # print(Mpartial.shape)
        M = scatter(Mpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
        # print(M.shape)
        # print(M)
        Minv = torch.linalg.pinv(M)
    #     debugPrint(Minv)

        e1 = torch.tensor([1,0,0], dtype=Minv.dtype, device=Minv.device)
        Me1 = torch.matmul(Minv,e1)
    #     debugPrint(Me1)


        pGpartial = torch.einsum('nd, nd -> n', Me1[i], bij) * \
            kernel * ((fluidArea / fluidDensity)[j])

#         pG = scatter(pGpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
    #     debugPrint(pG)

        return pGpartial

@torch.jit.script
def computeFluidAcceleration(fluidArea, fluidDensity, fluidRestDensity, fluidPressure2, fluidNeighbors, fluidDistances, fluidRadialDistances, support):
    with record_function("DFSPH - accel (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - accel (fluid) [gradient]"): 
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)

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
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)

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
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)
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
        alpha = fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2
        alpha = torch.clamp(alpha, -1, -1e-7)
        return alpha

@torch.jit.script
def computeSourceTermFluid(fluidActualArea, fluidPredictedVelocity, fluidNeighbors, fluidRadialDistances, fluidDistances, support, dt : float):
    with record_function("DFSPH - source (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - source (fluid) [term]"): 
            fac = - dt * fluidActualArea[j]
            vij = fluidPredictedVelocity[i] - fluidPredictedVelocity[j]
        with record_function("DFSPH - source (fluid) [gradient]"): 
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)
            prod = torch.einsum('nd, nd -> n', vij, grad)

        with record_function("DFSPH - source (fluid) [scatter]"): 
            source = scatter_sum(fac * prod, i, dim=0, dim_size=fluidActualArea.shape[0])

        return source

# class dfsphModule(Module):
#     def getParameters(self):
#         return [
#             Parameter('dfsph', 'minDensitySolverIterations', 'int', 2, required = False, export = True, hint = ''),
#             Parameter('dfsph', 'minDivergenceSolverIterations', 'int', 2, required = False, export = True, hint = ''),
#             Parameter('dfsph', 'maxDensitySolverIterations', 'int', 256, required = False, export = True, hint = ''),
#             Parameter('dfsph', 'maxDivergenceSolverIterations', 'int', 8, required = False, export = True, hint = ''),
#             Parameter('dfsph', 'densityThreshold', 'float', 1e-4, required = False, export = True, hint = ''),
#             Parameter('dfsph', 'divergenceThreshold', 'float', 1e-2, required = False, export = True, hint = ''),
#             Parameter('dfsph', 'divergenceSolver', 'bool', False, required = False, export = True, hint = ''),
#             Parameter('dfsph', 'relaxedJacobiOmega', 'float', 0.5, required = False, export = True, hint = '')
#         ]
        
#     def __init__(self):
#         super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
#     def initialize(self, simulationConfig, simulationState):
#         self.support = simulationConfig['particle']['support']
#         # self.kernel, self.gradientKernel = getKernelFunctions(simulationConfig['kernel']['defaultKernel'])
        
        
#         self.minDensitySolverIterations = simulationConfig['dfsph']['minDensitySolverIterations']
#         self.minDivergenceSolverIterations = simulationConfig['dfsph']['minDivergenceSolverIterations']
#         self.maxDensitySolverIterations = simulationConfig['dfsph']['maxDensitySolverIterations']
#         self.maxDivergenceSolverIterations = simulationConfig['dfsph']['maxDivergenceSolverIterations']
#         self.densityThreshold = simulationConfig['dfsph']['densityThreshold']
#         self.divergenceThreshold = simulationConfig['dfsph']['divergenceThreshold']
# #         self.divergenceSolver - simulationConfig['dfsph']['divergenceSolver']
#         self.relaxedJacobiOmega = simulationConfig['dfsph']['relaxedJacobiOmega']
    
#         self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        
#         self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
#         self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

#         self.pressureScheme = simulationConfig['simulation']['pressureTerm'] 
#         self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        
        
#     def computeAlpha(self, simulationState, simulation, density = True):
#         with record_function("DFSPH - alpha"): 
#             kSum1, kSum2 = computeAlphaFluidTerm(simulationState['fluidArea'], simulationState['fluidRestDensity'], simulationState['fluidActualArea'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)

#             if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
#                 kSum1 += scatter(simulationState['sdfBoundary']['ghostParticleGradientIntegral'], simulationState['sdfBoundary']['fluidToGhostNeighbors'][0], dim=0, dim_size=simulationState['numParticles'],reduce='add')
#             if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
#                 bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
#                 boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
#                 boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
#                 boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
#                 boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
#                 boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']

#                 grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)
#                 grad2 = torch.einsum('nd, nd -> n', grad, grad)

#                 fluidActualArea = simulationState['fluidActualArea']
#                 fluidArea = simulationState['fluidArea']

#                 termFluid = (boundaryActualArea**2 / (boundaryArea * boundaryRestDensity))[bb] * grad2
#                 termBoundary = (simulationState['fluidActualArea']**2 / (simulationState['fluidArea'] * simulationState['fluidRestDensity']))[bf] * grad2
#                 if self.pressureScheme == 'PBSPH':
#                     kSum1 += -scatter(boundaryActualArea[bb,None] * grad, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
#                     kSum2 += scatter(termFluid, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
#                     simulationState['akinciBoundary']['boundaryAlpha'] = -simulationState['dt']**2 * boundaryActualArea * scatter(termBoundary, bb, dim=0, dim_size=boundaryArea.shape[0],reduce='add')
#                 else:
#                     kSum1 += -scatter(boundaryArea[bb,None] * grad, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
#                     termBoundary = (boundaryArea**2 / (boundaryArea * boundaryRestDensity))[bb] * grad2
#                     kSum2 += scatter(termBoundary, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')

            
#             return computeAlphaFinal(kSum1, kSum2, simulationState['dt'], simulationState['fluidArea'], simulationState['fluidActualArea'], simulationState['fluidRestDensity'])
        

        
#     def computeSourceTerm(self, simulationState, simulation, density = True):
#         with record_function("DFSPH - source"): 
#             source = computeSourceTermFluid(simulationState['fluidActualArea'], simulationState['fluidPredictedVelocity'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])
            
#             if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
#                 boundaryTerm = scatter(simulationState['sdfBoundary']['ghostParticleGradientIntegral'], simulationState['sdfBoundary']['fluidToGhostNeighbors'][0], dim=0, dim_size=simulationState['numParticles'],reduce='add')
                
#                 source = source - simulationState['dt'] * torch.einsum('nd, nd -> n',  simulationState['fluidPredictedVelocity'],  boundaryTerm)
#             if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
#                 bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
#                 boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
#                 boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
#                 boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
#                 boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
#                 boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']
#                 boundaryPredictedVelocity = simulationState['akinciBoundary']['boundaryPredictedVelocity']
                
#                 grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)
#                 velDifference = boundaryPredictedVelocity[bb] - simulationState['fluidPredictedVelocity'][bf]
#                 prod = torch.einsum('nd, nd -> n',  velDifference,  grad)
#                 # debugPrint(simulationState['fluidPredictedVelocity'][bf,0])
#                 # debugPrint(simulationState['fluidPredictedVelocity'][bf,1])
#                 # debugPrint(prod)
#                 # debugPrint(grad[:,0])
#                 # debugPrint(grad[:,1])

#                 if self.pressureScheme == 'PBSPH':
#                     source = source - simulationState['dt'] * scatter(boundaryActualArea[bb] *prod, bf, dim = 0, dim_size = simulationState['numParticles'], reduce= 'add')
#                     boundarySource = - simulationState['dt'] * scatter(simulationState['fluidActualArea'][bf] *prod, bb, dim = 0, dim_size = boundaryArea.shape[0], reduce= 'add')
#                     simulationState['akinciBoundary']['boundarySource'] = 1. - simulationState['akinciBoundary']['boundaryDensity'] + boundarySource if density else boundarySource 
#                 else:
#                     fluidActualArea = simulationState['fluidActualArea']
#                     source = source - simulationState['dt'] * scatter(boundaryArea[bb] *prod, bf, dim = 0, dim_size = simulationState['numParticles'], reduce= 'add')

                                
#             return 1. - simulationState['fluidDensity'] + source if density else source            
        
        
#     def computeUpdatedPressure(self, simulationState, simulation, density = True):
#         with record_function("DFSPH - pressure"): 
#             kernelSum = computeUpdatedPressureFluidSum(simulationState['fluidActualArea'], simulationState['fluidPredAccel'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])
#             if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
#                 kernelSum += computeUpdatedPressureBoundarySum(simulationState['sdfBoundary']['fluidToGhostNeighbors'], simulationState['sdfBoundary']['ghostParticleGradientIntegral'], simulationState['fluidPredAccel'], simulationState['dt'])

#             if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
#                 bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
#                 boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
#                 boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
#                 boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
#                 boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
#                 boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']
#                 boundaryPredictedVelocity = simulationState['akinciBoundary']['boundaryPredictedVelocity']
                
#                 grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)


#                 facFluid = simulationState['dt']**2 * simulationState['fluidActualArea'][bf]
#                 facBoundary = simulationState['dt']**2 * boundaryActualArea[bb]
#                 aij = simulationState['fluidPredAccel'][bf]

#                 if self.pressureScheme == 'PBSPH':
#                     boundaryKernelSum = scatter_sum(torch.einsum('nd, nd -> n', facFluid[:,None] * aij, -grad), bb, dim=0, dim_size=boundaryArea.shape[0])

#                     simulationState['akinciBoundary']['boundaryResidual'] = boundaryKernelSum - simulationState['akinciBoundary']['boundarySource']
#                     boundaryPressure = simulationState['akinciBoundary']['boundaryPressure'] - self.relaxedJacobiOmega * simulationState['akinciBoundary']['boundaryResidual'] / simulationState['akinciBoundary']['boundaryAlpha']
#                     boundaryPressure = torch.clamp(boundaryPressure, min = 0.) if density else boundaryPressure
#                     if density and self.backgroundPressure:
#                         boundaryPressure = torch.clamp(boundaryPressure, min = (5**2) * simulationState['akinciBoundary']['boundaryRestDensity'])
#                     simulationState['akinciBoundary']['boundaryPressure'] = boundaryPressure
#                     kernelSum += scatter_sum(torch.einsum('nd, nd -> n', -facBoundary[:,None] * aij, grad), bf, dim=0, dim_size=simulationState['fluidActualArea'].shape[0])
#                 else:

#                     facFluid = simulationState['dt']**2 * simulationState['akinciBoundary']['boundaryVolume'][bb] / simulationState['fluidDensity'][bf]
#                     kernelSum += scatter_sum(torch.einsum('nd, nd -> n', -facFluid[:,None] * aij, grad), bf, dim=0, dim_size=simulationState['fluidActualArea'].shape[0])




#             residual = kernelSum - simulationState['fluidSourceTerm']

#             pressure = simulationState['fluidPressure'] - self.relaxedJacobiOmega * residual / simulationState['fluidAlpha']
#             pressure = torch.clamp(pressure, min = 0.) if density else pressure
#             if density and self.backgroundPressure:
#                 pressure = torch.clamp(pressure, min = (5**2) * simulationState['fluidRestDensity'])
#             if torch.any(torch.isnan(pressure)) or torch.any(torch.isinf(pressure)):
#                 raise Exception('Pressure solver became unstable!')

#             return pressure, residual

        
        
#     def computeAcceleration(self, simulationState, simulation, density = True):
#         with record_function("DFSPH - accel"):
#             fluidAccelTerm = computeFluidAcceleration(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], simulationState['fluidNeighbors'], simulationState['fluidDistances'], simulationState['fluidRadialDistances'], self.support)
            
#             if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
#                 simulationState['sdfBoundary']['boundaryPressure'], boundaryAccelTerm, simulationState['sdfBoundary']['boundaryPressureForce'] = \
#                     computeBoundaryAccelTerm(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], simulationState['sdfBoundary']['pgPartial'], \
#                                              simulationState['sdfBoundary']['ghostToFluidNeighbors2'], simulationState['sdfBoundary']['fluidToGhostNeighbors'], simulationState['sdfBoundary']['ghostParticleBodyAssociation'], simulationState['sdfBoundary']['ghostParticleGradientIntegral'], \
#                                              self.boundaryCounter, mlsPressure = self.pressureScheme == "deltaMLS", computeBodyForces = self.computeBodyForces)


#                 return fluidAccelTerm + boundaryAccelTerm

#             if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
#                 bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
#                 boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
#                 boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
#                 boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
#                 boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
#                 boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']
#                 boundaryPredictedVelocity = simulationState['akinciBoundary']['boundaryPredictedVelocity']
                
#                 grad = -spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)

#                 fac = -(boundaryArea * boundaryRestDensity)[bb]
                
#                 pi = (simulationState['fluidPressure2'] / (simulationState['fluidDensity'] * simulationState['fluidRestDensity'])**2)[bf]

#                 if self.pressureScheme == "mirrored":
#                     simulationState['akinciBoundary']['boundaryPressure2'] = 0
#                     pb = simulationState['fluidPressure2'][bf]

                    
#                 if self.pressureScheme == "deltaMLS":
#                     neighbors2 = simulationState['akinciBoundary']['boundaryToFluidNeighbors2']
                    
#                     simulationState['akinciBoundary']['boundaryPressure2'] = scatter(simulationState['akinciBoundary']['pgPartial'] * simulationState['fluidPressure2'][neighbors2[1]], neighbors2[0], dim=0, dim_size = boundaryArea.shape[0], reduce='add')

#                     simulationState['akinciBoundary']['boundaryGravity'] = torch.zeros_like(simulationState['akinciBoundary']['positions'])
#                     simulationState['akinciBoundary']['boundaryGravity'][:,1] = -1

#                     # simulationState['akinciBoundary']['boundaryPressure2'] += 2 * 2 * boundaryRestDensity * simulationState['akinciBoundary']['boundaryDensity'] * torch.einsum('nd, nd -> n', simulationState['akinciBoundary']['boundaryNormals'], simulationState['akinciBoundary']['boundaryGravity'])
#                     simulationState['akinciBoundary']['boundaryPressure2'][:] = torch.clamp(simulationState['akinciBoundary']['boundaryPressure2'][:],min = 0)
#                     simulationState['akinciBoundary']['boundaryPressure'][:] = simulationState['akinciBoundary']['boundaryPressure2'][:]
#                     pb = simulationState['akinciBoundary']['boundaryPressure2'][bb]
#                 if self.pressureScheme == "MLSPressure":
#                     pb = evalMLSBoundaries(simulationState['akinciBoundary']['M1'], simulationState['akinciBoundary']['vec'], simulationState['akinciBoundary']['bbar'], simulationState['akinciBoundary']['boundaryToFluidNeighbors'], simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances'], simulationState['fluidPosition'], simulationState['fluidActualArea'], simulationState['fluidPressure'], self.support)
#                     pb = torch.clamp(pb,min = 0)
#                     # simulationState['akinciBoundary']['boundaryGravity'] = torch.zeros_like(simulationState['akinciBoundary']['positions'])
#                     # simulationState['akinciBoundary']['boundaryGravity'][:,1] = -1
#                     # pb += 2 * 2 * boundaryRestDensity * simulationState['akinciBoundary']['boundaryDensity'] * torch.einsum('nd, nd -> n', simulationState['akinciBoundary']['boundaryNormals'], simulationState['akinciBoundary']['boundaryGravity'])
#                     simulationState['akinciBoundary']['boundaryPressure'][:] = simulationState['akinciBoundary']['boundaryPressure2'][:] = pb

#                     pb = simulationState['akinciBoundary']['boundaryPressure'][bb]


#                 if self.pressureScheme == "PBSPH":
#                     simulationState['akinciBoundary']['boundaryPressure2'][:] = simulationState['akinciBoundary']['boundaryPressure'][:]
#                     pb = simulationState['akinciBoundary']['boundaryPressure'][bb]

#                     pb =  pb / ((simulationState['boundaryDensity'][bb] * simulationState['akinciBoundary']['boundaryRestDensity'][bb])**2)
#                 else:
#                     fac = -(boundaryArea * boundaryRestDensity)[bb]
#                     pb =  pb / ((1 * simulationState['akinciBoundary']['boundaryRestDensity'][bb])**2)


#                 # pb =  pb / ((simulationState['akinciBoundary']['boundaryDensity'] * simulationState['akinciBoundary']['boundaryRestDensity'])**2)[bb]
#                 term = (fac * (pi + pb))[:,None] * grad

#                 # debugPrint(fac)
#                 # debugPrint(pi)
#                 # debugPrint(pb)
#                 # debugPrint(term)
#                 # debugPrint(grad)
                
#                 if self.computeBodyForces:
#                     force = -term * (simulationState['fluidArea'] * simulationState['fluidRestDensity'])[bf,None]
#                     # simulationState['akinciBoundary']['bodyAssociation']
#                     boundaryForces = scatter_sum(force, bb, dim=0, dim_size = simulationState['akinciBoundary']['boundaryDensity'].shape[0])
#                     simulationState['akinciBoundary']['pressureForces'] = scatter_sum(boundaryForces, simulationState['akinciBoundary']['bodyAssociation'], dim=0, dim_size = self.boundaryCounter)

#                 boundaryAccelTerm = scatter_sum(term, bf, dim=0, dim_size = simulationState['fluidArea'].shape[0])

#                 return fluidAccelTerm + boundaryAccelTerm



#             return fluidAccelTerm


#     def densitySolve(self, simulationState, simulation):
#         with record_function("DFSPH - densitySolve"): 
#             errors = []
#             i = 0
#             error = 0.
#             minIters = self.minDensitySolverIterations
# #             minIters = 32
#             if 'densityErrors' in simulationState:
#                 minIters = max(minIters, len(simulationState['densityErrors'])*0.75)

#             while((i < minIters or \
#                     error > self.densityThreshold) and \
#                     i <= self.maxDensitySolverIterations):
                
#                 with record_function("DFSPH - densitySolve (iteration)"): 
#                     with record_function("DFSPH - densitySolve (computeAccel)"): 
#                         simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, True)
#                         simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
#                         simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

#                     with record_function("DFSPH - densitySolve (updatePressure)"): 
#                         simulationState['fluidPressure2'], simulationState['residual'] = self.computeUpdatedPressure(simulationState, simulation, True)             
#                         simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
#                         # debugPrint(self.boundaryScheme)
#                         # debugPrint(self.pressureScheme)
#                         if self.boundaryScheme == 'Akinci' and self.pressureScheme == 'PBSPH':
#                             boundaryError = torch.sum(torch.clamp(simulationState['akinciBoundary']['boundaryResidual'], min = -self.densityThreshold))
#                             fluidError = torch.sum(torch.clamp(simulationState['residual'], min = -self.densityThreshold))
#                             error = (fluidError + boundaryError) / (simulationState['akinciBoundary']['boundaryResidual'].shape[0] + simulationState['residual'].shape[0])
#                             # debugPrint(boundaryError)
#                         else:
#                             error = torch.mean(torch.clamp(simulationState['residual'], min = -self.densityThreshold))# * simulationState['fluidArea'])
                        
#                     errors.append((error).item())
#                     i = i + 1
#             simulationState['densityErrors'] = errors
#             simulationState['densitySolverPressure'] = simulationState['fluidPressure']
#             return errors

#     def divergenceSolve(self, simulationState, simulation):
#         with record_function("DFSPH - divergenceSolve"): 
#             errors = []
#             i = 0
#             error = 0.
#             while((i < self.minDivergenceSolverIterations or error > self.divergenceThreshold) and i <= self.maxDivergenceSolverIterations):
                
#                 with record_function("DFSPH - divergenceSolve (iteration)"): 
#                     with record_function("DFSPH - divergenceSolve (computeAccel)"): 
#                         simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, False)
#                         simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
#                         simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

#                     with record_function("DFSPH - divergenceSolve (updatePressure)"): 
#                         simulationState['fluidPressure2'], simulationState['residual'] = self.computeUpdatedPressure(simulationState, simulation, False)                    
#                         simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
#                         error = torch.mean(torch.clamp(simulationState['residual'], min = -self.divergenceThreshold))# * simulationState['fluidArea'])
                        
#                     errors.append((error).item())
#                     i = i + 1
#             simulationState['divergenceErrors'] = errors
#             simulationState['divergenceSolverPressure'] = simulationState['fluidPressure']
#             return errors


#     def DFSPH(self, simulationState, simulation, density = True): 
#         with record_function("DFSPH - solver"): 
#             with record_function("DFSPH - predict velocity"): 
#                 simulationState['fluidPredAccel'] = torch.zeros(simulationState['fluidPosition'].shape, dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)
#                 simulationState['fluidPredictedVelocity'] = simulationState['fluidVelocity'] + simulationState['dt'] * simulationState['fluidAcceleration']
#                 simulationState['fluidActualArea'] = simulationState['fluidArea'] / simulationState['fluidDensity']

#             with record_function("DFSPH - compute alpha"): 
#                 if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None: 
#                     if self.pressureScheme == "deltaMLS":
#                         support = simulation.config['particle']['support']
#                         supports = torch.ones(simulationState['sdfBoundary']['ghostParticlePosition'].shape[0], device = simulationState['fluidActualArea'].device, dtype = simulationState['fluidActualArea'].dtype) * support

#                         neighbors, distances, radialDistances = simulation.neighborSearch.searchExisting(simulationState['sdfBoundary']['ghostParticlePosition'], supports * 2, simulationState, simulation, searchRadius = 2)
#                         simulationState['sdfBoundary']['ghostToFluidNeighbors2'] = neighbors
#                         simulationState['sdfBoundary']['pgPartial'] = precomputeMLS(simulationState['sdfBoundary']['ghostParticlePosition'], simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, neighbors, radialDistances)

#                 if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
#                     simulationState['akinciBoundary']['boundaryPressure'] = torch.zeros_like(simulationState['akinciBoundary']['boundaryVolume'])
#                     simulationState['akinciBoundary']['boundaryPressure2'] = torch.zeros_like(simulationState['akinciBoundary']['boundaryVolume'])
#                     simulationState['akinciBoundary']['boundaryActualArea'] = simulationState['akinciBoundary']['boundaryVolume'] / simulationState['akinciBoundary']['boundaryDensity']
#                     simulationState['akinciBoundary']['fluidPredAccel'] = torch.zeros(simulationState['akinciBoundary']['positions'].shape, dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)
#                     simulationState['akinciBoundary']['boundaryPredictedVelocity'] = simulationState['akinciBoundary']['boundaryVelocity'] + simulationState['dt'] * simulationState['akinciBoundary']['boundaryAcceleration']
#                     if self.pressureScheme == "deltaMLS":
#                         neighbors, distances, radDistances = simulation.neighborSearch.searchExisting(simulationState['akinciBoundary']['positions'], simulationState['akinciBoundary']['boundarySupport'] * 2, simulationState, simulation, searchRadius = 2)
#                         simulationState['akinciBoundary']['boundaryToFluidNeighbors2'] = neighbors  
#         # neighbors = radius(fluidPosition, queryPositions, support, max_num_neighbors = 256)
#                         simulationState['akinciBoundary']['pgPartial'] = precomputeMLS(simulationState['akinciBoundary']['positions'], simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, neighbors, radDistances)

#                     if self.pressureScheme == "MLSPressure":
#                         simulationState['akinciBoundary']['M1'], simulationState['akinciBoundary']['vec'], simulationState['akinciBoundary']['bbar'] = prepareMLSBoundaries(simulationState['akinciBoundary']['positions'], simulationState['akinciBoundary']['boundarySupport'], simulationState['akinciBoundary']['boundaryToFluidNeighbors'], simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances'], simulationState['fluidPosition'], simulationState['fluidActualArea'], self.support)

#         #                 neighbors, distances, radDistances = simulation.neighborSearch.searchExisting(simulationState['akinciBoundary']['positions'], simulationState['akinciBoundary']['boundarySupport'] * 2, simulationState, simulation, searchRadius = 2)
#         #                 simulationState['akinciBoundary']['boundaryToFluidNeighbors2'] = neighbors  
#         # # neighbors = radius(fluidPosition, queryPositions, support, max_num_neighbors = 256)
#         #                 simulationState['akinciBoundary']['pgPartial'] = precomputeMLS(simulationState['akinciBoundary']['positions'], simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, neighbors, radDistances)
        

#                 simulationState['fluidAlpha'] = self.computeAlpha(simulationState, simulation, density)
#                 simulation.periodicBC.syncQuantity(simulationState['fluidAlpha'], simulationState, simulation)

#             with record_function("DFSPH - compute source"):
#                 simulationState['fluidSourceTerm'] = self.computeSourceTerm(simulationState, simulation, density)
#                 simulation.periodicBC.syncQuantity(simulationState['fluidSourceTerm'], simulationState, simulation)
                
#             with record_function("DFSPH - initialize pressure"):
#                 simulationState['fluidPressure2'] =  torch.zeros(simulationState['numParticles'], dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)

#                 if 'densitySolverPressure' in simulationState and density:
#                     simulationState['fluidPressure2'] =  simulationState['densitySolverPressure'] * 0.5
#                 else:
#                     simulationState['fluidPressure2'] = torch.zeros(simulationState['numParticles'], dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)

#                 simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
#                 totalArea = torch.sum(simulationState['fluidArea'])


#             with record_function("DFSPH - solver step"):
#                 if density:
#                     errors = self.densitySolve(simulationState, simulation)
#                 else:
#                     errors = self.divergenceSolve(simulationState, simulation)

                
#             with record_function("DFSPH - compute accel"):
#                 simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, density)
#                 simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
#                 simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

#                 simulationState['fluidPredictedVelocity'] += simulationState['dt'] * simulationState['fluidPredAccel']

#             return errors
        
#     def incompressibleSolver(self, simulationState, simulation):
#         with record_function("DFSPH - incompressibleSolver"): 
#             return self.DFSPH(simulationState, simulation, True)
#     def divergenceSolver(self, simulationState, simulation):
#         with record_function("DFSPH - divergenceSolver"): 
#             return self.DFSPH(simulationState, simulation, False)


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
        
        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

        self.pressureScheme = simulationConfig['simulation']['pressureTerm'] 
        self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        
        
    def computeAlpha(self, simulationState, simulation, density = True):
        with record_function("DFSPH - alpha"): 
            kSum1, kSum2 = computeAlphaFluidTerm(simulationState['fluidArea'], simulationState['fluidRestDensity'], simulationState['fluidActualArea'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)

            if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
                kSum1 += scatter(simulationState['sdfBoundary']['ghostParticleGradientIntegral'], simulationState['sdfBoundary']['fluidToGhostNeighbors'][0], dim=0, dim_size=simulationState['numParticles'],reduce='add')
            if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
                bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
                boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
                boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
                boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
                boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
                boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']

                grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)
                grad2 = torch.einsum('nd, nd -> n', grad, grad)

                fluidActualArea = simulationState['fluidActualArea']
                fluidArea = simulationState['fluidArea']

                termFluid = (boundaryActualArea**2 / (boundaryArea * boundaryRestDensity))[bb] * grad2
                termBoundary = (simulationState['fluidActualArea']**2 / (simulationState['fluidArea'] * simulationState['fluidRestDensity']))[bf] * grad2
                if self.pressureScheme == 'PBSPH':
                    kSum1 += -scatter(boundaryActualArea[bb,None] * grad, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
                    kSum2 += scatter(termFluid, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
                    simulationState['akinciBoundary']['boundaryAlpha'] = -simulationState['dt']**2 * boundaryActualArea * scatter(termBoundary, bb, dim=0, dim_size=boundaryArea.shape[0],reduce='add')
                else:
                    area        = simulationState['akinciBoundary']['boundaryVolume']
                    restDensity = simulationState['akinciBoundary']['boundaryRestDensity']
                    density     = simulationState['akinciBoundary']['boundaryDensity']
                    actualArea  = area /1# density
                    
                    term1 = actualArea[bb][:,None] * grad
                    term2 = actualArea[bb]**2 / (area * restDensity)[bb] * grad2
                    
                    kSum1 += scatter(term1, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
                    kSum2 += scatter(term2, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')

            
            return computeAlphaFinal(kSum1, kSum2, simulationState['dt'], simulationState['fluidArea'], simulationState['fluidActualArea'], simulationState['fluidRestDensity'])
        

        
    def computeSourceTerm(self, simulationState, simulation, density = True):
        with record_function("DFSPH - source"): 
            source = computeSourceTermFluid(simulationState['fluidActualArea'], simulationState['fluidPredictedVelocity'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])
            
            if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
                boundaryTerm = scatter(simulationState['sdfBoundary']['ghostParticleGradientIntegral'], simulationState['sdfBoundary']['fluidToGhostNeighbors'][0], dim=0, dim_size=simulationState['numParticles'],reduce='add')
                
                source = source - simulationState['dt'] * torch.einsum('nd, nd -> n',  simulationState['fluidPredictedVelocity'],  boundaryTerm)
            if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
                bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
                boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
                boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
                boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
                boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
                boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']
                boundaryPredictedVelocity = simulationState['akinciBoundary']['boundaryPredictedVelocity']
                
                grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)
                velDifference = boundaryPredictedVelocity[bb] - simulationState['fluidPredictedVelocity'][bf]
                prod = torch.einsum('nd, nd -> n',  velDifference,  grad)
                # debugPrint(simulationState['fluidPredictedVelocity'][bf,0])
                # debugPrint(simulationState['fluidPredictedVelocity'][bf,1])
                # debugPrint(prod)
                # debugPrint(grad[:,0])
                # debugPrint(grad[:,1])

                if self.pressureScheme == 'PBSPH':
                    source = source - simulationState['dt'] * scatter(boundaryActualArea[bb] *prod, bf, dim = 0, dim_size = simulationState['numParticles'], reduce= 'add')
                    boundarySource = - simulationState['dt'] * scatter(simulationState['fluidActualArea'][bf] *prod, bb, dim = 0, dim_size = boundaryArea.shape[0], reduce= 'add')
                    simulationState['akinciBoundary']['boundarySource'] = 1. - simulationState['akinciBoundary']['boundaryDensity'] + boundarySource if density else boundarySource 
                else:
                    area        = simulationState['akinciBoundary']['boundaryVolume']
                    restDensity = simulationState['akinciBoundary']['boundaryRestDensity']
                    boundaryDensity     = simulationState['akinciBoundary']['boundaryDensity']
                    actualArea  = area / 1#boundaryDensity
                    
                    fac = - simulationState['dt'] * actualArea[bb]
                    boundarySource = scatter(fac * prod, bf, dim = 0, dim_size = simulationState['numParticles'], reduce= 'add')
                    
#                     fluidActualArea = simulationState['fluidActualArea']
                    source = source + boundarySource

                                
            return 1. - simulationState['fluidDensity'] + source if density else source            
        
        
    def computeUpdatedPressure(self, simulationState, simulation, density = True):
        with record_function("DFSPH - pressure"): 
            kernelSum = computeUpdatedPressureFluidSum(simulationState['fluidActualArea'], simulationState['fluidPredAccel'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])
            if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
                kernelSum += computeUpdatedPressureBoundarySum(simulationState['sdfBoundary']['fluidToGhostNeighbors'], simulationState['sdfBoundary']['ghostParticleGradientIntegral'], simulationState['fluidPredAccel'], simulationState['dt'])

            if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
                bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
                boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
                boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
                boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
                boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
                boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']
                boundaryPredictedVelocity = simulationState['akinciBoundary']['boundaryPredictedVelocity']
                
                grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)


                facFluid = simulationState['dt']**2 * simulationState['fluidActualArea'][bf]
                facBoundary = simulationState['dt']**2 * boundaryActualArea[bb]
                aij = simulationState['fluidPredAccel'][bf]

                if self.pressureScheme == 'PBSPH':
                    boundaryKernelSum = scatter_sum(torch.einsum('nd, nd -> n', facFluid[:,None] * aij, -grad), bb, dim=0, dim_size=boundaryArea.shape[0])

                    simulationState['akinciBoundary']['boundaryResidual'] = boundaryKernelSum - simulationState['akinciBoundary']['boundarySource']
                    boundaryPressure = simulationState['akinciBoundary']['boundaryPressure'] - self.relaxedJacobiOmega * simulationState['akinciBoundary']['boundaryResidual'] / simulationState['akinciBoundary']['boundaryAlpha']
                    boundaryPressure = torch.clamp(boundaryPressure, min = 0.) if density else boundaryPressure
                    if density and self.backgroundPressure:
                        boundaryPressure = torch.clamp(boundaryPressure, min = (5**2) * simulationState['akinciBoundary']['boundaryRestDensity'])
                    simulationState['akinciBoundary']['boundaryPressure'] = boundaryPressure
                    kernelSum += scatter_sum(torch.einsum('nd, nd -> n', -facBoundary[:,None] * aij, grad), bf, dim=0, dim_size=simulationState['fluidActualArea'].shape[0])
                else:
                    area        = simulationState['akinciBoundary']['boundaryVolume']
                    restDensity = simulationState['akinciBoundary']['boundaryRestDensity']
                    boundaryDensity     =  simulationState['akinciBoundary']['boundaryDensity']
                    actualArea  = area[bb] / 1 #simulationState['fluidDensity'][bf]
                    
                    fac = simulationState['dt']**2 * actualArea
                    boundarySum = scatter_sum(torch.einsum('nd, nd -> n', fac[:,None] * aij, -grad), bf, dim=0, dim_size=simulationState['fluidActualArea'].shape[0])
                    
                    kernelSum += boundarySum




            residual = kernelSum - simulationState['fluidSourceTerm']
            self.relaxedJacobiOmega
            pressure = simulationState['fluidPressure'] - 0.3 * residual / simulationState['fluidAlpha']
            pressure = torch.clamp(pressure, min = 0.) if density else pressure
            if density and self.backgroundPressure:
                pressure = torch.clamp(pressure, min = (5**2) * simulationState['fluidRestDensity'])
            if torch.any(torch.isnan(pressure)) or torch.any(torch.isinf(pressure)):
                raise Exception('Pressure solver became unstable!')

            return pressure, residual

        
        
    def computeAcceleration(self, simulationState, simulation, density = True):
        with record_function("DFSPH - accel"):
            fluidAccelTerm = computeFluidAcceleration(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], simulationState['fluidNeighbors'], simulationState['fluidDistances'], simulationState['fluidRadialDistances'], self.support)
            
            if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
                simulationState['sdfBoundary']['boundaryPressure'], boundaryAccelTerm, simulationState['sdfBoundary']['boundaryPressureForce'] = \
                    computeBoundaryAccelTerm(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], simulationState['sdfBoundary']['pgPartial'], \
                                             simulationState['sdfBoundary']['ghostToFluidNeighbors2'], simulationState['sdfBoundary']['fluidToGhostNeighbors'], simulationState['sdfBoundary']['ghostParticleBodyAssociation'], simulationState['sdfBoundary']['ghostParticleGradientIntegral'], \
                                             self.boundaryCounter, mlsPressure = self.pressureScheme == "deltaMLS", computeBodyForces = self.computeBodyForces)


                return fluidAccelTerm + boundaryAccelTerm

            if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
                bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
                boundaryDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborDistances']
                boundaryRadialDistances = simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances']
                boundaryActualArea = simulationState['akinciBoundary']['boundaryActualArea']
                boundaryArea = simulationState['akinciBoundary']['boundaryVolume']
                boundaryRestDensity = simulationState['akinciBoundary']['boundaryRestDensity']
                boundaryPredictedVelocity = simulationState['akinciBoundary']['boundaryPredictedVelocity']
                
                grad = -spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)

                fac = -(boundaryArea * boundaryRestDensity)[bb]
                
                pi = (simulationState['fluidPressure2'] / (simulationState['fluidDensity'] * simulationState['fluidRestDensity'])**2)[bf]

                if self.pressureScheme == "mirrored":
                    simulationState['akinciBoundary']['boundaryPressure2'] = 0
                    pb = simulationState['fluidPressure2'][bf]

                    
                if self.pressureScheme == "deltaMLS":
                    neighbors2 = simulationState['akinciBoundary']['boundaryToFluidNeighbors2']
                    
                    simulationState['akinciBoundary']['boundaryPressure2'] = scatter(simulationState['akinciBoundary']['pgPartial'] * simulationState['fluidPressure2'][neighbors2[1]], neighbors2[0], dim=0, dim_size = boundaryArea.shape[0], reduce='add')

                    simulationState['akinciBoundary']['boundaryGravity'] = torch.zeros_like(simulationState['akinciBoundary']['positions'])
                    simulationState['akinciBoundary']['boundaryGravity'][:,1] = -1

                    # simulationState['akinciBoundary']['boundaryPressure2'] += 2 * 2 * boundaryRestDensity * simulationState['akinciBoundary']['boundaryDensity'] * torch.einsum('nd, nd -> n', simulationState['akinciBoundary']['boundaryNormals'], simulationState['akinciBoundary']['boundaryGravity'])
                    simulationState['akinciBoundary']['boundaryPressure2'][:] = torch.clamp(simulationState['akinciBoundary']['boundaryPressure2'][:],min = 0)
                    simulationState['akinciBoundary']['boundaryPressure'][:] = simulationState['akinciBoundary']['boundaryPressure2'][:]
                    pb = simulationState['akinciBoundary']['boundaryPressure2'][bb]
                if self.pressureScheme == "MLSPressure":
                    pb = evalMLSBoundaries(simulationState['akinciBoundary']['M1'], simulationState['akinciBoundary']['vec'], simulationState['akinciBoundary']['bbar'], simulationState['akinciBoundary']['boundaryToFluidNeighbors'], simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances'], simulationState['fluidPosition'], simulationState['fluidActualArea'], simulationState['fluidPressure'], self.support)
                    pb = torch.clamp(pb,min = 0)
                    # simulationState['akinciBoundary']['boundaryGravity'] = torch.zeros_like(simulationState['akinciBoundary']['positions'])
                    # simulationState['akinciBoundary']['boundaryGravity'][:,1] = -1
                    # pb += 2 * 2 * boundaryRestDensity * simulationState['akinciBoundary']['boundaryDensity'] * torch.einsum('nd, nd -> n', simulationState['akinciBoundary']['boundaryNormals'], simulationState['akinciBoundary']['boundaryGravity'])
                    simulationState['akinciBoundary']['boundaryPressure'][:] = simulationState['akinciBoundary']['boundaryPressure2'][:] = pb

                    pb = simulationState['akinciBoundary']['boundaryPressure'][bb]


                if self.pressureScheme == "PBSPH":
                    simulationState['akinciBoundary']['boundaryPressure2'][:] = simulationState['akinciBoundary']['boundaryPressure'][:]
                    pb = simulationState['akinciBoundary']['boundaryPressure'][bb]

                    pb =  pb / ((simulationState['akinciBoundary']['boundaryDensity'][bb] * simulationState['akinciBoundary']['boundaryRestDensity'][bb])**2)
                    
                                    # pb =  pb / ((simulationState['akinciBoundary']['boundaryDensity'] * simulationState['akinciBoundary']['boundaryRestDensity'])**2)[bb]
                    term = (fac * (pi + pb))[:,None] * grad

                    # debugPrint(fac)
                    # debugPrint(pi)
                    # debugPrint(pb)
                    # debugPrint(term)
                    # debugPrint(grad)

                    if self.computeBodyForces:
                        force = -term * (simulationState['fluidArea'] * simulationState['fluidRestDensity'])[bf,None]
                        # simulationState['akinciBoundary']['bodyAssociation']
                        boundaryForces = scatter_sum(force, bb, dim=0, dim_size = simulationState['akinciBoundary']['boundaryDensity'].shape[0])
                        simulationState['akinciBoundary']['pressureForces'] = scatter_sum(boundaryForces, simulationState['akinciBoundary']['bodyAssociation'], dim=0, dim_size = self.boundaryCounter)

                    boundaryAccelTerm = scatter_sum(term, bf, dim=0, dim_size = simulationState['fluidArea'].shape[0])

                    
                else:
                    area        = simulationState['akinciBoundary']['boundaryVolume']
                    restDensity = simulationState['akinciBoundary']['boundaryRestDensity']
                    density     = simulationState['akinciBoundary']['boundaryDensity']
                    actualArea  = area / 1
                    
                    fac = -(area * restDensity)[bb]
                    pf = (simulationState['fluidPressure2'] / (simulationState['fluidDensity'] * simulationState['fluidRestDensity'])**2)[bf]
#                     pb = pb /  ((simulationState['fluidDensity'][bf] * restDensity[bb])**2)
                    pb = pb /  ((1 * restDensity[bb])**2)
                    
                    term = (fac * (pf + pb))[:,None] * grad
                    
                    boundaryAccelTerm = scatter_sum(term, bf, dim=0, dim_size=simulationState['fluidArea'].shape[0])


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
                        # debugPrint(self.boundaryScheme)
                        # debugPrint(self.pressureScheme)
                        if self.boundaryScheme == 'Akinci' and self.pressureScheme == 'PBSPH':
                            boundaryError = torch.sum(torch.clamp(simulationState['akinciBoundary']['boundaryResidual'], min = -self.densityThreshold))
                            fluidError = torch.sum(torch.clamp(simulationState['residual'], min = -self.densityThreshold))
                            error = (fluidError + boundaryError) / (simulationState['akinciBoundary']['boundaryResidual'].shape[0] + simulationState['residual'].shape[0])
                            # debugPrint(boundaryError)
                        else:
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
                if density and self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None: 
                    if self.pressureScheme == "deltaMLS":
                        support = simulation.config['particle']['support']
                        supports = torch.ones(simulationState['sdfBoundary']['ghostParticlePosition'].shape[0], device = simulationState['fluidActualArea'].device, dtype = simulationState['fluidActualArea'].dtype) * support

                        neighbors, distances, radialDistances = simulation.neighborSearch.searchExisting(simulationState['sdfBoundary']['ghostParticlePosition'], supports * 2, simulationState, simulation, searchRadius = 2)
                        simulationState['sdfBoundary']['ghostToFluidNeighbors2'] = neighbors
                        simulationState['sdfBoundary']['pgPartial'] = precomputeMLS(simulationState['sdfBoundary']['ghostParticlePosition'], simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, neighbors, radialDistances)

                if density and self.boundaryScheme == 'Akinci' and 'boundaryToFluidNeighbors' in simulationState['akinciBoundary'] and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
                    simulationState['akinciBoundary']['boundaryPressure'] = torch.zeros_like(simulationState['akinciBoundary']['boundaryVolume'])
                    simulationState['akinciBoundary']['boundaryPressure2'] = torch.zeros_like(simulationState['akinciBoundary']['boundaryVolume'])
                    simulationState['akinciBoundary']['boundaryActualArea'] = simulationState['akinciBoundary']['boundaryVolume'] / simulationState['akinciBoundary']['boundaryDensity']
                    simulationState['akinciBoundary']['fluidPredAccel'] = torch.zeros(simulationState['akinciBoundary']['positions'].shape, dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)
                    simulationState['akinciBoundary']['boundaryPredictedVelocity'] = simulationState['akinciBoundary']['boundaryVelocity'] + simulationState['dt'] * simulationState['akinciBoundary']['boundaryAcceleration']
                    if self.pressureScheme == "deltaMLS":
                        neighbors, distances, radDistances = simulation.neighborSearch.searchExisting(simulationState['akinciBoundary']['positions'], simulationState['akinciBoundary']['boundarySupport'] * 2, simulationState, simulation, searchRadius = 2)
                        simulationState['akinciBoundary']['boundaryToFluidNeighbors2'] = neighbors  
        # neighbors = radius(fluidPosition, queryPositions, support, max_num_neighbors = 256)
                        simulationState['akinciBoundary']['pgPartial'] = precomputeMLS(simulationState['akinciBoundary']['positions'], simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, neighbors, radDistances)

                    if self.pressureScheme == "MLSPressure":
                        simulationState['akinciBoundary']['M1'], simulationState['akinciBoundary']['vec'], simulationState['akinciBoundary']['bbar'] = prepareMLSBoundaries(simulationState['akinciBoundary']['positions'], simulationState['akinciBoundary']['boundarySupport'], simulationState['akinciBoundary']['boundaryToFluidNeighbors'], simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances'], simulationState['fluidPosition'], simulationState['fluidActualArea'], self.support)

        #                 neighbors, distances, radDistances = simulation.neighborSearch.searchExisting(simulationState['akinciBoundary']['positions'], simulationState['akinciBoundary']['boundarySupport'] * 2, simulationState, simulation, searchRadius = 2)
        #                 simulationState['akinciBoundary']['boundaryToFluidNeighbors2'] = neighbors  
        # # neighbors = radius(fluidPosition, queryPositions, support, max_num_neighbors = 256)
        #                 simulationState['akinciBoundary']['pgPartial'] = precomputeMLS(simulationState['akinciBoundary']['positions'], simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, neighbors, radDistances)
        

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
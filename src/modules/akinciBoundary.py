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

from ..ghostParticles import *


class akinciBoundaryModule(Module):


    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
        
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.active = True if 'solidBC' in simulationConfig else False
        self.maxNeighbors = simulationConfig['compute']['maxNeighbors']
        if not self.active:
            return
        self.numBodies = len(simulationConfig['solidBC'])
        self.boundaryObjects = simulationConfig['solidBC']
        simulationState['akinciBoundary'] = {}
        simulationState['akinciBoundary']['bodies'] =  simulationConfig['solidBC']
        # self.kernel, _ = getKernelFunctions(simulationConfig['kernel']['defaultKernel'])
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']  
        
        self.domainMin = torch.tensor(simulationConfig['domain']['min'], device = self.device)
        self.domainMax = torch.tensor(simulationConfig['domain']['max'], device = self.device)
        
        bptcls = []
        for b in simulationState['akinciBoundary']['bodies']:
            bdy = simulationState['akinciBoundary']['bodies'][b]
            packing = simulationConfig['particle']['packing'] * simulationConfig['particle']['support']
            ptcls,_ = samplePolygon(bdy['polygon'], packing, simulationConfig['particle']['support'], offset = packing / 2 if bdy['inverted'] else -packing /2)
            # debugPrint(ptcls)
            bptcls.append(torch.tensor(ptcls).type(self.dtype).to(self.device))
        # debugPrint(bptcls)
        bptcls = torch.cat(bptcls)
        simulationState['akinciBoundary']['positions'] = bptcls

        # simulationState['akinciBoundary']['positions']
        boundaryPositions = simulationState['akinciBoundary']['positions']

        bj, bi = radius(boundaryPositions, boundaryPositions, self.support)

        bbDistances = (boundaryPositions[bi] - boundaryPositions[bj])
        bbRadialDistances = torch.linalg.norm(bbDistances,axis=1)

        # fluidDistances[fluidRadialDistances < self.threshold,:] = 0
        # fluidDistances[fluidRadialDistances >= self.threshold,:] /= fluidRadialDistances[fluidRadialDistances >= self.threshold,None]
        bbRadialDistances /= self.support

        boundaryKernelTerm = kernel(bbRadialDistances, self.support)

        boundaryVolume = scatter(boundaryKernelTerm, bi, dim=0, dim_size = boundaryPositions.shape[0], reduce='add')
        boundaryDensity = scatter(boundaryKernelTerm * 0.6 / boundaryVolume[bj], bi, dim=0, dim_size = boundaryPositions.shape[0], reduce='add')

        gamma = 0.7
        simulationState['akinciBoundary']['boundaryDensityTerm'] = (boundaryDensity).type(self.dtype)
        simulationState['akinciBoundary']['boundaryVolume'] = 0.6 / boundaryVolume
        simulationState['akinciBoundary']['boundarySupport'] = torch.ones_like(boundaryVolume) * self.support
        simulationState['akinciBoundary']['boundaryRestDensity'] = torch.ones_like(boundaryVolume) * simulationConfig['fluid']['restDensity'] 
        simulationState['akinciBoundary']['boundaryVelocity'] = torch.zeros_like(boundaryPositions) 
        simulationState['akinciBoundary']['boundaryAcceleration'] = torch.zeros_like(boundaryPositions) 

        # compute Vb0 
        # add neighbor boundary search both ways
        # add density contribution
        # add code for boundary density (apparent volume denominator)
        # rest is done in dfsph and other codes

#         print(self.numBodies)
        
#         self.periodicX = simulationConfig['periodicBC']['periodicX']
#         self.periodicY = simulationConfig['periodicBC']['periodicY']
#         self.buffer = simulationConfig['periodicBC']['buffer']
#         self.domainMin = simulationConfig['domain']['virtualMin']
#         self.domainMax = simulationConfig['domain']['virtualMax']
#         self.dtype = simulationConfig['compute']['precision']
        
        
    def search(self, simulationState, simulation):
        if not self.active:
            return None, None, None
        neighbors, distances, radDistances = simulation.neighborSearch.searchExisting(simulationState['akinciBoundary']['positions'], simulationState['akinciBoundary']['boundarySupport'], simulationState, simulation)
        return neighbors, distances, radDistances
    

    def density(self, simulationState, simulation):
        density = torch.zeros(simulationState['fluidDensity'].shape, device=simulation.device, dtype= simulation.dtype)
        gradient = torch.zeros(simulationState['akinciBoundary']['positions'].shape, device=simulation.device, dtype= simulation.dtype)
        if 'akinciBoundary' in simulationState and simulationState['akinciBoundary']['boundaryToFluidNeighbors'] != None:
            bb,bf = simulationState['akinciBoundary']['boundaryToFluidNeighbors']
            k = kernel(simulationState['akinciBoundary']['boundaryToFluidNeighborRadialDistances'], self.support)

            boundaryDensityContribution = scatter(k * simulationState['akinciBoundary']['boundaryVolume'][bb], bf, dim=0, dim_size = simulationState['numParticles'], reduce = 'add')
            boundaryDensity = simulationState['akinciBoundary']['boundaryDensityTerm'] + scatter(k * simulationState['fluidArea'][bf], bb, dim=0, dim_size = simulationState['akinciBoundary']['positions'].shape[0], reduce = 'add') + 0.1

            return boundaryDensityContribution, boundaryDensity
            
        return density, gradient


# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)

# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# # solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
# # sphSimulation.simulationState['fluidToGhostNeighbors'], sphSimulation.simulationState['ghostToFluidNeighbors'], sphSimulation.simulationState['ghostParticleBodyAssociation'], \
# #     sphSimulation.simulationState['ghostParticlePosition'], sphSimulation.simulationState['ghostParticleDistance'], sphSimulation.simulationState['ghostParticleGradient'], \
# #     sphSimulation.simulationState['ghostParticleKernelIntegral'], sphSimulation.simulationState['ghostParticleGradientIntegral'] = solidBC.search(sphSimulation.simulationState, sphSimulation)
# solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
    
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
# #             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
#             solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")

# simulationState = sphSimulation.simulationState
# b = simulationState['solidBC']['domainBoundary']
# # polyDist, polyDer2 = domainDistanceAndDer(sphSimulation.simulationState['fluidPosition'], torch.Tensor(sphSimulation.config['domain']['min']), torch.Tensor(sphSimulation.config['domain']['max']))
# polyDist, polyDer, polyInt, polyGrad = sdPolyDerAndIntegral(b['polygon'], simulationState['fluidPosition'], solidBC.support, inverted = b['inverted'])
# polyDist2, polyDer2, polyInt2, polyGrad2 = domainDistanceDerAndIntegral(simulationState['fluidPosition'], b['polygon'], solidBC.support)
    

# fig, axis = plt.subplots(2,2, figsize=(9 *  1.09, 9), squeeze = False)
# for axx in axis:
#     for ax in axx:
#         ax.set_xlim(sphSimulation.config['domain']['virtualMin'][0], sphSimulation.config['domain']['virtualMax'][0])
#         ax.set_ylim(sphSimulation.config['domain']['virtualMin'][1], sphSimulation.config['domain']['virtualMax'][1])
#         ax.axis('equal')
#         ax.axvline(sphSimulation.config['domain']['min'][0], ls= '--', c = 'black')
#         ax.axvline(sphSimulation.config['domain']['max'][0], ls= '--', c = 'black')
#         ax.axhline(sphSimulation.config['domain']['min'][1], ls= '--', c = 'black')
#         ax.axhline(sphSimulation.config['domain']['max'][1], ls= '--', c = 'black')



# state = sphSimulation.simulationState

# positions = state['fluidPosition'].detach().cpu().numpy()
# # data = torch.linalg.norm(state['fluidUpdate'].detach(),axis=1).cpu().numpy()
# data = polyDer[:,0].detach().cpu().numpy()


# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = polyGrad[:,0].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 
# sc = axis[0,1].scatter(positions[:,0], positions[:,1], c = polyGrad[:,1].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[0,1])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 
# # axis[0,0].axis('equal')

# sc = axis[1,0].scatter(positions[:,0], positions[:,1], c = polyGrad2[:,0].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[1,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 
# sc = axis[1,1].scatter(positions[:,0], positions[:,1], c = polyGrad2[:,1].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[1,1])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()

# simulationState = sphSimulation.simulationState

# b = simulationState['solidBC']['domainBoundary']
# polyDist, polyDer, _, _, _, _ = sdPolyDer(b['polygon'], simulationState['fluidPosition'], inverted = b['inverted'])

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("old"): 
#             sdPolyDerAndIntegral(b['polygon'], simulationState['fluidPosition'], solidBC.support, inverted = b['inverted'])
#     for i in range(16):
#         with record_function("new"): 
#             sdPolyDerAndIntegral2(b['polygon'], simulationState['fluidPosition'], solidBC.support, inverted = b['inverted'])
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# # prof.export_chrome_trace("trace.json")

# fig, axis = plt.subplots(1,1, figsize=(9 *  1.09, 3), squeeze = False)

# x = torch.linspace(-1,1,1023)
# xt = torch.linspace(-1,1,1023)

# # axis[0,0].plot(xt,boundaryKernelAnalytic(xt,xt))
# axis[0,0].plot(x,gradK(x), label = 'gradient')
# axis[0,0].plot(x,numGradK(x,1e-2), label = 'numerical Gradient')
# axis[0,0].plot(x,k(x), label = 'kernel')
# axis[0,0].grid(True)

# axis[0,0].legend()

# fig.tight_layout()

# x = torch.linspace(-1,1,1023 * 512)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
# #     for i in range(16):
# #         with record_function("timing old"): 
# #             _ = boundaryKernelAnalytic(xt,xt)
#     for i in range(16):
#         with record_function("timing kernel"): 
#             _ = k(x)
#     for i in range(16):
#         with record_function("timing gradient"): 
#             _ = gradK(x)
#     for i in range(16):
#         with record_function("timing num gradient"): 
#             _ = numGradK(x)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# # prof.export_chrome_trace("trace.json")

# @torch.jit.script
# def sdPoly(poly, p):    
#     with record_function("sdPoly"): 
#         N = len(poly)

#         i = torch.arange(N, device = p.device, dtype = torch.int64)
#         i2 = (i + 1) % N
#         e = poly[i2] - poly[i]
#         v = p - poly[i][:,None]

#         ve = torch.einsum('npd, nd -> np', v, e)
#         ee = torch.einsum('nd, nd -> n', e, e)

#         pq = v - e[:,None] * torch.clamp(ve / ee[:,None], min = 0, max = 1)[:,:,None]

#         d = torch.einsum('npd, npd -> np', pq, pq)
#         d = torch.min(d, dim = 0).values

#         wn = torch.zeros((N, p.shape[0]), device = p.device, dtype = torch.int64)

#         cond1 = 0 <= v[i,:,1]
#         cond2 = 0 >  v[i2,:,1]
#         val3 = e[i,0,None] * v[i,:,1] - e[i,1,None] * v[i,:,0]

#         c1c2 = torch.logical_and(cond1, cond2)
#         nc1nc2 = torch.logical_and(torch.logical_not(cond1), torch.logical_not(cond2))

#         wn[torch.logical_and(c1c2, val3 > 0)] += 1
#         wn[torch.logical_and(nc1nc2, val3 < 0)] -= 1

#         wn = torch.sum(wn,dim=0)
#         s = torch.ones(p.shape[0], device = p.device, dtype = p.dtype)
#         s[wn != 0] = -1

#         return s * torch.sqrt(d)

# @torch.jit.script
# def boundaryKernelAnalytic(dr : torch.Tensor , q : torch.Tensor):
#     with record_function("boundaryKernelAnalytic"): 
#         d = dr + 0j
#         a = torch.zeros(d.shape, device = q.device, dtype=d.dtype)
#         b = torch.zeros(d.shape, device = q.device, dtype=d.dtype)

        
#         mask = torch.abs(d.real) > 1e-3
#         dno = d[mask]


#         a[mask] += ( 12 * dno**5 + 80 * dno**3) * torch.log(torch.sqrt(1 - dno**2) + 1)
#         a[mask] += (-12 * dno**5 - 80 * dno**3) * torch.log(1 - torch.sqrt(1 - dno**2))
#         a[mask] += (-12 * dno**5 - 80 * dno**3) * torch.log(torch.sqrt(1 - 4 * dno**2) + 1)
#         a[mask] += ( 12 * dno**5 + 80 * dno**3) * torch.log(1 - torch.sqrt(1 - 4 * dno**2))
#         a += -13 * torch.acos(2 * d)
#         a +=  16 * torch.acos(d)
#         a += torch.sqrt(1 - 4 * d**2) * (74 * d**3 + 49 * d)
#         a += torch.sqrt(1 - d **2) * (-136 * d**3 - 64 * d)


#         b += -36 * d**5 * torch.log(torch.sqrt(1-4 * d**2) + 1)
#         b[mask] += 36 * dno**5 * torch.log(1-torch.sqrt(1-4*dno**2))
#         b += 11 * torch.acos(2 * d)
# #         b += -36 * np.log(-1 + 0j) * d**5
#         b += -160j * d**4
#         b += torch.sqrt(1 -4 *d**2)*(62 *d**3 - 33*d)
#         b += 80j * d**2
#         res = (a + b) / (14 * np.pi)

#         gammaScale = 2.0
#         gamma = 1
# #         gamma = 1 + (1 - q / 2) ** gammaScale
#         # gamma = torch.log( 1 + torch.exp(gammaScale * q)) - np.log(1 + np.exp(-gammaScale) / np.log(2))

#         return res.real * gamma

# @torch.jit.script
# def sdPolyDer(poly, p, dh :float = 1e-4, inverted :bool = False):
#     with record_function("sdPolyDer"): 
#         dh = 1e-4
#         dpx = torch.zeros_like(p)
#         dnx = torch.zeros_like(p)
#         dpy = torch.zeros_like(p)
#         dny = torch.zeros_like(p)

#         dpx[:,0] += dh
#         dnx[:,0] -= dh
#         dpy[:,1] += dh
#         dny[:,1] -= dh

#         c = sdPoly(poly, p)
#         cpx = sdPoly(poly, p + dpx)
#         cnx = sdPoly(poly, p + dnx)
#         cpy = sdPoly(poly, p + dpy)
#         cny = sdPoly(poly, p + dny)

#         if inverted:
#             c = -c
#             cpx = -cpx
#             cnx = -cnx
#             cpy = -cpy
#             cny = -cny

#         grad = torch.zeros_like(p)
#         grad[:,0] = (cpx - cnx) / (2 * dh)
#         grad[:,1] = (cpy - cny) / (2 * dh)

#         gradLen = torch.linalg.norm(grad, dim =1)
#         grad[torch.abs(gradLen) > 1e-5] /= gradLen[torch.abs(gradLen)>1e-5,None]

#         return c, grad, cpx, cnx, cpy, cny

# @torch.jit.script
# def boundaryIntegralAndDer(poly, p, support : float, c, cpx, cnx, cpy, cny, dh : float = 1e-4):
#     k = boundaryKernelAnalytic(torch.clamp(c / support, min = -1, max = 1), c / support)   
#     kpx = boundaryKernelAnalytic(torch.clamp(cpx / support, min = -1, max = 1), c / support)
#     knx = boundaryKernelAnalytic(torch.clamp(cnx / support, min = -1, max = 1), c / support)  
#     kpy = boundaryKernelAnalytic(torch.clamp(cpy / support, min = -1, max = 1), c / support)  
#     kny = boundaryKernelAnalytic(torch.clamp(cny / support, min = -1, max = 1), c / support)   
        
#     kgrad = torch.zeros_like(p)
#     kgrad[:,0] = (kpx - knx) / (2 * dh)
#     kgrad[:,1] = (kpy - kny) / (2 * dh)
    
#     return k, kgrad
    
# @torch.jit.script
# def sdPolyDerAndIntegral(poly, p, support : float, masked : bool = False, inverted : bool = False):     
#     c, grad, cpx, cnx, cpy, cny = sdPolyDer(poly, p, dh = 1e-4, inverted = inverted)
#     k, kgrad = boundaryIntegralAndDer(poly, p, support, c, cpx, cnx, cpy, cny, dh = 1e-4)  
    
    
#     return c, grad, k, kgrad


# poly = simulationState['solidBC']['domainBoundary']['polygon']
# inverted = simulationState['solidBC']['domainBoundary']['inverted']
# p = simulationState['fluidPosition']
# support = solidBC.support

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
#             sdPolyDerAndIntegral(poly, p, support, inverted = inverted)
# #             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")
# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
# sphSimulation.simulationState['fluidToGhostNeighbors'], sphSimulation.simulationState['ghostToFluidNeighbors'], sphSimulation.simulationState['ghostParticleBodyAssociation'], \
#     sphSimulation.simulationState['ghostParticlePosition'], sphSimulation.simulationState['ghostParticleDistance'], sphSimulation.simulationState['ghostParticleGradient'], \
#     sphSimulation.simulationState['ghostParticleKernelIntegral'], sphSimulation.simulationState['ghostParticleGradientIntegral'] = solidBC.search(sphSimulation.simulationState, sphSimulation)
# solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
#             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")
# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
# sphSimulation.simulationState['fluidToGhostNeighbors'], sphSimulation.simulationState['ghostToFluidNeighbors'], sphSimulation.simulationState['ghostParticleBodyAssociation'], \
#     sphSimulation.simulationState['ghostParticlePosition'], sphSimulation.simulationState['ghostParticleDistance'], sphSimulation.simulationState['ghostParticleGradient'], \
#     sphSimulation.simulationState['ghostParticleKernelIntegral'], sphSimulation.simulationState['ghostParticleGradientIntegral'] = solidBC.search(sphSimulation.simulationState, sphSimulation)
# solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
    
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
#             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")
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

from typing import Dict, Optional

@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

from scipy.optimize import minimize

def genParticlesCentered(minCoord, maxCoord, radius, support, packing, dtype = torch.float32, device = 'cpu'):
    area = np.pi * radius**2
    
    gen_position = lambda r, i, j: torch.tensor([r * i, r * j], dtype=dtype, device = device)
        
    diff = maxCoord - minCoord
    center = (minCoord + maxCoord) / 2
    requiredSlices = torch.div(torch.ceil(diff / packing / support).type(torch.int64), 2, rounding_mode='floor')
    
    generatedParticles = []
#     print(requiredSlices)
    for i in range(-requiredSlices[0]-1, requiredSlices[0]+2):
        for j in range(-requiredSlices[1]-1, requiredSlices[1]+2):
            p = center
            g = gen_position(packing * support,i,j)
            pos = p + g
            if pos[0] <= maxCoord[0] + support * 0.2 and pos[1] <= maxCoord[1] + support * 0.2 and \
             pos[0] >= minCoord[0] - support * 0.2 and pos[1] >= minCoord[1] - support * 0.2:
                generatedParticles.append(pos)
                
    return torch.stack(generatedParticles)

def genParticles(minCoord, maxCoord, radius, packing, support, dtype, device):
    with record_function('config - gen particles'):
        area = np.pi * radius**2
#         support = np.sqrt(area * config['targetNeighbors'] / np.pi)
        
        gen_position = lambda r, i, j: torch.tensor([r * i, r * j], dtype=dtype, device = device)
        
    #     packing *= support
        
        diff = maxCoord - minCoord
        requiredSlices = torch.ceil(diff / packing / support).type(torch.int64)
        
    #     print(requiredSlices)
        generatedParticles = []
        for i in range(requiredSlices[0]+1):
            for j in range(requiredSlices[1]+1):
                p = minCoord
                g = gen_position(packing * support,i,j)
                pos = p + g
                if pos[0] <= maxCoord[0] + support * 0.2 and pos[1] <= maxCoord[1] + support * 0.2:
                    generatedParticles.append(pos)
        particles = torch.stack(generatedParticles)

        return particles



# def mlsInterpolation(simulationState, simulation, queryPositions, support):
#     with record_function("mls interpolation"): 
#         # queryPositions = simulationState['fluidPosition']
#         # queryPosition = pb
#         # support = simulation.config['particle']['support'] * 2

#         i, j = radius(simulationState['fluidPosition'], queryPositions, support, max_num_neighbors = 256)
#         neighbors = torch.stack([i, j], dim = 0)

#     #     debugPrint(neighbors)
#         # debugPrint(torch.min(neighbors[0]))
#         # debugPrint(torch.max(neighbors[0]))
#         # debugPrint(torch.min(neighbors[1]))
#         # debugPrint(torch.max(neighbors[1]))

#         distances = (simulationState['fluidPosition'][j] - queryPositions[i])
#         radialDistances = torch.linalg.norm(distances,axis=1)

#         distances[radialDistances < 1e-5,:] = 0
#         distances[radialDistances >= 1e-5,:] /= radialDistances[radialDistances >= 1e-5,None]
#         radialDistances /= support

#         kernel = wendland(radialDistances, support)

#         bij = simulationState['fluidPosition'][j] - queryPositions[i]
#         bij = torch.hstack((bij.new_ones((bij.shape[0]))[:,None], bij))
#     #     debugPrint(bij)

#         Mpartial = 2 * torch.einsum('nu, nv -> nuv', bij, bij) * \
#                 (simulationState['fluidArea'][j] / simulationState['fluidDensity'][j] * kernel)[:,None,None]

#         M = scatter(Mpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
#         Minv = torch.linalg.pinv(M)
#     #     debugPrint(Minv)

#         e1 = torch.tensor([1,0,0], dtype=Minv.dtype, device=Minv.device)
#         Me1 = torch.matmul(Minv,e1)
#     #     debugPrint(Me1)


#         pGpartial = torch.einsum('nd, nd -> n', Me1[i], bij) * \
#             kernel * simulationState['fluidPressure2'][j] * (simulationState['fluidArea'][j] / simulationState['fluidDensity'][j])

#         pG = scatter(pGpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
#     #     debugPrint(pG)

#         return pG

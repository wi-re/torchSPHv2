import numpy as np
from scipy.optimize import minimize
import torch
from torch_geometric.nn import radius
from torch_scatter import scatter
from torch.profiler import record_function

from . import kernel
from .solidBC import *


def genParticlesCentered(minCoord, maxCoord, radius, packing, config):
    area = np.pi * radius**2
    support = np.sqrt(area * config['targetNeighbors'] / np.pi)
    
    gen_position = lambda r, i, j: torch.tensor([r * i, r * j], dtype=config['precision'], device = config['device'])
        
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
def genParticles(minCoord, maxCoord, radius, packing, config):
    with record_function('config - gen particles'):
        area = np.pi * radius**2
        support = np.sqrt(area * config['targetNeighbors'] / np.pi)
        
        gen_position = lambda r, i, j: torch.tensor([r * i, r * j], dtype=config['precision'], device = config['device'])
        
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
        if 'solidBoundary' in config:
            for b in config['solidBoundary']:
                polyDist, polyDer, bIntegral, bGrad = sdPolyDerAndIntegral(b['polygon'], particles, config['support'], inverted = b['inverted'])
                # print('Particle count before filtering: ', particles.shape[0])
                particles = particles[polyDist >= config['spacing'] * config['support'] * 0.99,:]
                # print('Particle count after filtering: ', particles.shape[0])


        return particles

def evalPacking(arg, config):
    packing = torch.tensor(arg, dtype=config['precision'], device = config['device'])
    
    minDomain = torch.tensor([-2 * config['support'],-2 * config['support']], device = config['device'], dtype=config['precision'])
    maxDomain = torch.tensor([ 2 * config['support'], 2 * config['support']], device = config['device'], dtype=config['precision'])
    
    fluidPosition = genParticlesCentered(minDomain, maxDomain, config['radius'], packing, config)
    
    fluidArea = torch.ones(fluidPosition.shape[0], device = config['device'], dtype=config['precision']) * config['area']
    centralPosition = torch.tensor([[0,0]], device = config['device'], dtype=config['precision'])

    row, col = radius(centralPosition, fluidPosition, config['support'], max_num_neighbors = config['max_neighbors'])
    fluidNeighbors = torch.stack([row, col], dim = 0)
        
#     print(fluidNeighbors[0])
#     print(fluidNeighbors[1])
        
    fluidDistances = (centralPosition - fluidPosition[fluidNeighbors[0]])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)
    
    fluidRadialDistances /= config['support']
    rho = scatter(kernel.wendland(fluidRadialDistances, config['support']) * fluidArea[fluidNeighbors[1]], fluidNeighbors[1], dim=0, dim_size=centralPosition.size(0), reduce="add")
    
    return ((1 - rho)**2).detach().cpu().numpy()[0]

def evalSpacing(arg, config):
    s = torch.tensor(arg, dtype=config['precision'], device = config['device'])
    
    fluidPosition = genParticlesCentered(\
            torch.tensor([-config['support'] * 2,- config['support'] * 2], dtype=config['precision'], device=config['device']),\
            torch.tensor([config['support'] * 2,config['support'] * 2], dtype=config['precision'], device=config['device']), \
            config['radius'],config['packing'], config)
    fluidPosition = fluidPosition[fluidPosition[:,1] >= 0,:]
    centralPosition = torch.tensor([[0,0]], device = config['device'], dtype=config['precision'])

    row, col = radius(centralPosition, fluidPosition, config['support'], max_num_neighbors = config['max_neighbors'])
    fluidNeighbors = torch.stack([row, col], dim = 0)

    fluidDistances = (centralPosition - fluidPosition[fluidNeighbors[0]])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidRadialDistances /= config['support']
    rho = scatter(kernel.wendland(fluidRadialDistances, config['support']) * config['area'], fluidNeighbors[1], dim=0, dim_size=centralPosition.size(0), reduce="add")

    sdf, sdfGrad, b, bGrad = sdPolyDerAndIntegral(\
            torch.tensor([\
                [ -config['support'] * 2, -config['support'] * 2],\
                [  config['support'] * 2, -config['support'] * 2],\
                [  config['support'] * 2,  s * config['support']],\
                [ -config['support'] * 2,  s * config['support']],\
                         ], dtype= config['precision'], device = config['device']),\
            p = centralPosition, support = config['support']
    )

    return ((1- (rho + b))**2).detach().cpu().numpy()[0]



def evalSpacing(arg, config):
    s = torch.tensor(arg, dtype=config['precision'], device = config['device'])
    
    fluidPosition = genParticlesCentered(\
            torch.tensor([-config['support'] * 2,- config['support'] * 2], dtype=config['precision'], device=config['device']),\
            torch.tensor([config['support'] * 2,config['support'] * 2], dtype=config['precision'], device=config['device']), \
            config['radius'],config['packing'], config)
    fluidPosition = fluidPosition[fluidPosition[:,1] >= 0,:]
    centralPosition = torch.tensor([[0,0]], device = config['device'], dtype=config['precision'])

    row, col = radius(centralPosition, fluidPosition, config['support'], max_num_neighbors = config['max_neighbors'])
    fluidNeighbors = torch.stack([row, col], dim = 0)

    fluidDistances = (centralPosition - fluidPosition[fluidNeighbors[0]])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidRadialDistances /= config['support']
    rho = scatter(kernel.wendland(fluidRadialDistances, config['support']) * config['area'], fluidNeighbors[1], dim=0, dim_size=centralPosition.size(0), reduce="add")

    sdf, sdfGrad, b, bGrad = sdPolyDerAndIntegral(\
            torch.tensor([\
                [ -config['support'] * 2, -config['support'] * 2],\
                [  config['support'] * 2, -config['support'] * 2],\
                [  config['support'] * 2,  s * config['support']],\
                [ -config['support'] * 2,  s * config['support']],\
                         ], dtype= config['precision'], device = config['device']),\
            p = centralPosition, support = config['support']
    )

    return ((1- (rho + b))**2).detach().cpu().numpy()[0]


def evalContrib(arg, config):
    s = torch.tensor(arg, dtype=config['precision'], device = config['device'])
    centralPosition = torch.tensor([[0,0]], device = config['device'], dtype=config['precision'])

    sdf, sdfGrad, b, bGrad = sdPolyDerAndIntegral(\
            torch.tensor([\
                [ -config['support'] * 2, -config['support'] * 2],\
                [  config['support'] * 2, -config['support'] * 2],\
                [  config['support'] * 2,  s * config['support']],\
                [ -config['support'] * 2,  s * config['support']],\
                         ], dtype= config['precision'], device = config['device']),\
            p = centralPosition, support = config['support']
    )

    return b
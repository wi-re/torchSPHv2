import torch
from torch_geometric.nn import radius
from torch.profiler import record_function
from .solidBC import *

def neighborSearch(positionsA, positionsB, config, simulationState):
    with record_function("sph - neighborhood"): 
        row, col = radius(positionsA, positionsB, config['support'], max_num_neighbors = config['max_neighbors'])
        fluidNeighbors = torch.stack([row, col], dim = 0)
        
        fluidDistances = (positionsA[fluidNeighbors[1]] - positionsB[fluidNeighbors[0]])
        fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)
        
        fluidDistances[fluidRadialDistances < 1e-7,:] = 0
        fluidDistances[fluidRadialDistances >= 1e-7,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-7,None]
        fluidRadialDistances /= config['support']
        
        if('solidBoundary' in config):
            for ib, b in enumerate(config['solidBoundary']):
                i = fluidNeighbors[1]
                j = fluidNeighbors[0]
                
                polyDist, polyDer, _, _, _, _ = sdPolyDer(b['polygon'], simulationState['fluidPosition'], inverted = b['inverted'])
                cp = simulationState['fluidPosition'] - polyDist[:,None] * polyDer
                d = torch.einsum('nd,nd->n', polyDer, cp)
                neighDistances = torch.einsum('nd,nd->n', simulationState['fluidPosition'][j], polyDer[i]) - d[i]
                
                i = i[neighDistances >= 0]
                j = j[neighDistances >= 0]
                
                fluidNeighbors = torch.vstack((j,i))
                fluidDistances = fluidDistances[neighDistances >= 0]
                fluidRadialDistances = fluidRadialDistances[neighDistances >= 0]

        return fluidNeighbors, fluidDistances, fluidRadialDistances
    
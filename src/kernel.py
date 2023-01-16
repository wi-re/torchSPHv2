import torch
import numpy as np
from torch_scatter import scatter
from .periodicBC import *

def wendland(q, support):
    C = 7 / np.pi
#     print(q)
    
    b1 = torch.pow(1. - q, 4)
#     print(b1)
    b2 = 1.0 + 4.0 * q
#     print(b2)
    return b1 * b2 * C / support**2    

def wendlandGrad(q,r,support):
    C = 7 / np.pi
    
    return - r * C / support**3 * (20. * q * (1. -q)**3)[:,None]
    
@torch.jit.script
def w2(neighbors, fluidRadialDistances, numParticles, support):
    i = neighbors[1]
    j = neighbors[0]
    
    k  = wendland(fluidRadialDistances, support)
    return k

@torch.jit.script
def firstDerivative(d, r, neighbors, fluidPosition, numParticles, support):
#     print(neighbors.shape)
    i = neighbors[1]
    j = neighbors[0]
    
    
    a = fluidPosition[i]
    b = fluidPosition[j]
    
    df = d.flatten().reshape(d.shape[0] * d.shape[1])
    rf = r.flatten()
    qf = rf / support
    
    x_a = a[:,0]
    x = b[:,0]
    y_a = a[:,1]
    y = b[:,1]
    
    l = torch.sqrt((x_a - x)**2 + (y_a - y)**2).flatten()
    q = l / support
    dx = (x_a - x).flatten()
    dy = (y_a - y).flatten()
    
    df = torch.zeros((dx.shape[0],2), dtype=dx.dtype, device = dx.device)
#     print(a)
#     print('x_a', x_a.shape)
#     print('x', x.shape)
#     print('dx', dx.shape)
#     print('dy', dy.shape)
#     print('l', l.shape)
#     print('df',df.shape)

    c = 7. / np.pi
    h = support
    
    df[:,0] = -(20 * c * (h **(-7))* (dx) *pow(l -h, 3)).flatten()
    df[:,1] = -(20 * c * (h **(-7))* (dy) *pow(l -h, 3)).flatten()
    
    df[q>1,0] = 0
    df[q>1,1] = 0
    return df

@torch.jit.script
def secondDerivative(d, r, neighbors, fluidPosition, numParticles, support):
#     print(neighbors.shape)
    i = neighbors[1]
    j = neighbors[0]
    
    a = fluidPosition[i]
    b = fluidPosition[j]
    
    df = d.flatten().reshape(d.shape[0] * d.shape[1])
    rf = r.flatten()
    qf = rf / support
    
    x_a = a[:,0]
    x = b[:,0]
    y_a = a[:,1]
    y = b[:,1]
    
    
#     df[:,0,0] = 1.
    
    c = 7. / np.pi
    h = support
    
    l = torch.sqrt((x_a - x)**2 + (y_a - y)**2).flatten()
    q = l / support
    dx = (x_a - x).flatten()
    dy = (y_a - y).flatten()
    
    df = torch.zeros((dx.shape[0],2,2), device = dx.device, dtype=dx.dtype)
    
    
    df[:,0,0] = 20 * c / h**7 * (l - h)**3
    
    df[l>0,0,0] -= (60 * c / h**7 * (l-h)**2 * dx * -dx)[l>0] / l[l>0]
    
    df[l>0,0,1] = - (60 * c / h**7 * -dx*dy * (l - h)**2)[l>0] / l[l>0]
    df[:,1,0] = df[:,0,1]
    
    df[:,1,1] = 20 * c / h**7 * (l - h)**3
    df[l>0,1,1] -= (60 * c / h**7 * (l-h)**2 * -dy * dy)[l>0] / l[l>0]
    
    df[q>1,:,:] = 0.
    
    return df

@torch.jit.script
def evalKernel(fluidOmegas, fluidPosition, fluidNeighbors, fluidDistances, fluidRadialDistances, numParticles, support):
    k = w2(fluidNeighbors, fluidRadialDistances, numParticles, support)
    J = fluidOmegas[fluidNeighbors[0],None] * firstDerivative(-fluidDistances, fluidRadialDistances, fluidNeighbors, fluidPosition, numParticles, support)
    H = fluidOmegas[fluidNeighbors[0],None,None] * secondDerivative(-fluidDistances, fluidRadialDistances, fluidNeighbors, fluidPosition, numParticles, support)
    return k, J, H
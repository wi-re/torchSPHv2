
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time
import torch
from torch_geometric.loader import DataLoader
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

# from rbfConv import RbfConv
# from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
# %matplotlib notebook
import copy

import time
import torch
from torch_geometric.loader import DataLoader
from tqdm.notebook import trange, tqdm
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

# from rbfConv import RbfConv
# from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))


import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import math
from scipy import interpolate

import numpy as np
# %matplotlib notebook
import matplotlib.pyplot as plt

import scipy.special

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# import triangle as tr
from scipy.optimize import minimize

# np
from itertools import product

# seed = 0


# import random 
# import numpy as np
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# # print(torch.cuda.device_count())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print('running on: ', device)
# torch.set_num_threads(1)

# from joblib import Parallel, delayed

# # from cutlass import *
# # from rbfConv import *
# # from tqdm.notebook import tqdm

# # from datautils import *
# # # from sphUtils import *
# # from lossFunctions import *
# import math
# from scipy import interpolate

# import numpy as np
# %matplotlib notebook
# import matplotlib.pyplot as plt

# import scipy.special

# from numpy.random import MT19937
# from numpy.random import RandomState, SeedSequence

# import ipywidgets as widgets
# from ipywidgets import interact, interact_manual

# # import triangle as tr
# from scipy.optimize import minimize

# # np
# from itertools import product

import numpy as np


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant, rng = np.random.default_rng(seed=42)
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*rng.random((res[0]+1, res[1]+1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant, seed = 1337
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    rng = np.random.default_rng(seed=seed)
    
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant, rng
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise

@torch.jit.script
def sdPoly(poly, p):    
    with record_function("sdPoly"): 
        N = len(poly)

        i = torch.arange(N, device = p.device, dtype = torch.int64)
        i2 = (i + 1) % N
        e = poly[i2] - poly[i]
        v = p - poly[i][:,None]

        ve = torch.einsum('npd, nd -> np', v, e)
        ee = torch.einsum('nd, nd -> n', e, e)

        pq = v - e[:,None] * torch.clamp(ve / ee[:,None], min = 0, max = 1)[:,:,None]

        d = torch.einsum('npd, npd -> np', pq, pq)
        d = torch.min(d, dim = 0).values

        wn = torch.zeros((N, p.shape[0]), device = p.device, dtype = torch.int64)

        cond1 = 0 <= v[i,:,1]
        cond2 = 0 >  v[i2,:,1]
        val3 = e[i,0,None] * v[i,:,1] - e[i,1,None] * v[i,:,0]

        c1c2 = torch.logical_and(cond1, cond2)
        nc1nc2 = torch.logical_and(torch.logical_not(cond1), torch.logical_not(cond2))

        wn[torch.logical_and(c1c2, val3 > 0)] += 1
        wn[torch.logical_and(nc1nc2, val3 < 0)] -= 1

        wn = torch.sum(wn,dim=0)
        s = torch.ones(p.shape[0], device = p.device, dtype = p.dtype)
        s[wn != 0] = -1

        return s * torch.sqrt(d)
@torch.jit.script
def sdPolyDer(poly, p, dh :float = 1e-4, inverted :bool = False):
    with record_function("sdPolyDer"): 
#         dh = 1e-2
        dpx = torch.zeros_like(p)
        dnx = torch.zeros_like(p)
        dpy = torch.zeros_like(p)
        dny = torch.zeros_like(p)

        dpx[:,0] += dh
        dnx[:,0] -= dh
        dpy[:,1] += dh
        dny[:,1] -= dh

        c = sdPoly(poly, p)
        cpx = sdPoly(poly, p + dpx)
        cnx = sdPoly(poly, p + dnx)
        cpy = sdPoly(poly, p + dpy)
        cny = sdPoly(poly, p + dny)

        if inverted:
            c = -c
            cpx = -cpx
            cnx = -cnx
            cpy = -cpy
            cny = -cny

        grad = torch.zeros_like(p)
        grad[:,0] = (cpx - cnx) / (2 * dh)
        grad[:,1] = (cpy - cny) / (2 * dh)

        gradLen = torch.linalg.norm(grad, dim =1)
        grad[torch.abs(gradLen) > 1e-5] /= gradLen[torch.abs(gradLen)>1e-5,None]

        return c, grad, cpx, cnx, cpy, cny
    
def buildSDF(poly, minCoord = [-1,-1], maxCoord = [1,1], n = 256, dh = 1e-2):
    
    x = np.linspace(minCoord[0],maxCoord[0],n)
    y = np.linspace(minCoord[1],maxCoord[1],n)

    xx, yy = np.meshgrid(x,y)

    sdf, sdfGrad, _, _, _, _ = sdPolyDer(torch.tensor(poly[:-1,:]), torch.tensor(np.vstack((xx.flatten(),yy.flatten()))).mT, dh = dh)
    
    return xx, yy, sdf, sdfGrad
def plotMesh(xx,yy,z, axis, fig):
    im = axis.pcolormesh(xx,yy,z)
    axis.axis('equal')
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("bottom", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    
    
def createNoiseFunction(n = 256, res = 2, octaves = 2, lacunarity = 2, persistance = 0.5, seed = 1336):
    noise = generate_fractal_noise_2d(shape = (n,n), res = (res,res), octaves = octaves, persistence = persistance, lacunarity = lacunarity, tileable = (False, False), seed = seed)
#     noise = Octave(n, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)

#     noise[:,0] = noise[:,1] - noise[:,2] + noise[:,1]
#     noise[0,:] = noise[1,:] - noise[2,:] + noise[1,:]

#     noise = noise[:n,:n] / 255
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    xx, yy = np.meshgrid(x,y)

    f = interpolate.RegularGridInterpolator((x, y), noise, bounds_error = False, fill_value = None, method = 'linear')
    
    return f, noise

def createVelocityField(f, n = 256):
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)

    xx, yy = np.meshgrid(x,y)

    z = f((xx, yy))
    zxp = f((xx + 1e-4, yy))
    zxn = f((xx - 1e-4, yy))
    zyp = f((xx, yy + 1e-4))
    zyn = f((xx, yy - 1e-4))
    xv = (zxp - zxn) / (2 * 1e-4)
    yv = (zyp - zyn) / (2 * 1e-4)
#     print(xv)
#     print(yv)
    
    return np.stack((xv, yv), axis = 2), xx, yy, z

def createPotentialField(n = 256, res = 4, octaves = 2, lacunarity = 2, persistance = 0.5, seed = 1336):
    f, noise = createNoiseFunction(n = n, res = res, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = 1336)
#     noise = Octave(n, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)

#     noise[:,0] = noise[:,1] - noise[:,2] + noise[:,1]
#     noise[0,:] = noise[1,:] - noise[2,:] + noise[1,:]

#     noise = noise[:n,:n] / 255
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    xx, yy = np.meshgrid(x,y)

#     f = interpolate.RegularGridInterpolator((x, y), noise, bounds_error = False, fill_value = None, method = 'linear')
    
    return xx,yy,noise

def filterPotential(noise, sdf, d0 = 0.25):
    r = sdf / d0
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
#     ramped[r <= 0] = 0
    ramped[r <= -1] = -1
    
    return ramped * noise
    # ramped = r
    
def generateParticles(nd, nb, border = 3):
#     nd = 16
    nc = 2 * nd
#     nb = 32
    na = 2 * nb + nc
#     border = 3
    xi = np.arange(-border, na + border, dtype = int) + border
    dx = 2 / (na - 1)
    px = xi * dx - 1 - border * dx
    # print(xi)
    # print(x)
    xx, yy = np.meshgrid(px,px)
    xxi, yyi = np.meshgrid(xi,xi)

    c = np.ones_like(xx)
#     print(xx.shape)

    c[xxi < border] = -1
    c[xxi >= na + border] = -1
    c[yyi < border] = -1
    c[yyi >= na + border] = -1
#     print(np.sum(c > 0) - 96**2)
    # print(96**2)

    maskA = xxi >= border + nb
    maskB = yyi >= border + nb
    maskAB = np.logical_and(maskA, maskB)

    maskC = xxi < border + nb + nc
    maskD = yyi < border + nb + nc
    maskCD = np.logical_and(maskC, maskD)

    mask = np.logical_and(maskAB, maskCD)
#     print(np.sum(mask))
    c[mask] = -1

    maskA = xxi >= 2 * border + nb
    maskB = yyi >= 2 * border + nb
    maskAB = np.logical_and(maskA, maskB)

    maskC = xxi < border + nb + nc - border
    maskD = yyi < border + nb + nc - border
    maskCD = np.logical_and(maskC, maskD)

    mask = np.logical_and(maskAB, maskCD)
#     print(np.sum(mask))
    c[mask] = 0.25
    # c[:,:] = -0.5


    minDomain = -1 - dx / 2
    minCenter = - nd * dx# - dx / 2
#     print(dx)
#     print(-nd * dx)
#     print(minCenter)


#     fig, axis = plt.subplots(1, 1, figsize=(6,6), sharex = False, sharey = False, squeeze = False)

    ptcls = np.vstack((xx[c > 0.5], yy[c>0.5])).transpose()
    bdyPtcls = np.vstack((xx[c < -0.5], yy[c <-0.5])).transpose()
    return ptcls, bdyPtcls, minDomain, minCenter

def genNoisyParticles(nd = 8, nb = 16, border = 3, n = 256, res = 2, octaves = 4, lacunarity = 2, persistance = 0.25, seed = 1336, boundary = 0.25, dh = 1e-3):
    ptcls, bdyPtcls, minDomain, minCenter = generateParticles(nd, nb, border = border)

#     dh = 1e-3

#     boundary = 0.25

    c = -minCenter
    domainBoundary = np.array([[minDomain + boundary,minDomain + boundary],[-minDomain - boundary,minDomain + boundary], [-minDomain - boundary,-minDomain - boundary],[minDomain + boundary,-minDomain - boundary],[minDomain + boundary,minDomain + boundary]])
    centerBoundary = np.array([[-c,-c],[c,-c],[c,c],[-c,c],[-c,-c]])

    _, _, polySDF, polySDFGrad = buildSDF(centerBoundary, n = n, dh = dh)
    _, _, domainSDF, domainSDFGrad = buildSDF(domainBoundary, n = n, dh = dh)
    # _, _, domainSDF, domainSDFGrad = buildSDF(np.array([[-1.0 ,-1 ],[1 ,-1 ],\
    #                                                     [1 ,1 ],[-1 ,1 ],[-1 ,-1 ]]), n = 256, dh = dh)

    # poly, shape = buildPolygon()
    # xx, yy, polySDF, polySDFGrad = buildSDF(poly, n = 256)
    s = (- domainSDF + boundary).numpy()
    s = s.reshape(polySDF.shape)
    # s = - domainSDF



    xx, yy, noise = createPotentialField(n = n, res = res, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)
    filtered = noise
    
    filtered = filterPotential(torch.tensor(filtered).flatten(), torch.tensor(s).flatten(), d0 = boundary ).numpy().reshape(noise.shape)
    if nd > 0:
        filtered = filterPotential(torch.tensor(filtered).flatten(), torch.tensor(polySDF).flatten(), d0 = boundary).numpy().reshape(noise.shape)
        filtered[polySDF.reshape(noise.shape) < 0] = 0

    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    f = interpolate.RegularGridInterpolator((x, y), filtered, bounds_error = False, fill_value = None, method = 'linear')

    velocityField, xx, yy, potential = createVelocityField(f, n = n)  
#     print(filtered)
    
    f = interpolate.RegularGridInterpolator((x, y), velocityField, bounds_error = False, fill_value = None, method = 'linear')
    vel = f((ptcls[:,0], ptcls[:,1]))
    
    
    domainBoundaryActual = np.array([[minDomain,minDomain],[-minDomain,minDomain], [-minDomain,-minDomain],[minDomain,-minDomain],[minDomain,minDomain]])
    sdf, sdfDer, _, _, _, _ = sdPolyDer(torch.tensor(domainBoundaryActual[:-1]), torch.tensor(bdyPtcls), dh = 1e-2)
    domainPtcls = bdyPtcls[-sdf < 0]
    domainGhostPtcls = domainPtcls - 2 * (sdfDer[-sdf < 0] * (sdf[-sdf < 0,None])).numpy()

    csdf, csdfDer, _, _, _, _ = sdPolyDer(torch.tensor(centerBoundary[:-1]), torch.tensor(bdyPtcls), dh = 1e-2)
    centerPtcls = bdyPtcls[csdf < 0]
    centerGhostPtcls = centerPtcls - 2 * (csdfDer[csdf < 0] * (csdf[csdf < 0,None])).numpy()
    
    return ptcls, vel, domainPtcls, domainGhostPtcls, -sdf[-sdf < 0], -sdfDer[-sdf < 0], centerPtcls, centerGhostPtcls, csdf[csdf < 0], csdfDer[csdf < 0], minDomain, minCenter
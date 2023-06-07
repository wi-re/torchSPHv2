import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    


# Timing function for performance evaluation
import time
class catchtime:    
    def __init__(self, arg = 'Unnamed Context'):
#         print('__init__ called with', arg)
        self.context = arg
        
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.time
        self.readout = f'{self.context} took: {1000 * self.time:.3f} ms'
        print(self.readout)

# Math/parallelization library includes
import numpy as np
import torch

# Imports for neighborhood searches later on
from torch_geometric.nn import radius
from torch_scatter import scatter


# Plotting includes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.tri as tri
import random

# minDomain = -1
# maxDomain = 1

C = 5/4
def kernel(q, support):
    return C * (1-q)**3 * (1 + 3 * q) / (support)

def kernelGradient(q, dist, support):
    return -dist * C * 12 * (q-1)**2 * q / (support ** 2)

# Plotting function for convenience
def plot1DValues(x,counts, minDomain, maxDomain, scatter = False, xlabel = None, ylabel = None, title = None):
    if scatter:
        x = x
        y = np.zeros_like(x)
        c = counts

        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
        sc = axis[0,0].scatter(x, y, c = c, s = 1)
        ax1_divider = make_axes_locatable(axis[0,0])
        cax1 = ax1_divider.append_axes("bottom", size="20%", pad="2%")
        cb1 = fig.colorbar(sc, cax=cax1,orientation='horizontal')
        cb1.ax.tick_params(labelsize=8) 
        axis[0,0].axvline(minDomain, color = 'black', ls = '--')
        axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
        
        if not xlabel is None:
            axis[0,0].set_xlabel(xlabel)
        if not ylabel is None:
            cb1.ax.set_xlabel(ylabel)        
        if not title is None:
            fig.suptitle(title)
        
        fig.tight_layout()
        return fig, axis
    else:
        x = x
        y = counts
        idx = np.argsort(x)

        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
        axis[0,0].plot(x[idx], y[idx])
        axis[0,0].axvline(minDomain, color = 'black', ls = '--')
        axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
        
        if not xlabel is None:
            axis[0,0].set_xlabel(xlabel)
        if not ylabel is None:
            axis[0,0].set_ylabel(ylabel)
        if not title is None:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axis


# Same concept as before, just now moved to a separate function for legibility
def createGhostParticles(particles, minDomain, maxDomain):
    ghostParticlesLeft = particles - (maxDomain - minDomain)
    ghostParticlesRight = particles + (maxDomain - minDomain)

    allParticles = torch.hstack((ghostParticlesLeft, particles, ghostParticlesRight))
    return allParticles



def findNeighborhoods(particles, allParticles, support):
    # Call the external neighborhood search function
    row, col = radius(allParticles, particles, support, max_num_neighbors = 256)
    fluidNeighbors = torch.stack([row, col], dim = 0)
        
    # Compute the distances of all particle pairings
    fluidDistances = (allParticles[fluidNeighbors[1]] - particles[fluidNeighbors[0]])
    # This could also be done with an absolute value function
    fluidRadialDistances = torch.abs(fluidDistances)# torch.sqrt(fluidDistances**2)

    # Compute the direction, in 1D this is either 0 (i == j) or +-1 depending on the relative position
    fluidDistances[fluidRadialDistances < 1e-7] = 0
    fluidDistances[fluidRadialDistances >= 1e-7] /= fluidRadialDistances[fluidRadialDistances >= 1e-7]
    fluidRadialDistances /= support
    
    # Modify the neighbor list so that everything points to the original particles
    particleIndices = torch.arange(particles.shape[0])
    stackedIndices = torch.hstack((particleIndices, particleIndices, particleIndices))
    fluidNeighbors[1,:] = stackedIndices[fluidNeighbors[1,:]]    
    
    return fluidNeighbors, fluidRadialDistances, fluidDistances
def computeDensity(particles, particleArea, particleSupport, fluidRadialDistances, fluidNeighbors):
    pairWiseDensity = particleArea[fluidNeighbors[1]] * kernel(fluidRadialDistances, particleSupport)
    fluidDensity = scatter(pairWiseDensity, fluidNeighbors[0], dim=0, dim_size = particles.shape[0],reduce='add')
    
    return fluidDensity

def computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwisePressureForces = fluidAreas[j] * restDensity * \
            (fluidPressure[i] / (fluidDensity[i] * restDensity)**2 + fluidPressure[j] / (fluidDensity[j]* restDensity)**2) *\
            kernelGradient(fluidRadialDistances, fluidDistances, particleSupport)
    fluidPressureForces = scatter(pairwisePressureForces, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0],reduce='add')
    
    return fluidPressureForces


import scipy
import seaborn as sns

# @torch.jit.script
def computeLaminarViscosity(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, ui, uj, alpha : float, c0 : float, restDensity : float):
    gradW = kernelGradient(radialDistances, distances, support)

    uij = ui[i] - uj[j]
    rij = ri[j] - rj[i]
    rij = radialDistances * support
    rij2 = torch.abs(rij) + eps

    mui = rhoi[i] * alpha
    muj = rhoj[i] * alpha
    mj = Vj[j] * Vj[j] 

    nominator = 4 * mj * (mui + muj) * rij * gradW 
    denominator = (rhoi[i] + rhoj[j])**2 * (rij2)**2
    term = nominator / denominator
    
    term = alpha * 2 * Vj[j] / rhoj[j] * torch.abs(gradW) / rij2
    

#     xsphUpdate = scatter(pairwiseXSPH, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0],reduce='add')
    return scatter(term * uij, i, dim=0, dim_size = Vi.shape[0], reduce='add')


def computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensities, particleSupport, restDensity, diffusionCoefficient, fluidNeighbors, fluidRadialDistances, fluidDistances):
    laminarViscosity = computeLaminarViscosity(fluidNeighbors[0], fluidNeighbors[1], \
                                                                                      fluidPositions, fluidPositions, fluidAreas, fluidAreas,\
                                                                                      fluidDistances, fluidRadialDistances,\
                                                                                      particleSupport, fluidDensities.shape[0], 1e-7,\
                                                                                      fluidDensities, fluidDensities,\
                                                                                      fluidVelocities,fluidVelocities,
                                                                                      diffusionCoefficient, 10, restDensity)
    
    return laminarViscosity

# xsphConstant = 0.5 # We do not want 'real' viscosity, just a minor diffusion term for stability

def computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphConstant, fluidNeighbors, fluidRadialDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwiseXSPH = xsphConstant * fluidAreas[j] / ( fluidDensity[i] + fluidDensity[j]) * 2 \
                * (fluidVelocities[j] - fluidVelocities[i]) \
                * kernel(fluidRadialDistances, particleSupport) 
    
    xsphUpdate = scatter(pairwiseXSPH, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0],reduce='add')
    
    return xsphUpdate

def plotDistorted(simulationStates, minDomain, maxDomain, dt):
    fig, axis = plt.subplots(2, 1, figsize=(14,9), sharex = False, sharey = False, squeeze = False)

    im = axis[0,0].imshow(simulationStates[:,2].mT.numpy()[::-1,:], extent = [0,dt * simulationStates.shape[0], minDomain,maxDomain])
    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Density [1/m]')


    im = axis[1,0].imshow(simulationStates[:,1].mT.numpy()[::-1,:], extent = [0,dt * simulationStates.shape[0], minDomain,maxDomain], cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Velocity [m/s]')

    fig.tight_layout()

def scatterPlotAll(simulationStates, minDomain, maxDomain, dt):

    timeArray = torch.arange(simulationStates.shape[0])[:,None].repeat(1,simulationStates.shape[2]) * dt
    positionArray = simulationStates[:,0]

    fig, axis = plt.subplots(2, 1, figsize=(12,9), sharex = False, sharey = False, squeeze = False)

    trip = axis[0,0].scatter(timeArray.flatten(), positionArray.flatten(), s = 0.1, c = simulationStates[:,2].flatten(), vmin = torch.min(torch.abs(simulationStates[:,2])), vmax =torch.max(torch.abs(simulationStates[:,2])),cmap = 'viridis')

    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(trip, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_ylabel('Density [1/m]')
    axis[0,0].set_ylim(minDomain, maxDomain)
    axis[0,0].set_xlim(simulationStates.shape[0] * dt, 0)
    axis[0,0].set_xlim(0,simulationStates.shape[0] * dt)

    trip = axis[1,0].scatter(timeArray.flatten(), positionArray.flatten(), s = 0.1, c = simulationStates[:,1].flatten(), vmin = -torch.max(torch.abs(simulationStates[:,1])), vmax =torch.max(torch.abs(simulationStates[:,1])),cmap = 'RdBu')
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(trip, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_ylabel('Velocity [m/s]')
    axis[1,0].set_ylim(minDomain, maxDomain)
    axis[1,0].set_xlim(simulationStates.shape[0] * dt, 0)
    axis[1,0].set_xlim(0,simulationStates.shape[0] * dt)

    fig.tight_layout()

def plotSimulationState(simulationStates, minDomain, maxDomain, dt, timepoints = []):
    fig, axis = plt.subplots(2, 1, figsize=(9,6), sharex = True, sharey = False, squeeze = False)

    axis[0,0].axvline(minDomain, color = 'black', ls = '--')
    axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
    axis[1,0].axvline(minDomain, color = 'black', ls = '--')
    axis[1,0].axvline(maxDomain, color = 'black', ls = '--')

    axis[1,0].set_xlabel('Position')
    axis[1,0].set_ylabel('Velocity[m/s]')
    axis[0,0].set_ylabel('Density[1/m]')

    def plotTimePoint(i, c, simulationStates, axis):
        x = simulationStates[i,0,:]
        y = simulationStates[i,c,:]
        idx = torch.argsort(x)
        axis.plot(x[idx], y[idx], label = 't = %1.2g' % (i * dt))
    if timepoints == []:
        plotTimePoint(0,1, simulationStates, axis[1,0])
        plotTimePoint(0,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4*2,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4*2,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4*3,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4*3,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]-1,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]-1,2, simulationStates, axis[0,0])
    else:
        for t in timepoints:
            plotTimePoint(t,1, simulationStates, axis[1,0])
            plotTimePoint(t,2, simulationStates, axis[0,0])
            

#             plotTimePoint(16,1, simulationStates, axis[0,0])
#             plotTimePoint(16,2, simulationStates, axis[1,0])

#             plotTimePoint(128,1, simulationStates, axis[0,0])
#             plotTimePoint(128,2, simulationStates, axis[1,0])

#             plotTimePoint(256,1, simulationStates, axis[0,0])
#             plotTimePoint(256,2, simulationStates, axis[1,0])

#             plotTimePoint(511,1, simulationStates, axis[0,0])
#             plotTimePoint(511,2, simulationStates, axis[1,0])


    axis[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axis[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()




import scipy.integrate as integrate


def samplePDF(pdf, n = 2048, numParticles = 1024, plot = False, randomSampling = False):
    x = np.linspace(-1,1,n)
    if plot:
        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)

    n = 2048
    xs = np.linspace(-1,1,n)

    if plot:
        axis[0,0].plot(xs, pdf(xs))


    # integral, err = integrate.quad(pdf, -1, 1)

    normalized_pdf = lambda x: pdf(x) / np.sum(pdf(np.linspace(-1,1,n)))
    if plot:
        axis[0,0].plot(xs, normalized_pdf(xs))
        axis[0,0].axhline(0,ls= '--', color = 'black')


    xs = np.linspace(-1,1,n)
    fxs = normalized_pdf(xs)
    sampled_cdf = np.cumsum(fxs) - fxs[0]
    sampled_cdf = sampled_cdf / sampled_cdf[-1] 
    # axis[0,0].plot(xs, sampled_cdf)
    # print(cfxs.shape, cfxs)

    # n = 2048
    # cdf = []
    # for i in range(n):
    # #     cdf.append(i)
    #     cdf.append(integrate.quad(normalized_pdf, -1, -1 + i * 2 / (n-1))[0])
    # sampled_cdf = np.array(cdf)
    # axis[0,0].plot(xs, sampled_cdf)
    # sampled_pdf = normalized_pdf(np.linspace(-1,1,n))

    cdf = lambda x : np.interp(x, np.linspace(-1,1,n), sampled_cdf)
    inv_cdf = lambda x : np.interp(x, sampled_cdf, np.linspace(-1,1,n))

    # # fig, axis = plt.subplots(1, 2, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
    # # xs = np.linspace(-1,1,n)
    # # axis[0,0].plot(xs, cdf(xs))
    # # axis[0,1].plot(np.linspace(0,1,n), inv_cdf(np.linspace(0,1,n)))
    # # axis[0,1].plot(xs, normalized_pdf(xs))

    samples = np.random.uniform(size = numParticles)
    if not randomSampling:
        samples = np.linspace(0,1,numParticles, endpoint=False)
    sampled = inv_cdf(samples)

    # fig, axis = plt.subplots(1, 2, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
    # axis[0,0].scatter(sampled, sampled * 0, s = 1)
    return sampled

def plotDensity(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport):

    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)
    fig, axis = plot1DValues(fluidPositions, fluidDensity, minDomain, maxDomain, ylabel = 'Density')
    axis[0,0].axhline(torch.mean(fluidDensity), ls = '--', c = 'black', alpha = 0.5)
    axis[0,0].axhline(torch.max(fluidDensity), ls = '-', c = 'black', alpha = 0.5)
    axis[0,0].axhline(torch.min(fluidDensity), ls = '-', c = 'black', alpha = 0.5)
    
def computeUpdate(fluidPositions, fluidVelocities, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionCoefficient, xsphCoefficient, particleSupport, dt):
    #  1. Create ghost particles for our boundary conditions
    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    #  2. Find neighborhoods of all particles:
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    #  3. Compute \rho using an SPH interpolation
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)
    #  4. Compute the pressure of each particle using an ideal gas EOS
    fluidPressure = (fluidDensity - 1.0) * kappa * restDensity
    #  5. Compute the XSPH term and apply it to the particle velocities:    
    xsphUpdate = computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphCoefficient, fluidNeighbors, fluidRadialDistances)
#     fluidVelocities += xsphUpdate
    #  6. Compute pressure forces and resulting acceleration
    fluidPressureForces = computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances)
    fluidAccel = fluidPressureForces # / (fluidAreas * restDensity)
    
    laminarViscosity = computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensity, particleSupport, restDensity, diffusionCoefficient, fluidNeighbors, fluidRadialDistances, fluidDistances)
    # fluidAccel += laminarViscosity
    fluidAccel += xsphUpdate / dt + laminarViscosity
    return fluidAccel, fluidDensity, fluidPressure

from scipy.interpolate import RegularGridInterpolator  
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

def generate1DPeriodicNoise(numSamples = 1024, r = 0.75, freq = 4, octaves = 2, persistence = 0.75, lacunarity = 2, plot = False, seed = 1337):
    n = 1024
    # freq = 4
    # octaves = 2
    # persistence = 0.75
    # lacunarity = 2
    noise = generate_fractal_noise_2d([n, n], [freq,freq], octaves, persistence = persistence, lacunarity = lacunarity, seed = seed)

    interp = RegularGridInterpolator((np.linspace(-1,1,n), np.linspace(-1,1,n)), noise)


    r = 0.75
    # numSamples = 128
    thetas = np.linspace(0, 2 * np.pi, numSamples)

    x = np.cos(thetas) * r
    y = np.sin(thetas) * r

    sampled = interp(np.vstack((x,y)).transpose())
    if plot:
        fig, axis = plt.subplots(1, 2, figsize=(12,6), sharex = False, sharey = False, squeeze = False)

        circle1 = plt.Circle((0, 0), r, color='white', ls = '--', fill = False)
        axis[0,0].imshow(noise, extent =(-1,1,-1,1))
        axis[0,0].add_patch(circle1)
        axis[0,1].plot(thetas, sampled)

        fig.tight_layout()
    return sampled


from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator

def regularPlot(simulationStates, minDomain, maxDomain, dt, nx = 512, ny = 2048):
    timeArray = torch.arange(simulationStates.shape[0])[:,None].repeat(1,simulationStates.shape[2]) * dt
    positionArray = simulationStates[:,0]
    xys = torch.vstack((timeArray.flatten(), positionArray.flatten())).mT  


    # interpVelocity = LinearNDInterpolator(xys, simulationStates[:,1].flatten())
    # interpDensity = LinearNDInterpolator(xys, simulationStates[:,2].flatten())
    interpVelocity = NearestNDInterpolator(xys, simulationStates[:,1].flatten())
    interpDensity = NearestNDInterpolator(xys, simulationStates[:,2].flatten())

    X = torch.linspace(torch.min(timeArray), torch.max(timeArray), ny)
    Y = torch.linspace(torch.min(positionArray), torch.max(positionArray), nx)
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    # Z = interp(X, Y)

    fig, axis = plt.subplots(2, 1, figsize=(14,9), sharex = False, sharey = False, squeeze = False)


    im = axis[0,0].pcolormesh(X,Y,interpDensity(X,Y), cmap = 'viridis', vmin = torch.min(torch.abs(simulationStates[:,2])),vmax = torch.max(torch.abs(simulationStates[:,2])))
    # im = axis[0,0].imshow(simulationStates[:,2].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain])
    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Density [1/m]')

    im = axis[1,0].pcolormesh(X,Y,interpVelocity(X,Y), cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    # im = axis[1,0].imshow(simulationStates[:,1].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain], cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Velocity [m/s]')

    fig.tight_layout()


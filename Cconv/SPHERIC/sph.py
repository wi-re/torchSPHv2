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
def computeLaminarViscosity(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, ui, uj, alpha : float, beta: float, c0 : float, restDensity : float):
    gradW = kernelGradient(radialDistances, distances, support)

    uij = ui[i] - uj[j]
    rij = ri[i] - rj[j]
    rij = -distances * radialDistances
    
    mu_nom = support * (uij * rij)
    mu_denom = torch.abs(rij) + 0.01 * support**2
    mu = mu_nom / mu_denom
    
    
    nom = - alpha * c0 * mu + beta * mu**2
    denom = (rhoi[i] + rhoj[j]) / 2
    termL = Vi[j] * nom / denom
#     rab = rij / mu_denom    
    
    term = termL * gradW# * torch.sign(ri[j] - ri[i] + 0.01 * support **2)
#     term = alpha * 2 * Vj[j] / rhoj[j] * torch.abs(gradW) / rij2
    

#     xsphUpdate = scatter(pairwiseXSPH, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0],reduce='add')
    return scatter(term, i, dim=0, dim_size = Vi.shape[0], reduce='add')


def computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensities, particleSupport, restDensity, alpha, beta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances):
    laminarViscosity = computeLaminarViscosity(fluidNeighbors[0], fluidNeighbors[1], \
                                                                                      fluidPositions, fluidPositions, fluidAreas, fluidAreas,\
                                                                                      fluidDistances, fluidRadialDistances,\
                                                                                      particleSupport, fluidDensities.shape[0], 1e-7,\
                                                                                      fluidDensities, fluidDensities,\
                                                                                      fluidVelocities,fluidVelocities,
                                                                                      alpha, beta, c0, restDensity)
    
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


# SPH simulation step, returns dudt, dxdt as well as current density and pressure
def computeUpdate(fluidPositions, fluidVelocities, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphCoefficient, particleSupport, dt):
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
    #  6. Compute pressure forces and resulting acceleration
    fluidPressureForces = computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances)
    fluidAccel = fluidPressureForces # / (fluidAreas * restDensity)
    # 7. Compute kinematic viscosity
    laminarViscosity = computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensity, particleSupport, restDensity, diffusionAlpha, diffusionBeta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances) # currently broken for some reason
#     fluidAccel += laminarViscosity
    fluidAccel += xsphUpdate / dt + laminarViscosity
    return fluidAccel, fluidVelocities, fluidDensity, fluidPressure

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




from rbfConv import *
from trainingHelper import *


def getLoss(batch, lossFunction, model, optimizer, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked):
    optimizer.zero_grad()
    stackedPositions, features, groundTruth, stackedNeighbors, d = loadBatch(simulationStates, minDomain, maxDomain, particleSupport, batch, getFeatures, getGroundTruth, stacked)

    prediction = model((features[:,None], features[:,None]), stackedNeighbors, d)
    # compute the loss
    lossTerm = lossFunction(prediction, groundTruth)
    loss = torch.mean(lossTerm)
    # store the losses for later processing
#     losses.append(lossTerm.detach().cpu().numpy())
    # store the current weights before the update
#     weights.append(copy.deepcopy(model.state_dict()))
    return loss

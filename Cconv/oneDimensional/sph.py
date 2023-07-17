# Copyright 2023 <COPYRIGHT HOLDER>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Math/parallelization library includes
import numpy as np
import torch

# Imports for neighborhood searches later on
from torch_geometric.nn import radius
from cutlass import scatter_sum

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

# Wendland 2 kernel as per Dehnen and Aly 2012 https://arxiv.org/abs/1204.2471
C = 5/4
def kernel(q, support):
    return C * (1-q)**3 * (1 + 3 * q) / (support)

def kernelGradient(q, dist, support):
    return -dist * C * 12 * (q-1)**2 * q / (support ** 2)

# Ghost particle creation for periodic BC, these particles not actually used
# but only used as part of the neighborhood search
def createGhostParticles(particles, minDomain, maxDomain):
    ghostParticlesLeft = particles - (maxDomain - minDomain)
    ghostParticlesRight = particles + (maxDomain - minDomain)

    allParticles = torch.hstack((ghostParticlesLeft, particles, ghostParticlesRight))
    return allParticles

# Neighborhood search
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
    particleIndices = torch.arange(particles.shape[0]).to(particles.device)
    stackedIndices = torch.hstack((particleIndices, particleIndices, particleIndices))
    fluidNeighbors[1,:] = stackedIndices[fluidNeighbors[1,:]]    
    
    return fluidNeighbors, fluidRadialDistances, fluidDistances

def periodicNeighborSearch(fluidPositions, particleSupport, minDomain, maxDomain):
    distanceMat = fluidPositions[:,None] - fluidPositions
    distanceMat = torch.remainder(distanceMat + minDomain, maxDomain - minDomain) - maxDomain
    neighs = torch.abs(distanceMat) < particleSupport
    n0 = torch.sum(neighs, dim = 0)
    indices = torch.arange(fluidPositions.shape[0]).to(fluidPositions.device)
    indexMat = indices.expand(fluidPositions.shape[0], fluidPositions.shape[0])
    j, i = indexMat[neighs], indexMat.mT[neighs]
    distances = -distanceMat[neighs]
    directions = torch.sign(distances)    
    return torch.vstack((i, j)), torch.abs(distances)  / particleSupport, directions

# Summation density formulation. Note that we ignore the rest density here as this term cancels out everywhere
# that we use it and it makes learning this term more straight forward.
def computeDensity(particles, particleArea, particleSupport, fluidRadialDistances, fluidNeighbors):
    pairWiseDensity = particleArea[fluidNeighbors[1]] * kernel(fluidRadialDistances, particleSupport)
    fluidDensity = scatter_sum(pairWiseDensity, fluidNeighbors[0], dim=0, dim_size = particles.shape[0])
    
    return fluidDensity

# symmetric pressure gradient based on DJ Price 2010 https://arxiv.org/pdf/1012.1885.pdf
def computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwisePressureForces = fluidAreas[j] * restDensity * \
            (fluidPressure[i] / (fluidDensity[i] * restDensity)**2 + fluidPressure[j] / (fluidDensity[j]* restDensity)**2) *\
            kernelGradient(fluidRadialDistances, fluidDistances, particleSupport)
    fluidPressureForces = scatter_sum(pairwisePressureForces, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0])
    
    return fluidPressureForces

# Laminar viscosity term based on DJ Price 2010 https://arxiv.org/pdf/1012.1885.pdf
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
    
    term = termL * gradW
    return scatter_sum(term, i, dim=0, dim_size = Vi.shape[0])
# Helper function that calls the laminar viscosity function
def computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensities, particleSupport, restDensity, alpha, beta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances):
    laminarViscosity = computeLaminarViscosity(fluidNeighbors[0], fluidNeighbors[1], \
                                                                                      fluidPositions, fluidPositions, fluidAreas, fluidAreas,\
                                                                                      fluidDistances, fluidRadialDistances,\
                                                                                      particleSupport, fluidDensities.shape[0], 1e-7,\
                                                                                      fluidDensities, fluidDensities,\
                                                                                      fluidVelocities,fluidVelocities,
                                                                                      alpha, beta, c0, restDensity)
    
    return laminarViscosity
# XSPH based numerical viscosity term based on Monaghan 2005 https://ui.adsabs.harvard.edu/link_gateway/2005RPPh...68.1703M/doi:10.1088/0034-4885/68/8/R01
def computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphConstant, fluidNeighbors, fluidRadialDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwiseXSPH = xsphConstant * fluidAreas[j] / ( fluidDensity[i] + fluidDensity[j]) * 2 \
                * (fluidVelocities[j] - fluidVelocities[i]) \
                * kernel(fluidRadialDistances, particleSupport) 
    
    xsphUpdate = scatter_sum(pairwiseXSPH, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0])
    
    return xsphUpdate

# Function to sample particles such that their density equals a desired PDF
def samplePDF(pdf, n = 2048, numParticles = 1024, plot = False, randomSampling = False):
    x = np.linspace(-1,1,n)
    if plot:
        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)

    n = 2048
    xs = np.linspace(-1,1,n)

    if plot:
        axis[0,0].plot(xs, pdf(xs))

    normalized_pdf = lambda x: pdf(x) / np.sum(pdf(np.linspace(-1,1,n)))
    if plot:
        axis[0,0].plot(xs, normalized_pdf(xs))
        axis[0,0].axhline(0,ls= '--', color = 'black')


    xs = np.linspace(-1,1,n)
    fxs = normalized_pdf(xs)
    sampled_cdf = np.cumsum(fxs) - fxs[0]
    sampled_cdf = sampled_cdf / sampled_cdf[-1] 
    inv_cdf = lambda x : np.interp(x, sampled_cdf, np.linspace(-1,1,n))

    samples = np.random.uniform(size = numParticles)
    if not randomSampling:
        samples = np.linspace(0,1,numParticles, endpoint=False)
    sampled = inv_cdf(samples)

    return sampled


# SPH simulation step, returns dudt, dxdt as well as current density and pressure
def computeUpdate(fluidPositions, fluidVelocities, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphCoefficient, particleSupport, dt):
    # 1. Find neighborhoods of all particles:
    fluidNeighbors, fluidRadialDistances, fluidDistances = periodicNeighborSearch(fluidPositions, particleSupport, minDomain, maxDomain)
    # 2. Compute \rho using an SPH interpolation
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)
    # 3. Compute the pressure of each particle using an ideal gas EOS
    fluidPressure = (fluidDensity - 1.0) * kappa * restDensity
    # 4. Compute pressure forces and resulting acceleration
    fluidPressureForces = computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances)
    fluidAccel = fluidPressureForces # / (fluidAreas * restDensity)
    # 5. Compute the XSPH term and apply it to the particle velocities:    
    # xsphUpdate = computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphCoefficient, fluidNeighbors, fluidRadialDistances)
    # fluidAccel += xsphUpdate / dt
    # 6. Compute kinematic viscosity
    fluidAccel += computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensity, particleSupport, restDensity, diffusionAlpha, diffusionBeta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances) # currently broken for some reason
    return fluidAccel, fluidVelocities, fluidDensity, fluidPressure

from plotting import *
def initSimulation(pdf, numParticles, minDomain, maxDomain, baseArea, particleSupport, dtype, device, plot = False):
    # sample the pdf using the inverse CFD, plotting shows the pdf
    sampled = samplePDF(pdf, plot = False, numParticles = numParticles)
    # sample positions according to the given pdf
    fluidPositions = ((torch.tensor(sampled)/2 +0.5)* (maxDomain - minDomain) + minDomain).type(dtype).to(device)
    # initially zero velocity everywhere
    fluidVelocities = torch.zeros(fluidPositions.shape[0]).type(dtype).to(device)
    # and all particles with identical masses
    fluidAreas = torch.ones_like(fluidPositions) * baseArea
    # simulationStates holds all timestep information
    simulationStates = []
    # plot initial density field to show starting conditions
    if plot:
        density = plotDensityField(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport)
    return fluidPositions, fluidAreas, fluidVelocities
def runSimulation(fluidPositions_, fluidAreas_, fluidVelocities_, timesteps, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, dt):
    fluidPositions = torch.clone(fluidPositions_)
    fluidAreas = torch.clone(fluidAreas_)
    fluidVelocities = torch.clone(fluidVelocities_)
    simulationStates = []
    # run the simulation using RK4
    for i in tqdm(range(timesteps)):
        # Compute state for substep 1
        v1 = torch.clone(fluidVelocities)
        # RK4 substep 1
        dudt_k1, dxdt_k1, fluidDensity, fluidPressure = computeUpdate(fluidPositions, fluidVelocities, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, dt)   
        # Compute state for substep 2
        x_k1 = fluidPositions + 0.5 * dt * dxdt_k1
        u_k1 = fluidVelocities + 0.5 * dt * dudt_k1    
        # RK4 substep 2
        dudt_k2, dxdt_k2, _, _ = computeUpdate(x_k1, u_k1, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, 0.5 * dt)    
        # Compute state for substep 2
        x_k2 = fluidPositions + 0.5 * dt * dxdt_k2
        u_k2 = fluidVelocities + 0.5 * dt * dudt_k2
        # RK4 substep 3
        dudt_k3, dxdt_k3, _, _ = computeUpdate(x_k2, u_k2, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport,  0.5 * dt)    
        # Compute state for substep 4    
        x_k3 = fluidPositions + dt * dxdt_k3
        u_k3 = fluidVelocities + dt * dudt_k3
        # RK4 substep 4
        dudt_k4, dxdt_k4, _, _ = computeUpdate(x_k3, u_k3, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, dt)    
        # RK substeps done, store current simulation state for later processing/learning. density and pressure are based on substep 1 (i.e., the starting point for this timestep)
        simulationStates.append(torch.stack([fluidPositions, fluidVelocities, fluidDensity, fluidPressure, dt/6 * (dudt_k1 + 2* dudt_k2 + 2 * dudt_k3 + dudt_k4), dudt_k1, dudt_k2, dudt_k3, dudt_k4, dxdt_k1, dxdt_k2, dxdt_k3, dxdt_k4, fluidAreas]))
        # time integration using RK4 for velocity
    #     fluidVelocities = fluidVelocities + dt * dudt_k1 # semi implicit euler mode
        fluidVelocities = fluidVelocities + dt/6 * (dudt_k1 + 2* dudt_k2 + 2 * dudt_k3 + dudt_k4)
        fluidPositions = fluidPositions + dt * fluidVelocities
    # After the simulation has run we stack all the states into one large array for easier slicing and analysis
    simulationStates = torch.stack(simulationStates)
    return simulationStates

import os
from datetime import datetime
import h5py

def export(simulationStates, numParticles, timesteps, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleRadius, baseArea, particleSupport, dt, generator, generatorSettings):
    if not os.path.exists('./output/'):
        os.makedirs('./output/')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outFile = h5py.File('./output/out_%s_%08d_%s.hdf5' % (generator, generatorSettings['seed'], timestamp),'w')

    outFile.attrs['minDomain'] = minDomain
    outFile.attrs['maxDomain'] = maxDomain

    outFile.attrs['baseArea'] = baseArea
    outFile.attrs['particleRadius'] = particleRadius
    outFile.attrs['particleSupport'] = particleSupport

    outFile.attrs['xsphConstant'] = xsphConstant
    outFile.attrs['diffusionAlpha'] = diffusionAlpha
    outFile.attrs['diffusionBeta'] = diffusionBeta
    outFile.attrs['kappa'] = kappa
    outFile.attrs['restDensity'] = restDensity
    outFile.attrs['c0'] = c0
    outFile.attrs['dt'] = dt

    outFile.attrs['numParticles'] = numParticles
    outFile.attrs['timesteps'] = timesteps

    outFile.attrs['generator'] = generator

    grp = outFile.create_group('generatorSettings')
    grp.attrs.update(generatorSettings)

    grp = outFile.create_group('simulationData')

    grp.create_dataset('fluidPosition', data = simulationStates[:,0].detach().cpu().numpy())
    grp.create_dataset('fluidVelocities', data = simulationStates[:,1].detach().cpu().numpy())
    grp.create_dataset('fluidDensity', data = simulationStates[:,2].detach().cpu().numpy())
    grp.create_dataset('fluidPressure', data = simulationStates[:,3].detach().cpu().numpy())
    grp.create_dataset('fluidAreas', data = simulationStates[:,13].detach().cpu().numpy())

    grp.create_dataset('dudt', data = simulationStates[:,4].detach().cpu().numpy())
    grp.create_dataset('dudt_k1', data = simulationStates[:,5].detach().cpu().numpy())
    grp.create_dataset('dudt_k2', data = simulationStates[:,6].detach().cpu().numpy())
    grp.create_dataset('dudt_k3', data = simulationStates[:,7].detach().cpu().numpy())
    grp.create_dataset('dudt_k4', data = simulationStates[:,8].detach().cpu().numpy())

    grp.create_dataset('dxdt_k1', data = simulationStates[:,9].detach().cpu().numpy())
    grp.create_dataset('dxdt_k2', data = simulationStates[:,10].detach().cpu().numpy())
    grp.create_dataset('dxdt_k3', data = simulationStates[:,11].detach().cpu().numpy())
    grp.create_dataset('dxdt_k4', data = simulationStates[:,12].detach().cpu().numpy())
    outFile.close()
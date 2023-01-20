import torch
from torch_scatter import scatter

from .periodicBC import *
from .kernel import *
from .neighborhood import *
from .sph import *
from torch.profiler import record_function


def computeGasPressureAccel(config, simulationState):
    with record_function("sph - gas pressure accel"): 
        simulationState['fluidPressure'] = (simulationState['fluidDensity'] - 1.) * config['kappa'] * simulationState['fluidRestDensity']
        simulationState['fluidPressure'] = torch.clamp(simulationState['fluidPressure'], min = 0.)
        syncQuantity(simulationState['fluidPressure'], config, simulationState)
        
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[1]
        j = neighbors[0]
        
        grad = wendlandGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], config['support'])
        fac = - simulationState['fluidArea'][j] * simulationState['fluidRestDensity'][j]
        pi = (simulationState['fluidPressure'][i] / (simulationState['fluidDensity'][i] * simulationState['fluidRestDensity'][i])**2)
        pj = (simulationState['fluidPressure'][j] / (simulationState['fluidDensity'][j] * simulationState['fluidRestDensity'][j])**2)
        term = (fac * (pi + pj))[:,None] * grad
        gathered = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
        syncQuantity(gathered, config, simulationState)
        return gathered

def compressibleSimulation(config, simulationState):
    with record_function("sph - compressible step"): 
        enforcePeriodicBC(config, simulationState)

        simulationState['fluidNeighbors'], simulationState['fluidDistances'], simulationState['fluidRadialDistances'] = \
            neighborSearch(simulationState['fluidPosition'], simulationState['fluidPosition'], config, simulationState)

        simulationState['fluidDensity'] = sphDensity(config, simulationState)  
        simulationState['fluidPressureAccel'] = computeGasPressureAccel(config, simulationState)
        simulationState['fluidVelocity'] += config['dt'] * simulationState['fluidPressureAccel']
        velocityBC(config,simulationState)
        syncQuantity(simulationState['fluidVelocity'], config, simulationState)
        XSPHCorrection(config, simulationState)
        simulationState['fluidPosition'] += config['dt'] * simulationState['fluidVelocity']

        simulationState['time'] += config['dt']
        simulationState['timestep'] += 1
import torch
from torch_scatter import scatter
from torch.profiler import record_function


from .periodicBC import *
from .kernel import *

def sphDensity(config, simulationState):
    with record_function("sph - density"): 
        rho =  scatter(wendland(simulationState['fluidRadialDistances'], config['support']) * \
                    simulationState['fluidArea'][simulationState['fluidNeighbors'][0]], simulationState['fluidNeighbors'][1], dim=0, dim_size=simulationState['numParticles'], reduce="add")
        
        if 'boundaryNeighbors' in simulationState and simulationState['boundaryNeighbors'] != None:
            simulationState['boundaryDensity'] = torch.zeros(rho.shape, device=config['device'], dtype= config['precision'])
            simulationState['boundaryGradient'] = torch.zeros(rho.shape, device=config['device'], dtype= config['precision'])
            
            simulationState['boundaryDensity'] = scatter(simulationState['boundaryIntegrals'], simulationState['boundaryNeighbors'][0], dim = 0, dim_size = simulationState['numParticles'], reduce="add")
            simulationState['boundaryGradient'] = scatter(simulationState['boundaryIntegralGradients'], simulationState['boundaryNeighbors'][0], dim = 0, dim_size = simulationState['numParticles'], reduce="add")
            
            rho += simulationState['boundaryDensity'] 
            
        syncQuantity(rho, config, simulationState)    
        return rho

def XSPHCorrection(config, simulationState):
    with record_function("sph - xsph correction"): 
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[1]
        j = neighbors[0]
        
        fac = config['viscosityConstant'] * simulationState['fluidRestDensity'][j] * simulationState['fluidArea'][j]
        rho_i = simulationState['fluidDensity'][i] * simulationState['fluidRestDensity'][i]
        rho_j = simulationState['fluidDensity'][j] * simulationState['fluidRestDensity'][j]

        v_ij = simulationState['fluidVelocity'][j] - simulationState['fluidVelocity'][i]

        k = wendland(simulationState['fluidRadialDistances'], config['support'])
        
        term = (fac / (rho_i + rho_j) * 2. * k)[:,None] * v_ij
        
        correction = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
        syncQuantity(correction, config, simulationState)
        
        simulationState['fluidVelocity'] += correction

def gravity(config, simulationState):
    with record_function("sph - external accel"): 
        return torch.from_numpy(np.array(config['gravity'])).type(config['precision']).to(config['device'])
@torch.jit.script
def LinearCG(H, B, x0, i, j, tol=1e-5, verbose = False):    
    xk = x0
    rk = torch.zeros_like(x0)
    numParticles = rk.shape[0] // 2

    rk[::2]  += scatter(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
    rk[::2]  += scatter(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")

    rk[1::2] += scatter(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
    rk[1::2] += scatter(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")
    
    rk = rk - B
    
    pk = -rk
    rk_norm = torch.linalg.norm(rk)
    
    num_iter = 0

    if verbose:
        print('xk: ', x0)
        print('rk: ', rk)
        print('|rk|: ', rk_norm)
        print('pk: ', pk)


    while rk_norm > tol and num_iter < 32:
        apk = torch.zeros_like(x0)

        apk[::2]  += scatter(H[:,0,0] * pk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
        apk[::2]  += scatter(H[:,0,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")

        apk[1::2] += scatter(H[:,1,0] * pk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
        apk[1::2] += scatter(H[:,1,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")

        rkrk = torch.dot(rk, rk)
        
        alpha = rkrk / torch.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = torch.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        
        num_iter += 1

        rk_norm = torch.linalg.norm(rk)
        if verbose:
            print('iter: ', num_iter)
            print('\t|rk|: ', rk_norm)
            print('\talpha: ', alpha)
            
    return xk


def solveShifting(config, state, verbose = False):
    state['fluidOmegas'] = config['area'] / state['fluidDensity']
    syncQuantity(state['fluidOmegas'], config, state)

    K, J, H = evalKernel(state['fluidOmegas'], state['fluidPosition'], state['fluidNeighbors'], state['fluidDistances'], state['fluidRadialDistances'], state['numParticles'], config['support'])

    JJ = scatter(J, state['fluidNeighbors'][1], dim=0, dim_size=state['numParticles'], reduce= "add")
    JJ -= state['boundaryGradient']
    
    syncQuantity(JJ, config, state)
    

    B = torch.zeros(JJ.shape[0]*2, device = JJ.device, dtype=JJ.dtype)
    B[::2] = JJ[:,0]
    B[1::2] = JJ[:,1]
    

    i = state['fluidNeighbors'][1]
    j = state['fluidNeighbors'][0]
    
    x0 = torch.rand(state['numParticles'] * 2).to(config['device']).type(config['precision']) * config['support'] / 4
    diff = LinearCG(H, B, x0, i, j, verbose = verbose)
    
    dx = torch.zeros(J.shape[0], device = J.device, dtype=J.dtype)
    dy = torch.zeros(J.shape[0], device = J.device, dtype=J.dtype)
    dx = -diff[::2]
    dy = -diff[1::2]

    update = torch.vstack((dx,dy)).T
    syncQuantity(update, config, state)

    state['fluidUpdate'] = update

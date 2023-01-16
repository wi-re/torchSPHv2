import torch
from torch_geometric.nn import radius
from torch_scatter import scatter
from torch.profiler import record_function


from .kernel import *
from .neighborhood import *
from .periodicBC import *
from .solidBC import *
from .sph import *
from .plotting import *

def computeAlpha(config, simulationState, density = True):
    with record_function("dfsph - compute alpha"): 
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[1]
        j = neighbors[0]
        grad = wendlandGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], config['support'])
        grad2 = torch.einsum('nd, nd -> n', grad, grad)

        term1 = simulationState['fluidActualArea'][j][:,None] * grad
        term2 = simulationState['fluidActualArea'][j]**2 / (simulationState['fluidArea'][j] * simulationState['fluidRestDensity'][j]) * grad2

        kSum1 = scatter(term1, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
        kSum2 = scatter(term2, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        if density and 'boundaryNeighbors' in simulationState and simulationState['boundaryNeighbors'] != None:
            kSum1 += simulationState['boundaryGradient']
            
        fac = - config['dt'] **2 * simulationState['fluidActualArea']
        mass = simulationState['fluidArea'] * simulationState['fluidRestDensity']

        return fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2

def computeSourceTerm(config, simulationState, density = True):
    with record_function("dfsph - compute source term"): 
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[1]
        j = neighbors[0]
        fac = - config['dt'] * simulationState['fluidActualArea'][j]
        vij = simulationState['fluidPredictedVelocity'][i] - simulationState['fluidPredictedVelocity'][j]
        grad = wendlandGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], config['support'])
        prod = torch.einsum('nd, nd -> n', vij, grad)

        source = scatter(fac * prod, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        if density and 'boundaryNeighbors' in simulationState and simulationState['boundaryNeighbors'] != None:
            source = source - config['dt'] * torch.einsum('nd, nd -> n',  simulationState['fluidPredictedVelocity'],  simulationState['boundaryGradient'])
            
        return 1. - simulationState['fluidDensity'] + source if density else source


def computeUpdatedPressure(config, simulationState, density = True):
    with record_function("dfsph - update pressure"): 
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[1]
        j = neighbors[0]
        grad = wendlandGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], config['support'])
        
        fac = config['dt']**2 * simulationState['fluidActualArea'][j]
        aij = simulationState['fluidPredAccel'][i] - simulationState['fluidPredAccel'][j]
        kernelSum = scatter(torch.einsum('nd, nd -> n', fac[:,None] * aij, grad), i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        if density and 'boundaryNeighbors' in simulationState and simulationState['boundaryNeighbors'] != None:
            kernelSum = kernelSum + config['dt']**2 * torch.einsum('nd, nd -> n', simulationState['fluidPredAccel'], simulationState['boundaryGradient'])
            
            
        residual = kernelSum - simulationState['fluidSourceTerm']

        pressure = simulationState['fluidPressure'] - config['omega'] * residual / simulationState['fluidAlpha']
        pressure = torch.clamp(pressure, min = 0.) if density else pressure
        if density and config['dfsph']['backgroundPressure']:
            pressure = torch.clamp(pressure, min = (5**2) * simulationState['fluidRestDensity'])


        return pressure, residual

def computeAcceleration(config, simulationState, density = True):
    with record_function("dfsph - compute accel"): 
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[1]
        j = neighbors[0]
        grad = wendlandGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], config['support'])
        
        fac = - simulationState['fluidArea'][j] * simulationState['fluidRestDensity'][j]
        pi = simulationState['fluidPressure2'][i] / (simulationState['fluidDensity'][i] * simulationState['fluidRestDensity'][i])**2
        pj = simulationState['fluidPressure2'][j] / (simulationState['fluidDensity'][j] * simulationState['fluidRestDensity'][j])**2
        term = (fac * (pi + pj))[:,None] * grad
        fluidAccelTerm = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        if density and 'boundaryNeighbors' in simulationState and simulationState['boundaryNeighbors'] != None:
            with record_function("dfsph - compute accel boundary"): 
                neighbors = simulationState['boundaryNeighbors']
                i = neighbors[0]
                b = neighbors[1]
                
                boundaryPressure = simulationState['fluidPressure2'][i]
                
                bi, bb = simulationState['boundaryFluidNeighbors']
                
                pb = simulationState['fluidPosition'][i] - simulationState['boundaryDistances'][:, None] * simulationState['boundaryGradients'] * config['support']
                simulationState['pb'] = pb
        #         print(pb.shape)
        #         print('bi', bi.shape, bi)
        #         print('bb', bb.shape, bb)
                
                distances = torch.linalg.norm(simulationState['fluidPosition'][bi] - pb[bb], axis = 1) / config['support']
        #         print('distances', distances.shape, distances)
        #         print('kernel', wendland(distances, config['support']))

                kernel = wendland(distances, config['support'])
                pjb = simulationState['fluidPosition'][bi] - simulationState['boundaryFluidPositions'][bb]
                
                Mpartial = torch.einsum('nd, ne -> nde', pjb, pjb)
        #         print('Mpartial', Mpartial.shape, Mpartial)
                
                M = scatter(Mpartial, bb, dim=0, dim_size = pb.shape[0], reduce='add')
                
        #         print('pjb', pjb.shape, pjb)
                
                vecSum = scatter(pjb * (simulationState['fluidPressure2'][bi] * simulationState['fluidArea'][bi] * kernel)[:,None], bb, dim=0, dim_size = pb.shape[0], reduce='add')
                sumA = scatter(simulationState['fluidPressure2'][bi] * simulationState['fluidArea'][bi] * kernel, bb, dim=0, dim_size = pb.shape[0], reduce='add')
                sumB = scatter(simulationState['fluidArea'][bi] * kernel, bb, dim=0, dim_size = pb.shape[0], reduce='add')

        #         print('M', M.shape, M)
        #         print('vecSum', vecSum.shape, vecSum)
        #         print('sumA', sumA.shape, sumA)
        #         print('sumB', sumB.shape, sumB)
                
                fac = simulationState['fluidArea'][bi] * wendland(distances, config['support'])
                
                Mp = torch.linalg.pinv(M)
                
                alpha = sumA / sumB
                beta  = Mp[:,0,0] * vecSum[:,0] + Mp[:,0,1] * vecSum[:,1]
                gamma = Mp[:,1,0] * vecSum[:,0] + Mp[:,1,1] * vecSum[:,1]
                
                det = torch.linalg.det(Mp)
                
                simulationState['boundary sumA'] = sumA
                simulationState['boundary sumB'] = sumB
                simulationState['boundary vecSum'] = vecSum
                simulationState['boundary alpha'] = alpha
                simulationState['boundary beta'] = beta
                simulationState['boundary gamma'] = gamma
                simulationState['boundary det'] = det
                simulationState['boundary M'] = M
                simulationState['boundary Mp'] = Mp
                
                beta[torch.isnan(det)] = 0
                gamma[torch.isnan(det)] = 0
                
                boundaryPressure = alpha + beta * simulationState['boundaryFluidPositions'][:,0] + gamma * simulationState['boundaryFluidPositions'][:,1]
        #         print(boundaryPressure)
                boundaryPressure[torch.isnan(alpha)] = 0
                boundaryPressure[torch.isnan(boundaryPressure)] = 0     
                
                simulationState['boundaryPressure'] = boundaryPressure 
                
                # boundaryPressure = simulationState['fluidPressure2'][i]
                
                fac = - simulationState['fluidRestDensity'][i]
                pi = simulationState['fluidPressure2'][i] / (simulationState['fluidDensity'][i] * simulationState['fluidRestDensity'][i])**2
                pb = boundaryPressure / (1. * simulationState['fluidRestDensity'][i])**2
                grad = simulationState['boundaryIntegralGradients']
                
                boundaryAccelTerm = scatter((fac * (pi + pb))[:,None] * grad, simulationState['boundaryNeighbors'][0], dim = 0, dim_size = simulationState['numParticles'], reduce="add")

                simulationState['boundaryAccelTerm'] = boundaryAccelTerm
                
                boundaryAccelTerm2 = scatter((fac * (pb + pb))[:,None] * grad, simulationState['boundaryNeighbors'][0], dim = 0, dim_size = simulationState['numParticles'], reduce="add")

                force = -boundaryAccelTerm2 * (simulationState['fluidArea'] * simulationState['fluidRestDensity'])[:,None]
                simulationState['boundaryPressureForce'] = scatter(force[i], b, dim = 0, dim_size = len(config['solidBoundary']), reduce = "add")

                return fluidAccelTerm + boundaryAccelTerm
        
        return fluidAccelTerm
    

def densitySolve(config, simulationState):
    with record_function("dfsph - density solver"): 
        errors = []
        i = 0
        error = 0.
        minIters = config['dfsph']['minDensitySolverIterations']
        if 'densityErrors' in simulationState:
            minIters = max(minIters, len(simulationState['densityErrors'])*0.75)

        while((i < minIters or \
                error > config['dfsph']['densityThreshold']) and \
                i <= config['dfsph']['maxDensitySolverIterations']):
            with record_function("dfsph - density solver iteration"): 
                simulationState['fluidPredAccel'] = computeAcceleration(config, simulationState, True)
                syncQuantity(simulationState['fluidPredAccel'], config, simulationState)
                simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                simulationState['fluidPressure2'], simulationState['residual'] = computeUpdatedPressure(config, simulationState, True)
                syncQuantity(simulationState['fluidPressure2'], config, simulationState)

                error = torch.mean(torch.clamp(simulationState['residual'], min = -config['dfsph']['densityThreshold']))# * simulationState['fluidArea'])
                
                errors.append((error).item())
                i = i + 1
        simulationState['densityErrors'] = errors
        return errors
    
def divergenceSolve(config, simulationState):
    with record_function("dfsph - divergence solver"): 
        errors = []
        i = 0
        error = 0.
        while((i < config['dfsph']['minDivergenceSolverIterations'] or error > config['dfsph']['divergenceThreshold']) and i <= config['dfsph']['maxDivergenceSolverIterations']):
            with record_function("dfsph - divergence solver iteration"): 
                simulationState['fluidPredAccel'] = computeAcceleration(config, simulationState, False)
                syncQuantity(simulationState['fluidPredAccel'], config, simulationState)
                simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                simulationState['fluidPressure2'], simulationState['residual'] = computeUpdatedPressure(config, simulationState, False)
                syncQuantity(simulationState['fluidPressure2'], config, simulationState)

                error = torch.mean(torch.clamp(simulationState['residual'], min = -config['dfsph']['divergenceThreshold']))# * simulationState['fluidArea'])
                
                errors.append((error).item())
                i = i + 1
        simulationState['divergenceErrors'] = errors
        return errors
    

def DFSPH(config, simulationState, density = True): 
    with record_function("dfsph - solver"): 
        simulationState['fluidPredictedVelocity'] = simulationState['fluidVelocity'] + config['dt'] * simulationState['fluidAcceleration']
        simulationState['fluidActualArea'] = simulationState['fluidArea'] / simulationState['fluidDensity']

        simulationState['fluidAlpha'] = computeAlpha(config, simulationState, density)
        syncQuantity(simulationState['fluidAlpha'], config, simulationState)
        simulationState['fluidSourceTerm'] = computeSourceTerm(config, simulationState, density)
        syncQuantity(simulationState['fluidSourceTerm'], config, simulationState)
        if 'fluidPressure' in simulationState:
            simulationState['fluidPressure2'] = simulationState['fluidPressure'] * 0.5
        else:
            simulationState['fluidPressure2'] = torch.zeros(simulationState['numParticles'], dtype = config['precision'], device = config['device'])

        syncQuantity(simulationState['fluidPressure2'], config, simulationState)
        totalArea = torch.sum(simulationState['fluidArea'])

        if density:
            errors = densitySolve(config, simulationState)
        else:
            errors = divergenceSolve(config, simulationState)
    #         print(error / totalArea)
    #     print(i, ["{0:0.5f}".format(i) for i in errors])
        simulationState['fluidPredAccel'] = computeAcceleration(config, simulationState, density)
        syncQuantity(simulationState['fluidPredAccel'], config, simulationState)
        simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

        simulationState['fluidPredictedVelocity'] += config['dt'] * simulationState['fluidPredAccel']
    #     print(density, errors)
        return errors

def incompressibleSimulation(config, state):
    with record_function("sph - incompressible step"): 
        if config['export']['active']:
            
            state['exportCounter'] += 1
            grp = state['outFile'].create_group('%04d' %(state['exportCounter']))
            state['outGroup'] = grp
            

        enforcePeriodicBC(config, state)

        state['fluidNeighbors'], state['fluidDistances'], state['fluidRadialDistances'] = \
            neighborSearch(state['fluidPosition'], state['fluidPosition'], config, state)

        state['boundaryNeighbors'], state['boundaryDistances'], state['boundaryGradients'], \
            state['boundaryIntegrals'], state['boundaryIntegralGradients'], \
            state['boundaryFluidNeighbors'], state['boundaryFluidPositions'] = boundaryNeighborSearch(config, state)

        state['fluidDensity'] = sphDensity(config, state)  
        state['fluidAcceleration'][:] = 0.
        state['fluidAcceleration'] += gravity(config, state)
        syncQuantity(state['fluidAcceleration'], config, state)
        
        if config['export']['active']:
            grp.create_dataset('position', data = state['fluidPosition'].detach().cpu().numpy())
            grp.create_dataset('velocity', data = state['fluidVelocity'].detach().cpu().numpy())
            grp.create_dataset('area', data = state['fluidArea'].detach().cpu().numpy())
            grp.create_dataset('density', data = state['fluidDensity'].detach().cpu().numpy())
            grp.create_dataset('ghostIndices', data = state['ghostIndices'].detach().cpu().numpy())
            grp.create_dataset('UID', data = state['UID'].detach().cpu().numpy())
            grp.create_dataset('boundaryIntegral', data = state['boundaryIntegrals'].detach().cpu().numpy())
            grp.create_dataset('boundaryGradient', data = state['boundaryIntegralGradients'].detach().cpu().numpy())

        if config['dfsph']['divergenceSolver']:
            DFSPH(config, state, False)
            syncQuantity(state['fluidPredAccel'], config, state)
        state['densityIterations'] = DFSPH(config, state, True)
        syncQuantity(state['fluidPredAccel'], config, state)

        state['fluidAcceleration'] += state['fluidPredAccel']
        syncQuantity(state['fluidAcceleration'], config, state)

        state['fluidVelocity'] += config['dt'] * state['fluidAcceleration']
        if config['export']['active']:
            grp.create_dataset('velocityAfterSolver', data=state['fluidVelocity'].detach().cpu().numpy())
            
        syncQuantity(state['fluidVelocity'], config, state)
        velocityBC(config,state)
        if config['export']['active']:
            grp.create_dataset('gamma', data=state['fluidGamma'].detach().cpu().numpy())
            
        syncQuantity(state['fluidVelocity'], config, state)
        XSPHCorrection(config, state)
        syncQuantity(state['fluidVelocity'], config, state)
        boundaryFriction(config, state)
        syncQuantity(state['fluidVelocity'], config, state)
        

        state['fluidPosition'] += config['dt'] * state['fluidVelocity']
        
        if config['export']['active']:
            grp.create_dataset('velocityAfterBC', data=state['fluidVelocity'].detach().cpu().numpy())
            grp.create_dataset('positionAfterStep', data=state['fluidPosition'].detach().cpu().numpy())

        state['time'] += config['dt']
        state['timestep'] += 1

        # fullIndices = torch.arange(simulationState['numParticles'], dtype= torch.int64, device=config['device'])

        # indices = torch.arange(simulationState['realParticles'], dtype= torch.int64, device=config['device'])
        # ghosts = simulationState['ghostIndices'][simulationState['ghostIndices'] != -1]
        # indices = torch.cat((indices, ghosts))

        # uniqueIndices = set(indices.detach().cpu().numpy().tolist())

        # realParticles = filterVirtualParticles(simulationState['fluidPosition'], config)
        # newUniqueIndices = set(indices[realParticles].detach().cpu().numpy().tolist())

        # difference = torch.tensor(list(uniqueIndices - newUniqueIndices), dtype=torch.int64, device=config['device'])
        # if difference.shape[0] != 0:
        #     print(difference)
        # for d in difference:
        #     print('Difference detected for particle', d)
        #     print(fullIndices[indices ==d])
        #     ind = fullIndices[indices == d]
        #     for i in ind:
        #         printParticle(i, simulationState)

import torch
import numpy as np
from torch_scatter import scatter
from torch.profiler import record_function

def filterVirtualParticles(positions, config, state):
    with record_function('periodicBC - filtering'):
        # indices = torch.arange(positions.shape[0], dtype=torch.int64).to(config['device'])
        # virtualMin = config['domain']['virtualMin']
        # virtualMax = config['domain']['virtualMax']


        # if config['domain']['periodicX']:
        #     indices[positions[:,0] < virtualMin[0]] = -1
        #     indices[positions[:,0] >= virtualMax[0]] = -1
        #     # counter[positions[:,0] < virtualMin[0]] 
        # if config['domain']['periodicY']:        
        #     indices[positions[:,1] < virtualMin[1]] = -1    
        #     indices[positions[:,1] >= virtualMax[1]] = -1   
        


        # indices = indices[indices != -1]
        # if 'realParticles' in state and indices.shape[0] != state['realParticles']:
        #     print('panik mode')
        #     counter = torch.zeros(state['numParticles'], dtype=torch.int64).to(config['device'])
        #     print('particle counter ', counter.shape, counter)
        #     uidCounter = scatter(torch.ones(state['numParticles'], dtype=torch.int64).to(config['device']), state['UID'], dim = 0, dim_size=state['realParticles'])
        #     print('particle uid counter ', uidCounter.shape, uidCounter)

        #     virtualMin = config['domain']['virtualMin']
        #     virtualMax = config['domain']['virtualMax']
        #     if config['domain']['periodicX']:
        #         counter[positions[:,0] < virtualMin[0]] = -1
        #         counter[positions[:,0] >= virtualMax[0]] = -1
        #     if config['domain']['periodicY']:        
        #         counter[positions[:,1] < virtualMin[1]] = -1    
        #         counter[positions[:,1] >= virtualMax[1]] = -1
                
        #     deletionCounter = scatter(counter, state['UID'], dim = 0, dim_size=state['realParticles'])
        #     print('particle deletion counter ', deletionCounter.shape, deletionCounter)

        #     actualCounter = uidCounter + deletionCounter
        #     print('particle actual counter ', actualCounter.shape, actualCounter)

        #     problematicUIDs = state['UID'][:state['realParticles']][actualCounter != 1]
        #     print('problematic UIDs', problematicUIDs.shape, problematicUIDs)

        #     indices = torch.ones(state['numParticles'], dtype = torch.int64, device=config['device']) * -1
        #     indices[counter != -1] = state['UID'][counter != -1]
        #     print('indices before', indices.shape, indices)

        #     tempUIDs = torch.arange(state['numParticles'], dtype=torch.int64, device=config['device'])
        #     print('tempUIDs', tempUIDs.shape, tempUIDs)
        #     for uid in problematicUIDs:
        #         print(uid)
        #         relevantIndices = tempUIDs[state['UID'] == uid]
        #         relevantPositions = positions[relevantIndices,:]
        #         clippedPositions = positions[relevantIndices,:]
        #         print(relevantIndices)
        #         print(relevantPositions)
        #         clippedPositions[:,0] = torch.clamp(clippedPositions[:,0], min = config['domain']['virtualMin'][0], max = config['domain']['virtualMax'][0])
        #         clippedPositions[:,1] = torch.clamp(clippedPositions[:,1], min = config['domain']['virtualMin'][1], max = config['domain']['virtualMax'][1])
        #         print(clippedPositions)
        #         distances = torch.linalg.norm(clippedPositions - relevantPositions, axis =1)
        #         print(distances)
        #         iMin = torch.argmin(distances)
        #         print(iMin)
        #         for i in range(relevantIndices.shape[0]):
        #             indices[relevantIndices[i]] = state['UID'][relevantIndices[i]] if i == iMin else -1
        #             positions[relevantIndices[i]] = clippedPositions[i] if i == iMin else positions[relevantIndices[i]]

        #     indices = tempUIDs[indices != -1]
        #     print('indices after', indices.shape, indices)
        #     args = torch.argsort(state['UID'][indices])
        #     print('args', args.shape, args)
        #     indices = indices[args]
        #     print('indices after sorting', indices.shape, indices)
                



        counter = torch.zeros(state['numParticles'], dtype=torch.int64).to(config['device'])
        uidCounter = scatter(torch.ones(state['numParticles'], dtype=torch.int64).to(config['device']), state['UID'], dim = 0, dim_size=state['realParticles'])

        virtualMin = config['domain']['virtualMin']
        virtualMax = config['domain']['virtualMax']
        if config['domain']['periodicX']:
            counter[positions[:,0] < virtualMin[0]] = -1
            counter[positions[:,0] >= virtualMax[0]] = -1
        if config['domain']['periodicY']:        
            counter[positions[:,1] < virtualMin[1]] = -1    
            counter[positions[:,1] >= virtualMax[1]] = -1
            
        deletionCounter = scatter(counter, state['UID'], dim = 0, dim_size=state['realParticles'])
        actualCounter = uidCounter + deletionCounter
        problematicUIDs = state['UID'][:state['realParticles']][actualCounter != 1]
        indices = torch.ones(state['numParticles'], dtype = torch.int64, device=config['device']) * -1
        indices[counter != -1] = state['UID'][counter != -1]

        tempUIDs = torch.arange(state['numParticles'], dtype=torch.int64, device=config['device'])
        for uid in problematicUIDs:
            relevantIndices = tempUIDs[state['UID'] == uid]
            relevantPositions = positions[relevantIndices,:]
            clippedPositions = positions[relevantIndices,:]
            clippedPositions[:,0] = torch.clamp(clippedPositions[:,0], min = config['domain']['virtualMin'][0], max = config['domain']['virtualMax'][0])
            clippedPositions[:,1] = torch.clamp(clippedPositions[:,1], min = config['domain']['virtualMin'][1], max = config['domain']['virtualMax'][1])
            distances = torch.linalg.norm(clippedPositions - relevantPositions, axis =1)
            iMin = torch.argmin(distances)
            for i in range(relevantIndices.shape[0]):
                indices[relevantIndices[i]] = state['UID'][relevantIndices[i]] if i == iMin else -1
                positions[relevantIndices[i]] = clippedPositions[i] if i == iMin else positions[relevantIndices[i]]

        indices = tempUIDs[indices != -1]
        args = torch.argsort(state['UID'][indices])
        indices = indices[args]

        return indices

        indices = torch.arange(positions.shape[0], dtype=torch.int64).to(config['device'])
        virtualMin = config['domain']['virtualMin']
        virtualMax = config['domain']['virtualMax']


        if config['domain']['periodicX']:
            indices[positions[:,0] < virtualMin[0]] = -1
            indices[positions[:,0] >= virtualMax[0]] = -1
            # counter[positions[:,0] < virtualMin[0]] 
        if config['domain']['periodicY']:        
            indices[positions[:,1] < virtualMin[1]] = -1    
            indices[positions[:,1] >= virtualMax[1]] = -1
        
        


        indices = indices[indices != -1]

        
        return indices
    
def createGhostParticles(positions, config):
    with record_function('periodicBC - creating ghost particles'):
        indices = torch.arange(positions.shape[0], dtype=torch.int64).to(config['device'])
        virtualMin = config['domain']['virtualMin']
        virtualMax = config['domain']['virtualMax']
        
        mask_xp = positions[:,0] >= virtualMax[0] - config['domain']['buffer'] * config['support']
        mask_xn = positions[:,0] < virtualMin[0] + config['domain']['buffer'] * config['support']
        mask_yp = positions[:,1] >= virtualMax[1] - config['domain']['buffer'] * config['support']
        mask_yn = positions[:,1] < virtualMin[1] + config['domain']['buffer'] * config['support']

        filter_xp = indices[mask_xp]
        filter_xn = indices[mask_xn]
        filter_yp = indices[mask_yp]
        filter_yn = indices[mask_yn]
        
        mask_xp_yp = torch.logical_and(mask_xp, mask_yp)
        mask_xp_yn = torch.logical_and(mask_xp, mask_yn)
        mask_xn_yp = torch.logical_and(mask_xn, mask_yp)
        mask_xn_yn = torch.logical_and(mask_xn, mask_yn)
        
        filter_xp_yp = indices[torch.logical_and(mask_xp, mask_yp)]
        filter_xp_yn = indices[torch.logical_and(mask_xp, mask_yn)]
        filter_xn_yp = indices[torch.logical_and(mask_xn, mask_yp)]
        filter_xn_yn = indices[torch.logical_and(mask_xn, mask_yn)]
        
        main = filter_xp.shape[0] + filter_xn.shape[0] + filter_yp.shape[0] + filter_yn.shape[0]
        corner = filter_xp_yp.shape[0] + filter_xp_yn.shape[0] + filter_xn_yp.shape[0] + filter_xn_yn.shape[0]
        
        ghosts_xp = torch.zeros((filter_xp.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_xp[:,0] -=  virtualMax[0] - virtualMin[0]

        ghosts_yp = torch.zeros((filter_yp.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_yp[:,1] -=  virtualMax[1] - virtualMin[1]

        ghosts_xn = torch.zeros((filter_xn.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_xn[:,0] +=  virtualMax[0] - virtualMin[0]

        ghosts_yn = torch.zeros((filter_yn.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_yn[:,1] +=  virtualMax[1] - virtualMin[1]


        ghosts_xp_yp = torch.zeros((filter_xp_yp.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_xp_yp[:,0] -=  virtualMax[0] - virtualMin[0]
        ghosts_xp_yp[:,1] -=  virtualMax[1] - virtualMin[1]

        ghosts_xp_yn = torch.zeros((filter_xp_yn.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_xp_yn[:,0] -=  virtualMax[0] - virtualMin[0]
        ghosts_xp_yn[:,1] +=  virtualMax[1] - virtualMin[1]

        ghosts_xn_yp = torch.zeros((filter_xn_yp.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_xn_yp[:,0] +=  virtualMax[0] - virtualMin[0]
        ghosts_xn_yp[:,1] -=  virtualMax[1] - virtualMin[1]

        ghosts_xn_yn = torch.zeros((filter_xn_yn.shape[0], positions.shape[1]), dtype = config['precision'], device = config['device'])
        ghosts_xn_yn[:,0] +=  virtualMax[0] - virtualMin[0]
        ghosts_xn_yn[:,1] +=  virtualMax[1] - virtualMin[1]
        
        filters = []
        offsets = []
        if config['domain']['periodicX']:
            filters.append(filter_xp)
            filters.append(filter_xn)
            offsets.append(ghosts_xp)
            offsets.append(ghosts_xn)
        if config['domain']['periodicY']:
            filters.append(filter_yp)
            filters.append(filter_yn)
            offsets.append(ghosts_yp)
            offsets.append(ghosts_yn)
        if config['domain']['periodicX'] and config['domain']['periodicY']:
            filters.append(filter_xp_yp)
            filters.append(filter_xp_yn)
            filters.append(filter_xn_yp)
            filters.append(filter_xn_yn)
            offsets.append(ghosts_xp_yp)
            offsets.append(ghosts_xp_yn)
            offsets.append(ghosts_xn_yp)
            offsets.append(ghosts_xn_yn)
        
        return filters, offsets

# ghostIndices, ghostOffsets = createGhostParticles(simulationState['fluidPosition'])
# fig, axis = plt.subplots(1,1, figsize=(6,6), sharex = True, sharey = True, squeeze = False)

# sc = axis[0,0].scatter(simulationState['fluidPosition'][:,0], simulationState['fluidPosition'][:,1], c = simulationState['fluidDensity'], s = 16)
# axis[0,0].axis('equal')
# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 


# r = patches.Rectangle((config['domain']['min'][0], config['domain']['min'][1]), 
#                   config['domain']['max'][0] - config['domain']['min'][0], config['domain']['max'][1] - config['domain']['min'][1], linewidth=3, edgecolor='r', facecolor='none')
# axis[0,0].add_patch(r)
# r = patches.Rectangle((config['domain']['virtualMin'][0], config['domain']['virtualMin'][1]), 
#                   config['domain']['virtualMax'][0] - config['domain']['virtualMin'][0], config['domain']['virtualMax'][1] - config['domain']['virtualMin'][1], linewidth=3, edgecolor='g', facecolor='none')
# axis[0,0].add_patch(r)

# p = simulationState['fluidPosition']
# for indices, offsets in zip(ghostIndices, ghostOffsets):
#     axis[0,0].scatter(p[indices,0]+offsets[:,0], p[indices,1] + offsets[:,1],s=16)

# fig.tight_layout()

def enforcePeriodicBC(config, simulationState):
    with record_function('periodicBC - enforce BC'):
        if config['domain']['periodicX'] or config['domain']['periodicY']:
            if not 'realParticles' in simulationState:
                simulationState['realParticles'] = simulationState['numParticles']
    #         print('Old Particle Count: ', simulationState['numParticles'] )
            realParticles = filterVirtualParticles(simulationState['fluidPosition'], config, simulationState)
            for arr in simulationState:
                if not torch.is_tensor(simulationState[arr]):
                    continue
                if simulationState[arr].shape[0] == simulationState['numParticles']:
                    simulationState[arr] = simulationState[arr][realParticles]

            simulationState['numParticles'] = simulationState['fluidPosition'].shape[0]
            if 'realParticles' in simulationState:
                if simulationState['realParticles'] != simulationState['fluidPosition'].shape[0]:
                    print('panik, deleted or removed actual particles at time', simulationState['time'])

            simulationState['realParticles'] = simulationState['fluidPosition'].shape[0]
    #         print('After pruning: ', simulationState['numParticles'] )

            


            ghostIndices, ghostOffsets = createGhostParticles(simulationState['fluidPosition'], config)

            ghostIndices = torch.cat(ghostIndices)
            ghostOffsets = torch.vstack(ghostOffsets)

            realParticles = filterVirtualParticles(simulationState['fluidPosition'], config, simulationState)
            for arr in simulationState:
                if not torch.is_tensor(simulationState[arr]):
                    continue
                if simulationState[arr].shape[0] == simulationState['numParticles']:
                    if arr == 'fluidPosition':
                        simulationState[arr] = torch.cat((simulationState[arr],simulationState[arr][ghostIndices] + ghostOffsets))
                    else:
                        simulationState[arr] = torch.cat((simulationState[arr],simulationState[arr][ghostIndices]))

            simulationState['numParticles'] = simulationState['fluidPosition'].shape[0]
    #         print('New Particle Count: ', simulationState['numParticles'] )
            
            ones = torch.ones(simulationState['realParticles'], dtype = torch.int64, device=config['device']) * -1
            simulationState['ghostIndices'] = torch.cat((ones, ghostIndices))
            simulationState['ghosts'] = ghostIndices
    
def syncQuantity(qty, config, simulationState):
    with record_function('periodicBC - syncing quantity'):
        if config['domain']['periodicX'] or config['domain']['periodicY']:
            ghosts = simulationState['ghosts']
            qty[simulationState['numParticles'] - ghosts.shape[0]:] = qty[simulationState['ghosts']]


def velocityBC(config, state):
    if not 'velocitySources' in config:
        return
    with record_function('velocityBC - enforcing'):
        state['fluidGamma'] = torch.ones(state['fluidArea'].shape, device=config['device'], dtype=config['precision'])
        for source in config['velocitySources']:
        #     print(source)
            velTensor = torch.tensor(source['velocity'], device= config['device'], dtype=config['precision'])
            curSpeed = velTensor if not 'rampTime' in source else velTensor * np.clip(state['time'] / source['rampTime'], a_min = 0., a_max = 1.)
        #     print(curSpeed)

            xmask = torch.logical_and(state['fluidPosition'][:,0] >= source['min'][0], state['fluidPosition'][:,0] <= source['max'][0])
            ymask = torch.logical_and(state['fluidPosition'][:,1] >= source['min'][1], state['fluidPosition'][:,1] <= source['max'][1])

            mask = torch.logical_and(xmask, ymask)

            active = torch.any(mask)
            # print(xmask)
            # print(ymask)
            # print(mask)
            # print(active)
        #     print(mask)
        #     print(torch.any(mask))
            mu = 3.5
            xr = (state['fluidPosition'][:,0] - source['min'][0]) / (source['max'][0] - source['min'][0])

            if source['min'][0] < 0:
                xr = 1 - xr

            gamma = (torch.exp(torch.pow(torch.clamp(xr,min = 0, max = 1), mu)) - 1) / (np.exp(1) - 1)

            # gamma = 1 - (torch.exp(torch.pow(xr,mu)) - 1) / (np.exp(1) - 1)
            state['fluidGamma'] = torch.min(gamma, state['fluidGamma'])
            if active:
                # print(gamma.shape)
                # gamma = gamma[mask]
                state['fluidVelocity'][mask,:] = state['fluidVelocity'][mask,:] * (1 - gamma)[mask,None] + gamma[mask,None] * curSpeed
            

        #     print('\n')





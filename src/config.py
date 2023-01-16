from .generation import *
from torch.profiler import record_function

def adjustDomainForEmitters(config, fixSizes = True):
    with record_function('config - adjusting domain'):
        config['domain']['virtualMin'] = config['domain']['min']
        config['domain']['virtualMax'] = config['domain']['max']
        if config['domain']['periodicX'] or config['domain']['periodicY']:
            localPacking = config['packing'] / config['minCompression'] * config['support']

            minVolume = [config['maxValue'], config['maxValue']]
            maxVolume = [-config['maxValue'], -config['maxValue']]

            extentX = [config['maxValue'], -config['maxValue']]
            extentY = [config['maxValue'], -config['maxValue']]

            for emitter in config['emitters']:
                extentX[0] = min(extentX[0], emitter['min'][0])
                extentX[1] = max(extentX[1], emitter['max'][0])
                extentY[0] = min(extentY[0], emitter['min'][1])
                extentY[1] = max(extentY[1], emitter['max'][1])
            if fixSizes:
                for emitter in config['emitters']:
                    emitterPacking = config['packing'] / emitter['compression'] * config['support']
                    n = np.ceil((emitter['max'][0] - emitter['min'][0]) / emitterPacking)
                    m = np.ceil((emitter['max'][1] - emitter['min'][1]) / emitterPacking)

                    emitter['max'][0] = emitter['min'][0] + n * emitterPacking
                    emitter['max'][1] = emitter['min'][1] + m * emitterPacking

            for emitter in config['emitters']:
                minVolume[0] = min(minVolume[0], emitter['min'][0])
                maxVolume[0] = max(maxVolume[0], emitter['max'][0])
                minVolume[1] = min(minVolume[1], emitter['min'][1])
                maxVolume[1] = max(maxVolume[1], emitter['max'][1])

            # print(minVolume)
            # print(maxVolume)

            if config['domain']['min'][1] > minVolume[1] or config['domain']['max'][1] < maxVolume[1]:

                config['domain']['min'][1] = minVolume[1] - localPacking / 2
                config['domain']['max'][1] = maxVolume[1] + localPacking / 2

                config['domain']['virtualMin'][1] = config['domain']['min'][1]
                config['domain']['virtualMax'][1] = config['domain']['max'][1]

                config['domain']['min'][1] = config['domain']['min'][1] - config['support'] * config['domain']['buffer']
                config['domain']['max'][1] = config['domain']['max'][1] + config['support'] * config['domain']['buffer']
            else:
                config['domain']['virtualMin'][1] = config['domain']['min'][1]
                config['domain']['virtualMax'][1] = config['domain']['max'][1]
                config['domain']['min'][1] = config['domain']['min'][1] - config['support'] * config['domain']['buffer']
                config['domain']['max'][1] = config['domain']['max'][1] + config['support'] * config['domain']['buffer']

            if config['domain']['min'][0] > minVolume[0] or config['domain']['max'][0] < maxVolume[0]:
                config['domain']['min'][0] = minVolume[0] - localPacking / 2
                config['domain']['max'][0] = maxVolume[0] + localPacking / 2

                config['domain']['virtualMin'][0] = config['domain']['min'][0]
                config['domain']['virtualMax'][0] = config['domain']['max'][0]

                config['domain']['min'][0] = config['domain']['min'][0] - config['support'] * config['domain']['buffer']
                config['domain']['max'][0] = config['domain']['max'][0] + config['support'] * config['domain']['buffer']
            else:
                config['domain']['virtualMin'][0] = config['domain']['min'][0]
                config['domain']['virtualMax'][0] = config['domain']['max'][0]
                config['domain']['min'][0] = config['domain']['min'][0] - config['support'] * config['domain']['buffer']
                config['domain']['max'][0] = config['domain']['max'][0] + config['support'] * config['domain']['buffer']
        else:
            config['domain']['min'][0] = config['domain']['min'][0] - config['support'] * config['domain']['buffer']
            config['domain']['max'][0] = config['domain']['max'][0] + config['support'] * config['domain']['buffer']
        

def initializeSimulation(config):    
    with record_function('config - initializing simulation'):
        simulationState = {}
        positions = []
        areas = []
        emitterVelocities = []
        emitterDensities = []
        for emitter in config['emitters']:
            emitterPositions = genParticles(
                torch.tensor(emitter['min'], dtype = config['precision'], device=config['device']), 
                torch.tensor(emitter['max'], dtype = config['precision'], device=config['device']), 
                emitter['radius'], config['packing'] / emitter['compression'], config)
            emitterAreas = torch.ones(emitterPositions.shape[0], dtype = config['precision'], device=config['device']) * config['area']
            
            emitterVelocity = torch.ones((emitterPositions.shape[0], 2), dtype = config['precision'], device=config['device'])
            emitterVelocity[:,0] = emitter['velocity'][0]
            emitterVelocity[:,1] = emitter['velocity'][1]
            
            emitterDensity = torch.ones(emitterPositions.shape[0], dtype = config['precision'], device=config['device']) * emitter['density']
            
            positions.append(emitterPositions)
            areas.append(emitterAreas)
            emitterVelocities.append(emitterVelocity)
            emitterDensities.append(emitterDensity)
        #     break

        simulationState['fluidPosition'] =  torch.vstack(positions)
        simulationState['UID'] = torch.arange(simulationState['fluidPosition'].shape[0], dtype=torch.int64, device = config['device'])
        simulationState['ghostIndices'] = torch.ones(simulationState['fluidPosition'].shape[0], dtype=torch.int64, device = config['device']) * -1
        simulationState['fluidArea'] = torch.cat(areas)
        simulationState['fluidVelocity'] = torch.cat(emitterVelocities)
        simulationState['fluidAcceleration'] = torch.zeros(simulationState['fluidVelocity'].shape, device=config['device'], dtype=config['precision'])
        simulationState['fluidPressure'] = torch.zeros(simulationState['fluidArea'].shape, device=config['device'], dtype=config['precision'])
        simulationState['fluidRestDensity'] = torch.cat(emitterDensities)
        simulationState['numParticles'] = simulationState['fluidPosition'].shape[0]
        simulationState['time'] = 0.
        simulationState['timestep'] = int(0)
        
        return simulationState

from .plotting import *
from datetime import datetime
import os
import h5py
import copy

def setupSimulation(config, simFn, plotFn, nx = 256, ny = 256, saveFrames = False, figsize = (8,8)):    
    with record_function('config setup simulation'):
        state = initializeSimulation(config)
        if config['export']['active']:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            state['exportPath'] = './export/%s - %s.hdf5' %(config['export']['prefix'], timestamp)
            state['outFile'] = h5py.File(state['exportPath'],'w')
            state['exportCounter'] = 0
            if not os.path.exists('./export/'):
                os.makedirs(state['./export/'])
                
            state['outFile'].attrs['dt'] = config['dt']
            state['outFile'].attrs['area'] = config['area']
            state['outFile'].attrs['support'] = config['support']
            state['outFile'].attrs['radius'] = config['radius']
            # state['outFile'].attrs['config'] = config


            def convertDict(dic):
            #     print('dict:', dic)
                if type(dic) is list:
                    return [convertDict(d) for d in dic]
                d = copy.copy(dic)
                for v in d:
            #         print(v)
                    if type(d[v]) is np.ndarray:
                        d[v] = d[v].tolist()
                    if type(d[v]) is torch.Tensor:
            #             print(d[v])
                        d[v] = d[v].detach().cpu().numpy().tolist()
                return str(d)
                

            state['outFile'].attrs['domain'] = convertDict(config['domain'])
            state['outFile'].attrs['solidBoundary'] = convertDict(config['solidBoundary'])
            state['outFile'].attrs['velocitySources'] = convertDict(config['velocitySources'])
            state['outFile'].attrs['emitters'] = convertDict(config['emitters'])
            state['outFile'].attrs['dfsph'] = convertDict(config['dfsph'])

            state['outFile'].attrs['viscosityConstant'] = config['viscosityConstant']
            state['outFile'].attrs['boundaryViscosityConstant'] = config['boundaryViscosityConstant']
            state['outFile'].attrs['packing'] = config['packing']
            state['outFile'].attrs['spacing'] = config['spacing']
            state['outFile'].attrs['spacingContribution'] = config['spacingContribution'].detach().cpu().numpy().tolist()
            # state['outFile'].attr['spacing']


        simFn(config, state)


        fig, im, axis, cbar = initialPlot(config, state, plotFn, figsize = figsize)
        if saveFrames:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            state['path'] = './output/%s/' %timestamp
            if not os.path.exists(state['path']):
                os.makedirs(state['path'])
            imagePath = state['path'] + '%05d.png' % state['timestep']
            plt.savefig(imagePath)

        def simulate(steps, axis, config, state, simFn, plotFn, nx, ny):
            for i in tqdm(range(steps), leave=False):
                simFn(config, state)

                if state['timestep'] % 1 == 0:
                    updatePlot(config, state, fig, axis, im, cbar, plotFn, nx, ny)
                if saveFrames:
                    imagePath = state['path'] + '%05d.png' % state['timestep']
                    plt.savefig(imagePath)
            # updatePlot(config, state, fig, axis[0,0], im, cbar, plotFn, nx, ny, saveFrames = saveFrames)

        return state, lambda steps: simulate(steps, axis, config, state, simFn, plotFn, nx, ny)
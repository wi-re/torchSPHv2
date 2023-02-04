
import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .module import Module
from .parameter import Parameter
from .util import *
from .kernels import getKernelFunctions
from .modules.sdfBoundary import sdPolyDerAndIntegral

class SPHSimulation():
    def getBasicParameters(self):
        basicParticleParameters = [
            Parameter('particle', 'radius', 'float', 0.014426521330548324, required = False, export = True, hint = '')
        ]
        
        basicSimulationParameters = [
            Parameter('simulation', 'scheme', 'string', 'dfsph', required = False, export = True, hint = ''),
            Parameter('simulation', 'verbose', 'bool', True, required = False, export = True, hint = ''),
            Parameter('simulation', 'boundaryScheme', 'string', 'SDF', required = False, export = True, hint = ''),
            Parameter('simulation', 'bodyForces', 'bool', True, required = False, export = True, hint = ''),
            Parameter('simulation', 'densityScheme', 'string', 'summation', required = False, export = True, hint = ''),
            Parameter('akinciBoundary', 'gamma', 'float', 0.7, required = False, export = True, hint = ''),
            Parameter('pressure', 'gamma', 'float', 7.0, required = False, export = True, hint = ''),
            Parameter('pressure', 'kappa', 'float', 1.3, required = False, export = True, hint = ''),
            Parameter('pressure', 'fluidPressureTerm', 'str', 'TaitEOS', required = False, export = True, hint = ''),
            Parameter('pressure', 'boundaryPressureTerm', 'str', 'PBSPH', required = False, export = True, hint = '')
        ]
        
        basicKernelParameters = [
            Parameter('kernel', 'targetNeighbors', 'int', 20, required = False, export = True, hint = ''),
            Parameter('kernel', 'defaultKernel', 'string', 'wendland2', required = False, export = True, hint = '')
        ]
        
        basicComputeParameters = [
            Parameter('compute', 'maxNeighbors', 'int', 256, required = False, export = True, hint = ''),
            Parameter('compute', 'device', 'string', 'cuda', required = False, export = True, hint = ''),
            Parameter('compute', 'floatprecision', 'string', 'single', required = False, export = True, hint = '')
        ]   
        
        basicFluidParameters = [
            Parameter('fluid', 'restDensity', 'float', 1000, required = False, export = True, hint = '')  ,
            Parameter('fluid', 'gravity', 'float array', [0,0], required = False, export = True, hint = '')  ,
            Parameter('fluid', 'backgroundPressure', 'bool', False, required = False, export = True, hint = '')  ,
            Parameter('fluid', 'c0', 'float', -1, required = False, export = True, hint = '')  ,
        ]
        
        basicIntegrationParameters = [
            Parameter('integration', 'dt', 'float', 0.002, required = False, export = True, hint = '')
        ]
        
        basicViscosityParameters = [
            Parameter('diffusion', 'velocityScheme', 'string', 'xsph', required = False, export = True, hint = ''),
            Parameter('diffusion', 'densityScheme', 'string', 'MOG', required = False, export = True, hint = ''),
            Parameter('diffusion', 'alpha', 'float', 0.01, required = False, export = True, hint = ''),
            Parameter('diffusion', 'delta', 'float', 0.01, required = False, export = True, hint = ''),
            Parameter('diffusion', 'kinematic', 'float', 0.01, required = False, export = True, hint = ''),
            Parameter('diffusion', 'boundaryDiffusion', 'bool', True, required = False, export = True, hint = ''),
        ]
        
        basicDomainParameters = [
            Parameter('domain', 'min', 'float array', [-1, -1], required = False, export = True, hint = ''),
            Parameter('domain', 'max', 'float array', [ 1,  1], required = False, export = True, hint = ''),
            Parameter('domain', 'adjustDomain', 'bool', False,  required = False, export = True, hint = ''),
            Parameter('domain', 'adjustParticle', 'bool', False,  required = False, export = True, hint = '')
        ]
        
        basicExportParameters = [
            Parameter('export', 'active', 'bool', False, required = False, export = True, hint = ''),
            Parameter('export', 'prefix', 'string', 'unnamed', required = False, export = True, hint = '')
        ]
        
        basicPeriodicBCParameters = [
            Parameter('periodicBC', 'periodicX',  'bool', False, required = False, export = True, hint = ''),
            Parameter('periodicBC', 'periodicY',  'bool', False, required = False, export = True, hint = ''),
            Parameter('periodicBC',    'buffer', 'float',     2, required = False, export = True, hint = '')
            
        ]
        
        return basicParticleParameters + basicSimulationParameters + basicKernelParameters + basicComputeParameters + basicFluidParameters + \
            basicIntegrationParameters + basicViscosityParameters + basicDomainParameters + basicExportParameters + basicPeriodicBCParameters
    
    def evalPacking(self, arg):
        packing = torch.tensor(arg, dtype = self.dtype, device = self.device)

        minDomain = torch.tensor([\
                -2 * self.config['particle']['support'],\
                -2 * self.config['particle']['support']\
            ], device = self.device, dtype = self.dtype)
        maxDomain = torch.tensor([\
                 2 * self.config['particle']['support'],\
                 2 * self.config['particle']['support']\
            ], device = self.device, dtype = self.dtype)
        
        fluidPosition = genParticlesCentered(minDomain, maxDomain, \
                            self.config['particle']['radius'], self.config['particle']['support'], packing, \
                            self.dtype, self.device)

        fluidArea = torch.ones(fluidPosition.shape[0], device = self.device, dtype=self.dtype) * self.config['particle']['area']
        centralPosition = torch.tensor([[0,0]], device = self.device, dtype=self.dtype)

        row, col = radius(centralPosition, fluidPosition, \
                          self.config['particle']['support'], max_num_neighbors = self.config['compute']['maxNeighbors'])
        fluidNeighbors = torch.stack([row, col], dim = 0)

        fluidDistances = (centralPosition - fluidPosition[fluidNeighbors[0]])
        fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

        fluidRadialDistances /= self.config['particle']['support']
        rho = scatter(\
                self.kernel(fluidRadialDistances, self.config['particle']['support']) * fluidArea[fluidNeighbors[1]], \
                fluidNeighbors[1], dim=0, dim_size=centralPosition.size(0), reduce="add")

        return ((1 - rho)**2).detach().cpu().numpy()[0]

    def evalSpacing(self, arg):
        s = torch.tensor(arg, dtype=self.dtype, device = self.device)
        support = self.config['particle']['support']

        minDomain = torch.tensor([\
                -2 * self.config['particle']['support'],\
                -2 * self.config['particle']['support']\
            ], device = self.device, dtype = self.dtype)
        maxDomain = torch.tensor([\
                 2 * self.config['particle']['support'],\
                 2 * self.config['particle']['support']\
            ], device = self.device, dtype = self.dtype)
        
        fluidPosition = genParticlesCentered(minDomain, maxDomain, \
                            self.config['particle']['radius'], self.config['particle']['support'], \
                            self.config['particle']['packing'],self.dtype, self.device)
        
        fluidPosition = fluidPosition[fluidPosition[:,1] >= 0,:]
        centralPosition = torch.tensor([[0,0]], dtype = self.dtype, device=self.device)

        row, col = radius(centralPosition, fluidPosition, support, max_num_neighbors = self.config['compute']['maxNeighbors'])
        fluidNeighbors = torch.stack([row, col], dim = 0)

        fluidDistances = (centralPosition - fluidPosition[fluidNeighbors[0]])
        fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

        fluidRadialDistances /= support
        rho = scatter(self.kernel(fluidRadialDistances, support) * self.config['particle']['area'], fluidNeighbors[1], dim=0, dim_size=centralPosition.size(0), reduce="add")

        sdf, sdfGrad, b, bGrad = sdPolyDerAndIntegral(\
                torch.tensor([\
                    [ -support * 2, -support * 2],\
                    [  support * 2, -support * 2],\
                    [  support * 2,  s * support],\
                    [ -support * 2,  s * support],\
                             ], dtype= self.dtype, device = self.device),\
                p = centralPosition, support = support
        )

        return ((1- (rho + b))**2).detach().cpu().numpy()[0]
        
        
    def evalContrib(self):
        s = torch.tensor(self.config['particle']['spacing'], dtype=self.dtype, device = self.device)
        centralPosition = torch.tensor([[0,0]], dtype=self.dtype, device = self.device)

        support = self.config['particle']['support']
        
        sdf, sdfGrad, b, bGrad = sdPolyDerAndIntegral(\
                torch.tensor([\
                    [ -support * 2, -support * 2],\
                    [  support * 2, -support * 2],\
                    [  support * 2,  s * support],\
                    [ -support * 2,  s * support],\
                             ], dtype=self.dtype, device = self.device),\
                p = centralPosition, support = support
        )

        return b

    def processVelocitySources(self):
        if 'velocitySource' not in self.config:
            return
        for s in self.config['velocitySource']:
            source =self.config['velocitySource'][s]
#             print(emitter)
            if 'rampTime' not in source:
                source[ 'rampTime'] =1.0
            if 'min' not in source:
                raise Exception('Provided velocity source has no min extent, configuration invalid')
            if 'max' not in source:
                raise Exception('Provided velocity source has no max extent, configuration invalid')
            if 'velocity' not in source:
                raise Exception('Provided velocity source has no velocity, configuration invalid')
    
    def processEmitters(self):
        if 'emitter' not in self.config:
            return
        
        minCompression = self.config['compute']['maxValue']
        
        for emitterName in self.config['emitter']:
#             print(emitter)
            if self.verbose: print('processing emitter %s' % emitterName)
            emitter = self.config['emitter'][emitterName]
            emitter[ 'fillDomain'] = False if 'fillDomain' not in emitter else emitter['fillDomain']
            if emitter['fillDomain']:
                if 'min' in emitter or 'max' in emitter:
                    raise Exception('Extent provided for fillDomain emitter, configuration invalid')
                    
                spacing = self.config['particle']['spacing'] * self.config['particle']['support']
                packing = self.config['particle']['packing'] * self.config['particle']['support']

                emitter[        'min'] = [self.config['domain']['min'][0] + packing / 2, self.config['domain']['min'][1] + packing /2]
                emitter[        'max'] = [self.config['domain']['max'][0] - packing / 2, self.config['domain']['max'][1] - packing / 2]
                    
            else:
                if 'min' not in emitter or 'max' not in emitter:
                    raise Exception('Extent not provided for emitter, configuration invalid')
                
            emitter[     'radius'] = emitter['radius'] if 'radius' in emitter else self.config['particle']['radius']
            emitter['restDensity'] = emitter['restDensity'] if 'restDensity' in emitter else self.config['fluid']['restDensity']
            emitter[       'type'] = emitter['type'] if 'type' in emitter else 'once'
            emitter['compression'] = emitter['compression'] if 'compression' in emitter else 1.
            emitter[   'velocity'] = emitter['velocity'] if 'velocity' in emitter else [0.0,0.0]
            emitter[      'shape'] = emitter['shape'] if 'shape' in emitter else 'rectangle'
            emitter[      'adjust'] = emitter['adjust'] if 'adjust' in emitter else False
            
            if emitter['adjust']:
                spacing = self.config['particle']['spacing'] * self.config['particle']['support']
                packing = self.config['particle']['packing'] * self.config['particle']['support']
                if self.config['simulation']['boundaryScheme'] == 'solid':
                    emitter[        'min'] = [emitter['min'][0] + spacing / 2, emitter['min'][1] + spacing / 2]
                    emitter[        'max'] = [emitter['max'][0] - spacing / 2, emitter['max'][1] - spacing / 2]
                else:                    
                    # emitter[        'min'] = [emitter['min'][0] + packing / 2, emitter['min'][1] + packing / 2]
                    # emitter[        'max'] = [emitter['max'][0] - packing / 2, emitter['max'][1] - packing / 2]
                    emitter[        'min'] = [emitter['min'][0] + spacing, emitter['min'][1] + spacing]
                    emitter[        'max'] = [emitter['max'][0] - spacing, emitter['max'][1] - spacing]
                
                        
            minCompression = min(minCompression, emitter['compression'])
    
    def addBoundaryBoundaries(self):
        if self.config['periodicBC']['periodicX'] and self.config['periodicBC']['periodicY']:
            return
        if 'solidBC' not in self.config:
            self.config['solidBC'] = {}
        if self.config['periodicBC']['periodicX'] and not self.config['periodicBC']['periodicY']:
            minDomain = self.config['domain']['virtualMin']
            maxDomain = self.config['domain']['virtualMax']
            buffer = self.config['particle']['support'] * self.config['periodicBC']['buffer']

            self.config['solidBC']['bottomBoundary'] = {
                'vertices':[
                    [minDomain[0],minDomain[1]],
                    [maxDomain[0],minDomain[1]],
                    [maxDomain[0],minDomain[1] + buffer],
                    [minDomain[0],minDomain[1] + buffer]
                ],
                'inverted':False
            }
            self.config['solidBC']['topBoundary'] = {
                'vertices':[
                    [minDomain[0],maxDomain[1] - buffer],
                    [maxDomain[0],maxDomain[1] - buffer],
                    [maxDomain[0],maxDomain[1]],
                    [minDomain[0],maxDomain[1]]
                ],
                'inverted':False
            }
        if not self.config['periodicBC']['periodicX'] and self.config['periodicBC']['periodicY']:
            minDomain = self.config['domain']['virtualMin']
            maxDomain = self.config['domain']['virtualMax']
            buffer = self.config['particle']['support'] * self.config['periodicBC']['buffer']

            self.config['solidBC']['leftBoundary'] = {
                'vertices':[
                    [minDomain[0]         , minDomain[1]],
                    [minDomain[0] + buffer, minDomain[1]],
                    [minDomain[0] + buffer, maxDomain[1]],
                    [minDomain[0]         , maxDomain[1]]
                ],
                'inverted':False
            }
            self.config['solidBC']['rightBoundary'] = {
                'vertices':[
                    [maxDomain[0] - buffer, minDomain[1]],
                    [maxDomain[0]         , minDomain[1]],
                    [maxDomain[0]         , maxDomain[1]],
                    [maxDomain[0] - buffer, maxDomain[1]]
                ],
                'inverted':False
            }
        if not self.config['periodicBC']['periodicX'] and not self.config['periodicBC']['periodicY']:
            minDomain = self.config['domain']['virtualMin']
            maxDomain = self.config['domain']['virtualMax']
            buffer = self.config['particle']['support'] * self.config['periodicBC']['buffer']

            self.config['solidBC']['domainBoundary'] = {
                'vertices':[
                    [minDomain[0] + buffer, minDomain[1] + buffer],
                    [maxDomain[0] - buffer, minDomain[1] + buffer],
                    [maxDomain[0] - buffer, maxDomain[1] - buffer],
                    [minDomain[0] + buffer, maxDomain[1] - buffer]
                ],
                'inverted':True
            }

    def initializeSimulation(self):
        with record_function('config - initializing simulation'):
            self.simulationState = {}
            positions = []
            areas = []
            supports = []
            emitterVelocities = []
            emitterDensities = []
            for e in self.config['emitter']:
                if self.verbose: print(e)
                emitter = self.config['emitter'][e]
                if self.verbose: print(emitter)
                emitterPositions = genParticles(
                    torch.tensor(emitter['min'], dtype = self.dtype, device = self.device), 
                    torch.tensor(emitter['max'], dtype = self.dtype, device = self.device), 
                    emitter['radius'], self.config['particle']['packing'] / emitter['compression'], self.config['particle']['support'], self.dtype, self.device)
                
                

                if 'solidBC' in self.config:
                    if self.config['simulation']['boundaryScheme'] == 'SDF':
                        for bdy in self.config['solidBC']:
                            b = self.config['solidBC'][bdy]
                            polyDist, polyDer, bIntegral, bGrad = sdPolyDerAndIntegral(b['polygon'], emitterPositions, self.config['particle']['support'], inverted = b['inverted'])
                            # print('Particle count before filtering: ', particles.shape[0])
                            emitterPositions = emitterPositions[polyDist >= self.config['particle']['spacing'] * self.config['particle']['support'] * 0.99,:]
                            # print('Particle count after filtering: ', particles.shape[0])

                if emitter['shape'] == 'sphere':
                    center = (torch.tensor(emitter['max'], dtype = self.dtype, device = self.device) + \
                        torch.tensor(emitter['min'], dtype = self.dtype, device = self.device)) / 2
                    dist = (torch.tensor(emitter['max'], dtype = self.dtype, device = self.device) - \
                        torch.tensor(emitter['min'], dtype = self.dtype, device = self.device))
#                     debugPrint(center)
#                     debugPrint(dist)
                    rad = max(dist[0], dist[1]) / 2
#                     debugPrint(rad)
                    centerDist = torch.linalg.norm(emitterPositions - center,axis=1)
#                     debugPrint(centerDist)
                    emitterPositions = emitterPositions[centerDist <= rad,:]
#                     debugPrint(emitterPositions)
                    
                        
                emitterAreas = torch.ones(emitterPositions.shape[0], dtype = self.dtype, device=self.device) * self.config['particle']['area']
                emitterSupport = torch.ones(emitterPositions.shape[0], dtype = self.dtype, device=self.device) * self.config['particle']['support']

                emitterVelocity = torch.ones((emitterPositions.shape[0], 2), dtype = self.dtype, device=self.device)
                emitterVelocity[:,0] = emitter['velocity'][0]
                emitterVelocity[:,1] = emitter['velocity'][1]

                emitterDensity = torch.ones(emitterPositions.shape[0], dtype = self.dtype, device=self.device)

                positions.append(emitterPositions)
                areas.append(emitterAreas)
                supports.append(emitterSupport)
                emitterVelocities.append(emitterVelocity)
                emitterDensities.append(emitterDensity)
            #     break

            self.simulationState[    'fluidPosition'] = torch.vstack(positions)
            self.simulationState[              'UID'] = torch.arange(self.simulationState['fluidPosition'].shape[0], dtype=torch.int64, device = self.device)
            self.simulationState[     'ghostIndices'] = torch.ones(self.simulationState['fluidPosition'].shape[0], dtype=torch.int64, device = self.device) * -1
            self.simulationState[     'fluidDensity'] = torch.ones(self.simulationState['fluidPosition'].shape[0], dtype=torch.float32, device = self.device)
            self.simulationState[        'fluidArea'] = torch.cat(areas)
            self.simulationState[     'fluidSupport'] = torch.ones(self.simulationState['fluidPosition'].shape[0], dtype=torch.float32, device = self.device) * self.config['particle']['support']
            self.simulationState[    'fluidVelocity'] = torch.cat(emitterVelocities)
            self.simulationState['fluidAcceleration'] = torch.zeros(self.simulationState['fluidVelocity'].shape, device=self.device, dtype=self.dtype)
            self.simulationState[    'fluidPressure'] = torch.zeros(self.simulationState['fluidArea'].shape, device=self.device, dtype=self.dtype)
            self.simulationState[ 'fluidRestDensity'] = torch.cat(emitterDensities)
            self.simulationState[ 'dpdt'] = torch.zeros_like(self.simulationState[     'fluidSupport'])
            self.simulationState[     'numParticles'] = self.simulationState['fluidPosition'].shape[0]
            self.simulationState[    'realParticles'] = self.simulationState['fluidPosition'].shape[0]
            self.simulationState[             'time'] = 0.
            self.simulationState[         'timestep'] = int(0)
            self.simulationState[               'dt'] = self.config['integration']['dt']
            
            if self.verbose: print('Initializing modules')
            for module in self.modules:        
                module.initialize(self.config, self.simulationState)

            self.perennialState = self.saveState(copy = False)
            self.simulationState = self.setupSimulationState(self.perennialState)

    def resetState(self):
        for module in self.modules:
            module.resetState(self.simulationState)

        self.simulationState.pop('fluidAcceleration', None)

    def sync(self, tensor):
        if hasattr(self, 'periodicBC'):
            self.periodicBC.syncToGhost(tensor, self.simulationState, self)

    def saveState(self, copy = False):
        mask = (self.simulationState['ghostIndices'] == -1) if 'ghostIndices' in self.simulationState else self.simulationState['UID'] > -1
        
        perennialState = {}
        perennialState['UID'] = self.simulationState['UID'][mask]
        perennialState['fluidPosition'] = self.simulationState['fluidPosition'][mask] if not copy else torch.clone(self.simulationState['fluidPosition'][mask])
        perennialState['fluidVelocity'] = self.simulationState['fluidVelocity'][mask] if not copy else torch.clone(self.simulationState['fluidVelocity'][mask])
        perennialState['fluidDensity'] = self.simulationState['fluidDensity'][mask] if not copy else torch.clone(self.simulationState['fluidDensity'][mask])
        perennialState['fluidSupport'] = self.simulationState['fluidSupport'][mask] if not copy else torch.clone(self.simulationState['fluidSupport'][mask])
        perennialState['fluidRestDensity'] = self.simulationState['fluidRestDensity'][mask] if not copy else torch.clone(self.simulationState['fluidRestDensity'][mask])
        perennialState['fluidArea'] = self.simulationState['fluidArea'][mask] if not copy else torch.clone(self.simulationState['fluidArea'][mask])
        perennialState['fluidPressure'] = self.simulationState['fluidPressure'][mask] if not copy else torch.clone(self.simulationState['fluidPressure'][mask])
        perennialState['fluidAcceleration'] = self.simulationState['fluidAcceleration'][mask] if not copy else torch.clone(self.simulationState['fluidAcceleration'][mask])
        if self.config['simulation']['densityScheme'] == 'continuum':
            perennialState['dpdt'] = self.simulationState['dpdt'][mask] if not copy else torch.clone(self.simulationState['dpdt'][mask])
       


        perennialState['numParticles'] = perennialState['fluidPosition'].shape[0]
        perennialState['realParticles'] = perennialState['fluidPosition'].shape[0]
        
        perennialState['dt'] = self.simulationState['dt']
        perennialState['time'] = self.simulationState['time']
        perennialState['timestep'] = self.simulationState['timestep']
        
        for module in self.modules:
            module.saveState(perennialState, copy = copy)
        
        return perennialState
#             return simulationState
    def setupSimulationState(self, perennialState):
        simulationState = {}
        simulationState['UID'] = torch.clone(perennialState['UID'])
        simulationState['fluidPosition'] = torch.clone(perennialState['fluidPosition'])
        simulationState['fluidVelocity'] = torch.clone(perennialState['fluidVelocity'])
        simulationState['fluidDensity'] = torch.clone(perennialState['fluidDensity'])
        simulationState['fluidSupport'] = torch.clone(perennialState['fluidSupport'])
        simulationState['fluidRestDensity'] = torch.clone(perennialState['fluidRestDensity'])
        simulationState['fluidArea'] = torch.clone(perennialState['fluidArea'])
        simulationState['fluidPressure'] = torch.clone(perennialState['fluidPressure'])
        simulationState['fluidAcceleration'] = torch.clone(perennialState['fluidAcceleration'])
        if self.config['simulation']['densityScheme'] == 'continuum':
            simulationState['dpdt'] = torch.clone(perennialState['dpdt'])
        
        simulationState['numParticles'] = perennialState['numParticles']
        simulationState['realParticles'] =perennialState['realParticles']
        
        simulationState['dt'] = perennialState['dt']
        simulationState['time'] = perennialState['time']
        simulationState['timestep'] = perennialState['timestep']

        return simulationState

    
    def integrateValues(self, dt, dudt, dxdt, dpdt):
        self.simulationState['fluidVelocity'] = self.perennialState['fluidVelocity'] + dt * dudt
        self.simulationState['fluidPosition'] = self.perennialState['fluidPosition'] + dt * dxdt
        if self.config['simulation']['densityScheme'] == 'continuum':
            self.simulationState['fluidDensity'] = self.perennialState['fluidDensity'] + dt * dpdt / self.config['fluid']['restDensity'] 
        
    def integrate(self):
        self.perennialState = self.saveState()
        self.resetState()
        self.simulationState = self.setupSimulationState(self.perennialState)
        continuumDensity = self.config['simulation']['densityScheme'] == 'continuum'

        dt = self.simulationState['dt'] 
        
        if self.config['integration']['scheme'] == 'explicitEuler':
            dudt, dxdt, dpdt = self.timestep()     
            self.integrateValues(dt, dudt, dxdt, dpdt)

        if self.config['integration']['scheme'] == 'semiImplicitEuler':
            dudt, dxdt, dpdt = self.timestep()     
            self.integrateValues(dt, dudt, dxdt + dt * dudt, dpdt)                
        if self.config['integration']['scheme'] == 'PECE':
            dudt, dxdt, dpdt = self.timestep()     
            self.integrateValues(dt, dudt, dxdt, dpdt)                     
            dudt2, dxdt2, dpdt2 = self.timestep()  
            self.integrateValues(dt, 0.5 * ( dudt + dudt2), 0.5 * ( dxdt + dxdt2), 0.5 * ( dpdt + dpdt2) if continuumDensity else None)      
            
        if self.config['integration']['scheme'] == 'PEC':
            dudt, dxdt, dpdt = self.perennialState['fluidAcceleration'], self.perennialState['fluidVelocity'], self.perennialState['dpdt'] if self.config['simulation']['densityScheme'] == 'continuum' else None
            self.integrateValues(dt, dudt, dxdt, dpdt)                     
            dudt2, dxdt2, dpdt2 = self.timestep()  
            self.integrateValues(dt, 0.5 * ( dudt + dudt2), 0.5 * ( dxdt + dxdt2), (0.5 * ( dpdt + dpdt2)) if continuumDensity else None)    
            
        if self.config['integration']['scheme'] == 'RK4':
            dudt_k1, dxdt_k1, dpdt_k1 = self.timestep()    
            self.simulationState = self.setupSimulationState(self.perennialState)
            self.integrateValues(0.5 * dt, dudt_k1, dxdt_k1, dpdt_k1)    
            dudt_k2, dxdt_k2, dpdt_k2 = self.timestep()     
            self.simulationState = self.setupSimulationState(self.perennialState)
            self.integrateValues(0.5 * dt, dudt_k2, dxdt_k2, dpdt_k2)    
            dudt_k3, dxdt_k3, dpdt_k3 = self.timestep()     
            self.simulationState = self.setupSimulationState(self.perennialState)
            self.integrateValues(dt, dudt_k3, dxdt_k3, dpdt_k3)    
            dudt_k4, dxdt_k4, dpdt_k4 = self.timestep()     
            
            self.integrateValues(1/6 * dt, dudt_k1 + 2 * dudt_k2 + 2 * dudt_k3 + dudt_k4, dxdt_k1 + 2 * dxdt_k2 + 2 * dxdt_k3 + dxdt_k4, (dpdt_k1 + 2 * dpdt_k2 + 2 * dpdt_k3 + dpdt_k4) if continuumDensity else None)   
            


        step = '15 - Bookkeeping'
        if self.verbose: print(step)
        with record_function(step):
            self.simulationState['time'] += self.simulationState['dt']
            self.simulationState['timestep'] += 1

            self.simulationState['dt'] = self.adaptiveDT.updateTimestep(self.simulationState, self)



    def createPlot(self, plotScale = 1, plotDomain = True, plotEmitters = False, \
                   plotVelocitySources = False, plotSolids = True):
        vminDomain = np.array(self.config['domain']['virtualMin'])
        vmaxDomain = np.array(self.config['domain']['virtualMax'])

        aminDomain = np.array(self.config['domain']['min'])
        amaxDomain = np.array(self.config['domain']['max'])

        extent = vmaxDomain - vminDomain

        fig, axis = plt.subplots(1,1, figsize=(extent[0] * plotScale * 1.09, extent[1] * plotScale), squeeze = False)

        axis[0,0].set_xlim(vminDomain[0], vmaxDomain[0])
        axis[0,0].set_ylim(vminDomain[1], vmaxDomain[1])

        # axis[0,0].axis('equal')

        if plotDomain:
            axis[0,0].axvline(aminDomain[0], c = 'black', ls= '--')
            axis[0,0].axvline(amaxDomain[0], c = 'black', ls= '--')
            axis[0,0].axhline(aminDomain[1], c = 'black', ls= '--')
            axis[0,0].axhline(amaxDomain[1], c = 'black', ls= '--')

        if plotVelocitySources:
            if 'velocitySouce' in self.config:
                for vs in self.config['velocitySource']:
                    source = self.config['velocitySource'][vs]
                    rect = patches.Rectangle(source['min'], np.array(source['max']) - np.array(source['min']))
                    axis[0,0].add_patch(rect)

        if plotEmitters:
            if 'emitter' in self.config:
                for vs in self.config['emitter']:
                    source = self.config['emitter'][vs]
                    mi = np.array(source['min'])
                    ma = np.array(source['max'])
                    rect = patches.Rectangle(mi, ma[0] - mi[0], ma[1] - mi[1], linewidth = 1, edgecolor = 'b', hatch ='/', fill = False)
                    axis[0,0].add_patch(rect)

        if plotSolids:
            if 'solidBC' in self.config:
                for b in self.config['solidBC']:
                    bdy = self.config['solidBC'][b]
                    poly = patches.Polygon(bdy['vertices'], fill = False, hatch = None,  color = '#e0952b', alpha = 1.)
                    axis[0,0].add_patch(poly)
        return fig, axis
        
    def __init__(self, config):
        
        basicParams = self.getBasicParameters()
        # print('Parsing basic parameters of configuration')
        for param in basicParams:
            param.parseConfig(config)
        self.verbose = config['simulation']['verbose']
        if self.verbose: print('Basic parameters parsed succesfully')
        self.config = config
        
        self.parameters = basicParams
        
        if self.verbose: print('Setting Kernel parameters')
        self.kernel, self.kernelGrad = getKernelFunctions(self.config['kernel']['defaultKernel'])
        
        if self.verbose: print('Setting compute parameters')        
        self.config['compute']['precision'] = torch.float32 if self.config['compute']['floatprecision'] == 'single' else torch.float64
        self.config['compute']['maxValue'] = torch.finfo(config['compute']['precision']).max
        self.dtype = self.config['compute']['precision']
        self.device = self.config['compute']['device']
            
        if self.verbose: print('Setting generic fluid parameters')
        self.config['particle']['area'] = np.pi * self.config['particle']['radius']**2
        self.config['particle']['support'] = np.single(np.sqrt(self.config['particle']['area'] / np.pi * self.config['kernel']['targetNeighbors']))
        
        # print('Computing packing and spacing parameters')
        self.config['particle']['packing'] = minimize(lambda x: self.evalPacking(x), 0.5, method="nelder-mead").x[0]        
        # self.config['particle']['packing'] = np.float32(0.399023) # minimize(lambda x: self.evalPacking(x), 0.5, method="nelder-mead").x[0]        
        # print('Optimized packing: %g' % self.config['particle']['packing'])
        self.config['particle']['spacing'] = np.float32(0.316313)# -minimize(lambda x: self.evalSpacing(x), 0., method="nelder-mead").x[0]
        # print('Optimized spacing: %g' % self.config['particle']['spacing'])
        if self.config['simulation']['boundaryScheme'] == 'Akinci':
            self.config['particle']['spacing'] =  minimize(lambda x: evalBoundarySpacing(x, self.config['particle']['support'], self.config['particle']['packing'], self.config['particle']['radius'], gamma = self.config['akinciBoundary']['gamma']), 0., method="nelder-mead").x[0]
        if self.config['simulation']['boundaryScheme'] == 'solid':
            self.config['particle']['spacing'] = self.config['particle']['packing']

        if self.config['domain']['adjustParticle']:
            if self.verbose: print('Adjusting particle size to better match domain size')
            D = (self.config['domain']['max'][1] - self.config['domain']['min'][1])
            spacing = self.config['particle']['spacing']
            packing = self.config['particle']['packing']
            n = int(np.ceil((D / config['particle']['support'] - 2 * spacing)/packing))
            h = D / (2 * spacing + n * packing)
            area = h**2 / config['kernel']['targetNeighbors'] * np.pi
            radius = np.sqrt(area / np.pi)

            if self.verbose: print('Updated Radius  %g (%g : %g)' % (radius, config['particle']['radius'], radius - config['particle']['radius']))
            if self.verbose: print('Updated Area    %g (%g : %g)' % (area, config['particle']['area'], area - config['particle']['area']))
            if self.verbose: print('Updated Support %g (%g : %g)' % (h, config['particle']['support'], h - config['particle']['support']))

            self.config['particle']['radius'] = radius
            self.config['particle']['area'] = area
            self.config['particle']['support'] = h

#         config['particle']['packing'] = minimize(lambda x: evalSpacing(x,config), 0.5, method="nelder-mead").x[0]
        if self.verbose: print('Evaluating spacing contribution')
        
        self.config['particle']['spacingContribution'] = self.evalContrib()
        if self.verbose: print('Spacing contribution: %g' % self.config['particle']['spacingContribution'])
        
        if self.config['domain']['adjustDomain']:
            if self.verbose: print('Adjusting simulation domain to be integer multiple of particle packing')
            p = self.config['particle']['packing'] * self.config['particle']['support']
            nx = int(np.ceil((self.config['domain']['max'][0] - self.config['domain']['min'][0]) / p))
            ny = int(np.ceil((self.config['domain']['max'][1] - self.config['domain']['min'][1]) / p))
        #     print('nx', nx)
        #     print('prior', config['domain']['max'][0])
        
            if self.verbose: print('Domain was: [%g %g] - [%g %g]' %(self.config['domain']['min'][0], self.config['domain']['min'][1], self.config['domain']['max'][0], self.config['domain']['max'][1]))
            self.config['domain']['max'][0] = self.config['domain']['min'][0] + nx * p
            self.config['domain']['max'][1] = self.config['domain']['min'][1] + ny * p
            
            if self.verbose: print('Domain  is: [%g %g] - [%g %g]' %(self.config['domain']['min'][0], self.config['domain']['min'][1], self.config['domain']['max'][0], self.config['domain']['max'][1]))


        self.processEmitters()
        self.processVelocitySources()
        
        if self.verbose: print('Setting virtual domain limits')
        self.config['domain']['virtualMin'] = self.config['domain']['min'] - self.config['particle']['support'] * self.config['periodicBC']['buffer']
        self.config['domain']['virtualMax'] = self.config['domain']['max'] + self.config['particle']['support'] * self.config['periodicBC']['buffer']

        if self.verbose: print('Adding Boundary boundaries')
        self.addBoundaryBoundaries()
        
        if 'solidBC' in self.config:
            if self.verbose: print('Parsing boundary vertices to polygons')
            
            for b in self.config['solidBC']:
                boundary = self.config['solidBC'][b]
                boundary['polygon'] = torch.tensor(boundary['vertices'], device = self.device, dtype = self.dtype)
        
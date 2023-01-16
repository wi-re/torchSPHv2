import numpy as np
import torch
from scipy.optimize import minimize

from .generation import *
from .config import *

def getShockwaveConfig(radius = 0.01, dt = 0.005, c = .2, device = 'cpu', precision = torch.float64):

    config = {
        'radius': radius,
        'targetNeighbors': 20,
        'restDensity': 1000,
        'dt': dt,
        'kappa':1.5,
        'viscosityConstant':0.1,
        'max_neighbors':256,
#         'device':'cpu',
    #     'device':device,
#         'precision': torch.float32,
        'precision': precision,
        'device':device,
        
        'domain':{
            'periodicX':True,
            'periodicY':True,
            'min': [-1, -1],
            'max': [1,1],
            'buffer': 2
        }
    }
    config['maxValue'] = torch.finfo(config['precision']).max

    config['area'] = np.pi * config['radius']**2
    config['support'] = np.sqrt(config['area'] / np.pi * config['targetNeighbors'])


    config['packing'] = minimize(lambda x: evalPacking(x,config), 0.5, method="nelder-mead").x[0]
    # print(config['packing'])

    c = .2
    overallDomainMin = np.array([-1, -1])
    overallDomainMax = np.array([1, 1])
    centralRegionMin = np.array([-c, -c])
    centralRegionMax = np.array([c, c] )

    toMin = np.ceil((centralRegionMin - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toMax = np.ceil((centralRegionMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toDomain = np.ceil((overallDomainMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))

    spacing = (config['packing'] * config['support'] / 1.5)

    config['emitters'] = [
        {
        'min': overallDomainMin + spacing * np.array([0,0]),
        'max': overallDomainMin + spacing * np.array([toDomain[0]+ 1,toMin[1]]),
        'compression': 1.5
    },{
        'min': overallDomainMin + spacing * np.array([0,toMax[1]]),
        'max': overallDomainMin + spacing * np.array([toDomain[0] + 1,toDomain[1]+ 1]),
        'compression': 1.5
    }
        ,{
        'min': overallDomainMin + spacing * np.array([0,toMin[1] + 1]),
        'max': overallDomainMin + spacing * np.array([toMin[0],toMax[1]-1]),
        'compression': 1.5
    }    
    ,{
        'min': overallDomainMin + spacing * np.array([toMax[0]+1,toMin[1] + 1]),
        'max': overallDomainMin + spacing * np.array([toDomain[0]+1,toMax[1]-1]),
        'compression': 1.5
    }
        ,
        {
        'min': overallDomainMin + spacing * np.array([toMin[0],toMin[1]]),
        'max': overallDomainMin + spacing * np.array([toMax[0],toMax[1]]),
        'compression': 3.0
    }
    #     ,{
    #     'min': [-.180020360266536,-.180020360266536],
    #     'max': [.180020360266536, .180020360266536],
    #     'compression': 3.0
    # }
    ]


    for emitter in config['emitters']:
        if 'radius' not in emitter:
            emitter['radius'] = config['radius']
        if 'density' not in emitter:
            emitter['density'] = config['restDensity']
        if 'type' not in emitter:
            emitter['type'] = 'once'
        if 'compression' not in emitter:
            emitter['compression'] = 1.0
        if 'velocity' not in emitter:
            emitter['velocity'] = [0.0,0.0]

    if 'gravity' not in config:
        config['gravity'] = [0,0]


    minCompression = config['maxValue']
    for emitter in config['emitters']:
        minCompression = min(minCompression, emitter['compression'])
    config['minCompression'] = minCompression

    adjustDomainForEmitters(config, fixSizes = False)
    return config

def getTriangleBoundaryImpactConfig(radius = 0.01, dt = 0.005, device = 'cpu', precision = torch.float64):
    config = {
        'radius': radius,
        'omega':0.5,
        'targetNeighbors': 20,
        'restDensity': 1000,
        'dt':dt,
        'kappa':1.5,
        'viscosityConstant':0.01,
        'max_neighbors':256,
        'device':device,
    #     'device':device,
        'precision': precision,
        'domain':{
            'periodicX':True,
            'periodicY':True,
            'min': [-1, -1],
            'max': [1,1],
            'buffer': 2
        },
        'dfsph':{
            'minDensitySolverIterations': 2,
            'maxDensitySolverIterations': 256,
            'minDivergenceSolverIterations': 2,
            'maxDivergenceSolverIterations': 8,
            'densityThreshold': 1e-4,
            'divergenceSolver': True,
            'divergenceThreshold': 1e-2,
        },
        'solidBoundary':[
            {
                'vertices': [[0,0],[0.5,0.5],[0.5,-0.5]],
                'inverted': False
            }
    #         ,
    #         {
    #             'vertices': [[-0.75,-0.75],[0.,-0.75],[-0.75,0]],
    #             'inverted': False
    #         }
        ]
    }
    config['maxValue'] = torch.finfo(config['precision']).max

    config['area'] = np.pi * config['radius']**2
    config['support'] = np.sqrt(config['area'] / np.pi * config['targetNeighbors'])


    config['packing'] = minimize(lambda x: evalPacking(x,config), 0.5, method="nelder-mead").x[0]
    # print(config['packing'])

    c = .2
    overallDomainMin = np.array([-1, -1])
    overallDomainMax = np.array([1, 1])
    centralRegionMin = np.array([-c, -c])
    centralRegionMax = np.array([c, c] )

    toMin = np.ceil((centralRegionMin - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toMax = np.ceil((centralRegionMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toDomain = np.ceil((overallDomainMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))

    spacing = (config['packing'] * config['support'] / 1.5)

    config['emitters'] = [
        {
        'min': np.array([-2.5 * c,-c]),
        'max': np.array([-0.95 * config['support'], c]),
        'velocity': [1,0]
    }
    ]


    for emitter in config['emitters']:
        if 'radius' not in emitter:
            emitter['radius'] = config['radius']
        if 'density' not in emitter:
            emitter['density'] = config['restDensity']
        if 'type' not in emitter:
            emitter['type'] = 'once'
        if 'compression' not in emitter:
            emitter['compression'] = 1.0
        if 'velocity' not in emitter:
            emitter['velocity'] = [0.0,0.0]

    if 'gravity' not in config:
        config['gravity'] = [0,0]


    minCompression = config['maxValue']
    for emitter in config['emitters']:
        minCompression = min(minCompression, emitter['compression'])
    config['minCompression'] = minCompression

    adjustDomainForEmitters(config, fixSizes = False)

    for boundary in config['solidBoundary']:
        boundary['polygon'] = torch.tensor(boundary['vertices'], device = config['device'], dtype = config['precision'])
    return config


from .solidBC import *

def damBreakConfig(radius = 0.01, dt = 0.002, c = 0.2, device = 'cpu', precision = torch.float64):
    config = {
        'radius': radius,
        'omega':0.5,
        'targetNeighbors': 20,
        'restDensity': 1000,
        'dt':dt,
        'kappa':1.5,
        'viscosityConstant':0.01,
        'max_neighbors':256,
        'device':device,
    #     'device':device,
        'precision': precision,
        'domain':{
            'periodicX':False,
            'periodicY':False,
            'min': [-1, -1],
            'max': [1,1],
            'buffer': 2
        },
        'dfsph':{
            'minDensitySolverIterations': 2,
            'maxDensitySolverIterations': 256,
            'minDivergenceSolverIterations': 2,
            'maxDivergenceSolverIterations': 8,
            'densityThreshold': 1e-4,
            'divergenceSolver': True,
            'divergenceThreshold': 1e-2,
        },
        'gravity': [0,-9.81]
#         ,
#         'solidBoundary':[
#             {
#                 'vertices': [[0,0],[0.5,0.5],[0.5,-0.5], [0.25, -0.75], [-0.125, -0.5]],
#                 'inverted': False
#             }
    #         ,
    #         {
    #             'vertices': [[-0.75,-0.75],[0.,-0.75],[-0.75,0]],
    #             'inverted': False
    #         }
#         ]
    }
    config['maxValue'] = torch.finfo(config['precision']).max

    config['area'] = np.pi * config['radius']**2
    config['support'] = np.sqrt(config['area'] / np.pi * config['targetNeighbors'])


    config['packing'] = minimize(lambda x: evalPacking(x,config), 0.5, method="nelder-mead").x[0]
    config['spacing'] = -minimize(lambda x: evalSpacing(x,config), 0., method="nelder-mead").x[0]
    p = config['spacing']
    # print(config['packing'])

    overallDomainMin = np.array([-1, -1])
    overallDomainMax = np.array([1, 1])
    centralRegionMin = np.array([-c, -c])
    centralRegionMax = np.array([c, c] )

    toMin = np.ceil((centralRegionMin - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toMax = np.ceil((centralRegionMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toDomain = np.ceil((overallDomainMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))

    spacing = (config['packing'] * config['support'] / 1.5)

    config['emitters'] = [
        {
        'min': np.array([-1 + p,-1 + p]),
        'max': np.array([-1 + c / 2, -1 + c]),
        'velocity': [0,0]
    }
    ]


    for emitter in config['emitters']:
        if 'radius' not in emitter:
            emitter['radius'] = config['radius']
        if 'density' not in emitter:
            emitter['density'] = config['restDensity']
        if 'type' not in emitter:
            emitter['type'] = 'once'
        if 'compression' not in emitter:
            emitter['compression'] = 1.0
        if 'velocity' not in emitter:
            emitter['velocity'] = [0.0,0.0]
    if 'gravity' not in config:
        config['gravity'] = [0,0]


    minCompression = config['maxValue']
    for emitter in config['emitters']:
        minCompression = min(minCompression, emitter['compression'])
    config['minCompression'] = minCompression

    adjustDomainForEmitters(config, fixSizes = False)
    
    addBoundaryBoundaries(config)

    for boundary in config['solidBoundary']:
        boundary['polygon'] = torch.tensor(boundary['vertices'], device = config['device'], dtype = config['precision'])
        
    return config



def velocitSourceConfig(radius = 0.01, dt = 0.002, c = 0.2, device = 'cpu', precision = torch.float64):
    config = {
        'radius': radius,
        'omega':0.5,
        'targetNeighbors': 20,
        'restDensity': 1000,
        'dt':dt,
        'kappa':1.5,
        'viscosityConstant':0.01,
        'max_neighbors':256,
        'device':device,
    #     'device':device,
        'precision': precision,
        'domain':{
            'periodicX':True,
            'periodicY':False,
            'min': [-1, -1],
            'max': [1,1],
            'buffer': 2
        },
        'dfsph':{
            'minDensitySolverIterations': 2,
            'maxDensitySolverIterations': 256,
            'minDivergenceSolverIterations': 2,
            'maxDivergenceSolverIterations': 8,
            'densityThreshold': 1e-3,
            'divergenceSolver': False,
            'divergenceThreshold': 1e-2,
        },
        'gravity': [0,-9.81],
        'velocitySources':[
            {'min' : [-1,-1],
             'max' : [-0.9, 1],
             'rampTime' : 1,
             'velocity' : [1,0]}
            
        ]
#         ,
#         'solidBoundary':[
#             {
#                 'vertices': [[0,0],[0.5,0.5],[0.5,-0.5], [0.25, -0.75], [-0.125, -0.5]],
#                 'inverted': False
#             }
    #         ,
    #         {
    #             'vertices': [[-0.75,-0.75],[0.,-0.75],[-0.75,0]],
    #             'inverted': False
    #         }
#         ]
    }
    config['maxValue'] = torch.finfo(config['precision']).max

    config['area'] = np.pi * config['radius']**2
    config['support'] = np.sqrt(config['area'] / np.pi * config['targetNeighbors'])


    config['packing'] = minimize(lambda x: evalPacking(x,config), 0.5, method="nelder-mead").x[0]
    config['spacing'] = -minimize(lambda x: evalSpacing(x,config), 0., method="nelder-mead").x[0]
    p = config['spacing']
    # print(config['packing'])

    overallDomainMin = np.array([-1, -1])
    overallDomainMax = np.array([1, 1])
    centralRegionMin = np.array([-c, -c])
    centralRegionMax = np.array([c, c] )

    toMin = np.ceil((centralRegionMin - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toMax = np.ceil((centralRegionMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))
    toDomain = np.ceil((overallDomainMax - overallDomainMin) / (config['packing'] * config['support'] / 1.5))

    spacing = (config['packing'] * config['support'] / 1.5)

    config['emitters'] = [
        {
        'min': np.array([-1 + p / 2,-1 + p]),
        'max': np.array([1 - p / 2, -1 + p + c]),
        'velocity': [0,0]
    }
    ]


    for emitter in config['emitters']:
        if 'radius' not in emitter:
            emitter['radius'] = config['radius']
        if 'density' not in emitter:
            emitter['density'] = config['restDensity']
        if 'type' not in emitter:
            emitter['type'] = 'once'
        if 'compression' not in emitter:
            emitter['compression'] = 1.0
        if 'velocity' not in emitter:
            emitter['velocity'] = [0.0,0.0]
    if 'gravity' not in config:
        config['gravity'] = [0,0]


    minCompression = config['maxValue']
    for emitter in config['emitters']:
        minCompression = min(minCompression, emitter['compression'])
    config['minCompression'] = minCompression

    adjustDomainForEmitters(config, fixSizes = False)
    
    addBoundaryBoundaries(config)

    for boundary in config['solidBoundary']:
        boundary['polygon'] = torch.tensor(boundary['vertices'], device = config['device'], dtype = config['precision'])
        
#     print(config['solidBoundary'])
    return config
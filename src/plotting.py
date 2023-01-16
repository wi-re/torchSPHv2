import numpy as np
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
from torch_geometric.nn import radius
from torch_scatter import scatter
from torch.profiler import record_function


from .kernel import *

from tqdm.notebook import tqdm


def plotQuantity(qty, fig, axis, config, simulationState):    

    sc = axis.scatter(simulationState['fluidPosition'][:,0].detach().cpu(), 
                      simulationState['fluidPosition'][:,1].detach().cpu(), c = qty.detach().cpu(), s = 8)
    axis.axis('equal')
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(sc, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
#     axis.set_title('Density')
    axis.set_xlim(config['domain']['min'][0],config['domain']['max'][0])
    axis.set_ylim(config['domain']['min'][1],config['domain']['max'][1])

    r = patches.Rectangle((config['domain']['min'][0], config['domain']['min'][1]), 
                      config['domain']['max'][0] - config['domain']['min'][0], config['domain']['max'][1] - config['domain']['min'][1], linewidth=1, edgecolor='r', facecolor='none')
    axis.add_patch(r)
    r = patches.Rectangle((config['domain']['virtualMin'][0], config['domain']['virtualMin'][1]), 
                      config['domain']['virtualMax'][0] - config['domain']['virtualMin'][0], config['domain']['virtualMax'][1] - config['domain']['virtualMin'][1], linewidth=3, edgecolor='g', facecolor='none')
    axis.add_patch(r)

    fig.tight_layout()

def plotBoundary(axis, config):
    if 'solidBoundary' not in config:
        return
    for boundary in config['solidBoundary']:
        vertices = boundary['vertices']
        poly = patches.Polygon(vertices, fill = False, hatch = None, edgecolor='black')
        axis.add_patch(poly)

def plotSources(axis, config):
    if 'velocitySources' not in config:
        return
    for boundary in config['velocitySources']:
        pmin = boundary['min']
        pmax = boundary['max']

        r = patches.Rectangle((pmin[0], pmin[1]), 
                      pmax[0] - pmin[0], pmax[1] - pmin[1], linewidth=0.25, edgecolor='blue', facecolor='none')
        
        axis.add_patch(r)


def plotDomain(axis, config):
    axis.axis('equal')
    axis.set_xlim(config['domain']['min'][0],config['domain']['max'][0])
    axis.set_ylim(config['domain']['min'][1],config['domain']['max'][1])

    r = patches.Rectangle((config['domain']['min'][0], config['domain']['min'][1]), 
                      config['domain']['max'][0] - config['domain']['min'][0], config['domain']['max'][1] - config['domain']['min'][1], linewidth=0.25, edgecolor='r', facecolor='none')
    axis.add_patch(r)
    r = patches.Rectangle((config['domain']['virtualMin'][0], config['domain']['virtualMin'][1]), 
                      config['domain']['virtualMax'][0] - config['domain']['virtualMin'][0], config['domain']['virtualMax'][1] - config['domain']['virtualMin'][1], linewidth=0.25, edgecolor='g', facecolor='none')
    axis.add_patch(r)
# def scatterQuantity(qty, axis, label, simulationState):
#     densityScatter   = axis.scatter(simulationState['fluidPosition'][:,0].detach().cpu(), simulationState['fluidPosition'][:,1].detach().cpu(), c = qty, s = 2)
#     axis.set_title(label)
#     ax1_divider = make_axes_locatable(axis)
#     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
#     densityCbar = fig.colorbar(densityScatter, cax=cax1,orientation='vertical')
#     densityCbar.ax.tick_params(labelsize=8) 
    
#     return densityScatter, densityCbar

def getUVs(qty, config, simulationState, nx = 256, ny = 256):
    x = np.linspace(config['domain']['min'][0],config['domain']['max'][0],nx)
    y = np.linspace(config['domain']['min'][1],config['domain']['max'][1],ny)
                              
    xx, yy = np.meshgrid(x, y)

    xf = xx.flatten()
    yf = yy.flatten()

    gridPositions = torch.from_numpy(np.c_[xf, yf]).type(config['precision']).to(config['device'])

    row, col = radius(simulationState['fluidPosition'], gridPositions, config['support'], max_num_neighbors = config['max_neighbors'])
    edge_index = torch.stack([row, col], dim = 0)
        
    j = edge_index[1]
    i = edge_index[0]

    pseudo = (simulationState['fluidPosition'][j] - gridPositions[i])    
    q = torch.linalg.norm(pseudo,axis=1) / config['support']
    
    kernel = wendland(q, config['support'])
    factor = kernel * simulationState['fluidArea'][j] / simulationState['fluidDensity'][j]
    if len(qty.shape) > 1:
        term = factor[:,None] * qty[j]
    else:
        term = factor * qty[j]
#     print(term.shape)
    
    out = scatter(term, i, dim=0, dim_size=gridPositions.size(0), reduce="add")

#     out = torch.tensor(gridPositions)
#     out = gridPositions[:,0]
    
    if len(qty.shape) > 1:
        uv = out.detach().cpu().numpy().reshape(ny,nx,qty.shape[1])
    else:
        uv = out.detach().cpu().numpy().reshape(ny,nx,1)

    return xx,yy,uv

from inspect import signature

def initializeXYPlot(config, simulationState, axis, fn, nx = 256, ny = 256):
    data = fn(config, simulationState, nx, ny)
    
    im = axis.imshow(data, extent =(config['domain']['min'][0], config['domain']['max'][0], config['domain']['min'][1], config['domain']['max'][1]))
    return data, im

def initializeScatterPlot(config, simulationState, axis, fn):
    positions, data = fn(config, simulationState)
    
    im = axis.scatter(positions[:,0], positions[:,1], c = data, s = 1)
    return data, im
def initialPlot(config, state, plotFn, nx=256, ny = 256, figsize = (8,8), plotLayout = None):
    with record_function('plotting - initial'):
        if type(plotFn) is list:
            if plotLayout is None:
                plotLayout = (len(plotFn),1)
        plotLayout = (1,1) if plotLayout is None else plotLayout
        
        
        fig, axis = plt.subplots(plotLayout[0],plotLayout[1], figsize=(figsize[0],figsize[1]), squeeze = False)
        
        ims = []
        if type(plotFn) is list:
            for p, a in zip(plotFn, axis.flatten()):
                if len(signature(p).parameters) == 4:
                    data, im = initializeXYPlot(config, state, a,p, nx, ny)
                else:
                    data, im = initializeScatterPlot(config, state, a, p)
                ims.append(im)
        else:
            if len(signature(plotFn).parameters) == 4:
                data, im = initializeXYPlot(config, state, axis[0,0],plotFn, nx, ny)
            else:
                data, im = initializeScatterPlot(config, state, axis[0,0], plotFn)
            ims.append(im)

#         # axis[0,0].set_title('Density')

        cbars = []
        for a, im in zip(axis.flatten(), ims):
            a.axis('equal')
            a.set_xlim(config['domain']['min'][0],config['domain']['max'][0])
            a.set_ylim(config['domain']['min'][1],config['domain']['max'][1])
            ax1_divider = make_axes_locatable(a)
            cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
            cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            cbars.append(cbar)

        fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles']))
        if 'densityErrors' in state and not 'divergenceErrors' in state:
            fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles'], len(state['densityErrors'])))
        if 'divergenceErrors' in state and not 'densityErrors' in state:
            fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles'], len(state['divergenceErrors'])))
        if 'densityErrors' in state and 'divergenceErrors' in state:
            fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d], dfsph: [%3d, %3d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles'], len(state['densityErrors']),len(state['divergenceErrors'])))

        for a in axis.flatten():
            plotDomain(a, config)
            plotBoundary(a, config)
            plotSources(a, config)

        fig.tight_layout()
        
        return fig, ims, axis, cbars
    
def updatePlot(config, state, fig, axes, ims, cbars, plotFns, nx = 256, ny = 256):
    with record_function('plotting - updating'):
        if type(plotFns) is list:
            for axis, im, cbar, plotFn in zip(axes.flatten(), ims, cbars, plotFns):
                if len(signature(plotFn).parameters) == 4:
                    data = plotFn(config, state, nx, ny)
                    im.set_data(data)
                    cbar.mappable.set_clim(vmin=np.min(data), vmax=np.max(data))
                else:
                    positions, data = plotFn(config, state)
                    im.set_offsets(positions)
                    im.set_array(data)
                cbar.mappable.set_clim(vmin=np.min(data), vmax=np.max(data))
        else:            
            for axis, im, cbar in zip(axes.flatten(), ims, cbars):
                if len(signature(plotFns).parameters) == 4:
                    data = plotFns(config, state, nx, ny)
                    im.set_data(data)
                    cbar.mappable.set_clim(vmin=np.min(data), vmax=np.max(data))
                else:
                    positions, data = plotFns(config, state)
                    im.set_offsets(positions)
                    im.set_array(data)
                cbar.mappable.set_clim(vmin=np.min(data), vmax=np.max(data))
        fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles']))
        if 'densityErrors' in state and not 'divergenceErrors' in state:
            fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles'],len(state['densityErrors'])))
        if 'divergenceErrors' in state and not 'densityErrors' in state:
            fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles'],len(state['divergenceErrors'])))
        if 'densityErrors' in state and 'divergenceErrors' in state:
            fig.suptitle('t=%2.4f[%4d], ptcls: %5d[%5d], dfsph: [%3d, %3d]'%(state['time'], state['timestep'], state['numParticles'],state['realParticles'],len(state['densityErrors']),len(state['divergenceErrors'])))
        fig.canvas.draw()
        fig.canvas.flush_events()
        
def printParticle(i, simulationState):
    print('Printing  particle    : %6d' %  i)
    print('Ghost Index           : ',  simulationState['ghostIndices'][i].item())
    print('Area                  : ',  simulationState['fluidArea'][i].item())
    print('Rest Density          : ',  simulationState['fluidRestDensity'][i].item())
    
    print('Position              : ',  simulationState['fluidPosition'][i].detach().cpu().numpy())
    print('Velocity              : ',  simulationState['fluidVelocity'][i].detach().cpu().numpy())
    print('Acceleration          : ',  simulationState['fluidAcceleration'][i].detach().cpu().numpy())
                                                                  
    print('Density               : ',  simulationState['fluidDensity'][i].item())
    if 'boundaryDensity' in simulationState:
        print('boundaryDensity       : ',  simulationState['boundaryDensity'][i].detach().cpu().numpy())
        print('boundaryGradient      : ',  simulationState['boundaryGradient'][i].detach().cpu().numpy())
                                                                
    print('Pressure              : ',  simulationState['fluidPressure'][i].item())     
    if 'residual' in simulationState:
        print('Residual              : ',  simulationState['residual'][i].item())   
        print('Predicted Velocity    : ',  simulationState['fluidPredictedVelocity'][i].detach().cpu().numpy())   
        print('Predicted Acceleration: ',  simulationState['fluidPredAccel'][i].detach().cpu().numpy())   
        print('Actual Area           : ',  simulationState['fluidActualArea'][i].item())   
        print('Alpha                 : ',  simulationState['fluidAlpha'][i].item())   
        print('Source                : ',  simulationState['fluidSourceTerm'][i].item())   
        print('Pressure              : ',  simulationState['fluidPressure'][i].item())  
        print('Pressure2             : ',  simulationState['fluidPressure2'][i].item())
    
    mask = simulationState['fluidNeighbors'][0] == i
    
    print('fluidNeighbors   : ', simulationState['fluidNeighbors'][1, mask])
    if 'boundaryNeighbors' in simulationState:
        mask = simulationState['boundaryNeighbors'][0] == i
        print('boundaryNeighbors: ', simulationState['boundaryNeighbors'][1, mask])
        
def printBoundaryParticle(bi, simulationState):
    print('pb    : ', simulationState['pb'][bi])
    print('sumA  : ', simulationState['boundary sumA'][bi].item())
    print('sumB  : ', simulationState['boundary sumB'][bi].item())
    print('vecSum: ', simulationState['boundary vecSum'][bi])
    print('alpha : ', simulationState['boundary alpha'][bi].item())
    print('beta  : ', simulationState['boundary beta'][bi].item())
    print('gamma : ', simulationState['boundary gamma'][bi].item())
    print('det   : ', simulationState['boundary det'][bi].item())
    print('M     : ', simulationState['boundary M'][bi])
    print('Mp    : ', simulationState['boundary Mp'][bi])

# i = torch.argmax(state['fluidPosition'][:,0])
# printParticle(i, state)
# bi = torch.argmax(state['boundary alpha'])
# printBoundaryParticle(bi, state)
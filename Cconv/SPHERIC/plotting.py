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
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator


# Plotting function for convenience
def plot1DValues(x,counts, minDomain, maxDomain, scatter = False, xlabel = None, ylabel = None, title = None):
    if scatter:
        x = x
        y = np.zeros_like(x) if not torch.is_tensor(x) else torch.zeros_like(x)
        c = counts

        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
        if torch.is_tensor(x):
            axis[0,0].plot(x[idx].detach().cpu().numpy(), y[idx].detach().cpu().numpy())
        else:
            sc = axis[0,0].scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), c = c.detach().cpu().numpy(), s = 1)
        ax1_divider = make_axes_locatable(axis[0,0])
        cax1 = ax1_divider.append_axes("bottom", size="20%", pad="2%")
        cb1 = fig.colorbar(sc, cax=cax1,orientation='horizontal')
        cb1.ax.tick_params(labelsize=8) 
        axis[0,0].axvline(minDomain, color = 'black', ls = '--')
        axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
        
        if not xlabel is None:
            axis[0,0].set_xlabel(xlabel)
        if not ylabel is None:
            cb1.ax.set_xlabel(ylabel)        
        if not title is None:
            fig.suptitle(title)
        
        fig.tight_layout()
        return fig, axis
    else:
        x = x
        y = counts
        idx = np.argsort(x) if not torch.is_tensor(x) else torch.argsort(x)

        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
        if torch.is_tensor(x):
            axis[0,0].plot(x[idx].detach().cpu().numpy(), y[idx].detach().cpu().numpy())
        else:
            axis[0,0].plot(x[idx], y[idx])
        axis[0,0].axvline(minDomain, color = 'black', ls = '--')
        axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
        
        if not xlabel is None:
            axis[0,0].set_xlabel(xlabel)
        if not ylabel is None:
            axis[0,0].set_ylabel(ylabel)
        if not title is None:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axis
    

def plotDistorted(simulationStates, minDomain, maxDomain, dt):
    fig, axis = plt.subplots(2, 1, figsize=(14,9), sharex = False, sharey = False, squeeze = False)

    im = axis[0,0].imshow(simulationStates[:,2].mT.numpy()[::-1,:], extent = [0,dt * simulationStates.shape[0], minDomain,maxDomain])
    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Density [1/m]')


    im = axis[1,0].imshow(simulationStates[:,1].mT.numpy()[::-1,:], extent = [0,dt * simulationStates.shape[0], minDomain,maxDomain], cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Velocity [m/s]')

    fig.tight_layout()

def scatterPlotAll(simulationStates, minDomain, maxDomain, dt):

    timeArray = torch.arange(simulationStates.shape[0])[:,None].repeat(1,simulationStates.shape[2]) * dt
    positionArray = simulationStates[:,0]

    fig, axis = plt.subplots(2, 1, figsize=(12,9), sharex = False, sharey = False, squeeze = False)

    trip = axis[0,0].scatter(timeArray.flatten(), positionArray.flatten(), s = 0.1, c = simulationStates[:,2].flatten(), vmin = torch.min(torch.abs(simulationStates[:,2])), vmax =torch.max(torch.abs(simulationStates[:,2])),cmap = 'viridis')

    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(trip, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_ylabel('Density [1/m]')
    axis[0,0].set_ylim(minDomain, maxDomain)
    axis[0,0].set_xlim(simulationStates.shape[0] * dt, 0)
    axis[0,0].set_xlim(0,simulationStates.shape[0] * dt)

    trip = axis[1,0].scatter(timeArray.flatten(), positionArray.flatten(), s = 0.1, c = simulationStates[:,1].flatten(), vmin = -torch.max(torch.abs(simulationStates[:,1])), vmax =torch.max(torch.abs(simulationStates[:,1])),cmap = 'RdBu')
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(trip, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_ylabel('Velocity [m/s]')
    axis[1,0].set_ylim(minDomain, maxDomain)
    axis[1,0].set_xlim(simulationStates.shape[0] * dt, 0)
    axis[1,0].set_xlim(0,simulationStates.shape[0] * dt)

    fig.tight_layout()

def plotSimulationState(simulationStates, minDomain, maxDomain, dt, timepoints = []):
    fig, axis = plt.subplots(2, 1, figsize=(9,6), sharex = True, sharey = False, squeeze = False)

    axis[0,0].axvline(minDomain, color = 'black', ls = '--')
    axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
    axis[1,0].axvline(minDomain, color = 'black', ls = '--')
    axis[1,0].axvline(maxDomain, color = 'black', ls = '--')

    axis[1,0].set_xlabel('Position')
    axis[1,0].set_ylabel('Velocity[m/s]')
    axis[0,0].set_ylabel('Density[1/m]')

    def plotTimePoint(i, c, simulationStates, axis):
        x = simulationStates[i,0,:]
        y = simulationStates[i,c,:]
        idx = torch.argsort(x)
        axis.plot(x[idx].detach().cpu().numpy(), y[idx].detach().cpu().numpy(), label = 't = %1.2g' % (i * dt))
    if timepoints == []:
        plotTimePoint(0,1, simulationStates, axis[1,0])
        plotTimePoint(0,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4*2,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4*2,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4*3,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4*3,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]-1,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]-1,2, simulationStates, axis[0,0])
    else:
        for t in timepoints:
            plotTimePoint(t,1, simulationStates, axis[1,0])
            plotTimePoint(t,2, simulationStates, axis[0,0])
            

#             plotTimePoint(16,1, simulationStates, axis[0,0])
#             plotTimePoint(16,2, simulationStates, axis[1,0])

#             plotTimePoint(128,1, simulationStates, axis[0,0])
#             plotTimePoint(128,2, simulationStates, axis[1,0])

#             plotTimePoint(256,1, simulationStates, axis[0,0])
#             plotTimePoint(256,2, simulationStates, axis[1,0])

#             plotTimePoint(511,1, simulationStates, axis[0,0])
#             plotTimePoint(511,2, simulationStates, axis[1,0])


    axis[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axis[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()


def plotDensityField(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport):
    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)

    xs = fluidPositions.detach().cpu().numpy()
    densityField = fluidDensity.detach().cpu().numpy()
    fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
    numSamples = densityField.shape[-1]
    # xs = np.linspace(-1,1,numSamples)
    fs = numSamples/2
    fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
    x = densityField
    y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
    axis[0,0].plot(xs, densityField)
    axis[0,1].loglog(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'baseTarget')
    f, Pxx_den = scipy.signal.welch(densityField, fs, nperseg=len(x)//32)
    axis[0,2].loglog(f, Pxx_den, label = 'baseTarget')
    axis[0,2].set_xlabel('frequency [Hz]')
    axis[0,2].set_ylabel('PSD [V**2/Hz]')
    fig.tight_layout()
    return fluidDensity

def plotDensity(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport):
    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)
    fig, axis = plot1DValues(fluidPositions, fluidDensity, minDomain, maxDomain, ylabel = 'Density')
    axis[0,0].axhline(torch.mean(fluidDensity).detach().item(), ls = '--', c = 'white', alpha = 0.5)
    axis[0,0].axhline(torch.max(fluidDensity).detach().item(), ls = '--', c = 'white', alpha = 0.5)
    axis[0,0].axhline(torch.min(fluidDensity).detach().item(), ls = '--', c = 'white', alpha = 0.5)
    return fluidDensity


def regularPlot(simulationStates, minDomain, maxDomain, dt, nx = 512, ny = 2048):
    timeArray = torch.arange(simulationStates.shape[0])[:,None].repeat(1,simulationStates.shape[2]) * dt
    positionArray = simulationStates[:,0]
    xys = torch.vstack((timeArray.flatten().to(positionArray.device).type(positionArray.dtype), positionArray.flatten())).mT.detach().cpu().numpy()


    # interpVelocity = LinearNDInterpolator(xys, simulationStates[:,1].flatten())
    # interpDensity = LinearNDInterpolator(xys, simulationStates[:,2].flatten())
    interpVelocity = NearestNDInterpolator(xys, simulationStates[:,1].flatten().detach().cpu().numpy())
    interpDensity = NearestNDInterpolator(xys, simulationStates[:,2].flatten().detach().cpu().numpy())

    X = torch.linspace(torch.min(timeArray), torch.max(timeArray), ny).detach().cpu().numpy()
    Y = torch.linspace(torch.min(positionArray), torch.max(positionArray), nx).detach().cpu().numpy()
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    # Z = interp(X, Y)

    fig, axis = plt.subplots(2, 1, figsize=(14,9), sharex = False, sharey = False, squeeze = False)


    im = axis[0,0].pcolormesh(X,Y,interpDensity(X,Y), cmap = 'viridis', vmin = torch.min(torch.abs(simulationStates[:,2])),vmax = torch.max(torch.abs(simulationStates[:,2])))
    # im = axis[0,0].imshow(simulationStates[:,2].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain])
    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Density [1/m]')

    im = axis[1,0].pcolormesh(X,Y,interpVelocity(X,Y), cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    # im = axis[1,0].imshow(simulationStates[:,1].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain], cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('pseudo-position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Velocity [m/s]')

    fig.tight_layout()


def plotRandomWeights(samples, n = 8, basis = 'linear', windowFunction = 'Wendland2_1D', normalized = False):    
    # n = 8
    # basis = 'rbf square'
    randomWeights = []
    for i in range(samples):
        randomWeights.append(torch.rand(n) - 0.5)
    # randomWeights
    fig, axis = plt.subplots(1, 2, figsize=(16,4), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,511)
    # n = dict['weight'].shape[0]
    # internal function that is used for the rbf convolution
    fx = evalBasisFunction(n, x , which = basis, periodic=False)
    fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    windowFn = getWindowFunction(windowFunction) # window function that is applied after each network layer

    integrals = []
    for i in range(len(randomWeights)):
        integral = torch.sum(torch.sum(randomWeights[i][:,None] * fx,axis=0)) * 2 / 511
        integrals.append(integral)
    integrals = torch.hstack(integrals)
    norm = mpl.colors.Normalize(vmin=torch.min(integrals), vmax=torch.max(integrals))

    for i in range(len(randomWeights)):
        axis[0,0].plot(x,torch.sum(randomWeights[i][:,None] * fx,axis=0),ls='-',c=cmap(norm(integrals[i])), label = '$\Sigma_i w_i f_i(x)$', alpha = 0.75)
        axis[0,1].plot(x,windowFn(torch.abs(x)) * torch.sum(randomWeights[i][:,None] * fx,axis=0),ls='-',c=cmap(norm(integrals[i])), label = '$\Sigma_i w_i f_i(x)$', alpha = 0.75)
    # axis[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fancybox=True, shadow=False)
    axis[0,0].set_title('Random initializations %s [%2d]'% (basis,n))
    axis[0,1].set_title('Random initializations %s [%2d] /w window'% (basis,n))

    fig.tight_layout()


def buildLossLandscape(nx, targetWeights, targetBias, batch, lossFunction, model, optimizer, device, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked):
    center = torch.clone(torch.hstack((model.weight[:,0,0], model.bias)).detach())
    
    currentWeights = model.weight
    currentBias = model.bias
        
    loss = getLoss(batch, lossFunction, model, optimizer, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    loss.backward()

    currentGrad = model.weight.grad
    biasGrad = model.bias.grad

    idealWeights = torch.hstack((targetWeights, targetBias)).to(device)
    idealDirection = torch.clone(idealWeights - torch.hstack((currentWeights[:,0,0].detach(), currentBias.detach())).detach()).to(device)
    actualGradient = torch.clone(torch.hstack((currentGrad[:,0,0].detach().cpu(), biasGrad.detach().cpu()))).to(device)
    normIdealDirection = torch.clone(idealDirection / torch.linalg.norm(idealDirection))
    normCurrentDirection = torch.clone((actualGradient / torch.linalg.norm(actualGradient)))
    center = torch.clone(torch.hstack((currentWeights[:,0,0], currentBias)).detach())

    orthogonalDirection = normCurrentDirection - normIdealDirection.dot(normCurrentDirection) * normIdealDirection
    orthogonalDirection = orthogonalDirection / torch.linalg.norm(orthogonalDirection).to(device)
#     print(normIdealDirection)
#     print(normCurrentDirection)
#     print(orthogonalDirection)
#     print(normCurrentDirection.dot(normIdealDirection), normCurrentDirection.dot(orthogonalDirection))
#     print(orthogonalDirection.dot(normIdealDirection), orthogonalDirection.dot(normCurrentDirection))
    # startingWeights = 
    # with torch.no_grad():    
    #     model.weight[:,0,0] = torch.tensor(weightFn(np.linspace(-1,1,n)))
    #     model.weight[:,0,0] = kernel(torch.abs(torch.linspace(-1,1,n)),1) * baseArea / particleSupport
#     with torch.no_grad():
#         model.weight[:,0,0] = torch.nn.Parameter(center[:-1].type(model.weight.dtype)
#         model.bias = torch.nn.Parameter(center[-1].type(model.weight.dtype).to(model.weight.device))
        
    stackedPositions, features, groundTruth, stackedNeighbors, d = loadBatch(simulationStates, minDomain, maxDomain, particleSupport, np.arange(1), getFeatures, getGroundTruth, stacked)

    limit = torch.linalg.norm(idealDirection) * 1.1
#     nx = 127
    xExtent = [-limit,limit]
    yExtent = [-limit,limit]
    xs = torch.linspace(xExtent[0],xExtent[1],nx)
    ys = torch.linspace(yExtent[0],yExtent[1],nx)
    # ys = [0]
    with torch.no_grad():
        model.weight[:,0,0] = torch.nn.Parameter(center[:-1].type(model.weight.dtype).to(model.weight.device))
        model.bias = torch.nn.Parameter(center[-1].type(model.weight.dtype).to(model.weight.device))

    losses = []
    for xi in tqdm(xs,leave = False):
        lossx = []
        for yi in tqdm(ys,leave = False):
            currentWeights = center + xi * orthogonalDirection + yi * normIdealDirection
            with torch.no_grad():
                model.weight[:,0,0] = torch.nn.Parameter(currentWeights[:-1].type(model.weight.dtype).to(model.weight.device))
                model.bias = torch.nn.Parameter(currentWeights[-1].type(model.weight.dtype).to(model.weight.device))
    #             model.weight = torch.nn.Parameter(currentWeights.type(model.weight.dtype).to(model.weight.device))
                prediction = model((features[:,None], features[:,None]), stackedNeighbors, d)
                lossTerm = lossFunction(prediction, groundTruth)
                loss = torch.mean(lossTerm)
                lossx.append(loss)

        losses.append(torch.hstack(lossx))
    losses = torch.vstack(losses)

    with torch.no_grad():
        model.weight[:,0,0] = torch.nn.Parameter(center[:-1].type(model.weight.dtype).to(model.weight.device))
        model.bias = torch.nn.Parameter(center[-1].type(model.weight.dtype).to(model.weight.device))
        
        
    return losses, idealDirection, actualGradient, orthogonalDirection, [x.cpu() for x in xExtent], [x.cpu() for x in yExtent]


def plotLossLandscape(nx, idealWeights, batch, lossFunction, model, optimizer, device, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked):
    currentWeights = model.weight
    currentBias = model.bias
#     idealWeights = torch.clone(torch.hstack((kernel(torch.abs(torch.linspace(-1,1,n)),1) * baseArea / particleSupport,torch.tensor(0))))
    idealDirection = torch.clone(idealWeights - torch.hstack((currentWeights[:,0,0].detach().cpu(), currentBias.detach().cpu())).detach())

    lossLandscape, idealDirection, actualGradient, orthogonalDirection, xExtent, yExtent = buildLossLandscape(nx, idealWeights[:-1], idealWeights[-1], [0], lossFunction, model, optimizer, device, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)


    fig, axis = plt.subplots(1,1, figsize=(12,12), sharex = False, sharey = False, squeeze = False)
    losses = lossLandscape.detach().cpu().numpy()



    im = axis[0,0].imshow(losses.transpose()[::-1,:],extent=[xExtent[0],xExtent[1],yExtent[0],yExtent[1]], norm=LogNorm(vmin=np.min(losses), vmax=np.max(losses)), interpolation = 'bicubic')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].scatter(0,0, c = 'red', s = 2)

    idealPoint = torch.linalg.norm(idealDirection)

    curGradX = (idealDirection / torch.linalg.norm(idealDirection)).dot(actualGradient).cpu().numpy()
    curGradY = orthogonalDirection.dot(actualGradient).cpu().numpy()

    currentGradient = np.hstack((curGradX, curGradY))
    currentGradient = currentGradient if np.linalg.norm(currentGradient) < idealPoint else currentGradient / np.linalg.norm(currentGradient) * idealPoint.item()



    # axis[0,0].scatter(0,idealPoint.cpu().numpy() , c = 'white', s = 2)
    axis[0,0].plot([0,0],[0,idealPoint.cpu().numpy()], c = 'white', lw = 2)

    axis[0,0].scatter(currentGradient[0],currentGradient[1] , c = 'blue', s = 2)
    axis[0,0].plot([0,currentGradient[0]],[0,currentGradient[1]] , c = 'blue', lw = 2)
    fig.tight_layout()


def buildLossAndGradientLandscape(nx, targetWeights, targetBias, batch, lossFunction, model, optimizer, device, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked):
    center = torch.clone(torch.hstack((model.weight[:,0,0], model.bias)).detach())
    
    currentWeights = model.weight
    currentBias = model.bias
    
    optimizer.zero_grad()
    loss = getLoss(batch, lossFunction, model, optimizer, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    loss.backward()

    currentGrad = model.weight.grad
    biasGrad = model.bias.grad

    idealWeights = torch.hstack((targetWeights, targetBias)).to(device)
    idealDirection = torch.clone(idealWeights - torch.hstack((currentWeights[:,0,0].detach(), currentBias.detach())).detach()).to(device)
    actualGradient = torch.clone(torch.hstack((currentGrad[:,0,0].detach().cpu(), biasGrad.detach().cpu()))).to(device)
    normIdealDirection = torch.clone(idealDirection / torch.linalg.norm(idealDirection))
    normCurrentDirection = torch.clone((actualGradient / torch.linalg.norm(actualGradient)))
    center = torch.clone(torch.hstack((currentWeights[:,0,0], currentBias)).detach())

    orthogonalDirection = normCurrentDirection - normIdealDirection.dot(normCurrentDirection) * normIdealDirection
    orthogonalDirection = orthogonalDirection / torch.linalg.norm(orthogonalDirection).to(device)
        
    stackedPositions, features, groundTruth, stackedNeighbors, d = loadBatch(simulationStates, minDomain, maxDomain, particleSupport, np.arange(1), getFeatures, getGroundTruth, stacked)

    limit = min(torch.linalg.norm(actualGradient) * 1.1,torch.tensor(1).to(actualGradient.device).type(actualGradient.dtype))
#     nx = 127
    xExtent = [-limit,limit]
    yExtent = [-limit,limit]
    xs = torch.linspace(xExtent[0],xExtent[1],nx)
    ys = torch.linspace(yExtent[0],yExtent[1],nx)
    # ys = [0]
    with torch.no_grad():
        model.weight[:,0,0] = torch.nn.Parameter(center[:-1].type(model.weight.dtype).to(model.weight.device))
        model.bias = torch.nn.Parameter(center[-1].type(model.weight.dtype).to(model.weight.device))

    losses = []
    gradients = []
    for xi in tqdm(xs,leave = False):
        lossx = []
        gradx = []
        for yi in tqdm(ys,leave = False):
            currentWeights = center + xi * orthogonalDirection + yi * normIdealDirection
            with torch.no_grad():
                model.weight[:,0,0] = torch.nn.Parameter(currentWeights[:-1].type(model.weight.dtype).to(model.weight.device))
                model.bias = torch.nn.Parameter(currentWeights[-1].type(model.weight.dtype).to(model.weight.device))
            optimizer.zero_grad()
#             model.weight = torch.nn.Parameter(currentWeights.type(model.weight.dtype).to(model.weight.device))
            prediction = model((features[:,None], features[:,None]), stackedNeighbors, d)
            lossTerm = lossFunction(prediction, groundTruth)
            loss = torch.mean(lossTerm)
            lossx.append(loss)
            loss.backward()
            gradx.append(torch.linalg.norm(torch.hstack((model.weight.grad[:,0,0], model.bias.grad))).detach().cpu())

        losses.append(torch.hstack(lossx))
        gradients.append(torch.hstack(gradx))
    losses = torch.vstack(losses)
    gradients = torch.vstack(gradients)

    with torch.no_grad():
        model.weight[:,0,0] = torch.nn.Parameter(center[:-1].type(model.weight.dtype).to(model.weight.device))
        model.bias = torch.nn.Parameter(center[-1].type(model.weight.dtype).to(model.weight.device))
        
        
    return losses, gradients, idealDirection, actualGradient, orthogonalDirection, [x.cpu() for x in xExtent], [x.cpu() for x in yExtent]
def plotLossAndGradientLandscape(nx, idealWeights, batch, lossFunction, model, optimizer, device, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked):
    zeroWeights = torch.zeros_like(idealWeights)

    lossLandscape, gradientLandscape, idealDirection, actualGradient, orthogonalDirection, xExtent, yExtent = buildLossAndGradientLandscape(nx, zeroWeights[:-1], zeroWeights[-1], [0], lossFunction, model, optimizer, device, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)


    fig, axis = plt.subplots(1,2, figsize=(18,9), sharex = False, sharey = False, squeeze = False)
    losses = lossLandscape.detach().cpu().numpy()
    gradients = gradientLandscape.detach().cpu().numpy()


    axis[0,0].set_title('Loss Landscape')
    im = axis[0,0].imshow(losses[::-1,:],extent=[xExtent[0],xExtent[1],yExtent[0],yExtent[1]], norm=LogNorm(vmin=np.min(losses), vmax=np.max(losses)), interpolation = 'bilinear')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("bottom", size="2%", pad="5%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='horizontal')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].scatter(0,0, c = 'red', s = 2)


    axis[0,1].set_title('Gradient Magnitude Landscape')
    im = axis[0,1].imshow(gradients[::-1,:],extent=[xExtent[0],xExtent[1],yExtent[0],yExtent[1]], norm=LogNorm(vmin=np.min(gradients), vmax=np.max(gradients)), interpolation = 'bilinear')
    ax1_divider = make_axes_locatable(axis[0,1])
    cax1 = ax1_divider.append_axes("bottom", size="2%", pad="5%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='horizontal')
    cb1.ax.tick_params(labelsize=8) 
    # axis[0,0].scatter(0,0, c = 'red', s = 2)

    idealPoint = torch.linalg.norm(idealDirection)

    curGradX = (idealDirection / torch.linalg.norm(idealDirection)).dot(actualGradient).cpu().numpy()
    curGradY = orthogonalDirection.dot(actualGradient).cpu().numpy()

    currentGradient = np.hstack((curGradY, curGradX))
    currentGradient = currentGradient if np.linalg.norm(currentGradient) < idealPoint else currentGradient / np.linalg.norm(currentGradient) * idealPoint.item()


    # idealWeights = torch.hstack((targetWeights, targetBias)).to(device)
    # targetGradient = torch.clone(targetWeights.to(currentWeights.device) - torch.hstack((currentWeights[:,0,0].detach(), currentBias.detach())).detach()).to(device)

    # targetGradX = (idealDirection / torch.linalg.norm(idealDirection)).dot(targetGradient).cpu().numpy()
    # targetGradY = orthogonalDirection.dot(targetGradient).cpu().numpy()

    # targetGradient = np.hstack((targetGradX, targetGradY))
    # targetGradient = targetGradient if np.linalg.norm(targetGradient) < idealPoint else targetGradient / np.linalg.norm(targetGradient) * idealPoint.item()



    # axis[0,0].scatter(0,idealPoint.cpu().numpy() , c = 'white', s = 2)
    # axis[0,0].plot([0,0],[0,idealPoint.cpu().numpy()], c = 'white', lw = 2)

    # axis[0,0].scatter(currentGradient[0],currentGradient[1] , c = 'blue', s = 2)
    # axis[0,0].plot([0,currentGradient[0]],[0,currentGradient[1]] , c = 'blue', lw = 2)

    # axis[0,0].scatter(targetGradient[0],targetGradient[1] , c = 'red', s = 2)
    # axis[0,0].plot([0,targetGradient[0]],[0,targetGradient[1]] , c = 'red', lw = 2)

    xpts = np.linspace(xExtent[0], xExtent[1], losses.shape[0])
    ypts = np.linspace(yExtent[0], yExtent[1], losses.shape[0])
    XX, YY = np.meshgrid(xpts, ypts)

    testData = (XX - 5)**2 + YY**2

    xgrad, ygrad = np.gradient(losses)
    axis[0,0].streamplot(XX, YY, -ygrad, -xgrad, color = 'white')
    # axis[0,0].quiver(XX[::2,::2], YY[::2,::2], -ygrad[::2,::2], -xgrad[::2,::2], color ='white')

    xgrad, ygrad = np.gradient(gradients)
    axis[0,1].streamplot(XX, YY, -ygrad, -xgrad, color = 'white')

    fig.tight_layout()


def computeEvaluationLoss(model, weights, bdata, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked, batchSize = 128):  
    batched = np.array_split(bdata, len(bdata) // batchSize + 1)
    predictions = []
    groundTruths = []
    lossTerms = []
    losses = []
    for batch in tqdm(batched, leave = False):        
        with torch.no_grad():
#             print('1')
            storedWeights = copy.deepcopy(model.state_dict())
#             print('2')
            model.load_state_dict(weights)
#             print('3')
            # storedWeights = torch.clone(model.weight.detach())
            # model.weight = torch.nn.Parameter(torch.tensor(weights).type(model.weight.dtype).to(model.weight.device))

            stackedPositions, features, groundTruth, stackedNeighbors, d = loadBatch(simulationStates, minDomain, maxDomain, particleSupport, batch, getFeatures, getGroundTruth, stacked)

#             print(features, stackedNeighbors, d)
#             print(features.shape, stackedNeighbors.shape, d.shape)
#             print('4')
            # run the network layer
            prediction = model((features[:,None], features[:,None]), stackedNeighbors, d)
#             print('5')
            # model.weight = torch.nn.Parameter(storedWeights)
            model.load_state_dict(storedWeights)
#             print('6')
            # compute the loss
            lossTerm = lossFunction(prediction, groundTruth)
#             print('7')
            loss = torch.mean(lossTerm)
#             print('8')
            predictions.append(prediction)
            groundTruths.append(groundTruth)
            lossTerms.append(lossTerm)
            losses.append(loss)
#         print('9')
#     print(predictions)
#     print([p.shape for p in predictions])
#     print(np.concatenate(predictions, axis = 0))
#     print(np.hstack(groundTruths))
#     print(np.hstack(lossTerms))
#     print(np.hstack(losses))
    return torch.cat(predictions, axis = 0).cpu(), torch.cat(groundTruths, axis = 0).cpu(), torch.cat(lossTerms, axis = 0).cpu(), torch.hstack(losses).cpu()

def plotAB(fig, axisA, axisB, dataA, dataB, batchesA, batchesB, numParticles, cmap = 'viridis'):
    vmin = min(torch.min(dataA), torch.min(dataB))
    vmax = max(torch.max(dataA), torch.max(dataB))
    imA = axisA.imshow(dataA.reshape((batchesA.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', vmin = vmin, vmax = vmax, extent = [np.min(batchesA),np.max(batchesA),numParticles,0]) # uses some matrix reshaping to undo the hstack
    imB = axisB.imshow(dataB.reshape((batchesB.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', vmin = vmin, vmax = vmax, extent = [np.min(batchesB),np.max(batchesB),numParticles,0]) # uses some matrix reshaping to undo the hstack
    axisA.axis('auto')
    axisB.axis('auto')
    ax1_divider = make_axes_locatable(axisB)
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cbarPredFFT = fig.colorbar(imB, cax=cax1,orientation='vertical')
    cbarPredFFT.ax.tick_params(labelsize=8) 
def plotABLog(fig, axisA, axisB, dataA, dataB, batchesA, batchesB, numParticles, cmap = 'viridis'):
    vmin = min(np.percentile(dataA[dataA > 0], 1), np.percentile(dataB[dataB > 0],1))
    vmax = max(torch.max(dataA), torch.max(dataB))
    imA = axisA.imshow(dataA.reshape((batchesA.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', norm = LogNorm(vmin=vmin, vmax=vmax), extent = [np.min(batchesA),np.max(batchesA),numParticles,0]) # uses some matrix reshaping to undo the hstack
    imB = axisB.imshow(dataB.reshape((batchesB.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', norm = LogNorm(vmin=vmin, vmax=vmax), extent = [np.min(batchesB),np.max(batchesB),numParticles,0]) # uses some matrix reshaping to undo the hstack
    axisA.axis('auto')
    axisB.axis('auto')
    ax1_divider = make_axes_locatable(axisB)
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cbarPredFFT = fig.colorbar(imB, cax=cax1,orientation='vertical')
    cbarPredFFT.ax.tick_params(labelsize=8) 
    
def plotAll(model, device, weights, basis, normalized, iterations, epochs, numParticles, batchSize, lossArray, simulationStates, minDomain, maxDomain, particleSupport, timestamps, testBatch, lossFunction, getFeatures, getGroundTruth, stacked):
    trainingPrediction, trainingGroundTruth, trainingLossTerm, trainingLoss = computeEvaluationLoss(model, weights[-1][-1], timestamps, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    testingPrediction, testingGroundTruth, testingLossTerm, testingLoss = computeEvaluationLoss(model, weights[-1][-1], testBatch, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    fig, axis = plt.subplot_mosaic('''AABB
    AABB
    CCCD
    EEEF
    GGGH''', figsize=(16,10), sharex = False, sharey = False)
    fig.suptitle('Training results for basis %s%s %2d epochs %4d iterations batchSize %d: %2.6g' % (basis, '' if not normalized else ' (normalized)', epochs, iterations, batchSize, np.mean(lossArray[-1][-1])))

    batchedLosses = np.stack(lossArray, axis = 0).reshape(iterations * epochs, numParticles * batchSize)
    axis['A'].set_title('Learning progress')
    axis['A'].semilogy(np.mean(batchedLosses, axis = 1))
    axis['A'].semilogy(np.min(batchedLosses, axis = 1))
    axis['A'].semilogy(np.max(batchedLosses, axis = 1))

    axis['C'].set_title('Prediction (Training)') 
    axis['D'].set_title('Prediction (Testing)')
    axis['E'].set_title('Ground Truth (Training)') 
    axis['F'].set_title('Ground Truth (Testing)')
    axis['G'].set_title('Loss (Training)') 
    axis['H'].set_title('Loss (Testing)') 
    plotAB(fig, axis['C'], axis['D'], trainingPrediction, testingPrediction, timestamps, testBatch, numParticles, cmap = 'viridis')
    plotAB(fig, axis['E'], axis['F'], trainingGroundTruth, testingGroundTruth, timestamps, testBatch, numParticles, cmap = 'viridis')
    #     plotAB(fig, axis['G'], axis['H'], trainingLossTerm, testingLossTerm, timestamps, testBatch, positions, cmap = 'viridis')
    plotABLog(fig, axis['G'], axis['H'], trainingLossTerm, testingLossTerm, timestamps, testBatch, numParticles, cmap = 'viridis')
    axis['C'].set_xticklabels([])
    axis['D'].set_xticklabels([])
    axis['E'].set_xticklabels([])
    axis['F'].set_xticklabels([])
    axis['D'].set_yticklabels([])
    axis['F'].set_yticklabels([])
    axis['H'].set_yticklabels([])
    
    
    cm = mpl.colormaps['viridis']

    x =  torch.linspace(-1,1,511)[:,None].to(device)
    fx = torch.ones(511)[:,None].to(device)
    neighbors = torch.vstack((torch.zeros(511).type(torch.long), torch.arange(511).type(torch.long)))
    neighbors = torch.vstack((torch.arange(511).type(torch.long), torch.zeros(511).type(torch.long))).to(device)
    # internal function that is used for the rbf convolution
    #     n = weights[-1][-1]['weight'].shape[0]
    #     fx = evalBasisFunction(n, x , which = basis, periodic=False)
    #     fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    # print(neighbors)
    steps = iterations * epochs
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()

    # print(x, fx, neighbors)
#     model((fx,fx), neighbors, x)

    storedWeights = copy.deepcopy(model.state_dict())
    c = 0
    for i in tqdm(range(epochs), leave = False):
        for j in tqdm(range(iterations), leave = False):
            c = c + 1        
            if c + 1 in ls:           
                
                model.load_state_dict({k: v.to(device) for k, v in weights[i][j].items()})
#                 model = model.to(device)
                axis['B'].plot(x[:,0].detach().cpu().numpy(), model((fx,fx), neighbors, x).detach().cpu().numpy(),ls='--',c= cm(ls.index(c+1) / (len(ls) - 1)), alpha = 0.95)
    #             break

    model.load_state_dict(storedWeights)
    axis['B'].set_title('Weight progress')

    # fig, axis = plt.subplots(3, 2, figsize=(16,6), sharex = 'col', sharey = True, squeeze = False, gridspec_kw={'width_ratios': [3, 1]})

    
    fig.tight_layout()
    model.load_state_dict({k: v.to(device) for k, v in weights[-1][-1].items()})

    return fig, axis


def plotTrainingAndTesting1Layer(model, lossArray, weights, basis, normalized, iterations, epochs, numParticles, batchSize, testBatch,lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked):
    # Plot the learned convolution (only works for single layer models (for now))
    fig, axis = plt.subplots(2, 2, figsize=(16,8), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,511)
    # internal function that is used for the rbf convolution
    n = weights[-1][-1].shape[0]
    fx = evalBasisFunction(n, x , which = basis, periodic=False)
    fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    # # plot the individual basis functions with a weight of 1
    # for y in range(n):
    #     axis[1,0].plot(x, fx[y,:], label = '$f_%d(x)$' % y)
    # # plot the overall convolution basis for all weights equal to 1
    # axis[1,0].plot(x,torch.sum(fx, axis=0),ls='--',c='white', label = '$\Sigma_i f_i(x)$')
    # # axis[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fancybox=True, shadow=False)
    # axis[1,0].set_title('Basis Functions')

    # plot the individual basis functions with the learned weights
    for y in range(n):
        fy = model.weight[:,0][y].detach() * fx[y,:]
        axis[0,1].plot(x[fy != 0], fy[fy != 0], label = '$w_d f_%d(x)$' % y, ls = '--', alpha = 0.5)
    axis[0,1].plot(x,torch.sum(model.weight[:,0].detach() * fx,axis=0) + model.bias.detach(),ls='--',c='white', label = '$\Sigma_i w_i f_i(x)$')
    # axis[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fancybox=True, shadow=False)
    axis[0,1].set_title('Learned convolution')

    batchedLosses = np.stack(lossArray, axis = 0).reshape(iterations * epochs, numParticles * batchSize)
    axis[0,0].set_title('Learning progress')
    axis[0,0].semilogy(np.mean(batchedLosses, axis = 1))
    axis[0,0].semilogy(np.min(batchedLosses, axis = 1))
    axis[0,0].semilogy(np.max(batchedLosses, axis = 1))

    epochTestLosses = []
    for epoch in tqdm(range(epochs)):
        prediction, groundTruth, lossTerm, loss = computeEvaluationLoss(model, weights[-1][-1], testBatch, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
        epochTestLosses.append(lossTerm)
    epochTestLosses = torch.vstack(epochTestLosses).reshape(epochs, numParticles * len(testBatch)).detach().cpu().numpy()

    # batchedLosses = np.stack(testing, axis = 0).reshape(epochs, numParticles * ignoredTimesteps)
    axis[1,0].set_title('Testing progress')
    axis[1,0].semilogy(np.arange(0, epochs) * iterations, np.mean(epochTestLosses, axis = 1))
    axis[1,0].semilogy(np.arange(0, epochs) * iterations, np.min(epochTestLosses, axis = 1))
    axis[1,0].semilogy(np.arange(0, epochs) * iterations, np.max(epochTestLosses, axis = 1))

    cm = mpl.colormaps['viridis']

    steps = iterations * epochs
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()

    c = 0
    for i in range(epochs):
        for j in range(iterations):
            c = c + 1        
            if c + 1 in ls:
                axis[1,1].plot(x,torch.sum(torch.tensor(weights[i][j][:,0]) * fx,axis=0),ls='--',c= cm(ls.index(c+1) / (len(ls) - 1)), alpha = 0.95)
    axis[1,1].set_title('Weight progress')
    fig.tight_layout()   




def plotTraining1Layer(model, lossArray, weights, basis, normalized, iterations, epochs, numParticles, batchSize):
    # Plot the learned convolution (only works for single layer models (for now))
    fig, axis = plt.subplots(1, 2, figsize=(16,4), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,511)
    n = weights[-1][-1].shape[0]
    # internal function that is used for the rbf convolution
    fx = evalBasisFunction(n, x , which = basis, periodic=False)
    fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    # # plot the individual basis functions with a weight of 1
    # for y in range(n):
    #     axis[1,0].plot(x, fx[y,:], label = '$f_%d(x)$' % y)
    # # plot the overall convolution basis for all weights equal to 1
    # axis[1,0].plot(x,torch.sum(fx, axis=0),ls='--',c='white', label = '$\Sigma_i f_i(x)$')
    # # axis[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fancybox=True, shadow=False)
    # axis[1,0].set_title('Basis Functions')

#     # plot the individual basis functions with the learned weights
#     for y in range(n):
#         fy = model.weight[:,0][y].detach() * fx[y,:]
#         axis[0,1].plot(x[fy != 0], fy[fy != 0], label = '$w_d f_%d(x)$' % y, ls = '--', alpha = 0.5)
#     axis[0,1].plot(x,torch.sum(model.weight[:,0].detach() * fx,axis=0),ls='--',c='white', label = '$\Sigma_i w_i f_i(x)$')
#     # axis[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fancybox=True, shadow=False)
#     axis[0,1].set_title('Learned convolution')

    batchedLosses = np.stack(lossArray, axis = 0).reshape(iterations * epochs, numParticles * batchSize)
    axis[0,0].set_title('Learning progress')
    axis[0,0].semilogy(np.mean(batchedLosses, axis = 1))
    axis[0,0].semilogy(np.min(batchedLosses, axis = 1))
    axis[0,0].semilogy(np.max(batchedLosses, axis = 1))

#     epochTestLosses = []
#     for epoch in tqdm(range(epochs)):
#         prediction, groundTruth, lossTerm, loss = computeEvaluationLoss(weights[epoch][-1], testBatch)
#         epochTestLosses.append(lossTerm)
#     epochTestLosses = torch.vstack(epochTestLosses).reshape(epochs, numParticles * ignoredTimesteps).detach().cpu().numpy()

#     # batchedLosses = np.stack(testing, axis = 0).reshape(epochs, numParticles * ignoredTimesteps)
#     axis[1,0].set_title('Testing progress')
#     axis[1,0].semilogy(np.arange(0, epochs) * iterations, np.mean(epochTestLosses, axis = 1))
#     axis[1,0].semilogy(np.arange(0, epochs) * iterations, np.min(epochTestLosses, axis = 1))
#     axis[1,0].semilogy(np.arange(0, epochs) * iterations, np.max(epochTestLosses, axis = 1))

    cm = mpl.colormaps['viridis']

    steps = iterations * epochs
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()

    c = 0
    for i in range(epochs):
        for j in range(iterations):
            c = c + 1        
            if c + 1 in ls:
                axis[0,1].plot(x,torch.sum(torch.tensor(weights[i][j]['weight'][:,0]) * fx,axis=0) + weights[i][j]['bias'],ls='--',c= cm(ls.index(c+1) / (len(ls) - 1)), alpha = 0.95)
    axis[0,1].set_title('Weight progress')
    fig.tight_layout()   

def plotBatchedLoss(model, weights, batch, lossFunction, simulationStates, numParticles, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked):
    prediction, groundTruth, lossTerm, loss = computeEvaluationLoss(model, weights, batch, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    # Plot the 'testing' data
    fig, axis = plt.subplots(1, 3, figsize=(16,6), sharex = False, sharey = False, squeeze = False)

    axis[0,0].set_title('Prediction') 
    im = axis[0,0].imshow(prediction.reshape((batch.shape[0], numParticles)), interpolation = 'nearest') # uses some matrix reshaping to undo the hstack
    axis[0,0].axis('auto')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("bottom", size="5%", pad="15%")
    cbarPredFFT = fig.colorbar(im, cax=cax1,orientation='horizontal')
    cbarPredFFT.ax.tick_params(labelsize=8) 

    axis[0,1].set_title('GT')
    im = axis[0,1].imshow(groundTruth.reshape((batch.shape[0], numParticles)), interpolation = 'nearest')
    axis[0,1].axis('auto')
    ax1_divider = make_axes_locatable(axis[0,1])
    cax1 = ax1_divider.append_axes("bottom", size="5%", pad="15%")
    cbarPredFFT = fig.colorbar(im, cax=cax1,orientation='horizontal')
    cbarPredFFT.ax.tick_params(labelsize=8) 

    axis[0,2].set_title('Loss')
    im = axis[0,2].imshow(lossTerm.reshape((batch.shape[0], numParticles)), interpolation = 'nearest')
    axis[0,2].axis('auto')
    ax1_divider = make_axes_locatable(axis[0,2])
    cax1 = ax1_divider.append_axes("bottom", size="5%", pad="15%")
    cbarPredFFT = fig.colorbar(im, cax=cax1,orientation='horizontal')
    cbarPredFFT.ax.tick_params(labelsize=8) 

    fig.tight_layout()

def evalTestingAndTraining(model, weights, timestamps, testBatch, lossFunction, simulationStates, numParticles, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked, plot = True):
    trainingPrediction, trainingGroundTruth, trainingLossTerm, trainingLoss = computeEvaluationLoss(model, weights, timestamps, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    testingPrediction, testingGroundTruth, testingLossTerm, testingLoss = computeEvaluationLoss(model, weights, testBatch, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    if plot:
        fig, axis = plt.subplots(3, 2, figsize=(16,6), sharex = 'col', sharey = True, squeeze = False, gridspec_kw={'width_ratios': [3, 1]})

        axis[0,0].set_title('Prediction (Training)') 
        axis[0,1].set_title('Prediction (Testing)')
        axis[1,0].set_title('Ground Truth (Training)') 
        axis[1,1].set_title('Ground Truth (Testing)')
        axis[2,0].set_title('Loss (Training)') 
        axis[2,1].set_title('Loss (Testing)') 
        plotAB(fig, axis[0,0], axis[0,1], trainingPrediction, testingPrediction, timestamps, testBatch, numParticles, cmap = 'viridis')
        plotAB(fig, axis[1,0], axis[1,1], trainingGroundTruth, testingGroundTruth, timestamps, testBatch, numParticles, cmap = 'viridis')
    #     plotAB(axis[2,0], axis[2,1], trainingLossTerm, testingLossTerm, timestamps, testBatch, positions, cmap = 'viridis')
        plotABLog(fig, axis[2,0], axis[2,1], trainingLossTerm, testingLossTerm, timestamps, testBatch, numParticles, cmap = 'viridis')
        fig.tight_layout()
        
    return trainingPrediction, testingPrediction, trainingGroundTruth, testingGroundTruth, trainingLossTerm, testingLossTerm 

def plotMLP(model, weights):
    # Plot the learned convolution (only works for single layer models (for now))
    fig, axis = plt.subplots(1, 1, figsize=(16,4), sharex = False, sharey = False, squeeze = False)

    cm = mpl.colormaps['viridis']

    x =  torch.linspace(-1,1,511)[:,None]
    fx = torch.ones(511)[:,None]
    neighbors = torch.vstack((torch.zeros(511).type(torch.long), torch.arange(511).type(torch.long)))
    neighbors = torch.vstack((torch.arange(511).type(torch.long), torch.zeros(511).type(torch.long)))
    # internal function that is used for the rbf convolution
    #     n = weights[-1][-1]['weight'].shape[0]
    #     fx = evalBasisFunction(n, x , which = basis, periodic=False)
    #     fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    # print(neighbors)
    # steps = iterations * epochs
    # ls = np.logspace(0, np.log10(steps), num =  50)
    # ls = [int(np.floor(f)) for f in ls]
    # ls = np.unique(ls).tolist()

    # print(x, fx, neighbors)
#     model((fx,fx), neighbors, x)

    storedWeights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    axis[0,0].plot(x[:,0], model((fx,fx), neighbors, x).detach(),ls='--',c= 'white', alpha = 0.95)
    #             break

    model.load_state_dict(storedWeights)

    fig.tight_layout()
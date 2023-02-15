
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib as mpl

import numpy as np

from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

import matplotlib.pyplot as plt

# overallBins = []
# overallCounts = []
def histPlot(fig, axis, dataSet, binCount = 32, xLabel = None, yLabel = None, logScale = True):
    axis.set_yscale('log')

    histBins = []
    histCounts = []

    medians = []
    q1s = []
    q3s = []
    q0s = []
    q9s = []
    mins = []
    maxs = []

    for e in range(dataSet.shape[0]):
        data = dataSet[e,:]
        bs = 10 ** np.linspace(np.log10(np.min(data)), np.log10(np.max(data)), binCount) if logScale else np.linspace(np.min(data), np.max(data), binCount)

        cts, bins = np.histogram(dataSet[e,:], bins = bs)
        bins = (bins[:-1] + bins[1:]) / 2
        histBins.append(bins)
        histCounts.append(cts)

        nzd = data[data > 0]
        q0s.append(np.percentile(nzd, 5))
        q1s.append(np.percentile(nzd, 25))
        medians.append(np.median(nzd))
        q3s.append(np.percentile(nzd, 75))
        q9s.append(np.percentile(nzd, 95))
        mins.append(np.min(nzd))
        maxs.append(np.max(nzd))


    oCounts = np.hstack(histCounts)
    minVal = np.min(oCounts[oCounts > 0])
    maxVal = np.max(oCounts)

    for e, (cts, bins) in enumerate(zip(histCounts, histBins)):
        mapped = (cts[cts>0] - minVal ) / (maxVal - minVal)

        x = np.ones_like(bins)[cts > 0] * e
        y = bins[cts > 0]
        z = mapped

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cm.viridis, lw = 4)
        lc.set_array(z)

        sc = axis.scatter(np.ones_like(bins)[cts > 0] * e, bins[cts > 0], c = cts[cts > 0], alpha = 1.00 , s = 0.1, vmin = np.min(oCounts[oCounts > 0]), vmax = np.max(oCounts))
        # axis[0,0].plot(np.ones_like(bins)[cts > 0] * e, bins[cts > 0], \
            # c = cm.viridis(mapped), alpha = 1.00 , s = 1)
        axis.add_collection(lc)
    es = np.arange(len(medians))
    axis.plot(es, medians, c = 'white', alpha = 0.5)
    axis.plot(es, q1s, c = 'white', alpha = 0.5, ls = ':')
    axis.plot(es, q3s, c = 'white', alpha = 0.5, ls = ':')
    axis.plot(es, q0s, c = 'white', alpha = 0.5, ls = '--')
    axis.plot(es, q9s, c = 'white', alpha = 0.5, ls = '--')
    axis.plot(es, mins, c = 'white', alpha = 0.25, ls = 'dashdot')
    axis.plot(es, maxs, c = 'white', alpha = 0.25, ls = 'dashdot')

    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_alpha(1)
    cbar.set_label('count')
    if yLabel is not None:
        axis.set_ylabel(yLabel)
    if xLabel is not None:
        axis.set_xlabel(xLabel)

def plotLossesv1(epochLosses):
    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in epochLosses])
    indices = np.arange(overallLosses.shape[1])

    fig, axis = plt.subplots(3, 4, figsize=(16,12), sharex = False, sharey = False, squeeze = False)

    axis[1,0].set_yscale('log')
    axis[1,1].set_yscale('log')
    axis[1,2].set_yscale('log')
    axis[1,3].set_yscale('log')
    axis[1,0].set_xlabel('index')
    axis[1,1].set_xlabel('index')
    axis[1,2].set_xlabel('index')
    axis[1,3].set_xlabel('index')
    axis[1,0].set_ylabel('Loss')

    axis[0,0].set_xlabel('epoch')
    axis[0,1].set_xlabel('epoch')
    axis[0,2].set_xlabel('epoch')
    axis[0,3].set_xlabel('epoch')
    axis[0,0].set_ylabel('Loss')

    
    
    axis[2,0].set_xlabel('epoch')
    axis[2,1].set_xlabel('epoch')
    axis[2,2].set_xlabel('epoch')
    axis[2,3].set_xlabel('epoch')
    axis[2,0].set_ylabel('index')

    axis[0,0].set_title('mean Loss')
    axis[0,1].set_title('max Loss')
    axis[0,2].set_title('min Loss')
    axis[0,3].set_title('std dev Loss')
    axis[0,0].set_yscale('log')
    axis[0,1].set_yscale('log')
    axis[0,2].set_yscale('log')
    axis[0,3].set_yscale('log')
    
    
#     axis[0,0].grid(axis='y', which='both', alpha = 0.5)
#     axis[0,1].grid(axis='y', which='both', alpha = 0.5)
#     axis[0,2].grid(axis='y', which='both', alpha = 0.5)
#     axis[0,3].grid(axis='y', which='both', alpha = 0.5)
    
#     axis[1,0].grid(axis='y', which='both', alpha = 0.5)
#     axis[1,1].grid(axis='y', which='both', alpha = 0.5)
#     axis[1,2].grid(axis='y', which='both', alpha = 0.5)
#     axis[1,3].grid(axis='y', which='both', alpha = 0.5)
    
    
    
    for e in range(overallLosses.shape[0]):
        axis[0,0].scatter(np.ones_like(indices) * e, overallLosses[e,:,0] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
        axis[0,1].scatter(np.ones_like(indices) * e, overallLosses[e,:,1] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
        axis[0,2].scatter(np.ones_like(indices) * e, overallLosses[e,:,2] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
        sc = axis[0,3].scatter(np.ones_like(indices) * e, overallLosses[e,:,3] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
        axis[1,0].plot(indices, overallLosses[e,:,0] ,label = '%d' % e, c =  cm.viridis(e / overallLosses.shape[0]), alpha = 0.5)
        axis[1,1].plot(indices, overallLosses[e,:,1] ,label = '%d' % e,  c = cm.viridis(e / overallLosses.shape[0]), alpha = 0.5)
        axis[1,2].plot(indices, overallLosses[e,:,2] ,label = '%d' % e, c = cm.viridis(e / overallLosses.shape[0]), alpha = 0.5)
        pl = axis[1,3].plot(indices, overallLosses[e,:,3] ,label = '%d' % e,  c =cm.viridis( e / overallLosses.shape[0]), alpha = 0.5)

    ax1_divider = make_axes_locatable(axis[0,3])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_alpha(1)
    cbar.set_label('index')
    cbar.solids.set(alpha=1)

    ax1_divider = make_axes_locatable(axis[1,3])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")

    cbar = mpl.colorbar.ColorbarBase(cax1,orientation='vertical', norm=mpl.colors.Normalize(vmin=0, vmax=overallLosses.shape[0]))
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label('epoch')

    im = axis[2,0].imshow(overallLosses[:,:,0].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,0][overallLosses[:,:,0] > 0]), vmax=np.max(overallLosses[:,:,0])))
    axis[2,0].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,0])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 

    im = axis[2,1].imshow(overallLosses[:,:,1].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,1][overallLosses[:,:,1] > 0]), vmax=np.max(overallLosses[:,:,1])))
    axis[2,1].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,1])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 

    im = axis[2,2].imshow(overallLosses[:,:,2].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,2][overallLosses[:,:,2] > 0]), vmax=np.max(overallLosses[:,:,2])))
    axis[2,2].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,2])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 


    im = axis[2,3].imshow(overallLosses[:,:,3].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,3][overallLosses[:,:,3] > 0]), vmax=np.max(overallLosses[:,:,3])))
    axis[2,3].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,3])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label('loss')

    fig.tight_layout()


def histPlot(fig, axis, dataSet, binCount = 32, xLabel = None, yLabel = None, logScale = True):
    if logScale:
        axis.set_yscale('log')

    histBins = []
    histCounts = []

    medians = []
    q1s = []
    q3s = []
    q0s = []
    q9s = []
    mins = []
    maxs = []

    
    for e in range(dataSet.shape[0]):
        data = dataSet[e,:]
        if np.sum(data > 0) > 0:
            minV = max(1e-11, np.min(data[data > 0]))
            maxV = max(1e-9, np.max(data[data > 0]))
        else:
            minV = max(1e-11, np.min(data))
            maxV = max(1e-9, np.max(data))
        bs = 10 ** np.linspace(np.log10(minV), np.log10(maxV), binCount) if logScale else np.linspace(minV, maxV, binCount)
#         debugPrint(bs)

        cts, bins = np.histogram(dataSet[e,:], bins = bs)
        bins = (bins[:-1] + bins[1:]) / 2
        histBins.append(bins)
        histCounts.append(cts)

        nzd = data[data > 0]
        try:
            q0s.append(np.percentile(nzd, 5))
        except Exception:
            pass
        try:
            q1s.append(np.percentile(nzd, 25))
        except Exception:
            pass
        medians.append(np.median(nzd))
        try:
            q3s.append(np.percentile(nzd, 75))
        except Exception:
            pass
        try:
            q9s.append(np.percentile(nzd, 95))
        except Exception:
            pass
        try:
            mins.append(np.min(nzd))
        except Exception:
            pass
        try:
            maxs.append(np.max(nzd))
        except Exception:
            pass


    oCounts = np.hstack(histCounts)
    minVal = np.min(oCounts[oCounts > 0]) if np.sum(oCounts > 0) > 0 else 0
    maxVal = np.max(oCounts)


    for e, (cts, bins) in enumerate(zip(histCounts, histBins)):
        mapped = (cts - minVal ) / (maxVal - minVal) if (maxVal - minVal > 0) else 0

        x = np.ones_like(bins)* e
        y = bins
        z = mapped

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cm.viridis, lw = 4)
        lc.set_array(z)

        sc = axis.scatter(np.ones_like(bins) * e, bins, c = cts, alpha = 1.00 , s = 0.1, vmin = minVal, vmax = maxVal)
        # axis[0,0].plot(np.ones_like(bins)[cts > 0] * e, bins[cts > 0], \
            # c = cm.viridis(mapped), alpha = 1.00 , s = 1)
        axis.add_collection(lc)
    es = np.arange(len(medians))
    axis.plot(es, medians, c = 'white', alpha = 0.5)
    if len(q0s) == len(medians):
        axis.plot(es, q1s, c = 'white', alpha = 0.5, ls = ':')
    if len(q3s) == len(medians):
        axis.plot(es, q3s, c = 'white', alpha = 0.5, ls = ':')
    if len(q0s) == len(medians):
        axis.plot(es, q0s, c = 'white', alpha = 0.5, ls = '--')
    if len(q9s) == len(medians):
        axis.plot(es, q9s, c = 'white', alpha = 0.5, ls = '--')
    if len(mins) == len(medians):
        axis.plot(es, mins, c = 'white', alpha = 0.25, ls = 'dashdot')
    if len(maxs) == len(medians):
        axis.plot(es, maxs, c = 'white', alpha = 0.25, ls = 'dashdot')

    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_alpha(1)
    cbar.set_label('count')
    if yLabel is not None:
        axis.set_ylabel(yLabel)
    if xLabel is not None:
        axis.set_xlabel(xLabel)
        
        
def plotLossesv2(epochLosses, logScale = True):
    overallLosses = np.vstack([np.expand_dims(np.vstack((np.mean(e[:,:,0], axis = 1), np.max(e[:,:,1], axis = 1), np.min(e[:,:,2], axis = 1) , np.mean(e[:,:,3], axis = 1))),2).T for e in epochLosses])
    indices = np.arange(overallLosses.shape[1])

    fig, axis = plt.subplots(3, 4, figsize=(16,12), sharex = False, sharey = False, squeeze = False)

    axis[1,0].set_yscale('log')
    axis[1,1].set_yscale('log')
    axis[1,2].set_yscale('log')
    axis[1,3].set_yscale('log')
    axis[1,0].set_xlabel('index')
    axis[1,1].set_xlabel('index')
    axis[1,2].set_xlabel('index')
    axis[1,3].set_xlabel('index')
    axis[1,0].set_ylabel('Loss')

    axis[0,0].set_xlabel('epoch')
    axis[0,1].set_xlabel('epoch')
    axis[0,2].set_xlabel('epoch')
    axis[0,3].set_xlabel('epoch')
    axis[0,0].set_ylabel('Loss')

    
    
    axis[2,0].set_xlabel('epoch')
    axis[2,1].set_xlabel('epoch')
    axis[2,2].set_xlabel('epoch')
    axis[2,3].set_xlabel('epoch')
    axis[2,0].set_ylabel('index')

    axis[0,0].set_title('mean Loss')
    axis[0,1].set_title('max Loss')
    axis[0,2].set_title('min Loss')
    axis[0,3].set_title('std dev Loss')
    if logScale:
        axis[0,0].set_yscale('log')
        axis[0,1].set_yscale('log')
        axis[0,2].set_yscale('log')
        axis[0,3].set_yscale('log')
    
    
#     axis[0,0].grid(axis='y', which='both', alpha = 0.5)
#     axis[0,1].grid(axis='y', which='both', alpha = 0.5)
#     axis[0,2].grid(axis='y', which='both', alpha = 0.5)
#     axis[0,3].grid(axis='y', which='both', alpha = 0.5)
    
#     axis[1,0].grid(axis='y', which='both', alpha = 0.5)
#     axis[1,1].grid(axis='y', which='both', alpha = 0.5)
#     axis[1,2].grid(axis='y', which='both', alpha = 0.5)
#     axis[1,3].grid(axis='y', which='both', alpha = 0.5)
    
    
    
    for e in range(overallLosses.shape[0]):
#         axis[0,0].scatter(np.ones_like(indices) * e, overallLosses[e,:,0] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
#         axis[0,1].scatter(np.ones_like(indices) * e, overallLosses[e,:,1] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
#         axis[0,2].scatter(np.ones_like(indices) * e, overallLosses[e,:,2] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
#         sc = axis[0,3].scatter(np.ones_like(indices) * e, overallLosses[e,:,3] ,label = '%d' % e, s = 0.25, c = indices, vmin = 0, vmax = np.max(indices), alpha = 0.75)
        axis[1,0].plot(indices, overallLosses[e,:,0] ,label = '%d' % e, c =  cm.viridis(e / overallLosses.shape[0]), alpha = 0.5)
        axis[1,1].plot(indices, overallLosses[e,:,1] ,label = '%d' % e,  c = cm.viridis(e / overallLosses.shape[0]), alpha = 0.5)
        axis[1,2].plot(indices, overallLosses[e,:,2] ,label = '%d' % e, c = cm.viridis(e / overallLosses.shape[0]), alpha = 0.5)
        pl = axis[1,3].plot(indices, overallLosses[e,:,3] ,label = '%d' % e,  c =cm.viridis( e / overallLosses.shape[0]), alpha = 0.5)

#     ax1_divider = make_axes_locatable(axis[0,3])
#     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
#     cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
#     cbar.ax.tick_params(labelsize=8) 
#     cbar.set_alpha(1)
#     cbar.set_label('index')
#     cbar.solids.set(alpha=1)


    histPlot(fig, axis[0,0], overallLosses[:,:,0], xLabel = 'epoch', yLabel = 'Loss', logScale = logScale)
    histPlot(fig, axis[0,1], overallLosses[:,:,1], xLabel = 'epoch', logScale = logScale)
    histPlot(fig, axis[0,2], overallLosses[:,:,2], xLabel = 'epoch', logScale = logScale)
    histPlot(fig, axis[0,3], overallLosses[:,:,3], xLabel = 'epoch', logScale = logScale)
    
    ax1_divider = make_axes_locatable(axis[1,3])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")

    cbar = mpl.colorbar.ColorbarBase(cax1,orientation='vertical', norm=mpl.colors.Normalize(vmin=0, vmax=overallLosses.shape[0]))
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label('epoch')

    im = axis[2,0].imshow(overallLosses[:,:,0].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,0][overallLosses[:,:,0] > 0]), vmax=np.max(overallLosses[:,:,0])))
    axis[2,0].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,0])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 

    im = axis[2,1].imshow(overallLosses[:,:,1].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,1][overallLosses[:,:,1] > 0]), vmax=np.max(overallLosses[:,:,1])))
    axis[2,1].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,1])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 

    im = axis[2,2].imshow(overallLosses[:,:,2].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,2][overallLosses[:,:,2] > 0]), vmax=np.max(overallLosses[:,:,2])))
    axis[2,2].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,2])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 


    im = axis[2,3].imshow(overallLosses[:,:,3].transpose(), norm=LogNorm(vmin=np.min(overallLosses[:,:,3][overallLosses[:,:,3] > 0]), vmax=np.max(overallLosses[:,:,3])))
    axis[2,3].axis('auto')
    ax1_divider = make_axes_locatable(axis[2,3])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='vertical')
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label('loss')

    fig.tight_layout()

    return fig, axis

def findSize(n, maxX = 10):
    combinations = []    
    for i in range(1,min(n + 1, maxX + 1)):
        for j in range(i,min(n + 1, maxX + 1)):
            combinations.append(np.array([i,j]))
    combinations = np.array(combinations)
    combinations = combinations[combinations[:,0] * combinations[:,1] >= n]
    sizes = combinations[:,0] * combinations[:,1]
    return combinations[np.argmin((sizes - n))]

def plotNd(positions, fluidFeatures, label = 'Data'):
    x,y = findSize(fluidFeatures.shape[1], maxX = int(np.ceil(np.sqrt(fluidFeatures.shape[1]))))
    pos_x = positions[:,0].detach().cpu().numpy()
    pos_y = positions[:,1].detach().cpu().numpy()
    # axis[0,0].set_title('prediction')

    fig, axis = plt.subplots(x, y, figsize=(y*3*1.09,x*3), sharex = True, sharey = True, squeeze = False)
    c = 0
    for i in range(0, x):
        for j in range(0, y):

            if c >= fluidFeatures.shape[1]:
                axis[i,j].remove()
            else:
                axis[i,j].set_title('%s[:,%2d]' % (label, c))
                v = fluidFeatures[:,0].detach().cpu().numpy()
                # debugPrint(v)
                predSC = axis[i,j].scatter(pos_x,pos_y,c = fluidFeatures[:,c].detach().cpu().numpy(),s=2)
                axis[i,j].axis('equal')
                axis[i,j].set_xlim(-2.5,2.5)
                axis[i,j].set_ylim(-2.5,2.5)
                axis[i,j].axvline(-2)
                axis[i,j].axvline(2)
                axis[i,j].axhline(-2)
                axis[i,j].axhline(2)
                ax1_divider = make_axes_locatable(axis[i,j])
                cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")

                predCbar = fig.colorbar(predSC, cax=cax1,orientation='vertical')
                predCbar.ax.tick_params(labelsize=8) 


            c = c + 1

    fig.tight_layout()

def findSize(n, maxX = 10):
    combinations = []    
    for i in range(1,min(n + 1, maxX + 1)):
        for j in range(i,min(n + 1, maxX + 1)):
            combinations.append(np.array([i,j]))
    combinations = np.array(combinations)
    combinations = combinations[combinations[:,0] * combinations[:,1] >= n]
    sizes = combinations[:,0] * combinations[:,1]
    return combinations[np.argmin((sizes - n))]

def plotNd(positions, fluidFeatures, label = 'Data'):
    x,y = findSize(fluidFeatures.shape[1], maxX = int(np.ceil(np.sqrt(fluidFeatures.shape[1]))))
    pos_x = positions[:,0].detach().cpu().numpy()
    pos_y = positions[:,1].detach().cpu().numpy()
    # axis[0,0].set_title('prediction')

    fig, axis = plt.subplots(x, y, figsize=(y*3*1.09,x*3), sharex = True, sharey = True, squeeze = False)
    c = 0
    for i in range(0, x):
        for j in range(0, y):

            if c >= fluidFeatures.shape[1]:
                axis[i,j].remove()
            else:
                axis[i,j].set_title('%s[:,%2d]' % (label, c))
                v = fluidFeatures[:,0].detach().cpu().numpy()
                # debugPrint(v)
                predSC = axis[i,j].scatter(pos_x,pos_y,c = fluidFeatures[:,c].detach().cpu().numpy(),s=2)
                axis[i,j].axis('equal')
                axis[i,j].set_xlim(-2.5,2.5)
                axis[i,j].set_ylim(-2.5,2.5)
                axis[i,j].axvline(-2)
                axis[i,j].axvline(2)
                axis[i,j].axhline(-2)
                axis[i,j].axhline(2)
                ax1_divider = make_axes_locatable(axis[i,j])
                cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")

                predCbar = fig.colorbar(predSC, cax=cax1,orientation='vertical')
                predCbar.ax.tick_params(labelsize=8) 


            c = c + 1

    fig.tight_layout()
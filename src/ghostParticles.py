# Helpful statement for debugging, prints the thing entered as x and the output, i.e.,
# debugPrint(1+1) will output '1+1 [int] = 2'
import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import os
import os, sys
# sys.path.append(os.path.join('~/dev/pytorchSPH/', "lib"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import trange, tqdm
import yaml

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter

import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

import torch
# import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

from skimage import measure

from .modules.sdfBoundary import sdPoly, sdPolyDer

def shiftSDF(grid_z, pMin, pMax, dx):
    contours = measure.find_contours(grid_z.mT.numpy(), dx)
    maxLen = 0
    for contour in contours:
        maxLen = max(maxLen, len(contour))
#     debugPrint(maxLen)
    for contour in contours:
#     debugPrint(len(contour))
        if len(contour)  != maxLen: 
            continue
    #     debugPrint(len(contour))
        shiftedMesh = np.copy(contour)

        shiftedMesh[:,0] /= grid_z.shape[1] - 1
        shiftedMesh[:,1] /= grid_z.shape[0] - 1

        shiftedMesh = shiftedMesh * (pMax.numpy() - pMin.numpy()) + pMin.numpy()
    return shiftedMesh

def contourLength(contour):
    cumLength = 0
    for i in range(len(contour) -1):
        c = contour[i]
        n = contour[i+1]
#         debugPrint(c)
#         debugPrint(n)
        d = np.linalg.norm(c-n,axis=0)
#         debugPrint(d)
        cumLength += d
#         break
    
#     debugPrint(contour)
    return cumLength

def adjustSpacing(spacing, cLen):
#     debugPrint(spacing)
#     debugPrint(cLen)
#     debugPrint(cLen/spacing)
    a = cLen / spacing
    f = np.floor(cLen/spacing)
    c = np.ceil(cLen/spacing)
    af = np.abs(a-f)
    ac = np.abs(a-c)
#     debugPrint(af)
#     debugPrint(ac)
    return cLen / f if af < ac else cLen / c

# debugPrint(dx)
# debugPrint(spacing1)

def sampleContour(contour, spacing):
    cumLength = 0
    ptcls = []
    
    # ptcls.append(contour[0])
    offset = spacing
    
    for i in range(len(contour) -1):
        c = contour[i]
        n = contour[i+1]
        d = n - c
        curLength = np.linalg.norm(d,axis=0)
        d = d / curLength
        
        curr = cumLength - offset
#         offset = cumLength % spacing
# How much is left until the new particle
#         debugPrint(cumLength)
#         debugPrint(curLength)
#         debugPrint(spacing)
        while curr + spacing < cumLength + curLength - 1e-5:
            cOffset = curr + spacing - cumLength
#             cOffset = spacing + offset
#             debugPrint(cOffset)
            newP = c + d * cOffset
#             debugPrint(newP)
            ptcls.append(newP)
#             offset += spacing
            curr += spacing
#             debugPrint(offset)
        
        cumLength = cumLength + curLength
        offset = cumLength - curr
#         debugPrint(offset)
#         debugPrint(cumLength)
#         break
#     debugPrint(cumLength)
#     debugPrint(offset)
#     debugPrint(ptcls)
#     debugPrint(spacing)
    return np.array(ptcls)

def samplePolygon(poly, spacing, support, offset = 0, mirrored = False, inverted = False):
    pMin = torch.min(poly, dim = 0)[0] - 2 * support
    pMax = torch.max(poly, dim = 0)[0] + 2 * support
    pMin = pMin.type(torch.float32).to(poly.device)
    pMax = pMax.type(torch.float32).to(poly.device)

    gridSpacing = spacing / 4

    elemsX = torch.ceil((pMax[0] - pMin[0]) / gridSpacing ).to(torch.int32)
    elemsY = torch.ceil((pMax[1] - pMin[1]) / gridSpacing ).to(torch.int32)
    # debugPrint(elemsX)
    # debugPrint(elemsY)

    pMax[0] = pMin[0] + elemsX * gridSpacing
    pMax[1] = pMin[1] + elemsY * gridSpacing

    x = torch.linspace(pMin[0], pMax[0], elemsX)
    y = torch.linspace(pMin[1], pMax[1], elemsY)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    grid_z = grid_x**2 + grid_y**2
    
    grid_z = sdPoly(poly, torch.vstack((grid_x.flatten(), grid_y.flatten())).mT).reshape(grid_x.shape)
    
    sMeshn1 = shiftSDF(grid_z, pMin, pMax, offset)
    
    lenn1 = contourLength(sMeshn1)
    # debugPrint(lenn1)
    spacing1 = adjustSpacing(spacing, lenn1)
    ptcls = sampleContour(sMeshn1, spacing1)
    
    if mirrored and not(offset ==0):
        dist, grad, _, _, _, _ = sdPolyDer(poly, torch.tensor(ptcls).type(torch.float32).to(poly.device))
        o = offset if inverted else - offset
        offsetPtcls = torch.tensor(ptcls).type(torch.float32).to(poly.device) - (dist + offset)[:,None] * grad
#         debugPrint(dist + offset)
#         debugPrint(grad - ptcls)
        return ptcls, offsetPtcls
    if mirrored:
        dist, grad, _, _, _, _ = sdPolyDer(poly, torch.tensor(ptcls).type(torch.float32).to(poly.device))
        o = offset + spacing if inverted else - offset - spacing 
        offsetPtcls = torch.tensor(ptcls).type(torch.float32).to(poly.device) - (dist + o)[:,None] * grad
#         debugPrint(dist + offset)
#         debugPrint(grad - ptcls)
        return ptcls, offsetPtcls
    return ptcls, ptcls
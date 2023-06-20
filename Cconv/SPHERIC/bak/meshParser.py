# required for timing functions
from __future__ import print_function, division
# basic python includes 
import struct
import os
import math
import inspect
import re
import timeit
import time
from contextlib import contextmanager
from functools import partial
# numpy and some basic functions
import numpy as np
from numpy import pi, exp, sqrt
# imports required for 2d and 3d plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from mpl_toolkits.axes_grid1 import make_axes_locatable
#scipy optimization functions
from scipy import optimize
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import SR1
from scipy.integrate import dblquad
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
from numpy import sin, cos, tan, log, arcsin, arccos, sqrt
from bqplot import *
import pandas as pd
from tqdm import trange, tqdm
import random
import warnings
import yaml
# %matplotlib notebook
# warnings.filterwarnings(action='once')
import warnings
warnings.filterwarnings("ignore")

import time
import torch
from torch_geometric.loader import DataLoader
# from tqdm import trange, tqdm
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
import os

import inspect
import re
torch.set_num_threads(1)
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
from scipy.optimize import minimize 

from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter
import matplotlib.patches as patches
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x','--rbf_x', type=str, default='rbf gaussian')
parser.add_argument('-y','--rbf_y', type=str, default='rbf gaussian')
parser.add_argument('-n','--n', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('-t','--targetNeighbors', type=int, default=16)
parser.add_argument('-o','--outFile', type=str, default='out.hdf5')
parser.add_argument('-i','--basePath', type=str, default='/home/winchenbach/Downloads/1')

parser.add_argument('-v','--verbose', type=bool, default=False)
args = parser.parse_args()

basePath = args.basePath
basePath = os.path.expanduser(basePath)
meshFolder = basePath + '/constant/polyMesh/'

print('Mesh %s -> %s with t = %2d, n = %2d, rbf = %s x %s' %(args.basePath, args.outFile, args.targetNeighbors, args.n, args.rbf_x, args.rbf_y))


def parsePoints(meshFolder):
    pointFile = meshFolder + 'points'
    with open(pointFile) as file:
        lines = [line.rstrip() for line in file]
    numPoints = int(lines[19])
    points = [[float(e) for e in l[1:-1].split(' ')] for l in lines[21:21+numPoints]]
    return np.array(points)
def parseOwnerNeighbor(meshFolder, file = 'owner'):
    pointFile = meshFolder + file
    with open(pointFile) as file:
        lines = [line.rstrip() for line in file]
    numPoints = int(lines[20])
    points = [int(l) for l in lines[22:22+numPoints]]
    return np.array(points)
def getFaces(meshFolder):
    pointFile = meshFolder + 'faces'
    with open(pointFile) as file:
        lines = [line.rstrip() for line in file]
    numPoints = int(lines[19])
    points = [(int(l[0]), [int(e) for e in l[2:-1].split(' ')]) for l in lines[21:21+numPoints]]
    return points


points = parsePoints(meshFolder)
owners = parseOwnerNeighbor(meshFolder, file = 'owner')
neighbors = parseOwnerNeighbor(meshFolder, file = 'neighbour')
faces = getFaces(meshFolder)

# print(len(neighbors))

# print(faces[len(neighbors)])
# if args.verbose:
print('Loaded mesh with ', len(points), 'points')

faceIndices = np.arange(len(faces))
owned = faceIndices[owners == 0]

fs = []
for o in owned:
    fs.append(faces[o])

# print(fs)

def getVertices(f):
    numVertices, indices = f
#     print(numVertices, indices)
    vertices = points[indices]
#     print(vertices)
    return vertices

verts = np.vstack([getVertices(f) for f in fs])
if args.verbose:
    print('Number of vertices:', verts.shape)
singleSidedFaces = [f for f in faces if f[0] == 3]
if args.verbose:
    print('Number of single sided faces:', len(singleSidedFaces))
ssf_indices = [f for f in faceIndices if faces[f][0] == 3][::2]
ssf_owners = owners[ssf_indices]
ssf_faces = [faces[i] for i in ssf_indices]
def getTriFace(face):
    numIndices, indices = face
    vertices = points[indices]
    return np.array(vertices)[:,:2]
ssf_tris = np.array([getTriFace(ssf) for ssf in ssf_faces])
# print(ssf_tris.shape)
# print(len(ssf_owners))
# print(len(ssf_faces))

def getCellCenters(baseFolder, timestamp):
    pointFile = baseFolder + '/' + timestamp + '/C'
    with open(pointFile) as file:
        lines = [line.rstrip() for line in file]
    numPoints = int(lines[21])
    velocities = np.array([[float(e) for e in l[1:-1].split(' ')] for l in lines[23:23+numPoints]])[:,:2]
    return velocities
cellCenters = getCellCenters(basePath, '0')
# print(cellCenters.shape)
cmap = plt.colormaps['viridis']

def getVelocities(baseFolder, timestamp):
    pointFile = baseFolder + '/' + timestamp + '/U'
    with open(pointFile) as file:
        lines = [line.rstrip() for line in file]
    numPoints = int(lines[21])
    velocities = np.array([[float(e) for e in l[1:-1].split(' ')] for l in lines[23:23+numPoints]])[:,:2]
    return velocities
U = getVelocities(basePath, '0.01')
Umag = np.linalg.norm(U, axis = 1)
minU = np.min(Umag)
maxU = np.max(Umag)

Umapped = (Umag - minU) / (maxU - minU)
ssfRange = np.arange(len(ssf_tris))
# print(ssfRange)
cellCenters = np.average(ssf_tris, axis = 1)
# print('triangle 0:', ssf_tris[0])
# print('center computed: ', cellCentersC[0])
# print('center parsed: ', cellCenters[0])
f = ssf_tris[0]

cellAreas =       (  ssf_tris[:,0,0] * (ssf_tris[:,1,1] - ssf_tris[:,2,1]) \
                   + ssf_tris[:,1,0] * (ssf_tris[:,2,1] - ssf_tris[:,0,1]) \
                   + ssf_tris[:,2,0] * (ssf_tris[:,0,1] - ssf_tris[:,1,1]))/2

targetNeighbors = args.targetNeighbors
cellSupports = np.sqrt(np.abs(cellAreas) / np.pi * targetNeighbors)

maxSupport = np.max(cellSupports)
distances = np.linalg.norm(ssf_tris - cellCenters[:,None], axis=2)
maxDistance = np.max(distances)

if args.verbose:
    print('maximum cell support:', maxSupport)
if args.verbose:
    print('maximum cell distance:', maxDistance)

def triangleArea(p0,p1,p2):
    return 0.5 *(-p1[1]*p2[0] + p0[1]*(-p1[0] + p2[0]) + p0[0]*(p1[1] - p2[1]) + p1[0]*p2[1])
def pointInTriangle(p,p0,p1,p2):    
    Area = triangleArea(p0,p1,p2)
    s = 1/(2*Area)*(p0[1]*p2[0] - p0[0]*p2[1] + (p2[1] - p0[1])*p[0] + (p0[0] - p2[0])*p[1]);
    t = 1/(2*Area)*(p0[0]*p1[1] - p0[1]*p1[0] + (p0[1] - p1[1])*p[0] + (p1[0] - p0[0])*p[1]);
    u = 1.0 - s - t
    if (s > 0.0) & (t > 0.0) & (u > 0.0):
        return 1
    if (s >= 0.0) & (t >= 0.0) & (u >= 0.0):
        return 0
    return -1
def closestPoint(P, L, clipped = False):
    (A,B) = L
    ap = np.array([P[0] - A[0], P[1] - A[1]])
    ab = np.array([B[0] - A[0], B[1] - A[1]])
    ab2 = np.dot(ab,ab)
    apab = np.dot(ap,ab)
    t = apab / ab2
    if clipped:
        t = np.clip(t, 0, 1)
    return np.array([A[0] + ab[0] * t, A[1] + ab[1] * t])

def closestPointEdge(c, p1, p2, center, check = True):
    dC = (p2[1] - p1[1]) * c[0] - (p2[0] - p1[0]) * c[1] + p2[0] * p1[1] - p2[1] * p1[0]
    dT = (p2[1] - p1[1]) * center[0] - (p2[0] - p1[0]) * center[1] + p2[0] * p1[1] - p2[1] * p1[0]
    if (dC * dT < 0.0) | (not check):
        return closestPoint(c, (p1,p2))
    return c
    
def closestPointTriangle(P, Tri):
#     if (pointInTriangle(P, Tri[0], Tri[1], Tri[2]) >= 0):
    tCenter0 = np.sum(Tri,axis = 0) / (float)(Tri.shape[0])
    P01 = closestPointEdge(P, Tri[0],Tri[1],tCenter0, False)
    P12 = closestPointEdge(P, Tri[1],Tri[2],tCenter0, False)
    P20 = closestPointEdge(P, Tri[2],Tri[0],tCenter0, False)
    d01 = np.linalg.norm(P01 - P)
    d12 = np.linalg.norm(P12 - P)
    d20 = np.linalg.norm(P20 - P)
    if (d01 <= d12) & (d01 <= d20):
        return (P01, d01)
    elif (d12 <= d01) & (d12 <= d20):
        return (P12, d12)
    else:
        return (P20, d20)
def closestPointLine(P, A, B):
    v = B - A
    u = A - P
    vu = np.dot(v,u)
    vv = np.dot(v,v)
    t = -vu / vv
    t = np.clip(t, 0, 1)
    return (1-t) * A + t * B

def closestPointTriangle(P, Tri):
    P01 = closestPointLine(P, Tri[0],Tri[1])
    P12 = closestPointLine(P, Tri[1],Tri[2])
    P20 = closestPointLine(P, Tri[2],Tri[0])
    d01 = np.linalg.norm(P01 - P)
    d12 = np.linalg.norm(P12 - P)
    d20 = np.linalg.norm(P20 - P)
    if (d01 <= d12) & (d01 <= d20):
        return (P01, d01)
    elif (d12 <= d01) & (d12 <= d20):
        return (P12, d12)
    else:
        return (P20, d20)
    

@torch.jit.script
def closestPointLine(P, A, B):
#     print(P,A,B)
    v = B - A
    u = A - P
#     print(v.shape, v)
#     print(u.shape, u)
    vu = torch.einsum('d, nd->n',v,u)
    vv = torch.dot(v,v)
    t = -vu / vv
    t = torch.clip(t, 0, 1)
    return ((1-t) * A[:,None] + t * B[:,None]).mT

@torch.jit.script
def closestPointTriangle(P, Tri):
    P01 = closestPointLine(P, Tri[0],Tri[1])
    P12 = closestPointLine(P, Tri[1],Tri[2])
    P20 = closestPointLine(P, Tri[2],Tri[0])
    d01 = torch.linalg.norm(P01 - P[None,:], dim = 2)
    d12 = torch.linalg.norm(P12 - P[None,:], dim = 2)
    d20 = torch.linalg.norm(P20 - P[None,:], dim = 2)
        
    
    resT = P20
    mask1 = ((d01 <= d12) & (d01 <= d20))[0]
    mask2 = ((d12 <= d01) & (d12 <= d20))[0]
    resT[mask1] = P01[mask1]
    resT[mask2] = P12[mask2]
    
    return torch.min(torch.min(d01,d12),d20), resT
@torch.jit.script
def distanceToTriangle(P,Tri):
    P01 = closestPointLine(P, Tri[0],Tri[1])
    P12 = closestPointLine(P, Tri[1],Tri[2])
    P20 = closestPointLine(P, Tri[2],Tri[0])
    d01 = torch.linalg.norm(P01 - P[None,:], dim = 2)
    d12 = torch.linalg.norm(P12 - P[None,:], dim = 2)
    d20 = torch.linalg.norm(P20 - P[None,:], dim = 2)
    
    return torch.min(torch.min(d01,d12),d20)
    
testPoints = torch.from_numpy(np.random.uniform(size=(64,2))).type(torch.float64)

row, col = radius(torch.from_numpy(cellCenters), torch.from_numpy(cellCenters), maxSupport + maxDistance, max_num_neighbors = 2048)
edge_index = torch.stack([row, col], dim = 0)

if args.verbose:
    print('Overall neighborhood size:', edge_index.shape)
print('Average neighborhood size:', edge_index.shape[1]/ cellCenters.shape[0])

index = 38
neighbors = edge_index[1][edge_index[0] == index]
if args.verbose:
    print('Cell 38 has ', neighbors.shape, 'neighbors before filtering')

neighborTris = ssf_tris[neighbors]
# print(neighborTris.shape)

support = cellSupports[index]
center = torch.from_numpy(cellCenters[index])

distances = np.abs(np.array([distanceToTriangle(center[None,:], torch.from_numpy(t)).item() for t in neighborTris]))

# print(distances)

neighbors = neighbors[distances < support]
neighborTris = ssf_tris[neighbors]
# print(neighborTris.shape)
if args.verbose:
    print('Cell 38 has ', neighbors.shape, 'neighbors after filtering')

center = cellCenters[index]

nmc = 16 * 1024
r = torch.sqrt(torch.rand(size=(nmc,1)).to('cuda').type(torch.float32)) * cellSupports[index]
theta = torch.rand(size=(nmc,1)).to('cuda').type(torch.float32) *2 * np.pi

x = (r * torch.cos(theta)).detach().cpu().numpy() + center[0]
y = (r * torch.sin(theta)).detach().cpu().numpy() + center[1]

mcPoints = np.hstack((x,y))
# print(mcPoints)

# print(neighbors.shape[0])
neighborAreas = cellAreas[neighbors]

i = 0
s = 1 / (2 * neighborAreas[:]) * (neighborTris[:,0,1] * neighborTris[:,2,0] - neighborTris[:,0,0] * neighborTris[:,2,1] +\
                                 (neighborTris[:,2,1] - neighborTris[:,0,1]) * x + \
                                 (neighborTris[:,0,0] - neighborTris[:,2,0]) * y)
t = 1 / (2 * neighborAreas[:]) * (neighborTris[:,0,0] * neighborTris[:,1,1] - neighborTris[:,0,1] * neighborTris[:,1,0] +\
                                 (neighborTris[:,0,1] - neighborTris[:,1,1]) * x + \
                                 (neighborTris[:,1,0] - neighborTris[:,0,0]) * y)
u = 1 - s - t 

mask = np.logical_and(s >= 0, np.logical_and(t>=0, u>=0))
mcMask = np.sum(mask, axis = 1) > 0
# print().shape)

assigned = np.arange(neighbors.shape[0]).reshape((neighbors.shape[0],1)).repeat(nmc,1).T
# print(s.shape)
# print(t.shape)
# print(u.shape)
# print(mask.shape)
# print(assigned.shape)
assigned = assigned[mask]
# print(assigned.shape)
neighborAreas = cellAreas[neighbors]


windowFn = lambda r: torch.clamp(1-r**2, min = 0)
basis = 'rbf gaussian'
n = 8
periodic = False
from cutlass import * 
xrel = torch.from_numpy(x - center[0]).T / cellSupports[index]
yrel = torch.from_numpy(y - center[1]).T / cellSupports[index]
dist = torch.sqrt(xrel**2 + yrel**2)
window = windowFn(dist)[0]
# print(xrel)
# print(yrel)
# print(dist)
# print(window)
u = evalBasisFunction(n, torch.from_numpy(x - center[0]).T / cellSupports[index], which = basis, periodic = False).mT[0]
v = evalBasisFunction(n, torch.from_numpy(y - center[1]).T / cellSupports[index], which = basis, periodic = False).mT[0]
nuv = np.einsum('nu, nv -> nuv', u, v) * window[:,None,None].numpy()
from rbfConv import optimizeWeights2D
# optimizeWeights2D
weight = np.random.normal(size=(n,n))
def optimizedAndPlot(weight_npy, n, basis, periodic):
    weight = torch.from_numpy(weight_npy).type(torch.float32).to('cuda')
    normalizedWeights = (weight - torch.sum(weight) / weight.numel())/torch.std(weight)

    result, constr, fun, init, final = optimizeWeights2D(weights = weight, \
                                            basis = basis,
                                            periodicity = periodic,
                                            nmc = 2**21, targetIntegral = 1, \
                                            windowFn = windowFn, verbose = False)
    # print('Result of Optimization', result)
    print('Constraint Function Values', constr)
    print('Function Value at Optimum', fun)
    # print(init)
    
    
    def convert_polar_xticks_to_radians(ax):
        # Converts x-tick labels from degrees to radians

        # Get the x-tick positions (returns in radians)
        label_positions = ax.get_xticks()

        # Convert to a list since we want to change the type of the elements
        labels = list(label_positions)

        # Format each label (edit this function however you'd like)
        labels = [format_radians_label(label) for label in labels]

        ax.set_xticklabels(labels)
    def makePolar(axis):    
        ss = axis.get_subplotspec()
        axis.remove()
        axis = fig.add_subplot(ss, projection='polar')
        convert_polar_xticks_to_radians(axis)
        axis.set_rmin(0.)
        return axis
    def make3D(axis):    
        ss = axis.get_subplotspec()
        axis.remove()
        axis = fig.add_subplot(ss, projection='3d')
    #     convert_polar_xticks_to_radians(axis)
    #     axis.set_rmin(0.)
        return axis

    def meshify(weight, basis, periodic, nx = 128, ny = 128):
        numWeights = weight.shape[0] * weight.shape[1]    
        x3d = np.linspace(-1,1,nx)
        y3d = np.linspace(-1,1,ny)
        xv, yv = np.meshgrid(x3d, y3d)

        xvt = torch.from_numpy(xv.flatten()).type(torch.float32).to('cuda')
        yvt = torch.from_numpy(yv.flatten()).type(torch.float32).to('cuda')

        # print(xv.shape)

        u = evalBasisFunction(weight.shape[0], xvt,\
                              which = basis[0], periodic = periodic[0]).mT
        v = evalBasisFunction(weight.shape[1], yvt,\
                              which = basis[1], periodic = periodic[1]).mT

        window = weight.new_ones(xvt.shape[0]) if windowFn is None else windowFn(torch.sqrt(xvt**2 + yvt**2))
        # print(window.shape)
        # print('u', u.shape, u)

        nuv = torch.einsum('nu, nv -> nuv', u, v)
    #     nuv[:] = 1
        nuv = nuv * window[:,None, None]

        nuvw = torch.einsum('nuv, uv -> nuv', nuv, weight)

        # print('nuv', nuv.shape, nuv)
        nuvw = torch.sum(nuvw, dim=[1,2])
        # print('nuv', nuv.shape, nuv)

        return xvt.detach().cpu().numpy().reshape(nx,ny), yvt.detach().cpu().numpy().reshape(nx,ny), nuvw.detach().cpu().numpy().reshape(nx,ny)


    nmc = 16*1024
    r = torch.sqrt(torch.rand(size=(nmc,1)).to('cuda').type(torch.float32))
    theta = torch.rand(size=(nmc,1)).to('cuda').type(torch.float32) *2 * np.pi

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    u = evalBasisFunction(n[0], x.T, which = basis[0], periodic = periodic[0]).mT[0]
    v = evalBasisFunction(n[1], y.T, which = basis[1], periodic = periodic[1]).mT[0]


    window = normalizedWeights.new_ones(x.shape[0]) if windowFn is None else windowFn(torch.sqrt(x**2 + y**2))[:,0]


    nuv = torch.einsum('nu, nv -> nuv', u, v)
    nuv = nuv * window[:,None, None]

    nuvw = torch.einsum('nuv, uv -> nuv', nuv, normalizedWeights)
    sumMc = torch.sum(nuvw,dim=[1,2])

    nuvw = torch.einsum('nuv, uv -> nuv', nuv, result)
    resMc = torch.sum(nuvw,dim=[1,2])

    # fig, axis = plt.subplots(3,3, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
    # axis[0,0].scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), c = sumMc.detach().cpu().numpy(),s=0.25)
    # axis[0,0].axis('equal')

    # axis[0,1].scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), c = resMc.detach().cpu().numpy(),s=0.25)
    # axis[0,1].axis('equal')

    # axis[0,2].scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), c = (resMc - sumMc).detach().cpu().numpy(),s=0.25)
    # axis[0,2].axis('equal')

    # axis[1,0].imshow(normalizedWeights.detach().cpu().numpy())
    # axis[1,1].imshow(result.detach().cpu().numpy())
    # axis[1,2].imshow((result-normalizedWeights).detach().cpu().numpy())

    # a = make3D(axis[2,0])
    # x,y,za = meshify(normalizedWeights, basis, periodic)
    # a.plot_surface(x,y,za, cmap = cm.viridis)

    # b = make3D(axis[2,1])
    # x,y,zb = meshify(result, basis, periodic)
    # b.plot_surface(x,y,zb, cmap = cm.viridis)

    # c = make3D(axis[2,2])
    # # x,y,z = meshify(normalizedWeights, basis)
    # c.plot_surface(x,y,zb - za, cmap = cm.viridis)

    # fig.tight_layout()
    return result.detach().cpu().numpy()

# optimized = optimizedAndPlot(weight, basis = [basis,basis], periodic = [periodic,periodic], n = [n,n])

@torch.jit.script
def closestPointLine(P, A, B):
#     print(P,A,B)
    v = B - A
    u = A - P
#     print(v.shape, v)
#     print(u.shape, u)
    vu = torch.einsum('d, nd->n',v,u)
    vv = torch.dot(v,v)
    t = -vu / vv
    t = torch.clip(t, 0, 1)
    return ((1-t) * A[:,None] + t * B[:,None]).mT

@torch.jit.script
def closestPointTriangle(P, Tri):
    P01 = closestPointLine(P, Tri[0],Tri[1])
    P12 = closestPointLine(P, Tri[1],Tri[2])
    P20 = closestPointLine(P, Tri[2],Tri[0])
    d01 = torch.linalg.norm(P01 - P[None,:], dim = 2)
    d12 = torch.linalg.norm(P12 - P[None,:], dim = 2)
    d20 = torch.linalg.norm(P20 - P[None,:], dim = 2)
        
    
    resT = P20
    mask1 = ((d01 <= d12) & (d01 <= d20))[0]
    mask2 = ((d12 <= d01) & (d12 <= d20))[0]
    resT[mask1] = P01[mask1]
    resT[mask2] = P12[mask2]
    
    return torch.min(torch.min(d01,d12),d20), resT
@torch.jit.script
def distanceToTriangle(P,Tri):
    P01 = closestPointLine(P, Tri[0],Tri[1])
    P12 = closestPointLine(P, Tri[1],Tri[2])
    P20 = closestPointLine(P, Tri[2],Tri[0])
    d01 = torch.linalg.norm(P01 - P[None,:], dim = 2)
    d12 = torch.linalg.norm(P12 - P[None,:], dim = 2)
    d20 = torch.linalg.norm(P20 - P[None,:], dim = 2)
    
    return torch.min(torch.min(d01,d12),d20)
    

cellSupports_t = torch.from_numpy(cellSupports).type(torch.float32).to('cpu')
cellCenters_t = torch.from_numpy(cellCenters).type(torch.float32).to('cpu')
cellAreas_t = torch.from_numpy(cellAreas).type(torch.float32).to('cpu')
edge_index_t = edge_index.to('cpu')

mesh_t = torch.from_numpy(ssf_tris).type(torch.float32).to('cpu')


import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

@torch.jit.script
def filterNeighborhood(index, areaTensor, supportTensor, centerTensor, edge_index, meshTensor):
    with record_function("neighborhood filtering"):
        support = supportTensor[index]
        center = centerTensor[index]
        neighbors = edge_index[1][edge_index[0] == index]

        neighborTris = meshTensor[neighbors]

        distances = torch.abs(torch.stack([distanceToTriangle(center[None,:], t)[0,0] for t in neighborTris]))
#         print(distances)

        neighbors = neighbors[distances < support]
    return torch.vstack((neighbors.new_ones(neighbors.shape) * index, neighbors))



@torch.jit.script
def sampleNeighborhoods(mcPoints, mcNUV, neighbors, areaTensor, supportTensor, centerTensor, meshTensor):
    with record_function("neighborhood filtering"):
        support = supportTensor[neighbors[0]]
        center = centerTensor[neighbors[0]]
        neighborTris = meshTensor[neighbors[1]]
        neighborAreas = areaTensor[neighbors[1]]
    with record_function("mc preparation"):
        localPoints = (mcPoints[:,:,None] * support[None,:]).transpose(2,1) + center[None,:,:]
        x = localPoints[:,:,0].transpose(1,0)
        y = localPoints[:,:,1].transpose(1,0)
    with record_function("data allocation"):
        mcRange = torch.arange(mcPoints.shape[0]).type(torch.long).to(mcPoints.device).repeat(neighbors.shape[1],1)
        indexRange = torch.arange(neighbors.shape[1]).type(torch.long).to(mcPoints.device).repeat(mcPoints.shape[0],1).T
        neighborMatrix = x.new_zeros((neighbors.shape[1],mcNUV.shape[1],mcNUV.shape[2]))

    with record_function("pip test"):
        sc = neighborTris[:,0,1] * neighborTris[:,2,0] - neighborTris[:,0,0] * neighborTris[:,2,1]
        sx = (neighborTris[:,2,1] - neighborTris[:,0,1])[:,None] * x
        sy = (neighborTris[:,0,0] - neighborTris[:,2,0])[:,None] * y            
        s = 1 / (2 * neighborAreas[:,None]) * (sc[:,None] + sx + sy)

        tc = neighborTris[:,0,0] * neighborTris[:,1,1] - neighborTris[:,0,1] * neighborTris[:,1,0]
        tx = (neighborTris[:,0,1] - neighborTris[:,1,1])[:,None] * x
        ty = (neighborTris[:,1,0] - neighborTris[:,0,0])[:,None] * y    
        t = 1 / (2 * neighborAreas[:,None]) * (tc[:,None] + tx + ty)
        u = 1 - s - t 

        mask = torch.logical_and(s >= 0, torch.logical_and(t>=0, u>=0))
    with record_function("masking"):
        mcIndices = mcRange[mask]
        neighIndices = indexRange[mask]
    with record_function("scatter results"):
        scattered = scatter(mcNUV[mcIndices], neighIndices, dim=0, dim_size = neighbors.shape[1],reduce='add')
    return scattered

@torch.jit.script
def kernel(q: torch.Tensor):
    C = 7 / np.pi    
    b1 = torch.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C
@torch.jit.script
def kernelGrad(q: torch.Tensor, r : torch.Tensor):
    C = 7 / np.pi    
    return - r * C * (20. * q * (1. -q)**3)[:,None]
    

@torch.jit.script
def gaussianIntegral(mcPoints, mcNUV, neighbors, areaTensor, supportTensor, centerTensor, meshTensor):
    with record_function("neighborhood filtering"):
        support = supportTensor[neighbors[0]]
        center = centerTensor[neighbors[0]]
        neighborTris = meshTensor[neighbors[1]]
        neighborAreas = areaTensor[neighbors[1]]
    with record_function("mc preparation"):
        localPoints = (mcPoints[:,:,None] * support[None,:]).transpose(2,1) + center[None,:,:]
        x = localPoints[:,:,0].transpose(1,0)
        y = localPoints[:,:,1].transpose(1,0)
    with record_function("data allocation"):
        mcRange = torch.arange(mcPoints.shape[0]).type(torch.long).to(mcPoints.device).repeat(neighbors.shape[1],1)
        indexRange = torch.arange(neighbors.shape[1]).type(torch.long).to(mcPoints.device).repeat(mcPoints.shape[0],1).T
        neighborMatrix = x.new_zeros((neighbors.shape[1],mcNUV.shape[1],mcNUV.shape[2]))

    with record_function("pip test"):
        sc = neighborTris[:,0,1] * neighborTris[:,2,0] - neighborTris[:,0,0] * neighborTris[:,2,1]
        sx = (neighborTris[:,2,1] - neighborTris[:,0,1])[:,None] * x
        sy = (neighborTris[:,0,0] - neighborTris[:,2,0])[:,None] * y            
        s = 1 / (2 * neighborAreas[:,None]) * (sc[:,None] + sx + sy)

        tc = neighborTris[:,0,0] * neighborTris[:,1,1] - neighborTris[:,0,1] * neighborTris[:,1,0]
        tx = (neighborTris[:,0,1] - neighborTris[:,1,1])[:,None] * x
        ty = (neighborTris[:,1,0] - neighborTris[:,0,0])[:,None] * y    
        t = 1 / (2 * neighborAreas[:,None]) * (tc[:,None] + tx + ty)
        u = 1 - s - t 

        mask = torch.logical_and(s >= 0, torch.logical_and(t>=0, u>=0))
        
    
    filtered = mcPoints[torch.sum(mask,dim=0) > 0]
    
    q = torch.linalg.norm(filtered,dim=1)
#     print(torch.min(q), torch.mean(q), torch.max(q))
    
    r = torch.clone(filtered)
    r[q > 1e-5,:] = r[q > 1e-5,:] / q[q > 1e-5,None]
    
    k = torch.sum(kernel(q)) * np.pi / mcPoints.shape[0]
    grad = torch.sum(kernelGrad(q,r),dim=0) * np.pi / mcPoints.shape[0]
#     print(k, grad)
    return k, grad
    
    
    print(filtered.shape)
        
    return
    with record_function("masking"):
        mcIndices = mcRange[mask]
        neighIndices = indexRange[mask]
    with record_function("scatter results"):
        scattered = scatter(mcNUV[mcIndices], neighIndices, dim=0, dim_size = neighbors.shape[1],reduce='add')
    return scattered

@torch.jit.script
def sampleNeighborhood(index, mcPoints, mcNUV, neighbors, areaTensor, supportTensor, centerTensor, meshTensor):
    with record_function("neighborhood filtering"):
        support = supportTensor[index]
        center = centerTensor[index]
        neighborTris = meshTensor[neighbors]
        neighborAreas = areaTensor[neighbors]
        

    with record_function("mc preparation"):
        localPoints = torch.clone(mcPoints)
        localPoints = localPoints * support + center
        x = localPoints[:,0][:,None]
        y = localPoints[:,1][:,None]
    



    mcRange = torch.arange(mcPoints.shape[0]).type(torch.int32).to(mcPoints.device)
    neighborMatrix = x.new_zeros((len(neighbors),mcNUV.shape[1],mcNUV.shape[2]))
    
    with record_function("neighbor loop"):
        for i, neigh in enumerate(neighbors):            
            with record_function("loop iteration - pip"):
                tri = neighborTris[i]
                s = 1 / (2 * neighborAreas[i]) * (tri[0,1] * tri[2,0] - tri[0,0] * tri[2,1] +\
                                                 (tri[2,1] - tri[0,1]) * x + \
                                                 (tri[0,0] - tri[2,0]) * y)
                t = 1 / (2 * neighborAreas[i]) * (tri[0,0] * tri[1,1] - tri[0,1] * tri[1,0] +\
                                                 (tri[0,1] - tri[1,1]) * x + \
                                                 (tri[1,0] - tri[0,0]) * y)
                u = 1 - s - t 

                mask = torch.logical_and(s >= 0, torch.logical_and(t>=0, u>=0))[:,0]
#                 print(torch.sum(mask))
        #         print(mask.shape)
            with record_function("loop iteration - computing"):

                indices = mcRange[mask]
                nuvs = torch.index_select(mcNUV, 0, indices)
                summed = torch.sum(nuvs, dim = 0)
                neighborMatrix[i,:,:] = summed# * np.pi / mcPoints.shape[0]
        
    return neighborMatrix

windowFn = lambda r: torch.clamp(1-r**2, min = 0)
n = [args.n,args.n]
basis = [args.rbf_x, args.rbf_y]
periodic = [False, False]


weight = np.random.normal(size=(n[0],n[1]))

optimized = optimizedAndPlot(weight, basis = basis, periodic = periodic, n = n)

nmc = 2**19

r = torch.sqrt(torch.rand(size=(nmc,1)).to('cpu').type(torch.float32))
theta = torch.rand(size=(nmc,1)).to('cpu').type(torch.float32) *2 * np.pi

x = (r * torch.cos(theta))
y = (r * torch.sin(theta))

mcPoints = torch.hstack((x,y))


xrel = (x).T
yrel = (y).T
dist = torch.sqrt(xrel**2 + yrel**2)
window = weights.new_ones(x.shape[0]) if windowFn is None else windowFn(dist)[0]

u = evalBasisFunction(n[0], (x).T, which = basis[0], periodic = periodic[0]).mT[0]
v = evalBasisFunction(n[1], (y).T, which = basis[1], periodic = periodic[1]).mT[0]

nuv = torch.einsum('nu, nv -> nuv', u, v) * window[:,None,None]
mcNuv = nuv * np.pi / mcPoints.shape[0]


index = 1000
neighs = filterNeighborhood(index, cellAreas_t, cellSupports_t, cellCenters_t, edge_index_t, mesh_t)
mat = sampleNeighborhood(index, mcPoints, mcNuv, neighs[1], cellAreas_t, cellSupports_t, cellCenters_t, mesh_t)
print('Sanity check of integrals, should be 1: ', torch.einsum('nuv, uv-> ...', mat, torch.from_numpy(optimized).to('cpu').type(torch.float32)))


mcPoints_gpu = mcPoints.to('cuda')
mcNuv_gpu = mcNuv.to('cuda')
cellAreas_gpu = cellAreas_t.to('cuda')
cellSupports_gpu = cellSupports_t.to('cuda')
cellCenters_gpu = cellCenters_t.to('cuda')
mesh_gpu = mesh_t.to('cuda')
edge_index_gpu = edge_index_t.to('cuda')


cellIndices = np.arange(cellAreas_gpu.shape[0])
neighborsBatched = []
print('Filtering Neighborhoods')
print('Neighborhood shape before filtering: ', edge_index_t.shape)
for j in tqdm(cellIndices):
    neighs = filterNeighborhood(j, cellAreas_gpu, cellSupports_gpu, cellCenters_gpu, edge_index_gpu, mesh_gpu).cpu()
    neighborsBatched.append(neighs)
neighborhoods = torch.hstack(neighborsBatched)
print('Neighborhood shape after filtering: ', neighborhoods.shape)
# del edge_index_gpu

mcPoints_gpu = mcPoints.to('cuda')
mcNuv_gpu = mcNuv.to('cuda')
cellAreas_gpu = cellAreas_t.to('cuda')
cellSupports_gpu = cellSupports_t.to('cuda')
cellCenters_gpu = cellCenters_t.to('cuda')
mesh_gpu = mesh_t.to('cuda')

batched = torch.split(neighborhoods, 128,dim=1)

print('Sampling Neighborhoods')
matricesBatched = []
for batch in tqdm(batched):
    matrix = sampleNeighborhoods(mcPoints_gpu, mcNuv_gpu, batch.to('cuda'), cellAreas_gpu, cellSupports_gpu, cellCenters_gpu, mesh_gpu).cpu()
    matricesBatched.append(matrix)
matrices = torch.vstack(matricesBatched)

overall = torch.einsum('nuv, uv-> n', matrices, torch.from_numpy(optimized).to('cpu').type(torch.float32))
summed = scatter(overall, neighborhoods[0], dim=0, dim_size = cellIndices.shape[0], reduce='add')
# print(summed)

kernels = []
gradients = []
neighbors = []

print('Computing Boundary Integrals')
for index in tqdm(range(cellAreas_t.shape[0])):
    neighs = torch.vstack((neighborhoods[0][neighborhoods[0] == index], neighborhoods[1][neighborhoods[0] == index]))
    k, grad = gaussianIntegral(mcPoints_gpu, mcNuv_gpu, neighs.to('cuda'), cellAreas_gpu, cellSupports_gpu, cellCenters_gpu, mesh_gpu)
    
    kernels.append(k.cpu().item())
    gradients.append(grad.cpu().numpy())
    neighbors.append(neighs.shape[1])
    
kernels = np.array(kernels)
gradients = np.array(gradients)
neighbors = np.array(neighbors)
    

matrices.shape
neighborhoods.shape
import h5py
# configFiles = [basePath + '/' + f for f in os.listdir(basePath) if os.path.isdir(basePath + '/' + f) and  f[0].isdigit()]

# times = []
# fileNames = np.array([float(c.split('/')[-1]) for c in configFiles])
# indices = np.argsort(fileNames)

# configFiles = np.array(configFiles)[indices].tolist()

# for i, c in enumerate(configFiles):
#     f = c.split('/')[-1]
#     if f[0].isdigit() and f != "0":
#         times.append((f, c))
        
fileName = args.outFile
# print(fileName)
# outputFile.close()
outputFile = h5py.File(fileName,'w')

grp = outputFile.create_group('meshAttributes')


grp.create_dataset('areas', data=cellAreas_gpu.detach().cpu().numpy())
grp.create_dataset('supports', data=cellSupports_gpu.detach().cpu().numpy())
grp.create_dataset('centers', data=cellCenters_gpu.detach().cpu().numpy())
grp.create_dataset('vertices', data=mesh_gpu.detach().cpu().numpy())

# print(grp['vertices'].shape)

grp = outputFile.create_group('networkAttributes')

grp.create_dataset('neighbors', data=neighborhoods.cpu().numpy())
grp.create_dataset('filterMatrices', data=matrices.numpy())
grp.create_dataset('integralValues', data=summed.numpy())

grp = outputFile.create_group('integralAttributes')

grp.create_dataset('neighborCount', data=neighbors)
grp.create_dataset('splineKernel', data=kernels)
grp.create_dataset('splineGradient', data=gradients)


outputFile.attrs['n'] = n
outputFile.attrs['basis'] = basis
outputFile.attrs['periodic'] = periodic
outputFile.attrs['baseWeight'] = optimized



# mcPoints_gpu = mcPoints.to('cuda')
# mcNuv_gpu = mcNuv.to('cuda')
# cellAreas_gpu = cellAreas_t.to('cuda')
# cellSupports_gpu = cellSupports_t.to('cuda')
# cellCenters_gpu = cellCenters_t.to('cuda')
# mesh_gpu = mesh_t.to('cuda')
# edge_index_gpu = edge_index_t.to('cuda')


# ti = 0
# for t, f in tqdm(times):
#     dataFiles = [(d, f + '/' + d) for d in os.listdir(f) if not os.path.isdir(f + '/' + d) and not d.endswith('_0')]

#     g = '%05d' % ti
#     grp = outputFile.create_group(g)

#     for p, filename in dataFiles:
# #         print(p, filename)

#         with open(filename) as file:
#             lines = [line.rstrip() for line in file]
#         for il, l in enumerate(lines):
#             if len(l)> 0 and l[0].isdigit():
#                 cells = int(l)
#                 break
# #         print(cells)
#         firstData = lines[il + 2]
#         if firstData[0] == '(':
#             filteredLines = [l[1:-1].split(' ') for l in lines[il+2:il+2+cells]]
#             data = np.array(filteredLines).astype(float)
#         else:
#             data = np.array([float(l) for l in lines[il+2:il+2+cells]])
# #         print(data.shape)
#         ds = grp.create_dataset(filename.split('/')[-1], data = data)
    
# #     break
#     ti = ti + 1
outputFile.close()

import torch

from torch_geometric.nn import radius
from torch_scatter import scatter

from .kernel import *
import torch

from torch_geometric.nn import radius
from torch_scatter import scatter

from .kernel import *

@torch.jit.script
def sdPoly(poly, p):
    N = len(poly)
    
    i = torch.arange(N, device = p.device, dtype = torch.int64)
    i2 = (i + 1) % N
    e = poly[i2] - poly[i]
    v = p - poly[i][:,None]
    
    ve = torch.einsum('npd, nd -> np', v, e)
    ee = torch.einsum('nd, nd -> n', e, e)
    
    pq = v - e[:,None] * torch.clamp(ve / ee[:,None], min = 0, max = 1)[:,:,None]
    
    d = torch.einsum('npd, npd -> np', pq, pq)
    d = torch.min(d, dim = 0).values
    
    wn = torch.zeros((N, p.shape[0]), device = p.device, dtype = torch.int64)
    
    cond1 = 0 <= v[i,:,1]
    cond2 = 0 >  v[i2,:,1]
    val3 = e[i,0,None] * v[i,:,1] - e[i,1,None] * v[i,:,0]
    
    c1c2 = torch.logical_and(cond1, cond2)
    nc1nc2 = torch.logical_and(torch.logical_not(cond1), torch.logical_not(cond2))
    
    wn[torch.logical_and(c1c2, val3 > 0)] += 1
    wn[torch.logical_and(nc1nc2, val3 < 0)] -= 1
    
    wn = torch.sum(wn,dim=0)
    s = torch.ones(p.shape[0], device = p.device, dtype = p.dtype)
    s[wn != 0] = -1
    
    return s * torch.sqrt(d)

@torch.jit.script
def boundaryKernelAnalytic(d : torch.Tensor , q : torch.Tensor):
    a = torch.zeros(d.shape, device = q.device, dtype=d.dtype)
    b = torch.zeros(d.shape, device = q.device, dtype=d.dtype)
    
    mask = torch.abs(d.real) > 1e-3
    dno = d[mask]
    
    
    a[mask] += ( 12 * dno**5 + 80 * dno**3) * torch.log(torch.sqrt(1 - dno**2) + 1)
    a[mask] += (-12 * dno**5 - 80 * dno**3) * torch.log(1 - torch.sqrt(1 - dno**2))
    a[mask] += (-12 * dno**5 - 80 * dno**3) * torch.log(torch.sqrt(1 - 4 * dno**2) + 1)
    a[mask] += ( 12 * dno**5 + 80 * dno**3) * torch.log(1 - torch.sqrt(1 - 4 * dno**2))
    a += -13 * torch.acos(2 * d)
    a +=  16 * torch.acos(d)
    a += torch.sqrt(1 - 4 * d**2) * (74 * d**3 + 49 * d)
    a += torch.sqrt(1 - d **2) * (-136 * d**3 - 64 * d)
        
        
    b += -36 * d**5 * torch.log(torch.sqrt(1-4 * d**2) + 1)
    b[mask] += 36 * dno**5 * torch.log(1-torch.sqrt(1-4*dno**2))
    b += 11 * torch.acos(2 * d)
    b += -36 * torch.log(-1 + 0j) * d**5
    b += -160j * d**4
    b += torch.sqrt(1 -4 *d**2)*(62 *d**3 - 33*d)
    b += 80j * d**2
    res = (a + b) / (14 * np.pi)

    gammaScale = 2.0

    gamma = 1 + (1 - q / 2) ** gammaScale
    # gamma = torch.log( 1 + torch.exp(gammaScale * q)) - np.log(1 + np.exp(-gammaScale) / np.log(2))

    return res.real * gamma

@torch.jit.script
def sdPolyDer(poly, p, dh :float = 1e-4, inverted :bool = False):
    dh = 1e-4
    dpx = torch.zeros_like(p)
    dnx = torch.zeros_like(p)
    dpy = torch.zeros_like(p)
    dny = torch.zeros_like(p)
    
    dpx[:,0] += dh
    dnx[:,0] -= dh
    dpy[:,1] += dh
    dny[:,1] -= dh
    
    c = sdPoly(poly, p)
    cpx = sdPoly(poly, p + dpx)
    cnx = sdPoly(poly, p + dnx)
    cpy = sdPoly(poly, p + dpy)
    cny = sdPoly(poly, p + dny)

    if inverted:
        c = -c
        cpx = -cpx
        cnx = -cnx
        cpy = -cpy
        cny = -cny
        
    grad = torch.zeros_like(p)
    grad[:,0] = (cpx - cnx) / (2 * dh)
    grad[:,1] = (cpy - cny) / (2 * dh)
    
    gradLen = torch.linalg.norm(grad, dim =1)
    grad[torch.abs(gradLen) > 1e-5] /= gradLen[torch.abs(gradLen)>1e-5,None]
    
    return c, grad, cpx, cnx, cpy, cny

@torch.jit.script
def boundaryIntegralAndDer(poly, p, support : float, c, cpx, cnx, cpy, cny, dh : float = 1e-4):
    k = boundaryKernelAnalytic(torch.clamp(c / support, min = -1, max = 1).type(torch.complex64), c / support)   
    kpx = boundaryKernelAnalytic(torch.clamp(cpx / support, min = -1, max = 1).type(torch.complex64), c / support)
    knx = boundaryKernelAnalytic(torch.clamp(cnx / support, min = -1, max = 1).type(torch.complex64), c / support)  
    kpy = boundaryKernelAnalytic(torch.clamp(cpy / support, min = -1, max = 1).type(torch.complex64), c / support)  
    kny = boundaryKernelAnalytic(torch.clamp(cny / support, min = -1, max = 1).type(torch.complex64), c / support)   
        
    kgrad = torch.zeros_like(p)
    kgrad[:,0] = (kpx - knx) / (2 * dh)
    kgrad[:,1] = (kpy - kny) / (2 * dh)
    
    return k, kgrad
    
@torch.jit.script
def sdPolyDerAndIntegral(poly, p, support : float, masked : bool = False, inverted : bool = False):     
    c, grad, cpx, cnx, cpy, cny = sdPolyDer(poly, p, dh = 1e-4, inverted = inverted)
    k, kgrad = boundaryIntegralAndDer(poly, p, support, c, cpx, cnx, cpy, cny, dh = 1e-4)  
    
    
    return c, grad, k, kgrad

from torch.profiler import record_function

def boundaryNeighborSearch(config, simulationState):
    if 'solidBoundary' not in config:
        return None, None, None, None, None, None, None
    with record_function('solidBC - neighborhood'):
        particleIndices = torch.arange(simulationState['numParticles'], device = config['device'], dtype = torch.int64 )

        sdfDistances = []
        sdfGradients = []
        sdfRows = []
        sdfCols = []
        sdfIntegrals = []
        sdfIntegralDerivatives = []
        sdfFluidNeighbors = []
        sdfFluidNeighborsRows = []
        sdfFluidNeighborsCols = []
        sdfNeighbors = []


        for ib, b in enumerate(config['solidBoundary']):
            polyDist, polyDer, bIntegral, bGrad = sdPolyDerAndIntegral(b['polygon'], simulationState['fluidPosition'], config['support'], inverted = b['inverted'])
            
            adjacent = polyDist <= config['support']
            polyDer = polyDer / torch.linalg.norm(polyDer,axis=1)[:,None]
            polyDist = polyDist / config['support']
            if polyDer[adjacent].shape[0] == 0:
                continue

            i = particleIndices[adjacent]
            j = torch.ones(i.shape, device = config['device'], dtype = torch.int64) *ib
            
            pb = simulationState['fluidPosition'][adjacent] - polyDist[adjacent, None] * polyDer[adjacent,:] * config['support']
    #         print(polyDist[adjacent])
    #         print(polyDer[adjacent])
    #         print(pb)
            
            row, col = radius(pb, simulationState['fluidPosition'], config['support'], max_num_neighbors = config['max_neighbors'])

            # for ib, b in enumerate(config['solidBoundary']):
            #     i = col
            #     j = row
                
            #     polyDist, polyDer, polyDerLen = sdPolyDer(b['polygon'], pb, config, inverted = b['inverted'])
            #     cp = pb - polyDist[:,None] * polyDer
            #     d = torch.einsum('nd,nd->n', polyDer, cp)
            #     neighDistances = torch.einsum('nd,nd->n', simulationState['fluidPosition'][j], polyDer[i]) - d[i]
                
            #     col = col[neighDistances >= 0]
            #     row = row[neighDistances >= 0]
                
                # fluidNeighbors = torch.vstack((j,i))


    #         print('pb', pb.shape, pb)
    #         print('row', row.shape, row)
    #         print('col', col.shape, col)
            
            
    #         print(row,col)
            
            pbNeighbors = torch.stack([row, col], dim = 0)
    #         print(pbNeighbors)
            

            sdfDistances.append(polyDist[adjacent])
            sdfGradients.append(polyDer[adjacent])
            sdfIntegrals.append(bIntegral[adjacent])
            sdfIntegralDerivatives.append(bGrad[adjacent])
            # sdfFluidNeighbors.append(pbNeighbors)
            sdfRows.append(i)
            sdfCols.append(j)
            sdfFluidNeighborsRows.append(row)
            sdfFluidNeighborsCols.append(col)
        if len(sdfRows) > 0:
            boundaryDistances = torch.cat(sdfDistances)
            boundaryGradients = torch.cat(sdfGradients)
            sdfRows = torch.cat(sdfRows)
            sdfCols = torch.cat(sdfCols)
            boundaryNeighbors = torch.stack((sdfRows, sdfCols))
            sdfFluidNeighborsRows = torch.cat(sdfFluidNeighborsRows)
            sdfFluidNeighborsCols = torch.cat(sdfFluidNeighborsCols)
            boundaryFluidNeighbors = torch.stack((sdfFluidNeighborsRows, sdfFluidNeighborsCols))
            boundaryIntegrals = torch.cat(sdfIntegrals)
            boundaryIntegralGradients = torch.cat(sdfIntegralDerivatives)
            # boundaryFluidNeighbors = torch.cat(sdfFluidNeighbors)
            del particleIndices
            
            neighbors = boundaryNeighbors
            i = neighbors[0]
            b = neighbors[1]
            
            pb = simulationState['fluidPosition'][i] - boundaryDistances[:, None] * boundaryGradients * config['support']
            
    #         print('i', i.shape, i)
    #         print('b', b.shape, b)
    #         print('pb', pb.shape, pb)
            
            bi, bb  = boundaryFluidNeighbors
    #         print('bi', bi.shape, bi)
    #         print('bb', bb.shape, bb)
            
            distances = torch.linalg.norm(simulationState['fluidPosition'][bi] - pb[bb], axis = 1) / config['support']
    #         print('distances', distances.shape, distances)
    #         print('kernel', wendland(distances, config['support']))
            fac = simulationState['fluidArea'][bi] * wendland(distances, config['support'])
            
    #         print('fac', fac.shape, fac)
    #         print('distances', distances.shape, distances)
            
            d_sum = scatter(fac, bb, dim = 0, dim_size = pb.shape[0])
    #         print('d_bar', d_bar.shape, d_bar)
            d_bar = scatter(fac[:,None] * simulationState['fluidPosition'][bi], bb, dim = 0, dim_size = pb.shape[0])
    #         print('d_sum', d_sum.shape, d_sum)
    #         print('d_sum', d_sum.shape, d_sum)
    #         print('d_bar', d_bar.shape, d_bar)
            d_bar[d_sum > 0,:] /= d_sum[d_sum > 0,None]
        
        
    #         print('bi', bi.shape, bi)
    #         print('bb', bb.shape, bb)
    #         print('distances', distances.shape, distances)
    #         print('d_bar', d_bar.shape, d_bar)
    #         print('pb', pb.shape, pb)
            x_b = pb - d_bar
    #         print('x_b', x_b.shape, x_b)
            
    #         x_b *= simulationState['fluidRestDensity']
            
    #         row, col = radius(x_b, simulationState['fluidPosition'], config['support'], max_num_neighbors = config['max_neighbors'])
    #         boundaryFluidNeighbors = torch.stack([row, col], dim = 0)
    #         print('row', row)
    #         print('col', col)
            
    #         print(pbNeighbors)
            
            return boundaryNeighbors, boundaryDistances, boundaryGradients, boundaryIntegrals, boundaryIntegralGradients, boundaryFluidNeighbors, d_bar
        else:
            return None, None, None, None, None, None, None


# nx = 64
# ny = 64
# x = np.linspace(config['domain']['min'][0],config['domain']['max'][0],nx)
# y = np.linspace(config['domain']['min'][1],config['domain']['max'][1],ny)

# xx, yy = np.meshgrid(x, y)

# xf = xx.flatten()
# yf = yy.flatten()

# gridPositions = torch.from_numpy(np.c_[xf, yf]).type(config['precision']).to(config['device'])


# res = np.zeros_like(xf)
# resx = np.zeros_like(xf)
# resy = np.zeros_like(yf)

# for b in config['solidBoundary']:
#     polygon = b['polygon']
#     sdf, sdfgrad, b, bgrad = sdPolyDerAndIntegral(polygon, gridPositions, config)
#     print(sdf[torch.isnan(sdfgrad)[:,0]])
#     sdf = torch.clamp(sdf / config['support'], min = -1, max = 1)
    
#     res += sdf.detach().cpu().numpy()
#     resx += sdfgrad[:,0].detach().cpu().numpy()
#     resy += sdfgrad[:,1].detach().cpu().numpy()
    


# fig, axis = plt.subplots(1,3, figsize=(9,3), sharex = True, sharey = True, squeeze = False)

# im = axis[0,0].imshow(res.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[0,0], config)
# plotDomain(axis[0,0], config)
# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[0,1].imshow(resx.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[0,1], config)
# plotDomain(axis[0,1], config)
# ax1_divider = make_axes_locatable(axis[0,1])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[0,2].imshow(resy.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[0,2], config)
# plotDomain(axis[0,2], config)
# ax1_divider = make_axes_locatable(axis[0,2])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# fig.tight_layout()

# for boundary in config['solidBoundary']:
#     boundary['polygon'] = torch.tensor(boundary['vertices'], device = config['device'], dtype = config['precision'])
#     vertices = boundary['vertices']
#     polygon = torch.tensor(boundary['vertices'], device = config['device'], dtype = config['precision'])

# nx = 64
# ny = 64
# x = np.linspace(config['domain']['min'][0],config['domain']['max'][0],nx)
# y = np.linspace(config['domain']['min'][1],config['domain']['max'][1],ny)

# xx, yy = np.meshgrid(x, y)

# xf = xx.flatten()
# yf = yy.flatten()

# gridPositions = torch.from_numpy(np.c_[xf, yf]).type(config['precision']).to(config['device'])

# sd = sdPoly(polygon, gridPositions, config)
# fig, axis = plt.subplots(1,1, figsize=(8,8), sharex = True, sharey = True, squeeze = False)

# im = axis[0,0].imshow(sd.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# plotDomain(axis[0,0], config)

# plotBoundary(axis[0,0], config)


# nx = 64
# ny = 64
# x = np.linspace(config['domain']['min'][0],config['domain']['max'][0],nx)
# y = np.linspace(config['domain']['min'][1],config['domain']['max'][1],ny)

# xx, yy = np.meshgrid(x, y)

# xf = xx.flatten()
# yf = yy.flatten()

# gridPositions = torch.from_numpy(np.c_[xf, yf]).type(config['precision']).to(config['device'])

# res = np.zeros_like(xf)
# resx = np.zeros_like(xf)
# resy = np.zeros_like(yf)

# for b in config['solidBoundary']:
#     polygon = b['polygon']
#     sdf, sdfgrad, b, bgrad = sdPolyDerAndIntegral(polygon, gridPositions, config)
#     print(sdf[torch.isnan(sdfgrad)[:,0]])
#     sdf = torch.clamp(sdf / config['support'], min = -1, max = 1)
    
#     res += sdf.detach().cpu().numpy()
#     resx += sdfgrad[:,0].detach().cpu().numpy()
#     resy += sdfgrad[:,1].detach().cpu().numpy()
    

# resAutograd = np.zeros_like(xf)
# resAutogradx = np.zeros_like(xf)
# resAutogrady = np.zeros_like(yf)

# for b in config['solidBoundary']:
#     polygon = b['polygon']
#     sdf, sdfgrad, b, bgrad = sdPolyDerAndIntegralAutograd(polygon, gridPositions, config)
#     print(sdf[torch.isnan(sdfgrad)[:,0]])
#     sdf = torch.clamp(sdf / config['support'], min = -1, max = 1)
    
#     sdfgrad /= torch.linalg.norm(sdfgrad, axis = 1)[:,None]
    
#     resAutograd += sdf.detach().cpu().numpy()
#     resAutogradx += sdfgrad[:,0].detach().cpu().numpy()
#     resAutogrady += sdfgrad[:,1].detach().cpu().numpy()
    


# fig, axis = plt.subplots(3,3, figsize=(9,9), sharex = True, sharey = True, squeeze = False)


# im = axis[0,0].imshow(res.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[0,0], config)
# plotDomain(axis[0,0], config)
# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[0,1].imshow(resx.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[0,1], config)
# plotDomain(axis[0,1], config)
# ax1_divider = make_axes_locatable(axis[0,1])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[0,2].imshow(resy.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[0,2], config)
# plotDomain(axis[0,2], config)
# ax1_divider = make_axes_locatable(axis[0,2])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[1,0].imshow(resAutograd.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[1,0], config)
# plotDomain(axis[1,0], config)
# ax1_divider = make_axes_locatable(axis[1,0])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[1,1].imshow(resAutogradx.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[1,2].imshow(resAutogrady.reshape(nx,ny), extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[1,2], config)
# plotDomain(axis[1,2], config)
# ax1_divider = make_axes_locatable(axis[1,2])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[2,0].imshow((resAutograd.reshape(nx,ny) - res.reshape(nx,ny))**2, extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[2,0], config)
# plotDomain(axis[2,0], config)
# ax1_divider = make_axes_locatable(axis[2,0])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[2,1].imshow((resAutogradx.reshape(nx,ny) - resx.reshape(nx,ny))**2, extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[2,1], config)
# plotDomain(axis[2,1], config)
# ax1_divider = make_axes_locatable(axis[2,1])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# im = axis[2,2].imshow((resAutogrady.reshape(nx,ny) - resy.reshape(nx,ny))**2, extent=(config['domain']['min'][0],config['domain']['max'][0],config['domain']['min'][1],config['domain']['max'][1]))

# plotBoundary(axis[2,2], config)
# plotDomain(axis[2,2], config)
# ax1_divider = make_axes_locatable(axis[2,2])
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
# cb1.ax.tick_params(labelsize=8) 

# fig.tight_layout()

def addBoundaryBoundaries(config):
    if config['domain']['periodicX'] and config['domain']['periodicY']:
        return
    if 'solidBoundary' not in config:
        config['solidBoundary'] = []
    if config['domain']['periodicX'] and not config['domain']['periodicY']:
        minDomain = config['domain']['min']
        maxDomain = config['domain']['max']
        buffer = config['support'] * config['domain']['buffer']
        
        config['solidBoundary'].append({
            'vertices':[
                [minDomain[0],minDomain[1]],
                [maxDomain[0],minDomain[1]],
                [maxDomain[0],minDomain[1] + buffer],
                [minDomain[0],minDomain[1] + buffer]
            ],
            'inverted':False
        })
        config['solidBoundary'].append({
            'vertices':[
                [minDomain[0],maxDomain[1] - buffer],
                [maxDomain[0],maxDomain[1] - buffer],
                [maxDomain[0],maxDomain[1]],
                [minDomain[0],maxDomain[1]]
            ],
            'inverted':False
        })
    if not config['domain']['periodicX'] and config['domain']['periodicY']:
        minDomain = config['domain']['min']
        maxDomain = config['domain']['max']
        buffer = config['support'] * config['domain']['buffer']
        
        config['solidBoundary'].append({
            'vertices':[
                [minDomain[0]         , minDomain[1]],
                [minDomain[0] + buffer, minDomain[1]],
                [minDomain[0] + buffer, maxDomain[1]],
                [minDomain[0]         , maxDomain[1]]
            ],
            'inverted':False
        })
        config['solidBoundary'].append({
            'vertices':[
                [maxDomain[0] - buffer, minDomain[1]],
                [maxDomain[0]         , minDomain[1]],
                [maxDomain[0]         , maxDomain[1]],
                [maxDomain[0] - buffer, maxDomain[1]]
            ],
            'inverted':False
        })
    if not config['domain']['periodicX'] and not config['domain']['periodicY']:
        minDomain = config['domain']['min']
        maxDomain = config['domain']['max']
        buffer = config['support'] * config['domain']['buffer']
        
        config['solidBoundary'].append({
            'vertices':[
                [minDomain[0] + buffer, minDomain[1] + buffer],
                [maxDomain[0] - buffer, minDomain[1] + buffer],
                [maxDomain[0] - buffer, maxDomain[1] - buffer],
                [minDomain[0] + buffer, maxDomain[1] - buffer]
            ],
            'inverted':True
        })

def boundaryFriction(config, state):
    with record_function('solidBC - friction'):
        # print(state)
        # print(state['boundaryNeighbors'])
        if 'boundaryNeighbors' in state and state['boundaryNeighbors'] != None:
            neighbors = state['boundaryNeighbors']
            i = neighbors[0]
            b = neighbors[1]
            sdfs = state['boundaryDistances']
            sdfgrads = state['boundaryGradients']
            
        #     print(i.shape)
        #     print(b.shape)
        #     print(sdfs.shape)
        #     print(sdfgrads.shape)
            
            fluidVelocity = state['fluidVelocity'][i]
            
        #     print(fluidVelocity.shape)
            
            fluidVelocityOrthogonal = torch.einsum('nd, nd -> n', fluidVelocity, sdfgrads)[:,None] * sdfgrads
            fluidVelocityParallel = fluidVelocity - fluidVelocityOrthogonal
        #     print(fluidVelocity)
        #     print(fluidVelocityOrthogonal)
        #     print(fluidVelocityParallel)
            velocities = []
            for sb in config['solidBoundary']:
                if 'velocity' in sb:
                    velocities.append(torch.tensor(sb['velocity'],device=config['device'],dtype=config['precision']))
                else:
                    velocities.append(torch.tensor([0,0],device=config['device'],dtype=config['precision']))
                    
            boundaryVelocities = torch.stack(velocities)
            fac = config['boundaryViscosityConstant'] * state['fluidRestDensity'][i]
            rho_i = state['fluidDensity'][i] * state['fluidRestDensity'][i]
            rho_b = state['fluidRestDensity'][i]
            
            v_ib = boundaryVelocities[b] - fluidVelocityParallel
            
            k = state['boundaryIntegrals']
            
            term = (fac / (rho_i + rho_b))[:,None] * v_ib
            
            correction = scatter(term, i, dim = 0, dim_size=state['numParticles'], reduce='add')
            # print(correction[i])

            state['fluidVelocity'] += correction
            force = -correction / config['dt'] * (state['fluidArea'] * state['fluidRestDensity'])[:,None]
            state['boundaryFrictionForce'] = scatter(force[i], b, dim = 0, dim_size = len(config['solidBoundary']), reduce = "add")
            
            
            
        

        

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))


import torch
from torch.profiler import record_function
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import numpy as np


def getSpacing(n, periodic = False):
    if n == 1:
        return 2.
    else:
        return 2. / n if periodic else 2./(n-1)


centroidCache = {False:{'cuda':{},'cpu':{}},True:{'cuda':{},'cpu':{}}}
def getDistances(n, x, periodic = False):
    if n in centroidCache[periodic][x.device.type]:
        centroids = centroidCache[periodic][x.device.type][n]
        if periodic:
            spacing = getSpacing(n, True)
            offset = -1 + spacing / 2.
            ra = torch.unsqueeze(x,axis=0) - centroids
            rb = torch.unsqueeze(x,axis=0) - centroids - 2.
            rc = torch.unsqueeze(x,axis=0) - centroids + 2.
            return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        else:
            spacing = getSpacing(n, False)
            
            # centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.constant([0.], device = x.device)
        #     tx = torch.constant(x, dtype='float32')
            r = torch.unsqueeze(x,axis=0) - centroids
            return torch.abs(r)  / spacing


    if periodic:
        spacing = getSpacing(n, True)
        offset = -1 + spacing / 2.
        
#         tx = torch.constant(x, dtype='float32')
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],axis=1)
        centroidCache[periodic][x.device.type][n] = centroids

        ra = torch.unsqueeze(x,axis=0) - centroids
        rb = torch.unsqueeze(x,axis=0) - centroids - 2.
        rc = torch.unsqueeze(x,axis=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, axis = 1)
    centroidCache[periodic][x.device.type][n] = centroids
#     tx = torch.constant(x, dtype='float32')
    r = torch.unsqueeze(x,axis=0) - centroids
    return torch.abs(r)  / spacing


def getDistancesRel(n, x, periodic = False):
    if n in centroidCache[periodic][x.device.type]:
        centroids = centroidCache[periodic][x.device.type][n]
        if periodic:
            spacing = getSpacing(n, True)
            offset = -1 + spacing / 2.
            ra = torch.unsqueeze(x,axis=0) - centroids
            rb = torch.unsqueeze(x,axis=0) - centroids - 2.
            rc = torch.unsqueeze(x,axis=0) - centroids + 2.
            return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        else:
            spacing = getSpacing(n, False)
            
            # centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.constant([0.], device = x.device)
        #     tx = torch.constant(x, dtype='float32')
            r = torch.unsqueeze(x,axis=0) - centroids
            return r  / spacing


    if periodic:
        spacing = getSpacing(n, True)
        offset = -1 + spacing / 2.
        
#         tx = torch.constant(x, dtype='float32')
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],axis=1)
        centroidCache[periodic][x.device.type][n] = centroids

        ra = torch.unsqueeze(x,axis=0) - centroids
        rb = torch.unsqueeze(x,axis=0) - centroids - 2.
        rc = torch.unsqueeze(x,axis=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, axis = 1)
    centroidCache[periodic][x.device.type][n] = centroids
#     tx = torch.constant(x, dtype='float32')
    r = torch.unsqueeze(x,axis=0) - centroids
    return r  / spacing


def evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = False, adjustSpacing = False, normalized = False):    

    k = int(epsilon)
    if adjustSpacing:
        if which == 'gaussian' or which == 'inverse_quadric' or which == 'inverse_multiquadric' or 'spline' in which  or 'wendland' in which:
            x = x * (1 - 2/n)
        if which == 'bump':
            x = x * (1 - 4/n)
    
    rRel = getDistancesRel(n, x, periodic)
    r = torch.abs(rRel)
    if n == 1:
        return torch.ones_like(r)
    
    cpow = lambda x, p: torch.maximum(x, torch.zeros_like(r))**p
    
    funLib = {
        'linear': lambda r:  torch.clamp(1. - r / epsilon,0,1),
        'gaussian': lambda r:  torch.exp(-(epsilon * r)**2),
        'multiquadric': lambda r: torch.sqrt(1. + (epsilon * r) **2),
        'inverse_quadric': lambda r: 1. / ( 1 + (epsilon * r) **2),
        'inverse_multiquadric': lambda r: 1. / torch.sqrt(1. + (epsilon * r) **2),
        'polyharmonic': lambda r: torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r)),
        'bump': lambda r: torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r)),
        'cubic_spline': lambda r: cpow(1-r/(epsilon * 1.),3) - 4. * cpow(1/2-r/(epsilon * 1.),3),
        'quartic_spline': lambda r: cpow(1-r/(epsilon * 1.),4) - 5 * cpow(3/5-r/(epsilon * 1.),4) + 10 * cpow(1/5-r/(epsilon * 1.),4),
        'quintic_spline': lambda r: cpow(1-r/(epsilon * 1.),5) - 6 * cpow(2/3-r/(epsilon * 1.),5) + 15 * cpow(1/3-r/(epsilon * 1.),5),
        'wendland2': lambda r: cpow(1 - r/(epsilon * 1.), 4) * (1 + 4 * r/(epsilon * 1.)),
        'wendland4': lambda r: cpow(1 - r/(epsilon * 1.), 6) * (1 + 6 * r/(epsilon * 1.) + 35/3 * (r/(epsilon * 1.))**2),
        'wendland6': lambda r: cpow(1 - r/(epsilon * 1.), 8) * (1 + 8 * r/(epsilon * 1.) + 25 * (r/(epsilon * 1.)) **2 + 32 * (r * (epsilon * 1.))**3),
        'poly6': lambda r: cpow(1 - (r/epsilon)**2, 3),
        'spiky': lambda r: cpow(1 - r/epsilon, 3),
        'square': lambda r: torch.where(torch.logical_and(rRel > -0.5 * epsilon, rRel <= 0.5 * epsilon), torch.ones_like(r), torch.zeros_like(r))
    }
    normalizedFunLib = {
        'linear': lambda r:  torch.clamp(1. - r / epsilon,0,1),
        'gaussian': lambda r:  torch.exp(-(epsilon * r)**2),
        'multiquadric': lambda r: torch.sqrt(1. + (epsilon * r) **2),
        'inverse_quadric': lambda r: 1. / ( 1 + (epsilon * r) **2),
        'inverse_multiquadric': lambda r: 1. / torch.sqrt(1. + (epsilon * r) **2),
        'polyharmonic': lambda r: torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r)),
        'bump': lambda r: torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r)),
        'cubic_spline': lambda r: cpow(1-r/(epsilon * 1.732051),3) - 4. * cpow(1/2-r/(epsilon * 1.732051),3),
        'quartic_spline': lambda r: cpow(1-r/(epsilon * 1.936492),4) - 5 * cpow(3/5-r/(epsilon * 1.936492),4) + 10 * cpow(1/5-r/(epsilon * 1.732051),4),
        'quintic_spline': lambda r: cpow(1-r/(epsilon * 2.121321),5) - 6 * cpow(2/3-r/(epsilon * 2.121321),5) + 15 * cpow(1/3-r/(epsilon * 2.121321),5),
        'wendland2': lambda r: cpow(1 - r/(epsilon * 1.620185), 4) * (1 + 4 * r/(epsilon * 1.620185)),
        'wendland4': lambda r: cpow(1 - r/(epsilon * 1.936492), 6) * (1 + 6 * r/(epsilon * 1.936492) + 35/3 * (r/(epsilon * 1.936492))**2),
        'wendland6': lambda r: cpow(1 - r/(epsilon * 2.207940), 8) * (1 + 8 * r/(epsilon * 2.207940) + 25 * (r/(epsilon * 2.207940)) **2 + 32 * (r * (epsilon * 2.207940))**3),
        'poly6': lambda r: cpow(1 - (r/epsilon)**2, 3),
        'spiky': lambda r: cpow(1 - r/epsilon, 3),
        'square': lambda r: torch.where(torch.logical_and(rRel > -0.5 * epsilon, rRel <= 0.5 * epsilon), torch.ones_like(r), torch.zeros_like(r))
    }    
    adjustedFunLib = {
        'linear': lambda r:  torch.clamp(1. - r / 1,0,1),
        'gaussian': lambda r:  torch.exp(-(0.9919394235466537 * r)**2),
        'multiquadric': lambda r: torch.sqrt(1. + (1 * r) **2),
        'inverse_quadric': lambda r: 1. / ( 1 + (1.1480214948705423 * r) **2),
        'inverse_multiquadric': lambda r: 1. / torch.sqrt(1. + (1.6382510991695163 * r) **2),
        'polyharmonic': lambda r: torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r)),
        'bump': lambda r: torch.where(r < 1./0.38739618954567656, torch.exp(-1./(1- (0.38739618954567656 * r)**2)), torch.zeros_like(r)),
        'cubic_spline': lambda r: cpow(1-r/(epsilon * 2.009770395701026),3) - 4. * cpow(1/2-r/(epsilon * 2.009770395701026),3),
        'quartic_spline': lambda r: cpow(1-r/(epsilon * 2.4318514899853443),4) - 5 * cpow(3/5-r/(epsilon * 2.4318514899853443),4) + 10 * cpow(1/5-r/(epsilon * 2.4318514899853443),4),
        'quintic_spline': lambda r: cpow(1-r/(epsilon * 2.8903273082559844),5) - 6 * cpow(2/3-r/(epsilon * 2.8903273082559844),5) + 15 * cpow(1/3-r/(epsilon * 2.8903273082559844),5),
        'wendland2': lambda r: cpow(1 - r/(epsilon * 3.6238397655105032), 4) * (1 + 4 * r/(epsilon * 3.6238397655105032)),
        'wendland4': lambda r: cpow(1 - r/(epsilon * 3.7338788470933073), 6) * (1 + 6 * r/(epsilon * 3.7338788470933073) + 35/3 * (r/(epsilon * 3.7338788470933073))**2),
        'wendland6': lambda r: cpow(1 - r/(epsilon * 1.3856863702979971), 8) * (1 + 8 * r/(epsilon * 1.3856863702979971) + 25 * (r/(epsilon * 1.3856863702979971)) **2 + 32 * (r * (epsilon * 1.3856863702979971))**3),
        'poly6': lambda r: cpow(1 - (r/ 2.6936980947728384)**2, 3),
        'spiky': lambda r: cpow(1 - r/3, 3),
        'square': lambda r: torch.where(torch.logical_and(rRel > -0.5 * 1, rRel <= 0.5 * 1), torch.ones_like(r), torch.zeros_like(r))

    }
    
    rbf = funLib[which]
    if adjustSpacing:
        rbf = adjustedFunLib[which]
    if normalized:
        rbf = normalizedFunLib[which]
    res = rbf(r)
    if normalized:
        res = res / torch.sum(res, dim = 0)
#     if periodic:
#         return torch.maximum(rbf(r[0]), torch.maximum(rbf(r[1]), rbf(r[2])))
        # return torch.clip_by_value(torch.maximum(rbf(r[0]), torch.maximum(rbf(r[1]), rbf(r[2]))),0,1)   
    return res
    
def evalChebSeries(n,x):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)
sqrt_pi_1 = 1. / np.sqrt(np.pi)

def evalChebSeries2(n,x):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(2 * x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)

def fourier(n, x):
    if n == 0:
        return torch.ones_like(x) / np.sqrt(2. * np.pi)
    elif n % 2 == 0:
        return torch.cos((n // 2 + 1) * x) * sqrt_pi_1
    return torch.sin((n // 2 + 1) * x) * sqrt_pi_1

def evalFourierSeries(n, x):
    fs = []
    for i in range(n):
        fs.append(fourier(i, x))
    return torch.stack(fs)

def evalBasisFunction(n, x, which = 'chebyshev', periodic = False):   
    s = which.split()    
#     print(s)
    if s[0] == 'chebyshev':
        return evalChebSeries(n, x)
    if s[0] == 'chebyshev2':
        return evalChebSeries2(n, x)
    if s[0] == 'fourier':
        return evalFourierSeries(n, x * np.pi)
    if s[0] == 'linear':
        return evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = periodic)        
    if s[0] == 'rbf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic)     
    if s[0] == 'abf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, adjustSpacing = True)     
    if s[0] == 'ubf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, normalized = True)

    
# def evalChebSeries(n,x):
#     cs = []
#     for i in range(n):
#         if i == 0:
#             cs.append(torch.ones_like(x))
#         elif i == 1:
#             cs.append(x)
#         else:
#             cs.append(2. * x * cs[i-1] - cs[i-2])
#     return torch.stack(cs)
# sqrt_pi_1 = 1. / np.sqrt(np.pi)

# def evalChebSeries2(n,x):
#     cs = []
#     for i in range(n):
#         if i == 0:
#             cs.append(torch.ones_like(x))
#         elif i == 1:
#             cs.append(2 * x)
#         else:
#             cs.append(2. * x * cs[i-1] - cs[i-2])
#     return torch.stack(cs)

# def fourier(n, x):
#     if n == 0:
#         return torch.ones_like(x) / np.sqrt(2. * np.pi)
#     elif n % 2 == 0:
#         return torch.cos((n // 2 + 1) * x) * sqrt_pi_1
#     return torch.sin((n // 2 + 1) * x) * sqrt_pi_1

# def evalFourierSeries(n, x):
#     fs = []
#     for i in range(n):
#         fs.append(fourier(i, x))
#     return torch.stack(fs)

# def evalBasisFunction(n, x, which = 'chebyshev', periodic = False):   
#     s = which.split()    
# #     print(s)
#     if s[0] == 'chebyshev':
#         return evalChebSeries(n, x)
#     if s[0] == 'chebyshev2':
#         return evalChebSeries2(n, x)
#     if s[0] == 'fourier':
#         return evalFourierSeries(n, x * np.pi)
#     if s[0] == 'linear':
#         return evalRBFSeries(n, x, which = 'linear', epsilon = float(s[1]) if len(s) > 1 else 1., periodic = periodic)        
#     if s[0] == 'rbf':
#         eps = 1. if len(s) < 3 else float(s[2])
#         return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic)


class cutlass(torch.autograd.Function):
    @staticmethod
    # @profile
    def forward(ctx, edge_index, features_i, features_j, edge_attr, edge_weights, weight, 
                dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized = False):
        with record_function("cutlass forward step"): 
            ctx.save_for_backward(edge_index, features_i, features_j, edge_attr, edge_weights, weight)
            ctx.dimensions = len(size)
            ctx.dim_size = dim_size
            ctx.dim = dim
            ctx.size = size
            ctx.rbfs = rbfs
            ctx.periodic = periodic
            ctx.forwardBatchSize = forwardBatchSize
            ctx.backwardBatchSize = backwardBatchSize
            ctx.normalized = normalized
            
            aggr = aggr_resolver('sum')

            with record_function("cutlass forward batchprep"): 
                x_j = features_j[edge_index[1]]#torch.index_select(features, 0, edge_index[1])
                x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]

                indices = torch.arange(0,edge_attr.shape[0]).to(features_j.device)
            
                batches = torch.split(indices, ctx.forwardBatchSize * 1024)
                # convs = []
            out = features_i.new_zeros((features_i.shape[0], weight.shape[-1])).type(features_i.dtype)

            for batch in batches:
                if ctx.dimensions == 1:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T

                        with record_function("cutlass forward einsum"): 
                            conv = torch.einsum('nu, uio,ni -> no',u,weight, torch.index_select(features_j,0, edge_index[1,batch]) * edge_weights[batch])
                        del u
                        out += aggr(conv, index = edge_index[1,batch], ptr = None, dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv
                if ctx.dimensions == 2:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                            v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T

                        with record_function("cutlass forward einsum"): 

                            # conv = torch.einsum('nu, nv, uvio,ni -> no',u,v,weight, torch.index_select(features,0, edge_index[1,batch]) * edge_weights[batch])
                            if ctx.normalized:
                                normalizationFactor = torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                conv = torch.einsum('nu, nv, uvio,ni -> no',u,v,weight, x_j[batch]) * normalizationFactor[:,None]
                            else:
                                conv = torch.einsum('nu, nv, uvio,ni -> no',u,v,weight, x_j[batch])

                        # print('u', u.dtype, u.shape)
                        # print('v', v.dtype, v.shape)
                        # print('weight', weight.dtype, weight.shape)
                        # print('x_j', x_j.dtype, x_j.shape)
                        # print('x_j[batch]', x_j[batch].dtype, x_j[batch].shape)
                        # print('u', u.dtype, u.shape)

                        del u,v
                        out += aggr(conv, index = edge_index[0,batch], ptr = None, dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv
                if ctx.dimensions == 3:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                            v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                            w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T
                        print('feat', features_j.shape, features_j)
                        print('batch', batch.shape, batch)
                        print('eib', edge_index[1,batch].shape, edge_index[1,batch])
                        print('ew', edge_weights.shape, edge_weights)
                        print('ewb', edge_weights[batch,None].shape, edge_weights[batch,None])
                        s = torch.index_select(features_j,0, edge_index[1,batch])
                        print('is', s.shape, s)
                        with record_function("cutlass forward einsum"): 
                            conv = torch.einsum('nu, nv, nw, uvwio,ni -> no',u,v,w,weight, \
                                torch.index_select(features_j,0, edge_index[1,batch]) * edge_weights[batch,None])
                        del u,v,w
                        out += aggr(conv, index = edge_index[1,batch], ptr = None, dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv

            # with record_function("cutlass forward stacking"): 
            #     out = torch.vstack(convs)
            
            
            # with record_function("cutlass forward aggregation"): 
                # out = aggr(torch.vstack(convs), index = edge_index[1], ptr = None, dim_size = ctx.dim_size, dim = ctx.dim)
            return out
    
    @staticmethod
    def backward(ctx, grad_output):
        with record_function("cutlass backward step"): 
            edge_index, features_i, features_j, edge_attr, edge_weights, weight = ctx.saved_tensors
            
            featureGrad = None
            weightGrad = None
            
            with record_function("cutlass backward batching"): 
                x_j = torch.index_select(features_j, 0, edge_index[1])
                # debugPrint(x_j)
                x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
                # debugPrint(x_j)
                # debugPrint(edge_weights)
                gradFeatures = torch.index_select(grad_output, 0, edge_index[0])

                indices = torch.arange(0,edge_attr.shape[0]).to(features_i.device)
            
                batches = torch.split(indices, ctx.backwardBatchSize * 1024)
            
            if ctx.needs_input_grad[2] and not ctx.needs_input_grad[5]:  
                # debugPrint('if ctx.needs_input_grad[2] and not ctx.needs_input_grad[5]:')
                with record_function("cutlass backward feature grad"):    
                    
                    transposedWeights = torch.transpose(weight, 2, 3)        
                    aggr = aggr_resolver('sum')

                    convs = []
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                
                                with record_function("cutlass backward feature grad einsum"):    
                                    convs.append(torch.einsum('nu, nv, uvio,ni -> no',u, transposedWeights, gradFeatures[batch]))
                                del u
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                
                                with record_function("cutlass backward feature grad einsum"):    
                                    if ctx.normalized:
                                        if edge_weights is not None:
                                            normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, n, uvio,ni -> no',u,v, edge_weights[batch], transposedWeights, gradFeatures[batch]) * normalizationFactor[:,None])
                                        else:
                                            normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, uvio,ni -> no',u,v, transposedWeights, gradFeatures[batch])*normalizationFactor[:,None])
                                    else:       
                                        if edge_weights is not None: 
                                            convs.append(torch.einsum('nu, nv, n, uvio,ni -> no',u,v, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                        else:
                                            convs.append(torch.einsum('nu, nv, uvio,ni -> no',u,v, transposedWeights, gradFeatures[batch]))
                                del u,v
                        if ctx.dimensions == 3:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T
                                
                                with record_function("cutlass backward feature grad einsum"):    
                                    convs.append(torch.einsum('nu, nv,nw, uvwio,ni -> no',u,v,w, transposedWeights, gradFeatures[batch]))
                                del u,v,w
                    with record_function("cutlass backward feature grad stacking"):   
                        out = torch.vstack(convs)
                    with record_function("cutlass backward feature grad aggregation"):   
                        featureGrad = aggr(out, index = edge_index[1], ptr = None, dim_size = features_j.shape[0], dim = ctx.dim)       
            if ctx.needs_input_grad[5] and not ctx.needs_input_grad[2]:   
                # debugPrint('if ctx.needs_input_grad[5] and not ctx.needs_input_grad[2]:')
                with record_function("cutlass backward weight grad"):    
                    weightGrad = weight.new_zeros(weight.shape)                    
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    localGrad = torch.einsum('nu, ni, no -> uvio', u, x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    if ctx.normalized:
                                        normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                        localGrad = torch.einsum('nu, nv, n, ni, no -> uvio', u, v, normalizationFactor, x_j[batch], gradFeatures[batch])
                                    else:                                        
                                        localGrad = torch.einsum('nu, nv, ni, no -> uvio', u, v,x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u,v
                        if ctx.dimensions == 3:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    localGrad = torch.einsum('nu, nv, nw, ni, no -> uvwio', u, v, w,x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u,v,w

            if ctx.needs_input_grad[2] and ctx.needs_input_grad[5]:  
                # debugPrint('if ctx.needs_input_grad[2] and ctx.needs_input_grad[5]:')
                with record_function("cutlass backward"):      
                    weightGrad = weight.new_zeros(weight.shape)
                    
                    transposedWeights = torch.transpose(weight, 2, 3)        
                    aggr = aggr_resolver('sum')

                    convs = []
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                            with record_function("cutlass backward einsum features"):   
                                convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch]))
                            with record_function("cutlass backward einsum grad"):   
                                io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nu, nio -> uio', u, io)
                                weightGrad += localGrad
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                            with record_function("cutlass backward einsum uvw"):   
                                if ctx.normalized:
                                    normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                    uvw = torch.einsum('nu, nv -> nuv', u, v) * normalizationFactor[:,None,None]
                                else:
                                    uvw = torch.einsum('nu, nv -> nuv', u, v) 
                                # del u,v
                            with record_function("cutlass backward einsum features"):   
                                if edge_weights is not None:
                                    convs.append(torch.einsum('nuv, n, uvio,ni -> no',uvw, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                else:
                                    convs.append(torch.einsum('nuv, uvio,ni -> no',uvw, transposedWeights, gradFeatures[batch]))
                            with record_function("cutlass backward einsum grad"):   
                                io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nuv, nio -> uvio', uvw, io)

                                # print('u', u.dtype, u.shape)
                                # print('v', v.dtype, v.shape)
                                # print('uvw', uvw.dtype, uvw.shape)
                                # print('io', io.dtype, io.shape)
                                # print('localGrad', localGrad.dtype, localGrad.shape)
                                # print('x_j[batch]', x_j[batch].dtype, x_j[batch].shape)
                                # print('gradFeatures[batch]', gradFeatures[batch].dtype, gradFeatures[batch].shape)

                                weightGrad += localGrad
                        if ctx.dimensions == 3:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T
                            with record_function("cutlass backward einsum uvw"):   
                                uvw = torch.einsum('nu, nv, nw -> nuvw', u, v, w)
                                del u,v,w
                            with record_function("cutlass backward einsum features"):   
                                convs.append(torch.einsum('nuvw, uvwio,ni -> no',uvw, transposedWeights, gradFeatures[batch]))
                            with record_function("cutlass backward einsum grad"):   
                                io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nuvw, nio -> uvwio', uvw, io)
                                weightGrad += localGrad

                    with record_function("cutlass backward stacking"):   
                        out = torch.vstack(convs)
                    with record_function("cutlass backward aggregation"):   
                        # print('out', out.dtype, out.shape)
                        featureGrad = aggr(out, index = edge_index[1], ptr = None, dim_size = features_j.shape[0], dim = ctx.dim)     
                        # print('featureGrad', featureGrad.dtype, featureGrad.shape)  
            
            # print('index:       ', edge_index)
            # print('features:    ', features)
            # print('attr:        ', edge_attr)
            # print('weights:     ', edge_weights)
            # print('weight:      ', weight)
            # print('featureGrad: ', featureGrad)
            # print('weightGrad:  ', weightGrad)
            return None, None, featureGrad, None, None, weightGrad, None, None, None, None, None, None, None, None 

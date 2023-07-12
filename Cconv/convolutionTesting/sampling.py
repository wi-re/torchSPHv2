# Math/parallelization library includes
import numpy as np
import torch
from convTesting import *
from convolution import *
import scipy
import seaborn as sns
import matplotlib as mpl
cmap = mpl.colormaps['viridis']

def sampleCentered(n, samples, dxScale = 8, clamped = True, seed = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator().manual_seed(torch.Generator().seed())
    means = torch.linspace(-1,1,n)
    dx = 2 / (n-1)
    
    sampled = []
    for s in range(samples):
        s = torch.normal(means, dx / dxScale, generator = generator)
        if clamped:
            s = torch.clamp(s, -1, 1)
        sampled.append(s)
    return sampled

def sampleOffCentered(n, samples, dxScale = 8, clamped = True, seed = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator().manual_seed(torch.Generator().seed())
    means = torch.linspace(-1,1,n)
    means = means[:-1] + torch.diff(means) / 2
    dx = 2 / (n-1)
    
    sampled = []
    for s in range(samples):
        s = torch.normal(means, dx / dxScale, generator = generator)
        if clamped:
            s = torch.clamp(s, -1, 1)
        sampled.append(s)
    return sampled

def sampleUniform(n, samples, seed = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator().manual_seed(torch.Generator().seed())
    sampled = []
    for s in range(samples):
        s = torch.rand(n, generator = generator)
        sampled.append(s *2 - 1)
    return sampled

def sampleNormal(n, samples, dx = 0.125, clamped = True, seed = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator().manual_seed(torch.Generator().seed())
    sampled = []
    for s in range(samples):
        s = torch.normal(torch.zeros(n), torch.ones(n) * dx, generator = generator)
        if clamped:
            s = torch.clamp(s, -1, 1)
        sampled.append(s)
    return sampled

def sampleCheb1(n, samples, dxScale = 8, clamped = True, seed = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator().manual_seed(torch.Generator().seed())
    cpts = np.polynomial.chebyshev.chebpts1(n)
    dx = np.diff(cpts)
    dx2 = (dx[:-1] + dx[1:]) / 2
    dx = np.hstack((dx[0], dx2, dx[-1]))
    means = torch.tensor(cpts)
    var = torch.tensor(dx) / dxScale
    
    sampled = []
    for s in range(samples):
        s = torch.normal(means, var, generator = generator)
        if clamped:
            s = torch.clamp(s, -1, 1)
        sampled.append(s)
    return sampled
def sampleCheb2(n, samples, dxScale = 8, clamped = True, seed = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator().manual_seed(torch.Generator().seed())
    cpts = np.polynomial.chebyshev.chebpts2(n)
    dx = np.diff(cpts)
    dx2 = (dx[:-1] + dx[1:]) / 2
    dx = np.hstack((dx[0], dx2, dx[-1]))
    means = torch.tensor(cpts)
    var = torch.tensor(dx) / dxScale
    
    sampled = []
    for s in range(samples):
        s = torch.normal(means, var, generator = generator)
        if clamped:
            s = torch.clamp(s, -1, 1)
        sampled.append(s)
    return sampled

def sampleRegular(n, samples):
    sampled = []
    for s in range(samples):
        sampled.append(torch.linspace(-1,1,n))
    return sampled


def sample(n, samples, method = 'centered', dxScale = 8, dx = 1/3, clamped = True, seed = None):
    if method == 'centered':
        return sampleCentered(n, samples, dxScale = dxScale, clamped = clamped, seed = seed)
    if method == 'offCentered':
        return sampleOffCentered(n, samples, dxScale = dxScale, clamped = clamped, seed = seed)
    if method == 'uniform':
        return sampleUniform(n, samples, seed = seed)
    if method == 'normal':
        return sampleNormal(n, samples, dx = dx, clamped = clamped, seed = seed)
    if method == 'cheb1':
        return sampleCheb1(n, samples, dxScale = dxScale, clamped = clamped, seed = seed)
    if method == 'cheb2':
        return sampleCheb2(n, samples, dxScale = dxScale, clamped = clamped, seed = seed)
    if method == 'regular':
        return sampleRegular(n, samples)
    
def plotDataset(dataset, samples, target):
    fig, axis = plt.subplots(3, 1, figsize=(12,8), height_ratios = [4,4,1], sharex = True, sharey = False, squeeze = False)

    xb = torch.linspace(-1,1,2048)
    axis[0,0].plot(xb, target(xb), c = 'white',lw = 0.5)

    for s in range(samples):
        axis[0,0].scatter(dataset[s,:],target(dataset[s,:]),ls='-',color=mpl.colormaps['PuRd'](1 / (samples - 1)* s), label = 'sample %d' % s, s = 2)    


    for s in range(samples):
        axis[1,0].scatter(dataset[s,:],s * torch.ones(dataset[s,:].shape),ls='-',color=mpl.colormaps['PuRd'](1 / (samples - 1)* s), label = 'sample %d' % s, s = 1)

    dx = 2 / (n-1)
    for i in range(n):
        axis[0,0].axvline(-1 + dx * i, ls = '--', c = 'white', alpha = 0.5)
        axis[1,0].axvline(-1 + dx * i, ls = '--', c = 'white', alpha = 0.5)

    sns.histplot(data = dataset.flatten(), bins = 128, ax = axis[2,0],kde = True, kde_kws={'bw_adjust':0.1})
    fig.tight_layout()
    
def plotTraining(n, basis, normalizedBasis, target, weightList, gradList):
    fig, axis = plt.subplots(2, 2, figsize=(16,8), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,255)
    fx = evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis)

    import matplotlib.colors as colors
    norm = colors.LogNorm(vmin=1, vmax=len(weightList))

    for i, w in enumerate(weightList):
        c = cmap(1 / (len(weightList) - 1)* i)
        c = cmap(norm(i + 1))
        for y in range(n):
            fy = w[y].detach() * fx[y,:]
            axis[1,0].plot(x[fy != 0], fy[fy != 0], label = '$w_d f_%d(x)$' % y, ls = '--', alpha = 0.5,c = c)

        axis[0,0].plot(x,torch.sum(w[:,None].detach() * fx, axis=0),ls='-',c=c, label = 'epoch %d' % i)
        loss = (torch.sum(w.detach()[:,None] * fx, axis=0) - target(x))**2
        axis[0,1].plot(x, loss,c=c)

    gradListTensor = torch.vstack(gradList)
    for i in range(n):
        axis[1,1].plot(gradListTensor[:,i])

    axis[1,1].plot(torch.sum(gradListTensor, axis=1), c = 'white', ls = '--')
    axis[0,0].plot(x, target(x),label = 'target',lw=2,c='red')    
    axis[0,0].set_title('Learning Progress')
    axis[0,1].set_title('Loss')
    axis[1,0].set_title('Base Functions')
    axis[1,1].set_title('Gradients')
    axis[0,0].set_title('Interpolant')
    fig.tight_layout()
    
def plotTrainingv2(n, basis, normalizedBasis, target, weightList, gradList, lossList, windowFn = None, plotInterval = 1):
    fig, axis = plt.subplots(1, 2, figsize=(16,8), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,255)
    fx = evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis) if windowFn is None else evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis) * windowFn(x)[None,:]

    import matplotlib.colors as colors
    norm = colors.LogNorm(vmin=1, vmax=len(weightList))
    losses = []
    for i, w in enumerate(weightList):
        c = cmap(1 / (len(weightList) - 1)* i)
        c = cmap(norm(i + 1))
    #     for y in range(n):
    #         fy = w[y].detach() * fx[y,:]
    #         axis[1,0].plot(x[fy != 0], fy[fy != 0], label = '$w_d f_%d(x)$' % y, ls = '--', alpha = 0.5,c = c)
        if i % plotInterval == 0:
            axis[0,0].plot(x,torch.sum(w[:,None].detach() * fx, axis=0),ls='-',c=c, label = 'epoch %d' % i)
        loss = torch.mean((torch.sum(w.detach()[:,None] * fx, axis=0) - target(x))**2)
        losses.append(loss)
    #     axis[0,1].plot(x, loss,c=c)

    axis[0,1].loglog(torch.hstack(losses), label = 'error')
    axis[0,1].loglog(torch.hstack(lossList), label = 'training loss')
    axis[0,1].legend()

    # gradListTensor = torch.vstack(gradList)
    # for i in range(n):
    #     axis[1,1].plot(gradListTensor[:,i])

    # axis[1,1].plot(torch.sum(gradListTensor, axis=1), c = 'white', ls = '--')
    axis[0,0].plot(x, target(x),label = 'target',lw=2,c='red')    
    axis[0,0].set_title('Learning Progress')
    axis[0,1].set_title('Loss')
    # axis[1,0].set_title('Base Functions')
    # axis[1,1].set_title('Gradients')
    axis[0,0].set_title('Interpolant')
    fig.tight_layout()

from torch_geometric.loader import DataLoader

def generateLoaders(dataset, batch_size = 4, shuffleDataset = True, shuffled = True, shuffleSeed = None):
    if shuffled:
        npGenerator = np.random.default_rng(seed = shuffleSeed)
        t = np.arange(n * len(dataset))
        npGenerator.shuffle(t)

        shuffledDataset = dataset.flatten()[t].reshape(dataset.shape)
        train_dataloader = DataLoader(shuffledDataset, shuffle=shuffleDataset, batch_size = batch_size)
        train_iter = iter(train_dataloader)
        return train_dataloader, train_iter

    train_dataloader = DataLoader(dataset, shuffle=shuffleDataset, batch_size = batch_size)
    train_iter = iter(train_dataloader)
    return train_dataloader, train_iter

def sampleDataLoader():
    global train_dataloder, train_iter
    try:
        bdata = next(train_iter)
    except:
        train_iter = iter(train_dataloader)
        bdata = next(train_iter)
    return bdata

def generateTargetFunction(freq = 4, octaves = 2, seed = 1234, r = 1 / (2 * np.pi), persistence = 0.5, numSamples = 2**16, normalized = True, periodicNoise = True, window = None, baseOverride = None, plot = False):    
#     baseTarget = lambda x: (torch.sin(x * np.pi * 4) + 0.5 * torch.sin(x * np.pi * 2) + 0.25 * torch.sin(x * np.pi)).numpy()
#     baseTarget = lambda x: np.sin( np.pi * 100 * x)

    noise  = generate1DPeriodicNoise(numSamples = numSamples, r = r, freq = freq, octaves = octaves, plot = False, seed = seed, persistence = persistence) if periodicNoise else generate1DNoise(numSamples = numSamples, r = r, freq = freq, octaves = octaves, plot = False, seed = seed, persistence = persistence)
#     noise  = 
    baseTarget = lambda x : np.interp(x, np.linspace(-1,1,numSamples), noise) 
    if baseOverride is not None:
        baseTarget = baseOverride
    if window is not None:
        windowFunc = getWindowFunction(window)
        windowFn = lambda x : windowFunc(torch.abs(torch.tensor(x))).numpy()
        intermediate = lambda x: baseTarget(x) * windowFn(x) 
    else:
        intermediate = lambda x: baseTarget(x)

    xs = np.linspace(-1,1,2**16)
    maxValue = intermediate(xs[np.argmax(np.abs(intermediate(xs)))])

    target = lambda x: intermediate(x) 
    if normalized:
        target = lambda x: intermediate(x) / maxValue
    
#     target = lambda x: intermediate(x) / maxValue
    if plot:
        xs = np.linspace(-1,1,numSamples)
        signal = target(xs)

        fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
        axis[0,0].plot(xs, baseTarget(xs), label = 'baseTarget')
        axis[0,0].plot(xs, windowFn(xs), label = 'window')
        axis[0,0].plot(xs, signal, label = 'target')
        axis[0,0].legend()

        fs = numSamples/2
        fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
        x = baseTarget(xs)
        y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
        axis[0,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'baseTarget')
        x = windowFn(xs)
        y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
        axis[0,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'window')
        x = target(xs)
        y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
        axis[0,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'target')
        axis[0,1].legend()

        f, Pxx_den = scipy.signal.welch(baseTarget(x), fs, nperseg=len(x)//32)
        axis[0,2].loglog(f, Pxx_den, label = 'baseTarget')
        f, Pxx_den = scipy.signal.welch(windowFn(x), fs, nperseg=len(x)//32)
        axis[0,2].loglog(f, Pxx_den, label = 'window')
        f, Pxx_den = scipy.signal.welch(target(x), fs, nperseg=len(x)//32)
        axis[0,2].loglog(f, Pxx_den, label = 'target')
        axis[0,2].set_xlabel('frequency [Hz]')
        axis[0,2].set_ylabel('PSD [V**2/Hz]')
        axis[0,2].legend()

        fig.tight_layout()
    return target

def evalGroundTruth(samples, dataset, target, noise = True, noiseType = 'normal', noiseVar = 0.1):
    gts = []

    for s in range(samples):
        points = dataset[s]
        gt = torch.tensor(target(points))
        if noise:
            if noiseType == 'normal':
                gt = gt + torch.normal(torch.zeros(gt.shape), torch.ones(gt.shape) * noiseVar)
            if noiseType == 'uniform':
                gt = gt + (torch.rand(gt.shape) * 2 - 1) * noiseVar
        gts.append(gt)
    gts = torch.vstack(gts)
    return gts


def plotDatasetAndGroundtruth(dataset, samples, target, gt):
    fig, axis = plt.subplots(3, 1, figsize=(12,8), height_ratios = [4,4,1], sharex = True, sharey = False, squeeze = False)

    xb = torch.linspace(-1,1,2048)
    axis[0,0].plot(xb, target(xb), c = 'white',lw = 0.5)

    for s in range(samples):
        axis[0,0].scatter(dataset[s,:],gt[s,:],ls='-',color=mpl.colormaps['PuRd'](1 / (samples - 1)* s), label = 'sample %d' % s, s = 2)    


    for s in range(samples):
        axis[1,0].scatter(dataset[s,:],s * torch.ones(dataset[s,:].shape),ls='-',color=mpl.colormaps['PuRd'](1 / (samples - 1)* s), label = 'sample %d' % s, s = 1)

    dx = 2 / (n-1)
    for i in range(n):
        axis[0,0].axvline(-1 + dx * i, ls = '--', c = 'white', alpha = 0.5)
        axis[1,0].axvline(-1 + dx * i, ls = '--', c = 'white', alpha = 0.5)

    sns.histplot(data = dataset.flatten(), bins = 128, ax = axis[2,0],kde = True, kde_kws={'bw_adjust':0.1})
    fig.tight_layout()
    

def plotTrainingv3(n, basis, normalizedBasis, target, weightList, gradList, lossList, windowFn = None, plotInterval = 1, plotInitial = True):
    fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,255)
    fx = evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis) if windowFn is None else evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis) * windowFn(x)[None,:]

    import matplotlib.colors as colors
    norm = colors.LogNorm(vmin=1, vmax=len(weightList))
    losses = []
    evals = []
    for i, w in enumerate(weightList):
        c = cmap(1 / (len(weightList) - 1)* i)
        c = cmap(norm(i + 1))
    #     for y in range(n):
    #         fy = w[y].detach() * fx[y,:]
    #         axis[1,0].plot(x[fy != 0], fy[fy != 0], label = '$w_d f_%d(x)$' % y, ls = '--', alpha = 0.5,c = c)
#         if i % plotInterval == 0:
        evals.append(torch.sum(w[:,None].detach() * fx, axis=0))
        loss = torch.mean((torch.sum(w.detach()[:,None] * fx, axis=0) - target(x))**2)
        losses.append(loss)
    #     axis[0,1].plot(x, loss,c=c)
    im = axis[0,2].imshow(torch.vstack(evals), aspect = 'auto')
    ax1_divider = make_axes_locatable(axis[0,2])
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    
    axis[0,0].plot(x, target(x),label = 'Groundtruth',lw=2,c='red',alpha = 0.5)
    axis[0,0].plot(x, evals[-1], label = 'Final Output')
    if plotInitial:
        axis[0,0].plot(x, evals[0], label = 'Initial Output')
    axis[0,1].loglog(torch.hstack(losses), label = 'error')
    axis[0,1].loglog(torch.hstack(lossList), label = 'training loss')
    axis[0,1].legend()

    # gradListTensor = torch.vstack(gradList)
    # for i in range(n):
    #     axis[1,1].plot(gradListTensor[:,i])

    # axis[1,1].plot(torch.sum(gradListTensor, axis=1), c = 'white', ls = '--')
    axis[0,0].legend()   
    axis[0,2].set_title('Learning Progress')
    axis[0,1].set_title('Loss')
    # axis[1,0].set_title('Base Functions')
    # axis[1,1].set_title('Gradients')
    axis[0,0].set_title('Interpolant')
    fig.tight_layout()
    

def plotTrainingv4(n, basis, normalizedBasis, target, weightList, gradList, lossList, r2List, windowFn = None, plotInterval = 1, plotInitial = True):
    fig, axis = plt.subplots(1, 4, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,255)
    fx = evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis) if windowFn is None else evalBasis(n, x, basis, periodic = False, normalized = normalizedBasis) * windowFn(x)[None,:]

    import matplotlib.colors as colors
    norm = colors.LogNorm(vmin=1, vmax=len(weightList))
    losses = []
    evals = []
    r2s = []
    for i, w in enumerate(tqdm(weightList)):
        c = cmap(1 / (len(weightList) - 1)* i)
        c = cmap(norm(i + 1))
    #     for y in range(n):
    #         fy = w[y].detach() * fx[y,:]
    #         axis[1,0].plot(x[fy != 0], fy[fy != 0], label = '$w_d f_%d(x)$' % y, ls = '--', alpha = 0.5,c = c)
#         if i % plotInterval == 0:
        evals.append(torch.sum(w[:,None].detach() * fx, axis=0))
        loss = torch.mean((torch.sum(w.detach()[:,None] * fx, axis=0) - target(x))**2)
        losses.append(loss)
        
        r2s.append(r2_score(target(x), evals[-1].detach().cpu().numpy()))
    #     axis[0,1].plot(x, loss,c=c)
    im = axis[0,3].imshow(torch.vstack(evals), aspect = 'auto')
    ax1_divider = make_axes_locatable(axis[0,3])
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    
    axis[0,0].plot(x, target(x),label = 'Groundtruth',lw=2,c='red',alpha = 0.5)
    axis[0,0].plot(x, evals[-1], label = 'Final Output')
    if plotInitial:
        axis[0,0].plot(x, evals[0], label = 'Initial Output')
    axis[0,1].loglog(torch.hstack(losses), label = 'error')
    axis[0,1].loglog(torch.hstack(lossList), label = 'training loss')
    axis[0,1].legend()
    
    
    axis[0,2].semilogx(np.hstack(r2s), label = 'r2')
    axis[0,2].semilogx(np.hstack(r2List), label = 'training r2')
    axis[0,2].legend()
    

    # gradListTensor = torch.vstack(gradList)
    # for i in range(n):
    #     axis[1,1].plot(gradListTensor[:,i])

    # axis[1,1].plot(torch.sum(gradListTensor, axis=1), c = 'white', ls = '--')
    axis[0,0].legend()   
    axis[0,3].set_title('Learning Progress')
    axis[0,1].set_title('Loss %g / %g' % (losses[-1], lossList[-1]))
    axis[0,2].set_title('R2 %g / %g' % (r2s[-1], r2List[-1]))
    # axis[1,0].set_title('Base Functions')
    # axis[1,1].set_title('Gradients')
    axis[0,0].set_title('Interpolant')
    fig.tight_layout()
    
# plotTrainingv3(n, basis, normalizedBasis, target, weightList, gradList, lossList, plotInterval = 32)


def plotDatasetAndGroundtruthv2(dataset, samples, target, gt):
    fig, axis = plt.subplots(3, 1, figsize=(12,8),gridspec_kw={'height_ratios': [4,4,1]}, sharex = True, sharey = False, squeeze = False)

    xb = torch.linspace(-1,1,2048)
    axis[0,0].plot(xb, target(xb), c = 'white',lw = 0.5)
    
    if samples > 1:
        for s in tqdm(range(samples)):
            axis[0,0].plot(dataset[s,:],gt[s,:],ls='-',color=mpl.colormaps['viridis'](1 / (samples - 1)* s), label = 'sample %d' % s, alpha = 0.5)    
        for s in tqdm(range(samples)):
            axis[1,0].scatter(dataset[s,:],s * torch.ones(dataset[s,:].shape),ls='-',color=mpl.colormaps['PuRd'](1 / (samples - 1)* s), label = 'sample %d' % s, s = 1)
    else:    
        for s in tqdm(range(samples)):
            axis[0,0].plot(dataset[s,:],gt[s,:],ls='-',color=mpl.colormaps['viridis'](1.), label = 'sample %d' % s, alpha = 1.)    
        for s in tqdm(range(samples)):
            axis[1,0].scatter(dataset[s,:],s * torch.ones(dataset[s,:].shape),ls='-',color=mpl.colormaps['PuRd'](1), label = 'sample %d' % s, s = 8)

    n = len(dataset[0])
    dx = 2 / (n-1)
    for i in range(n):
        axis[0,0].axvline(-1 + dx * i, ls = '--', c = 'white', alpha = 0.5)
        axis[1,0].axvline(-1 + dx * i, ls = '--', c = 'white', alpha = 0.5)

    sns.histplot(data = dataset.flatten(), bins = 128, ax = axis[2,0],kde = True, kde_kws={'bw_adjust':0.1})
    fig.tight_layout()

def plotTraining(yTest, testEvals, lossList, testLosses, r2List, testr2s, weightList):
    fig, axis = plt.subplots(1, 5, figsize=(18,6), sharex = False, sharey = False, squeeze = False)

    axis[0,0].plot(np.linspace(-1,1,yTest.shape[0]), yTest, label = 'target')
    axis[0,0].plot(np.linspace(-1,1,yTest.shape[0]), testEvals[0], label = 'initial')
    axis[0,0].plot(np.linspace(-1,1,yTest.shape[0]), testEvals[-1], label = 'final')
    axis[0,0].set_title('Interpolant')
    axis[0,0].legend()

    axis[0,1].loglog(lossList, label = 'training Losses')
    axis[0,1].loglog(testLosses, label = 'test Losses')
    axis[0,1].set_title('Loss %g : %g' % (lossList[-1], testLosses[-1]))
    axis[0,1].legend()

    axis[0,2].semilogx(r2List, label = 'training r2')
    axis[0,2].semilogx(testr2s, label = 'test r2')
    axis[0,2].set_title('R2 %g : %g' % (r2List[-1], testr2s[-1]))
    axis[0,2].legend()

    im = axis[0,3].imshow(np.vstack(testEvals),aspect = 'auto', extent = [-1,1,len(testEvals),1])
    axis[0,3].set_yscale('log')
    axis[0,3].set_title('Training Progress')
    ax1_divider = make_axes_locatable(axis[0,3])
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 

    im = axis[0,4].imshow(torch.vstack(weightList).detach().numpy(), aspect = 'auto', extent = [1,8,len(testEvals),1])
    axis[0,4].set_yscale('log')
    axis[0,4].set_title('Weight Progress')
    ax1_divider = make_axes_locatable(axis[0,4])
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 

#     fig.tight_layout()
    
    return fig, axis

def plotTrainingMLP(yTest, testEvals, lossList, testLosses, r2List, testr2s, ws):
    fig, axis = plt.subplots(1, 5, figsize=(18,6), sharex = False, sharey = False, squeeze = False)

    axis[0,0].plot(np.linspace(-1,1,yTest.shape[0]), yTest, label = 'target')
    axis[0,0].plot(np.linspace(-1,1,yTest.shape[0]), testEvals[0], label = 'initial')
    axis[0,0].plot(np.linspace(-1,1,yTest.shape[0]), testEvals[-1], label = 'final')
    axis[0,0].set_title('Interpolant')
    axis[0,0].legend()

    axis[0,1].loglog(lossList, label = 'training Losses')
    axis[0,1].loglog(testLosses, label = 'test Losses')
    axis[0,1].set_title('Loss %g : %g' % (lossList[-1], testLosses[-1]))
    axis[0,1].legend()

    axis[0,2].semilogx(r2List, label = 'training r2')
    axis[0,2].semilogx(testr2s, label = 'test r2')
    axis[0,2].set_title('R2 %g : %g' % (r2List[-1], testr2s[-1]))
    axis[0,2].legend()

    im = axis[0,3].imshow(np.vstack(testEvals),aspect = 'auto', extent = [-1,1,len(testEvals),1])
    axis[0,3].set_yscale('log')
    axis[0,3].set_title('Training Progress')
    ax1_divider = make_axes_locatable(axis[0,3])
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 

    weightList = [torch.hstack([weights[w].flatten() for w in weights]) for weights in ws]
    im = axis[0,4].imshow(torch.vstack(weightList).detach().numpy(), aspect = 'auto', extent = [1,8,len(testEvals),1])
    axis[0,4].set_yscale('log')
    axis[0,4].set_title('Weight Progress')
    ax1_divider = make_axes_locatable(axis[0,4])
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    
    return fig, axis

from scipy import interpolate

def generateTargetAndGradientFunction(freq = 4, octaves = 2, seed = 1234, r = 1 / (2 * np.pi), persistence = 0.5, numSamples = 2**16, normalized = True, periodicNoise = True, window = None, baseOverride = None, plot = False, kind = 'linear'):    
#     baseTarget = lambda x: (torch.sin(x * np.pi * 4) + 0.5 * torch.sin(x * np.pi * 2) + 0.25 * torch.sin(x * np.pi)).numpy()
#     baseTarget = lambda x: np.sin( np.pi * 100 * x)

    noise  = generate1DPeriodicNoise(numSamples = numSamples, r = r, freq = freq, octaves = octaves, plot = False, seed = seed, persistence = persistence) if periodicNoise else generate1DNoise(numSamples = numSamples, r = r, freq = freq, octaves = octaves, plot = False, seed = seed, persistence = persistence)
#     noise  = 
    baseTargetFn = interpolate.interp1d(np.linspace(-1,1,numSamples), noise, kind = kind, fill_value = 'extrapolate')
    baseTarget = lambda x : baseTargetFn(x)
    if baseOverride is not None:
        baseTarget = baseOverride
    if window is not None:
        windowFunc = getWindowFunction(window)
        windowFn = lambda x : windowFunc(torch.abs(torch.tensor(x))).numpy()
        intermediate = lambda x: baseTarget(x) * windowFn(x) 
    else:
        intermediate = lambda x: baseTarget(x)

    xs = np.linspace(-1,1,numSamples)
    maxValue = np.abs(intermediate(xs[np.argmax(np.abs(intermediate(xs)))]))

    target = lambda x: intermediate(x) 
    if normalized:
        target = lambda x: intermediate(x) / maxValue
    
    gradxs = np.linspace(-1,1,256)
    grad = np.gradient(target(gradxs), gradxs)
    gradFn = interpolate.interp1d(gradxs, grad, kind = kind, fill_value = 'extrapolate')
    targetDerivative = lambda x: gradFn(x)
    
#     target = lambda x: intermediate(x) / maxValue
    if plot:
        xs = np.linspace(-1,1,numSamples)
        signal = target(xs)

        fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
        axis[0,0].plot(xs, baseTarget(xs), label = 'baseTarget')
        if window is not None:
            axis[0,0].plot(xs, windowFn(xs), label = 'window')
        axis[0,0].plot(xs, signal, label = 'target')
        axis[0,0].plot(xs, targetDerivative(xs), label = 'derivative')
        axis[0,0].legend()

        fs = numSamples/2
        fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
        x = baseTarget(xs)
        y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
        axis[0,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'baseTarget')
        if window is not None:
            x = windowFn(xs)
            y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
            axis[0,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'window')
        x = target(xs)
        y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
        axis[0,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'target')
        x = targetDerivative(xs)
        y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
        axis[0,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'gradient')
        axis[0,1].legend()

        f, Pxx_den = scipy.signal.welch(baseTarget(x), fs, nperseg=len(x)//32)
        axis[0,2].loglog(f, Pxx_den, label = 'baseTarget')
        if window is not None:
            f, Pxx_den = scipy.signal.welch(windowFn(x), fs, nperseg=len(x)//32)
            axis[0,2].loglog(f, Pxx_den, label = 'window')
            f, Pxx_den = scipy.signal.welch(target(x), fs, nperseg=len(x)//32)
        axis[0,2].loglog(f, Pxx_den, label = 'target')
        f, Pxx_den = scipy.signal.welch(targetDerivative(x), fs, nperseg=len(x)//32)
        axis[0,2].loglog(f, Pxx_den, label = 'gradient')
        axis[0,2].set_xlabel('frequency [Hz]')
        axis[0,2].set_ylabel('PSD [V**2/Hz]')
        axis[0,2].legend()

        fig.tight_layout()
    return target, targetDerivative

# Code to adust an rbf to make it optimal, will probably be useful later on
w = np.linspace(0.2, 4.5, 2048)
x =  torch.linspace(-1,1,511)
    
fig, axis = plt.subplot_mosaic('''
AB
CD
EE''', figsize=(18,7))

# fig, axis = plt.subplots(2, 2, figsize=(16,6), sharex = False, sharey = False, squeeze = False)

means = []
stds = []

basis = 'rbf wendland2'

for i, width in enumerate(w):
    fx = evalBasisFunction(n, x , which = '%s %g' % (basis, width), periodic=False)
    # torch.sum(fx, axis = 0)
    funs = torch.sum(fx[:,torch.logical_and(x>= -0.5, x <= 0.5)], axis = 0)
    fMean = torch.mean(funs)
    fStd = torch.std(funs)
    means.append(fMean)
    stds.append(fStd)
    c = cmap(-1 + 2 / (len(w) - 1)* i)
    if i % 8 == 0:
        axis['A'].plot(x, torch.sum(fx, axis = 0), c = c)


axis['A'].set_title('Variable Width Convolutions')
axis['B'].set_title('Variable Width Statistics')
        
axis['B'].semilogy(w, means, label = 'Mean')
axis['B'].semilogy(w, stds, label ='StdDev')

axis['B'].legend()
# print(fMean, fStd)

# print(stds)
print(torch.argmin(torch.hstack(stds)), stds[torch.argmin(torch.hstack(stds))], w[torch.argmin(torch.hstack(stds))] )
idealWidth = w[torch.argmin(torch.hstack(stds))]

w = np.linspace(0.5, 2.5, 2048)

means = []
stds = []

for i, width in enumerate(w):
    fx = evalBasisFunction(n, x , which = '%s %g' % (basis, idealWidth), periodic=False, spacing = width)
    # torch.sum(fx, axis = 0)
    funs = torch.sum(fx, axis = 0)
    fMean = torch.mean(funs)
    fStd = torch.std(funs)
    means.append(fMean)
    stds.append(fStd)
    c = cmap(-1 + 2 / (len(w) - 1)* i)
    if i % 8 == 0:
#         axis[1,0].plot(x[torch.logical_and(x>= -0.5, x <= 0.5)], funs, c = c)
        axis['C'].plot(x, torch.sum(fx, axis = 0), c = c)

    
axis['D'].semilogy(w, torch.hstack(means), label = 'Mean')
axis['D'].semilogy(w, torch.hstack(stds), label ='StdDev')

axis['C'].set_title('Variable Spacing Convolutions')
axis['D'].set_title('Variable Spacing Statistics')

axis['D'].legend()
# print(fMean, fStd)

print(torch.argmin(torch.hstack(stds)), stds[torch.argmin(torch.hstack(stds))], w[torch.argmin(torch.hstack(stds))] )
idealSpacing = w[torch.argmin(torch.hstack(stds))]

x =  torch.linspace(-1,1,511)
fx = evalBasisFunction(n, x , which = '%s %g' % (basis, idealWidth), periodic=False, spacing = 1.0 - 2/n)
# fx = fx / torch.sum(fx, axis = 0)[None,:]

for y in range(n):
#     print(y)
    axis['E'].plot(x, fx[y,:], label = '$f_%d(x)$' % y)
axis['E'].plot(x,torch.sum(fx, axis=0),ls='--',c='white', label = '$\Sigma_i f_i(x)$')
# axis[0,0].legend()
axis['E'].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=9, fancybox=True, shadow=False)
axis['E'].set_title('AdustedBasis Functions')

fig.tight_layout()

# Plot gradient landscapes of basis functions
nx = 256
basis = 'fourier'
grads = []
for i in tqdm(range(nx)):
    x = torch.tensor([-1 + 2 / (nx - 1) * i], dtype = torch.float32)
    weights.grad = torch.zeros(weights.shape)
    weights.requires_grad = True
    fx = evalBasisFunction(n, x , which = basis, periodic=False)
#     fx = fx / torch.sum(fx, axis = 0)[None,:]
    wfx = torch.sum(weights[:,None] * fx, axis = 0)
    wfx.backward()
    grads.append(torch.clone(weights.grad))
grads = torch.vstack(grads)

fig, axis = plt.subplots(1, 2, figsize=(16,6), sharex = False, sharey = False, squeeze = False)

im = axis[0,0].imshow(grads.mT, aspect = 'auto', extent = [-1,1, weights.shape[0],0], interpolation = 'nearest')
ax1_divider = make_axes_locatable(axis[0,0])
cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
cb1.ax.tick_params(labelsize=8) 

axis[0,0].set_yticks([x + 0.5 for x in range(weights.shape[0])])
axis[0,0].set_yticklabels([x for x in range(weights.shape[0])])
for i in range(weights.shape[0]):
    axis[0,0].axhline(i + 1, ls = '--', alpha = 0.5, c = 'white')
axis[0,0].set_title('Basis Gradient (landscape)')
for y in range(weights.shape[0]):
    axis[0,1].plot(torch.linspace(-1,1,nx), grads[:,y], label = '$\\nabla f_%d(x)$' % y)
axis[0,1].plot(torch.linspace(-1,1,nx), torch.sum(grads,axis=1),ls='--',c='white', label = '$\Sigma_i \\nabla f_i(x)$')
    
axis[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
          ncol=5, fancybox=True, shadow=False)

axis[0,1].set_title('Basis Gradients (individual)')
fig.tight_layout()

# Plot window functions 
splines = ['cubicSpline', 'quarticSpline', 'quinticSpline']
wendlands = ['Wendland2_1D', 'Wendland4_1D', 'Wendland6_1D', 'Wendland2', 'Wendland4', 'Wendland6']
misc = ['Hoct4', 'Spiky', 'Mueller', 'poly6', 'Parabola', 'Linear']


fig, axis = plt.subplots(1, 3, figsize=(18,4), sharex = False, sharey = False, squeeze = False)
numSamples = 2**16
xs = np.linspace(-1,1,numSamples)

for s in splines:
    windowFunction = getWindowFunction(s)
    windowFn = lambda x: windowFunction(torch.abs(torch.tensor(x))).numpy()
    
    axis[0,0].plot(xs, windowFn(xs), label = '%s' % s)

#     fs = numSamples/2
#     fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
#     x = windowFn(xs)
#     y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
#     axis[1,0].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = s)

#     f, Pxx_den = scipy.signal.welch(windowFn(x), fs, nperseg=len(x)//32)
#     axis[2,0].loglog(f, Pxx_den, label = s)
for s in wendlands:
    windowFunction = getWindowFunction(s)
    windowFn = lambda x: windowFunction(torch.abs(torch.tensor(x))).numpy()
    
    axis[0,1].plot(xs, windowFn(xs), label = '%s' % s)

#     fs = numSamples/2
#     fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
#     x = windowFn(xs)
#     y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
#     axis[1,1].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = s)

#     f, Pxx_den = scipy.signal.welch(windowFn(x), fs, nperseg=len(x)//32)
#     axis[2,1].loglog(f, Pxx_den, label = s)
for s in misc:
    windowFunction = getWindowFunction(s)
    windowFn = lambda x: windowFunction(torch.abs(torch.tensor(x))).numpy()
    
    axis[0,2].plot(xs, windowFn(xs), label = '%s' % s)

#     fs = numSamples/2
#     fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
#     x = windowFn(xs)
#     y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
#     axis[1,2].semilogx(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = s)

#     f, Pxx_den = scipy.signal.welch(windowFn(x), fs, nperseg=len(x)//32)
#     axis[2,2].loglog(f, Pxx_den, label = s)
    
axis[0,0].legend()
# axis[1,0].legend()
# axis[2,0].set_xlabel('frequency [Hz]')
# axis[2,0].set_ylabel('PSD [V**2/Hz]')
# axis[2,0].legend()
axis[0,1].legend()
# axis[1,1].legend()
# axis[2,1].set_xlabel('frequency [Hz]')
# axis[2,1].set_ylabel('PSD [V**2/Hz]')
# axis[2,1].legend()
axis[0,2].legend()
# axis[1,2].legend()
# axis[2,2].set_xlabel('frequency [Hz]')
# axis[2,2].set_ylabel('PSD [V**2/Hz]')
# axis[2,2].legend()


fig.tight_layout()

# Plot sampling

dx = 2 / (n-1)
sampled = sample(n, 1024, method = 'normal', dxScale = 8, dx = 1/3, clamped = True, seed = None)

fig, axis = plt.subplots(2, 1, figsize=(12,4), height_ratios = [4,1], sharex = True, sharey = False, squeeze = False)
for s in range(len(sampled)):
    axis[0,0].scatter(sampled[s],s * torch.ones(sampled[s].shape),ls='-',color=mpl.colormaps['PuRd'](1 / (len(sampled) - 1)* s), label = 'sample %d' % i, s = 1)
    
for i in range(n):
    axis[0,0].axvline(-1 + dx * i, ls = '--', c = 'white', alpha = 0.5)

# axis[1,0].hist(torch.vstack(sampled))
sns.histplot(data = torch.vstack(sampled).flatten(), bins = 128, ax = axis[1,0],kde = True, kde_kws={'bw_adjust':0.1})
fig.tight_layout()
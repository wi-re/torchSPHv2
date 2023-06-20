
import tensorflow as tf
import open3d.ml.tf as ml3d
from abc import ABC, abstractmethod
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], 'GPU')

constant = 7 / np.pi

@tf.function
def wendland(r):
    q = tf.sqrt(r) / support
    
    b1 = tf.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * constant / support**2    

# w = wendland(dista)
# print('w', w)


n = 32
weights = tf.random.normal((n,1), dtype='float32')

@tf.function
def lookupWeights(r, weights):
#     print(r)
    n = weights.shape[0]
    ld = tf.sqrt(r) / support * (n-1)
    left = tf.floor(ld)
#     w = tf.cast(ld - left, dtype= 'float64')
    w = ld -left
#     w = tf.expand_dims(w,axis=1)
#     print(w)
    left = tf.cast(left, dtype='int32')

    right = tf.cast(tf.math.ceil(ld), dtype='int32')
    wl = tf.gather(weights, left)
    wr = tf.gather(weights, right)
    weight = w * wl + (1. - w) * wr
    return weight
    
# lookupWeights(d32, weights[:,0])

@tf.function
def clenshaw(r, weights):
    bk1 = tf.zeros_like(r)
    bk2 = tf.zeros_like(r)
    for k in range(weights.shape[0] - 1, 0, -1):
        bk = weights[k] + 2. * r * bk1 - bk2
        bk2 = bk1
        bk1 = bk
    return weights[0] + 1. * r * bk1 - bk2
    
# k = clenshaw(tf_distances.values, weights)
# # print(k.shape)
# # print(weights.shape)
# lin = clenshaw(tf.linspace(0.,1.,256), weights)

# fig, axis = plt.subplots(1, 1, figsize=(6,6), sharex = False, sharey = False, squeeze = False)
# axis[0,0].plot(np.linspace(0.,1.,256), lin)
# # axis[0,0].set_yscale('log')

@tf.function
def antiDerivative(coeffs):
    n = coeffs.shape[0]
    ak = tf.concat([coeffs, tf.zeros((2), dtype='float32')], axis =0)
    rk = tf.range(2, n+1, dtype='float32')
    bk = 0.5 * (ak[1:n] - ak[3:n+2])/ rk
    bk1 = ak[0] - .5 * ak[2]
    bks = bk1 + tf.reduce_sum(bk[1::2]) - tf.reduce_sum(bk[::2])
    return tf.concat([tf.expand_dims(bks,axis =0), tf.expand_dims(bk1,axis =0), bk] , axis = 0)
    
@tf.function
def integrateCheb(coeffs, a, b):
    antiDeriv = antiDerivative(coeffs)
    integral = clenshaw(tf.constant([a,b], dtype='float32'), antiDeriv)
    return integral[1] - integral[0]

# tfInt = antiDerivative(weights[:,0])

# tfIntegral2 = clenshaw(tf.constant([-1.,1.]), tfInt)
# # print(tfInt)
# print(tfIntegral2)

from contextlib import contextmanager
from time import time

@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start

    print(f"{description}: {ellapsed_time * 1000.}ms")

@tf.function
def raggedWeights(tf_distances, weights, row_splits):
    kernelValues = area * lookupWeights(tf_distances.values, weights)
    rag = tf.RaggedTensor.from_row_splits(kernelValues, row_splits)
    return rag

# rag = tf.reduce_sum(raggedWeights(tf_distances, weights[:,0], row_splits), axis = 1)
# print(rag)

@tf.function
def raggedChebyshev(tf_distances, weights, row_splits):
    kernelValues = area * clenshaw(tf_distances.values * 2. - 1., weights)
    rag = tf.RaggedTensor.from_row_splits(kernelValues, row_splits)
    return rag

# rag = tf.reduce_sum(raggedChebyshev(tf.sqrt(tf_distances)/support, weights, row_splits),axis=1)


# class linearInterpolationLayer(tf.keras.layers.Layer):
#     def __init__(self, num_weights):
#         super(linearInterpolationLayer, self).__init__()
        
#         self.num_weights = num_weights
        
#         self.kernel = self.add_weight("kernel", initializer='random_normal',
#                                   shape=[self.num_weights])

#     def build(self, input_shape):
#         return
# #         print(input_shape[0])

#     def call(self, inputs):
#         positions, features, neighborhood = inputs
#         neighbors_index, row_splits, distances = neighborhood
#         k = tf.expand_dims(self.kernel, axis=1)
# #         print(k, self.kernel)
#         r32 = tf.reduce_sum(raggedWeights(distances, self.kernel, row_splits),axis=1)
#         return r32
        
# #         print(self.kernel)
#         return rho(positions, neighbors_index, row_splits, distances, features, self.kernel)

        
# layer = linearInterpolationLayer(16)

# out = layer([p32,f32,[neighbors_index, row_splits, tf_distances]])
# # print(out)
# r32 = tf.cast(data['rho'], dtype = tf.float32)
# # print(r32)
# # print(out - r32)

# class chebyshevLayer(tf.keras.layers.Layer):
#     def __init__(self, num_weights):
#         super(chebyshevLayer, self).__init__()
        
#         self.num_weights = num_weights
        
#         self.kernel = self.add_weight("kernel", initializer='random_normal',
#                                   shape=[self.num_weights])

#     def build(self, input_shape):
#         return
# #         print(input_shape[0])

#     def call(self, inputs):
#         positions, features, neighborhood = inputs
#         neighbors_index, row_splits, distances = neighborhood
        
# #         print('Distances: ', distances.values, (tf.sqrt(distances)/support).values)
# #         print('Kernel: ', self.kernel)
# #         print('Splits: ', row_splits)
#         dist = tf.sqrt(distances)/support
# #         print(dist)
#         r32 = tf.reduce_sum(raggedChebyshev(dist, self.kernel, row_splits),axis=1)
#         integral = 2. * np.pi * integrateCheb(self.kernel, -1, 1)
#         return r32 #/ integral
        
# #         print(self.kernel)
#         return rho(positions, neighbors_index, row_splits, distances, features, self.kernel)

        
# layer = chebyshevLayer(32)

# out = layer([p32,f32,[neighbors_index, row_splits, tf_distances]])
# # print('out:', out)
# # r32 = tf.cast(data['rho'], dtype = tf.float32)
# # print(r32)
# # print(out - r32)

# def prepData(index):
#     positions = np.copy(dataSet[index]['positions'])
#     positions[:,2] = 0.
#     positions = tf.constant(positions, dtype=tf.float32)
#     features = tf.constant(tf.Variable(dataSet[index]['areas']), dtype=tf.float32)
#     features = tf.expand_dims(features,axis=1)
    
#     gt = tf.cast(tf.constant(dataSet[index]['rho']), dtype=tf.float32)
#     return [positions, features, gt]

# class customLayer(tf.keras.Model):
#     def __init__(self, num_weights):
#         super().__init__(name=type(self).__name__)
# #         self.layer = linearInterpolationLayer(num_weights)
#         self.layer = chebyshevLayer(num_weights)
#         self.radiusSearch = ml3d.layers.FixedRadiusSearch(return_distances=True)
        
#     def call(self, inputs):        
#         positions, features = inputs
        
#         neighbors_index, row_splits, distances = self.radiusSearch(positions, positions, support)
#         tf_distances = tf.RaggedTensor.from_row_splits(distances, row_splits)
        
#         ans = features
        
# #         print(positions, features, tf_distances)
# #         print('....')
#         ans = self.layer([positions, features, [neighbors_index, row_splits, tf_distances]])
        
#         return ans
#     def init(self):
#         p32, f32, g32 = prepData(0)
#         _ = self.__call__([p32,f32])
        
# model = customLayer(8)
# model.init()

# trainable_count = sum(
#     tf.keras.backend.count_params(w) for w in model.trainable_weights)
# non_trainable_count = sum(
#     tf.keras.backend.count_params(w)
#     for w in model.non_trainable_weights)
# print("###################################")
# print("Parameter count '{}':".format(model.name))
# print(" Total params: {:,}".format(trainable_count +
#                                       non_trainable_count))
# print(" Trainable params: {:,}".format(trainable_count))
# print(" Non-trainable params: {:,}".format(non_trainable_count))
# print("-----------------------------------")

# learning_rate = 1e-2
# optimizer_sv = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-6)

# epoch_losses = []
# epoch_loss = []
# epoch_weights = []

# epochs = 1024
# breadth = 4
# t = trange(epochs)
# with tf.device('/device:CPU:0'):
#     for i in t:
#         with tf.GradientTape() as tape:   
#             losses = []
#             l2_sum = 0.
#             for i in range(breadth):
#                 p32, f32, gt = prepData(i)
# #                 print('Ground Truth: ', gt)
#                 pred = model([p32, f32])
# #                 print('Prediction: ', pred)

#                 loss = (pred - gt)**2. / 2.
# #                 print('Loss: ', loss)
#                 mse = tf.reduce_sum(loss) / loss.shape[0]
#                 losses.append(mse)

#             l2_sum = tf.reduce_sum(losses) / breadth
#             tfgrads = tape.gradient(l2_sum, model.trainable_weights)
# #             print('gradients:', tfgrads)
# #             print('gradients:', tape.gradient(l2_sum, f32))
#             epoch_losses.append(losses)
#             epoch_loss.append(l2_sum)
#             epoch_weights.append(np.copy(model.trainable_weights[0].numpy()))

#             optimizer_sv.apply_gradients(zip(tfgrads, model.trainable_weights))
#         t.set_postfix_str('Epoch Loss: %g' % l2_sum)
# epoch_losses =np.array(epoch_losses)
# epoch_loss =np.array(epoch_loss)

# epoch_losses =np.array(epoch_losses)
# epoch_loss =np.array(epoch_loss)
# fig, axis = plt.subplot_mosaic(
#     """
#     AAA
#     BBD
#     EEC
#     """, 
#     constrained_layout=True, figsize=(9,9),
#     gridspec_kw={
#         "height_ratios": [1, 2 ,1],
#         "width_ratios": [1, 1, 1],
#     },)

# axis['A'].plot(epoch_loss)
# axis['A'].set_yscale('log')
# axis['A'].grid(which='both')

# ls = tf.linspace(0.,1.,256)
# axis['B'].plot(ls, clenshaw(ls * 2 - 1, epoch_weights[-1]))
# axis['B'].plot(ls, wendland(ls**2 * support **2),ls='--')
# axis['B'].grid(which='both')
# # axis['B'].axhline(0,color='black')

# n = len(epoch_weights)
# mat = np.zeros((n, 256))
# for i in range(n):
#     mat[i,:] = clenshaw(ls, epoch_weights[i])
# axs = axis['D']
# im = axs.imshow(mat, interpolation='nearest', aspect='auto')
# ax1_divider = make_axes_locatable(axs)
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# plt.colorbar(im, cax=cax1)


# axs = axis['C']
# im = axs.imshow(epoch_weights, interpolation='nearest', aspect='auto')
# ax1_divider = make_axes_locatable(axs)
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# plt.colorbar(im, cax=cax1)

# axs = axis['E']
# im = axs.imshow(epoch_losses.transpose(), interpolation='nearest', aspect='auto', norm = LogNorm())
# ax1_divider = make_axes_locatable(axs)
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# plt.colorbar(im, cax=cax1)

# fig.tight_layout()


class densityNetwork(tf.keras.Model):
    def __init__(self, architecture):
        super().__init__(name=type(self).__name__)
        
        self.convs = []
        for layer in architecture:
            filters = layer['filters'] if 'filters' in layer else 1
            size = layer['size'] if 'size' in layer else 5
            bias = layer['bias'] if 'bias' in layer else False
            dense_for_center = layer['dense_for_center'] if 'dense_for_center' in layer else False
            normalize = layer['normalize'] if 'normalize' in layer else False
            ignoreQuery = layer['ignore_query'] if 'ignore_query' in layer else False
            activation = layer['activation'] if 'ignore_query' in layer else None
            l = ml3d.layers.ContinuousConv(
                filters=filters, 
                kernel_size=[1,size,size], 
                use_bias=bias,
                kernel_initializer = tf.keras.initializers.Constant(0.01),
                coordinate_mapping= 'ball_to_cube_volume_preserving',
                radius_search_ignore_query_points = ignoreQuery,
                use_dense_layer_for_center = dense_for_center,
                normalize=normalize,
                activation = activation
            )
            self.convs.append(l)
        
        
    def call(self, inputs):
        positions = inputs[0]
        features = inputs[1]
        output = inputs[2]
        
        ans = features
        
        for layer in self.convs:
            ans = layer(ans, positions, positions, extents  = float(2. * support))
        return ans
    
    def init(self):
        dummyP = tf.random.normal([32,3])
        dummyD = tf.random.normal([32,1])
        outputP = tf.convert_to_tensor([np.array([0.,0.,0.], dtype = 'float32')])
        _ = self.__call__([tf.convert_to_tensor(dummyP), tf.convert_to_tensor(dummyD), outputP])

# arch = [
#     {'filters': 2, 'size': 8, 'activation': None, 'ignore_query': False},
#     {'filters': 1, 'size': 8, 'activation': None, 'ignore_query': False}
# ]

# modelCConv = densityNetwork(arch)
# modelCConv.init()

# trainable_count = sum(
#     tf.keras.backend.count_params(w) for w in modelCConv.trainable_weights)
# non_trainable_count = sum(
#     tf.keras.backend.count_params(w)
#     for w in modelCConv.non_trainable_weights)
# print("###################################")
# print("Parameter count '{}':".format(modelCConv.name))
# print(" Total params: {:,}".format(trainable_count +
#                                       non_trainable_count))
# print(" Trainable params: {:,}".format(trainable_count))
# print(" Non-trainable params: {:,}".format(non_trainable_count))
# print("-----------------------------------")

# optimizer_sv = tf.keras.optimizers.Adam(learning_rate=1e-2, epsilon=1e-6)

# breadth = 4
# losses = []
# lossesArray = []
# def trainIteration():
#     with tf.GradientTape() as tape:
#         localLoss = []
#         localLosses = []
#         n = breadth
#         for data in range(n):
#             positions, features, gt = prepData(data)
            
#             pred = modelCConv([positions, features, positions])[:,0]
            
            
            
#             loss = tf.reduce_sum(((pred - gt)**2)/2) / pred.shape[0]
# #             print(pred)
# #             print(gt)
# #             print(pred - gt)
            
#             localLosses.append(loss.numpy())
#             localLoss.append(loss)
            
#         total_loss = tf.reduce_sum(localLoss) / n
# #         print('total: ', total_loss, '\nlocal:', localLoss)
#         loss = total_loss
# #         if loss != loss: 
# #             return loss.numpy(), localLosses
        
#         grads = tape.gradient(total_loss, modelCConv.trainable_weights)
# #         clipped_gradients = [tf.clip_by_norm(g, 1.) for g in grads
# #         f = tf.squeeze(grads)
# #         print(f)
# #         inan = tf.math.is_nan(f)
# #         if tf.math.reduce_any(inan):
# #             return loss.numpy(), localLosses

#         optimizer_sv.apply_gradients(zip(grads, modelCConv.trainable_weights))
#         return loss.numpy(), localLosses
    
# epochs = 1024

# t = trange(epochs)
# for i in t:
#     loss, ll = trainIteration()
#     losses.append(loss)
#     lossesArray.append(ll)
#     t.set_postfix_str('Epoch Loss: %g' % loss)



# # for i in tqdm(range(256)):
# #     losses.append(trainIteration())
# losses = np.array([losses]).flatten()
# lossesArray = np.array(lossesArray)

# fig, axis = plt.subplots(1, 3, figsize=(9,6), sharex = False, sharey = False, squeeze = False)
# axis[0,0].plot(losses)
# axis[0,0].set_yscale('log')

# axs = axis[0,1]
# im = axs.imshow(modelCConv.convs[0].kernel[0,:,:,0,0], interpolation='bilinear', aspect='equal')
# ax1_divider = make_axes_locatable(axs)
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# plt.colorbar(im, cax=cax1)

# axs = axis[0,2]
# im = axs.imshow(lossesArray, interpolation='nearest', aspect='auto', norm = LogNorm())
# ax1_divider = make_axes_locatable(axs)
# cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# plt.colorbar(im, cax=cax1)

# fig.tight_layout()

import matplotlib.gridspec as gridspec

@tf.function
def getSpacing(n, periodic = False):
    if n == 1:
        return 2.
    else:
        return 2. / n if periodic else 2./(n-1)

@tf.function
def getDistances(n, x, periodic = False):
    if periodic:
        spacing = getSpacing(n, True)
        offset = -1 + spacing / 2.
        
#         tx = tf.constant(x, dtype='float32')
        centroids = tf.linspace(-1.,1.,n+1)[:n]
        ra = tf.expand_dims(x,axis=0) - tf.expand_dims(centroids, axis=1)
        rb = tf.expand_dims(x,axis=0) - tf.expand_dims(centroids, axis=1) - 2.
        rc = tf.expand_dims(x,axis=0) - tf.expand_dims(centroids, axis=1) + 2.
        return tf.minimum(tf.minimum(tf.abs(ra)/spacing, tf.abs(rb)/spacing), tf.abs(rc)/spacing)
        return tf.abs(ra) / spacing, tf.abs(rb) / spacing, tf.abs(rc) / spacing
        
    spacing = getSpacing(n, False)
    
    centroids = tf.linspace(-1.,1.,n) if n > 1 else tf.constant([0.])
#     tx = tf.constant(x, dtype='float32')
    r = tf.expand_dims(x,axis=0) - tf.expand_dims(centroids, axis=1)
    return tf.abs(r)  / spacing

@tf.function
def evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = False):    
    k = int(epsilon)
    r = getDistances(n, x, periodic)    
    
    cpow = lambda x, p: tf.maximum(x, tf.zeros_like(r))**p
    
    funLib = {
        'linear': lambda r:  tf.clip_by_value(1. - r / epsilon,0,1),
        'gaussian': lambda r:  tf.exp(-(epsilon * r)**2),
        'multiquadric': lambda r: tf.sqrt(1. + (epsilon * r) **2),
        'inverse_quadric': lambda r: 1. / ( 1 + (epsilon * r) **2),
        'inverse_multiquadric': lambda r: 1. / tf.sqrt(1. + (epsilon * r) **2),
        'polyharmonic': lambda r: tf.pow(r, k) if k % 2 == 1 else tf.pow(r,k-1) * tf.math.log(tf.pow(r,r)),
        'bump': lambda r: tf.where(r < 1./epsilon, tf.exp(-1./(1- (epsilon * r)**2)), tf.zeros_like(r)),
        'cubic_spline': lambda r: cpow(1-r/epsilon,3) - 4. * cpow(1/2-r/epsilon,3),
        'quartic_spline': lambda r: cpow(1-r/epsilon,4) - 5 * cpow(3/5-r/epsilon,4) + 10 * cpow(1/5-r/epsilon,4),
        'quintic_spline': lambda r: cpow(1-r/epsilon,5) - 6 * cpow(2/3-r/epsilon,5) + 15 * cpow(1/3-r/epsilon,5),
        'wendland2': lambda r: cpow(1 - r/epsilon, 4) * (1 + 4 * r/epsilon),
        'wendland4': lambda r: cpow(1 - r/epsilon, 6) * (1 + 6 * r/epsilon + 35/3 * (r/epsilon)**2),
        'wendland6': lambda r: cpow(1 - r/epsilon, 8) * (1 + 8 * r/epsilon + 25 * (r/epsilon) **2 + 32 * (r/epsilon)**3),
        'poly6': lambda r: cpow(1 - (r/epsilon)**2, 3),
        'spiky': lambda r: cpow(1 - r/epsilon, 3),
        'square': lambda r: tf.where(r <= epsilon, tf.ones_like(r), tf.zeros_like(r))
    }
    rbf = funLib[which]
    
#     if periodic:
#         return tf.maximum(rbf(r[0]), tf.maximum(rbf(r[1]), rbf(r[2])))
        # return tf.clip_by_value(tf.maximum(rbf(r[0]), tf.maximum(rbf(r[1]), rbf(r[2]))),0,1)   
    return rbf(r)#tf.clip_by_value(rbf(r),0,1)
    
@tf.function
def cheb(n, x):
    if n == 0:
        return tf.ones_like(x)
    if n == 1:
        return x
    return 2. * x * cheb(n-1, x) - cheb(n-2, x)
@tf.function
def evalChebSeries(n,x):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(tf.ones_like(x))
        elif i == 1:
            cs.append(x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return tf.stack(cs)
sqrt_pi_1 = 1. / np.sqrt(np.pi)
@tf.function
def fourier(n, x):
    if n == 0:
        return tf.ones_like(x) / tf.sqrt(2. * np.pi)
    elif n % 2 == 0:
        return tf.cos((n // 2 + 1) * x) * sqrt_pi_1
    return tf.sin((n // 2 + 1) * x) * sqrt_pi_1
@tf.function
def evalFourierSeries(n, x):
    fs = []
    for i in range(n):
        fs.append(fourier(i, x))
    return tf.stack(fs)

@tf.function
def evalBasisFunction(n, x, which = 'chebyshev', periodic = False):   
    s = which.split()    
    
    if s[0] == 'chebyshev':
        return evalChebSeries(n, x)
    if s[0] == 'fourier':
        return evalFourierSeries(n, x * np.pi)
    if s[0] == 'linear':
        return evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = periodic)        
    if s[0] == 'rbf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic)
    
def plotBasisFunction(n, m, which = 'chebyshev', mapping = None):
    x = np.linspace(-1,1,m, dtype='float32')

    periodic = True if mapping == 'polar' else False
    
    b = evalBasisFunction(n, tf.constant(x), which = which, periodic = periodic)
    
    
    fig, axis = plt.subplots(1,1, figsize=(7,6), sharex = False, sharey = False, squeeze = False, subplot_kw={'projection': 'polar'} if mapping == 'polar' else None)
    
    
    for i in range(n):
        axis[0,0].plot(x * np.pi if mapping == 'polar' else x, b[i], label = '%d'%i)
    axis[0,0].legend(bbox_to_anchor=(1.2, 1.0))
    if mapping != 'polar':
        axis[0,0].axis('equal')
    fig.tight_layout()

def reshape(M, na, nr):
    return M.numpy().reshape(na,nr)

def plotBasisFunctions(n, m, weights = None, a = 'chebyshev', b = 'fourier', polar = False, na = 32, nr = 32, minPlot = False, norm = False):
    W = tf.constant(np.random.uniform(-1,1,size = (n,m)), dtype='float32') if weights is None else tf.constant(weights,dtype='float32')
    if polar:
        thetas = np.linspace(-np.pi, np.pi, na, dtype='float32')
        radii = np.linspace(0,1,nr, dtype='float32')

        tt, rr = np.meshgrid(thetas, radii)

        ttf = tt.flatten()
        rrf = rr.flatten()


        c = evalBasisFunction(n, tf.constant(rrf * 2. - 1.), which = a)
        f = evalBasisFunction(m, tf.constant(ttf / np.pi), which = b, periodic = True)
        
        bc = evalBasisFunction(n, tf.constant(radii * 2. - 1.), which = a)
        bf = evalBasisFunction(m, tf.constant(thetas / np.pi), which = b, periodic = True)

        M = tf.einsum("un,vn->nuv", c, f)
#         print(M.shape)
#         print(tf.reduce_sum(M, axis = [1,2]).shape)
        if norm:
            M = M / tf.reduce_sum(M, axis = [1,2])[:,None,None]
#         W = tf.constant(weights,dtype='float32')
        O = M * W[None,:]
        if minPlot:
            fig, axis = plt.subplots(2, 2, figsize=(8,8), sharex = False, sharey = False, squeeze = False, subplot_kw={'projection': 'polar'})
            
            gs = axis[0, 0].get_gridspec()
            axis[0,0].remove()
            axbig = fig.add_subplot(gs[0, 0])
            axbig.imshow(W.numpy(), interpolation='nearest', aspect='auto')
            axbig.set_xticks([])
            axbig.set_yticks([])
            axbig.set_title('Weights')
            
            a_s = a.split()
            a_s = a_s[1] if len(a_s) > 1 else a
            b_s = b.split()
            b_s = b_s[1] if len(b_s) > 1 else b

            axis[1,0].remove()
            axbig = fig.add_subplot(gs[1, 0])
            axbig.plot(radii * 2. -1., bc.numpy().transpose())
            axbig.set_xticks([])
            axbig.set_yticks([])
            axbig.set_title('[%s]' %(a_s))

            axis[0, 1].plot(thetas, bf.numpy().transpose())
            axis[0, 1].set_xticks([])
            axis[0, 1].set_yticks([])
            axis[0, 1].set_title('[%s]' %(b_s))

            axis[1,1].grid(False)
            #        axis[i,j].imshow(d[i,:,j,:])
            axis[1,1].pcolormesh(thetas, radii, \
                                     reshape(tf.reduce_sum(O, axis =[1,2]),na,nr)\
                                     , edgecolors='face',shading='gouraud')
            axis[1,1].set_xticks([])
            axis[1,1].set_yticks([])
            axis[1,1].set_title('[:,:]')


            fig.tight_layout()
            
        else:
            fig, axis = plt.subplots(n+2, m+2, figsize=((n+2)*2,(m+2)*2), sharex = False, sharey = False, squeeze = False, subplot_kw={'projection': 'polar'})
    #         axis[0,0].remove()
            gs = axis[0, 0].get_gridspec()
            axis[0,0].remove()
            axbig = fig.add_subplot(gs[0, 0])
            axbig.imshow(W.numpy(), interpolation='nearest', aspect='auto')
            axbig.set_xticks([])
            axbig.set_yticks([])
            axbig.set_title('Weights')

    #         axis[0,m+1].remove()
    #         axis[n+1,0].remove()

            a_s = a.split()
            a_s = a_s[1] if len(a_s) > 1 else a
            b_s = b.split()
            b_s = b_s[1] if len(b_s) > 1 else b

            for i in range(n):           
                gs = axis[i+1, 0].get_gridspec()
                axis[i+1,0].remove()
                axbig = fig.add_subplot(gs[i+1, 0])
                axbig.plot(radii * 2. -1., bc[i])
                axbig.set_xticks([])
                axbig.set_yticks([])
                axbig.set_title('[%s %d]' %(a_s,i))
            axis[n+1,0].remove()
            axbig = fig.add_subplot(gs[n+1, 0])
            axbig.plot(radii * 2. -1., bc.numpy().transpose())
            axbig.set_xticks([])
            axbig.set_yticks([])
            axbig.set_title('[%s]' %(a_s))


            for j in range(m):
                axis[0,j+1].plot(thetas, bf[j])
                axis[0, j + 1].set_xticks([])
                axis[0, j + 1].set_yticks([])
                axis[0, j + 1].set_title('[%s %d]' %(b_s,j))

            axis[0,m+1].plot(thetas, bf.numpy().transpose())
            axis[0, m + 1].set_xticks([])
            axis[0, m + 1].set_yticks([])
            axis[0, m + 1].set_title('[%s]' %(b_s))

    #         return
            for i in range(n):
                for j in range(m):
                    axis[i+1,j+1].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
                    axis[i+1,j+1].pcolormesh(thetas, radii, \
                                         reshape(M[:,i,j], na, nr)\
                                         , edgecolors='face',shading='gouraud')
                    axis[i+1,j+1].set_xticks([])
                    axis[i+1,j+1].set_yticks([])
                    axis[i+1,j+1].set_title('[%d,%d]' %(i,j))


            for i in range(n):
                axis[i+1,m+1].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
                axis[i+1,m+1].pcolormesh(thetas, radii, \
                                     reshape(tf.reduce_sum(O[:,i,:], axis=1), na, nr)\
                                         , edgecolors='face',shading='gouraud')
                axis[i+1,m+1].set_xticks([])
                axis[i+1,m+1].set_yticks([])
                axis[i+1,m+1].set_title('[%d,:]' %(i))

            for j in range(m):
                axis[n+1,j+1].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
                axis[n+1,j+1].pcolormesh(thetas, radii, \
                                     reshape(tf.reduce_sum(O[:,:,j], axis=1), na, nr)\
                                         , edgecolors='face',shading='gouraud')
                axis[n+1,j+1].set_xticks([])
                axis[n+1,j+1].set_yticks([])
                axis[n+1,j+1].set_title('[:,%d]' %(j))

            axis[n+1,m+1].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
            axis[n+1,m+1].pcolormesh(thetas, radii, \
                                     reshape(tf.reduce_sum(O, axis =[1,2]),na,nr)\
                                     , edgecolors='face',shading='gouraud')
            axis[n+1,m+1].set_xticks([])
            axis[n+1,m+1].set_yticks([])
            axis[n+1,m+1].set_title('[:,:]')


            fig.tight_layout()
    else:
        x = np.linspace(-1,1,nr, dtype='float32')
        y = np.linspace(-1,1,na, dtype='float32')

        xx, yy = np.meshgrid(x, y)

        xxf = xx.flatten()
        yyf = yy.flatten()

        c = evalBasisFunction(n, tf.constant(xxf), which = a)
        f = evalBasisFunction(m, tf.constant(yyf), which = b)
        
        bc = evalBasisFunction(n, tf.constant(x), which = a)
        bf = evalBasisFunction(m, tf.constant(y), which = b)

        M = tf.einsum("un,vn->nuv", c, f)
        if norm:
            M = M / tf.reduce_sum(M, axis = [1,2])[:,None,None]
#         W = tf.constant(weights,dtype='float32')
        O = M * W[None,:]

        if minPlot:
            fig, axis = plt.subplots(2,2, figsize=(8,8), sharex = False, sharey = False, squeeze = False)
    #         axis[0,0].remove()
            axis[0,0].imshow(W.numpy().transpose(), interpolation='nearest', aspect='auto')
            axis[0,0].set_xticks([])
            axis[0,0].set_yticks([])
            axis[0,0].set_title('Weights')

    #         axis[0,m+1].remove()
    #         axis[n+1,0].remove()

            a_s = a.split()
            a_s = a_s[1] if len(a_s) > 1 else a
            b_s = b.split()
            b_s = b_s[1] if len(b_s) > 1 else b

            axis[1,0].plot(x, bf.numpy().transpose())
            axis[1,0].set_xticks([])
            axis[1,0].set_yticks([])
            axis[1,0].set_title('[%s]' %(a_s))
            axis[0, 1].plot(y, bf.numpy().transpose())
            axis[0, 1].set_xticks([])
            axis[0, 1].set_yticks([])
            axis[0, 1].set_title('[%s]' %(b_s))
            
            axis[1, 1].imshow(reshape(tf.reduce_sum(O, axis =[1,2]),na,nr), extent=(-1,1,-1,1))
            axis[1, 1].set_xticks([])
            axis[1, 1].set_yticks([])
            axis[1, 1].set_title('[:,:]')
            circle = plt.Circle((0, 0), 1., color='orange', fill = False, ls= '--')
            axis[1, 1].add_patch(circle)


            fig.tight_layout()            
        else:
            fig, axis = plt.subplots(n+2, m+2, figsize=((n+2)*2,(m+2)*2), sharex = False, sharey = False, squeeze = False)
    #         axis[0,0].remove()
            axis[0,0].imshow(W.numpy().transpose(), interpolation='nearest', aspect='auto')
            axis[0,0].set_xticks([])
            axis[0,0].set_yticks([])
            axis[0,0].set_title('Weights')

    #         axis[0,m+1].remove()
    #         axis[n+1,0].remove()

            a_s = a.split()
            a_s = a_s[1] if len(a_s) > 1 else a
            b_s = b.split()
            b_s = b_s[1] if len(b_s) > 1 else b

            for i in range(n):
                axis[i+1,0].plot(x, bc[i])
                axis[i + 1,0].set_xticks([])
                axis[i + 1,0].set_yticks([])
                axis[i + 1,0].set_title('[%s %d]' %(a_s,i))
            axis[n + 1,0].plot(x, bf.numpy().transpose())
            axis[n + 1,0].set_xticks([])
            axis[n + 1,0].set_yticks([])
            axis[n + 1,0].set_title('[%s]' %(a_s))
            for j in range(m):
                axis[0, j + 1].plot(y, bf[j])
                axis[0, j + 1].set_xticks([])
                axis[0, j + 1].set_yticks([])
                axis[0, j + 1].set_title('[%s %d]' %(b_s,j))
            axis[0,m+1].plot(y, bf.numpy().transpose())
            axis[0, m + 1].set_xticks([])
            axis[0, m + 1].set_yticks([])
            axis[0, m + 1].set_title('[%s]' %(b_s))

    #         return
            for i in range(n):
                for j in range(m):
                    axis[i + 1,j + 1].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
                    axis[i + 1,j + 1].imshow(reshape(M[:,i,j], na,nr), extent=(-1,1,-1,1))
                    axis[i + 1,j + 1].set_xticks([])
                    axis[i + 1,j + 1].set_yticks([])
                    axis[i + 1,j + 1].set_title('[%d,%d]' %(i,j))
                    circle = plt.Circle((0, 0), 1., color='orange', fill = False, ls= '--')
                    axis[i + 1,j + 1].add_patch(circle)


            for i in range(n):
                axis[i + 1,m].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
                axis[i + 1,m + 1].imshow(reshape(tf.reduce_sum(O[:,i,:], axis=1), na, nr), extent=(-1,1,-1,1))
                axis[i + 1,m + 1].set_xticks([])
                axis[i + 1,m + 1].set_yticks([])
                axis[i + 1,m + 1].set_title('[%d,:]' %(i))
                circle = plt.Circle((0, 0), 1., color='orange', fill = False, ls= '--')
                axis[i + 1,m + 1].add_patch(circle)

            for j in range(m):
                axis[n,j].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
                axis[n + 1,j + 1].imshow(reshape(tf.reduce_sum(O[:,:,j], axis=1), na, nr), extent=(-1,1,-1,1))
                axis[n + 1,j + 1].set_xticks([])
                axis[n + 1,j + 1].set_yticks([])
                axis[n + 1,j + 1].set_title('[:,%d]' %(j))
                circle = plt.Circle((0, 0), 1., color='orange', fill = False, ls= '--')
                axis[n + 1,j + 1].add_patch(circle)

            axis[n,m].grid(False)
            #         axis[i,j].imshow(d[i,:,j,:])
            axis[n + 1,m + 1].imshow(reshape(tf.reduce_sum(O, axis =[1,2]),na,nr), extent=(-1,1,-1,1))
            axis[n + 1,m + 1].set_xticks([])
            axis[n + 1,m + 1].set_yticks([])
            axis[n + 1,m + 1].set_title('[:,:]')
            circle = plt.Circle((0, 0), 1., color='orange', fill = False, ls= '--')
            axis[n + 1,m + 1].add_patch(circle)


            fig.tight_layout()
    return W

def plotKernel(W, i, j, n, m, a, b, axis, c, l, polar = False, na = 32, nr = 32):
#     axis[c,l].grid(False)
#     axis[c,l].set_xticks([])
#     axis[c,l].set_yticks([])

    
    if polar:
#         ax1_divider = make_axes_locatable(axis[c,l])
#         cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        gs = axis[c,l].get_gridspec()
        
        axis[c,l].remove()
        axbig = fig.add_subplot(gs[c,l], projection = 'polar')
        axbig.grid(False)
        axbig.set_xticks([])
        axbig.set_yticks([])
    
        thetas = np.linspace(-np.pi, np.pi, na, dtype='float32')
        radii = np.linspace(0,1,nr, dtype='float32')

        tt, rr = np.meshgrid(thetas, radii)

        ttf = tt.flatten()
        rrf = rr.flatten()
        
        c = evalBasisFunction(n, tf.constant(rrf * 2. - 1.), which = a)
        f = evalBasisFunction(m, tf.constant(ttf / np.pi), which = b, periodic = True)
        M = tf.einsum("un,vn->nuv", c, f)
        O = tf.einsum("unm,ijnm->uij", M, W)
        

        pm = axbig.pcolormesh(thetas, radii, \
                                 reshape(O[:,i,j],na,nr)\
                                 , edgecolors='face')
#         plt.colorbar(pm, cax=cax1)
        return axbig
        
    else:
        x = np.linspace(-1, 1, na, dtype='float32')
        y = np.linspace(-1, 1, nr, dtype='float32')

        xx, yy = np.meshgrid(x, y)

        xf = xx.flatten()
        yf = yy.flatten()
        
        cb = evalBasisFunction(n, tf.constant(xf), which = a)
        fb = evalBasisFunction(m, tf.constant(yf), which = b, periodic = False)
        M = tf.einsum("un,vn->nuv", cb, fb)
        O = tf.einsum("unm,ijnm->uij", M, W)
        res = reshape(O[:,i,j],na,nr)
        im = axis[c,l].imshow(res)
        return im
    return


class rbfLayer(tf.keras.layers.Layer):
    def __init__(self, n, m, a, b, periodic = True, normalization = False, fin = 1, fout = 1):
        super(rbfLayer, self).__init__()
        
        self.n = n
        self.m = m
        
        self.rbf_a = a
        self.rbf_b = b
        self.periodic = periodic
        self.norm = normalization
        
        self.fin = fin
        self.fout = fout
        
        print('Creating RBF Layer [%d x %d] with %s x %s Basis functions (Periodic: %d) for %d->%d Features'%(n,m,a,b,periodic,fin,fout))
        
        self.kernel = self.add_weight("kernel", initializer='glorot_normal',
                                  shape=[fout,fin,n,m])

    def build(self, input_shape):
        return
#         print(input_shape[0])

    def call(self, inputs):
        positions, features, tf_neighbors, row_splits = inputs
        tf_positions = (tf.RaggedTensor.from_row_splits(tf.gather(positions, tf_neighbors), row_splits) \
                        - positions[:,None])
        if not self.periodic:
            x = tf_positions.values[:,0] / support
            y = tf_positions.values[:,1] / support
            c = evalBasisFunction(self.n, tf.constant(x), which = self.rbf_a)
            f = evalBasisFunction(self.m, tf.constant(y), which = self.rbf_b, periodic = False)
            
        else:
            r = tf.linalg.norm(tf_positions.values, axis = 1) / support
            theta = tf.atan2(tf_positions.values[:,1], tf_positions.values[:,0])
            c = evalBasisFunction(self.n, tf.constant(r * 2. - 1.), which = self.rbf_a)
            f = evalBasisFunction(self.m, tf.constant(theta / np.pi), which = self.rbf_b, periodic = True)
            
        M = tf.einsum("un,vn->nuv", c, f)
#         print('Base Shape: ', M.shape)
        if self.norm:
            M = M / tf.reduce_sum(M, axis = [1,2])[:,None,None]
#         print('Kernel Shape: ', self.kernel.shape)
        O = tf.einsum("unm,ijnm->uij", M, self.kernel)
        result = tf.reduce_sum(O, axis =[1,2])
#         print('O Shape: ', O.shape)
#         OO = tf.reduce_sum(O, axis = 2)
        
#         print(features)
#         print(tf_neighbors)
    
        featureVec = tf.gather(features, tf_neighbors)
#         print(featureVec.shape)
#         print(O.shape)
        FO = tf.einsum("nij,nj->ni", O, featureVec)
#         print(self.kernel)
#         print(W)
#         O = M * self.kernel[None,:]
#         result = tf.reduce_sum(O, axis =[1,2])
#         print(FO)
#         print(featureVec)
        tfo_result = tf.RaggedTensor.from_row_splits(FO, row_splits)
        tf_result = tf.RaggedTensor.from_row_splits(result, row_splits)
        
#         print(tfo_result - tf.expand_dims(tf_result, axis = 2))
#         print(tf_result)
        kSum = tf.reduce_sum(tf_result, axis=1)
        kSumo = tf.reduce_sum(tfo_result, axis=1)
#         print(kSumo - tf.expand_dims(kSum, axis = 1))
        
        return kSumo
    
# layer = rbfLayer(n,m,a,b,polar,fin=1, fout=1)

# out = layer([p32,f32,neighbors_index, row_splits])
# layer2 = rbfLayer(n,m,a,b,polar,fin=2, fout=1)
# out = layer2([p32, out, neighbors_index, row_splits])
# print(out)

class rbfNetwork(tf.keras.Model):
    def __init__(self, arch):
        super().__init__(name=type(self).__name__)
        
        self.convLayers = []
        
        mapValue = lambda i, v, d: arch[i][v] if v in arch[i] else d
        for i, larch in enumerate(arch):
            self.convLayers.append(rbfLayer(\
                n = mapValue(i, 'n', 8),\
                m = mapValue(i, 'm', 8),\
                a = mapValue(i, 'a', 'gaussian'),\
                b = mapValue(i, 'b', 'gaussian'),\
                periodic = mapValue(i, 'polar', True), \
                normalization = mapValue(i,'normalized', False),\
                fin = mapValue(i, ' inputs', arch[i-1]['outputs'] if i > 0 and 'outputs' in arch[i-1] else 1),\
                fout = mapValue(i, 'outputs', 1)
                )\
            )
        self.radiusSearch = ml3d.layers.FixedRadiusSearch(return_distances=True)
        
    def call(self, inputs):        
        positions, features = inputs
        
        neighbors_index, row_splits, distances = self.radiusSearch(positions, positions, support)
        
        ans = features
        for layer in self.convLayers:
            ans = layer([positions, ans, neighbors_index, row_splits])
            
        return ans
    def init(self):
        p32, f32, g32 = prepData(0)
        _ = self.__call__([p32,f32])
        
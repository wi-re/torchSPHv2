# main file that includes all relevant sph functionality
from sph import *

# main file that includes all learning relevant functionality, not necessary to understand
from torch.optim import Adam
from rbfConv import *
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm
# plotting/UI related imports
import matplotlib as mpl
import copy
plt.style.use('dark_background')
cmap = mpl.colormaps['viridis']


def loadBatch(simulationStates, minDomain, maxDomain, particleSupport, bdata, getFeatures, getGroundTruth, stacked):
    positions = [simulationStates[i,0,:] for i in bdata]
    areas = [simulationStates[i,-1,:] for i in bdata]
    velocities = [simulationStates[i,1,:] for i in bdata]
    updates = [simulationStates[i,-2,:] for i in bdata]
    # compute ghost particles for batch for neighborhood search
    ghosts = [createGhostParticles(p, minDomain, maxDomain) for p in positions]
    # perform neighborhood search for batch and split the data into 3 separate lists
    neighborInformation = [findNeighborhoods(p, g, particleSupport) for p,g in zip(positions, ghosts)]
    neighbors = [n[0] for n in neighborInformation]
    radialDistances = [n[1] for n in neighborInformation]
    distances = [n[2] for n in neighborInformation]
    # compute the density on the given batch data
    densities = [computeDensity(p, a, particleSupport, r, n) for p,a,r,n in zip(positions,areas,radialDistances, neighbors)]
    densities = [simulationStates[i,2,:] for i in bdata]

    # all data so far is in lists of equal length, merge lists with special attention to the neighborlist to make sure indices are pointing to the correct particles
    stackedPositions = torch.hstack(positions).type(torch.float32)
    stackedAreas = torch.hstack(areas).type(torch.float32)
    stackedVelocities = torch.hstack(velocities).type(torch.float32)
    stackedUpdates = torch.hstack(updates).type(torch.float32)
    stackedNeighbors = torch.hstack([i * positions[0].shape[0] + neighbors[i] for i in range(len(neighbors))])
    stackedRadialDistances = torch.hstack(radialDistances).type(torch.float32)
    stackedDistances = torch.hstack(distances).type(torch.float32)
    stackedDensities = torch.hstack(densities).type(torch.float32)
    # tensor of ones to make learning easier
    ones = torch.ones_like(stackedAreas)
    # compute the signed distances needed for the network layer, uses the radialDistances and directions computed before                
    d = stackedRadialDistances[:,None] * torch.sign(stackedDistances[:,None])  
    
    return stackedPositions, getFeatures(stackedPositions, stackedAreas, stackedVelocities, stackedUpdates), getGroundTruth(bdata, stacked, simulationStates), stackedNeighbors, d
# iterative training script, lifted from some other code of mine for convenience
def processDataLoaderIter(pb, iterations, epoch, lr, dataLoader, dataIter, batchSize, model, optimizer, simulationStates, minDomain, maxDomain, particleSupport, lossFunction, getFeatures, getGroundTruth, stacked, train = True, prefix = '', augmentAngle = False, augmentJitter = False, jitterAmount = 0.01):
    with record_function("process data loader"): 
        losses = []
        batchIndices = []
        weights = []

        if train:
            model.train(True)
        else:
            model.train(False)

        i = 0
        for b in (pbl := tqdm(range(iterations), leave=False)):
            # get next batch from dataLoader, if all batches have been processed get a new iterator (which shuffles the batch order)
            try:
                bdata = next(dataIter)
                if len(bdata) < batchSize:
                    raise Exception('batch too short')
            except:
                dataIter = iter(dataLoader)
                bdata = next(dataIter)
            # the actual batch processing step
            with record_function("process data loader[batch]"): 
                # reset optimizer gradients
                if train:
                    optimizer.zero_grad()
                # load data for batch                
                stackedPositions, features, groundTruth, stackedNeighbors, d = loadBatch(simulationStates, minDomain, maxDomain, particleSupport, bdata, getFeatures, getGroundTruth, stacked)
                
                
                # run the network layer
                prediction = model((features[:,None], features[:,None]), stackedNeighbors, d)
                # compute the loss
                lossTerm = lossFunction(prediction, groundTruth)
                loss = torch.mean(lossTerm)
                # store the losses for later processing
                losses.append(lossTerm.detach().cpu().numpy())
                # store the current weights before the update
                weights.append(copy.deepcopy(model.state_dict()))
                # weights.append(torch.clone(model.weight.detach().cpu()).numpy())
                # update the network weights
                if train:
                    loss.backward()
                    optimizer.step()
                # create some information to put on the tqdm progress bars
                batchString = str(np.array2string(np.array(bdata), formatter={'float_kind':lambda x: "%.2f" % x, 'int':lambda x:'%04d' % x}))
                pbl.set_description('%8s[gpu %d]: %3d [%1d] @ %1.1e: :  %s -> %.2e' %(prefix, 0, epoch, 0, lr, batchString, loss.detach().cpu().numpy()))
                pb.set_description('[gpu %d] %90s - Learning: %1.4e' %(0, "", np.mean(np.vstack(losses))))
                pb.update()
                batchIndices.append(bdata)
        # stack the processed batches and losses for further processing
        bIndices  = np.hstack(batchIndices)
#         losses = np.vstack(losses)
#         losses = np.vstack(losses)
        # and return
        return bIndices, losses, weights
    
# useful function for learning, returns non normalized windows
def getWindowFunction(windowFunction):
    windowFn = lambda r: torch.ones_like(r)
    if windowFunction == 'cubicSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3 - 4 * torch.clamp(1/2 - r, min = 0) ** 3
    if windowFunction == 'quarticSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 4 - 5 * torch.clamp(3/5 - r, min = 0) ** 4 + 10 * torch.clamp(1/5- r, min = 0) ** 4
    if windowFunction == 'quinticSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 5 - 6 * torch.clamp(2/3 - r, min = 0) ** 5 + 15 * torch.clamp(1/3 - r, min = 0) ** 5
    if windowFunction == 'Wendland2_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3 * (1 + 3 * r)
    if windowFunction == 'Wendland4_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 5 * (1 + 5 * r + 8 * r**2)
    if windowFunction == 'Wendland6_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 7 * (1 + 7 * r + 19 * r**2 + 21 * r**3)
    if windowFunction == 'Wendland2':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 4 * (1 + 4 * r)
    if windowFunction == 'Wendland4':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 6 * (1 + 6 * r + 35/3 * r**2)
    if windowFunction == 'Wendland6':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 8 * (1 + 8 * r + 25 * r**2 + 32 * r**3)
    if windowFunction == 'Hoct4':
        def hoct4(x):
            alpha = 0.0927 # Subject to 0 = (1 − α)** nk−2 + A(γ − α)**nk−2 + B(β − α)**nk−2
            beta = 0.5 # Free parameter
            gamma = 0.75 # Free parameter
            nk = 4 # order of kernel

            A = (1 - beta**2) / (gamma ** (nk - 3) * (gamma ** 2 - beta ** 2))
            B = - (1 + A * gamma ** (nk - 1)) / (beta ** (nk - 1))
            P = -nk * (1 - alpha) ** (nk - 1) - nk * A * (gamma - alpha) ** (nk - 1) - nk * B * (beta - alpha) ** (nk - 1)
            Q = (1 - alpha) ** nk + A * (gamma - alpha) ** nk + B * (beta - alpha) ** nk - P * alpha

            termA = P * x + Q
            termB = (1 - x) ** nk + A * (gamma - x) ** nk + B * (beta - x) ** nk
            termC = (1 - x) ** nk + A * (gamma - x) ** nk
            termD = (1 - x) ** nk
            termE = 0 * x

            termA[x > alpha] = 0
            termB[x <= alpha] = 0
            termB[x > beta] = 0
            termC[x <= beta] = 0
            termC[x > gamma] = 0
            termD[x <= gamma] = 0
            termD[x > 1] = 0
            termE[x < 1] = 0

            return termA + termB + termC + termD + termE

        windowFn = lambda r: hoct4(r)
    if windowFunction == 'Spiky':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3
    if windowFunction == 'Mueller':
        windowFn = lambda r: torch.clamp(1 - r ** 2, min = 0) ** 3
    if windowFunction == 'poly6':
        windowFn = lambda r: torch.clamp((1 - r)**3, min = 0)
    if windowFunction == 'Parabola':
        windowFn = lambda r: torch.clamp(1 - r**2, min = 0)
    if windowFunction == 'Linear':
        windowFn = lambda r: torch.clamp(1 - r, min = 0)
    return windowFn

def plotWeights(dict, basis, normalized):
    # Plot the learned convolution (only works for single layer models (for now))
    fig, axis = plt.subplots(1, 2, figsize=(16,4), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,511)
    n = dict['weight'].shape[0]
    # internal function that is used for the rbf convolution
    fx = evalBasisFunction(n, x , which = basis, periodic=False)
    fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    # plot the individual basis functions with a weight of 1
    for y in range(n):
        axis[0,0].plot(x, fx[y,:], label = '$f_%d(x)$' % y)
    # plot the overall convolution basis for all weights equal to 1
    axis[0,0].plot(x,torch.sum(fx, axis=0),ls='--',c='white', label = '$\Sigma_i f_i(x)$')
    # axis[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fancybox=True, shadow=False)
    axis[0,0].set_title('Basis Functions')

    # plot the individual basis functions with the learned weights
    # for y in range(n):
    #     fy = model.weight[:,0][y].detach() * fx[y,:]
    #     axis[0,1].plot(x[fy != 0], fy[fy != 0], label = '$w_d f_%d(x)$' % y, ls = '--', alpha = 0.5)
    axis[0,1].plot(x,torch.sum(dict['weight'][:,0].detach() * fx,axis=0) + dict['bias'].detach(),ls='--',c='white', label = '$\Sigma_i w_i f_i(x)$')
    # axis[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5, fancybox=True, shadow=False)
    axis[0,1].set_title('Learned convolution')

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
    return torch.cat(predictions, axis = 0), torch.cat(groundTruths, axis = 0), torch.cat(lossTerms, axis = 0), torch.hstack(losses)

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
    
def plotAll(model, weights, basis, normalized, iterations, epochs, numParticles, batchSize, lossArray, simulationStates, minDomain, maxDomain, particleSupport, timestamps, testBatch, lossFunction, getFeatures, getGroundTruth, stacked):
    trainingPrediction, trainingGroundTruth, trainingLossTerm, trainingLoss = computeEvaluationLoss(model, weights[-1][-1], timestamps, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    testingPrediction, testingGroundTruth, testingLossTerm, testingLoss = computeEvaluationLoss(model, weights[-1][-1], testBatch, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    fig, axis = plt.subplot_mosaic('''AABB
    AABB
    CCCD
    EEEF
    GGGH''', figsize=(16,10), sharex = False, sharey = False)
    fig.suptitle('Training results for basis %s%s %2d epochs %4d iterations batchSize %d' % (basis, '' if not normalized else ' (normalized)', epochs, iterations, batchSize))

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

    x =  torch.linspace(-1,1,511)[:,None]
    fx = torch.ones(511)[:,None]
    neighbors = torch.vstack((torch.zeros(511).type(torch.long), torch.arange(511).type(torch.long)))
    neighbors = torch.vstack((torch.arange(511).type(torch.long), torch.zeros(511).type(torch.long)))
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
    model((fx,fx), neighbors, x)

    storedWeights = copy.deepcopy(model.state_dict())
    c = 0
    for i in tqdm(range(epochs), leave = False):
        for j in tqdm(range(iterations), leave = False):
            c = c + 1        
            if c + 1 in ls:                
                model.load_state_dict(weights[i][j])
                axis['B'].plot(x[:,0], model((fx,fx), neighbors, x).detach(),ls='--',c= cm(ls.index(c+1) / (len(ls) - 1)), alpha = 0.95)
    #             break

    model.load_state_dict(storedWeights)
    axis['B'].set_title('Weight progress')

    # fig, axis = plt.subplots(3, 2, figsize=(16,6), sharex = 'col', sharey = True, squeeze = False, gridspec_kw={'width_ratios': [3, 1]})

    
    fig.tight_layout()

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
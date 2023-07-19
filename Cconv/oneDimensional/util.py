import torch
from sph import *

def loadBatch(particleData, settings, dataSet, bdata, device):
    dataEntries = [dataSet[b] for b in bdata]
    
    positions = [particleData[f]['positions'][t,:].to(device) for f,t in dataEntries]
    velocities = [particleData[f]['velocity'][t,:].to(device) for f,t in dataEntries]
    areas = [particleData[f]['area'][t,:].to(device) for f,t in dataEntries]
    dudts = [particleData[f]['dudt'][t,:].to(device) for f,t in dataEntries]
    densities = [particleData[f]['density'][t,:].to(device) for f,t in dataEntries]
    setup = [settings[f] for f,t in dataEntries]
    
    return positions, velocities, areas, dudts, densities, setup
    
def batchedNeighborsearch(positions, setup):
    neighborLists = [periodicNeighborSearch(p, s['particleSupport'], s['minDomain'], s['maxDomain']) for p, s in zip(positions, setup)]
    
    neigh_i = [n[0][0] for n in neighborLists]
    neigh_j = [n[0][1] for n in neighborLists]
    neigh_distance = [n[1] for n in neighborLists]
    neigh_direction = [n[2] for n in neighborLists]
    
    for i in range(len(neighborLists) - 1):
        neigh_i[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])
        neigh_j[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])
        
    neigh_i = torch.hstack(neigh_i)
    neigh_j = torch.hstack(neigh_j)
    neigh_distance = torch.hstack(neigh_distance)
    neigh_direction = torch.hstack(neigh_direction)
    
    return neigh_i, neigh_j, neigh_distance, neigh_direction

def flatten(positions, velocities, areas, density, dudts):
    return torch.hstack(positions), torch.hstack(velocities), torch.hstack(areas), torch.hstack(density), torch.hstack(dudts)

def loadFrames(particleData, settings, dataSet, f, frames, device):
#     dataEntries = [dataSet[b] for b in bdata]
    
    positions = [particleData[f]['positions'][t,:] for t in frames]
    velocities = [particleData[f]['velocity'][t,:] for t in frames]
    areas = [particleData[f]['area'][t,:] for t in frames]
    dudts = [particleData[f]['dudt'][t,:] for t in frames]
    densities = [particleData[f]['density'][t,:] for t in frames]
    setup = [settings[f] for t in frames]
    
    return positions, velocities, areas, dudts, densities, setup


def modelStep(model, positions, velocities, areas, densities, dudts, setup, getFeatures, device):
    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, dudt = flatten(positions, velocities, areas, densities, dudts)
    x = x[:,None]
    distance = distance[:,None]
    # features = torch.vstack((v, v)).mT
    features = getFeatures(positions, velocities, areas, densities, dudts)[:,None]
    
    prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))
    
    return prediction.reshape((len(positions), u.shape[0]))



def processDataLoaderIter(pb, iterations, epoch, lr, 
                          dataLoader, dataIter, batchSize, 
                          model, optimizer, 
                          particleData, settings, dataSet, 
                          
                          lossFunction, getFeatures, getGroundTruth, stacked, train = True, prefix = '', augmentAngle = False, augmentJitter = False, jitterAmount = 0.01):
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
                if len(bdata) < batchSize :
                    raise Exception('batch too short')
            except:
                dataIter = iter(dataLoader)
                bdata = next(dataIter)
            # the actual batch processing step
            with record_function("process data loader[batch]"): 
                # reset optimizer gradients
                if train:
                    optimizer.zero_grad()
                #
                positions, velocities, areas, dudts, density, setup = loadFrames(particleData, settings, dataSet, trainingFiles[0], 0 + np.arange(4), device)
#                 positions, velocities, areas, dudts, setup = loadBatch(particleData, settings, dataSet, bdata, device)
                
                i, j, distance, direction = batchedNeighborsearch(positions, setup)
                x, u, v, dudt = flatten(positions, velocities, density, dudts)
                x = x[:,None]
                distance = (distance * direction)[:,None]
                features = torch.vstack((u, torch.ones_like(v))).mT
                features = torch.vstack((v, v)).mT
                features = v[:,None]
#                 print(features.shape)
#                 print(i.shape)
#                 print(features[i].shape)
                
                    
#                 load data for batch                
#                 stackedPositions, features, groundTruth, stackedNeighbors, d = loadBatch(simulationStates, minDomain, maxDomain, particleSupport, bdata, getFeatures, getGroundTruth, stacked)
                # run the network layer
                prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))
                # compute the loss
                groundTruth = getGroundTruth(positions, velocities, areas, dudts).to(device) / dt
                lossTerm = lossFunction(prediction, groundTruth)
                loss = torch.mean(lossTerm)
                # store the losses for later processing
                losses.append(lossTerm.detach().cpu().numpy())
                # store the current weights before the update
                weights.append(copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
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
        # and return
        return bIndices, losses, weights
    
from rbfNet import getWindowFunction, RbfNet
from torch.optim import Adam

def trainModelOverfit(particleData, settings, dataSet, trainingFiles, n = 16, basis = 'linear', layers = [1], window = None, windowNorm = None, epochs = 5, iterations = 1000, initialLR = 1e-2, groundTruthFn = None, featureFn = None, lossFn = None, device = 'cpu'):   
    windowFn = getWindowFunction(window, norm = windowNorm) if window is not None else None
    model = RbfNet(fluidFeatures = 1, 
                   layers = layers, 
                   denseLayer = True, activation = 'ReLU', coordinateMapping = 'cartesian', 
                   n = n, windowFn = windowFn, rbf = basis, batchSize = 32, ignoreCenter = True, normalized = False).to(device)   
    lr = initialLR
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)
    
    positions, velocities, areas, dudts, density, setup = loadFrames(particleData, settings, dataSet, trainingFiles[0], 0 + np.arange(1), device)
    #                 positions, velocities, areas, dudts, setup = loadBatch(particleData, settings, dataSet, bdata, device)

    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, rho, dudt = flatten(positions, velocities, areas, density, dudts)

    x = x[:,None].to(device)
    groundTruth = groundTruthFn(positions, velocities, areas, density, dudts).to(device)
    distance = (distance * direction)[:,None].to(device)
    features = featureFn(x, u, v, rho, dudt).to(device)
    
    predictions = []
    losses = []
    for e in (pb := tqdm(range(epochs), leave = False)):
        for b in (pbl := tqdm(range(iterations), leave=False)):
            optimizer.zero_grad()
            prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
            lossTerm = lossFn(prediction, groundTruth)
            loss = torch.mean(lossTerm)
            
            loss.backward()
            optimizer.step()
            predictions.append(prediction.detach().cpu())
            losses.append(lossTerm.detach().cpu())
            pbl.set_description('%8s[gpu %d]: %3d [%1d] @ %1.1e: :  %s -> %.2e' %('overfitting', 0, 0, 0, lr, '0', loss.detach().cpu().numpy()))
        if True: #epoch % 1 == 0 and epoch > 0:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * param_group['lr']
                
    return {'model': model, 'optimizer': optimizer, 'finalLR': lr, 'predictions': predictions, 'losses': losses, 
            'window': window, 'windowNorm': windowNorm, 'n':n, 'basis':basis, 'layers':layers, 'epochs': epochs, 'iterations': iterations,
            'x': x, 'features': features, 'gt': groundTruth, 'i' : i, 'j':j, 'distances': distance
           }
from scipy.signal import lombscargle
def generateSyntheticData(n, device):
    xp =  torch.linspace(-1,1,n)[:,None].to(device)
    fx = torch.ones(n).to(device) 
    neighbors = torch.vstack((torch.arange(n).type(torch.long), torch.zeros(n).type(torch.long))).to(device)
    
    return xp, fx[:,None], neighbors[0,:], neighbors[1,:]

def plotModelOverfit(modelState, device, baseArea, particleSupport):
    fig, axis = plt.subplot_mosaic('''DBC
    ABC
    EBC
    FBC''', figsize=(16,12), sharex = False, width_ratios=[2,1,1])

    x = modelState['x'][:,0].detach().cpu().numpy()
    
    def plot(fig, axis, mat, title, cmap = 'viridis', norm = 'linear'):
        im = axis.imshow(mat, cmap = cmap, norm = norm)
        axis.axis('auto')
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("bottom", size="2%", pad="6%")
        cb1 = fig.colorbar(im, cax=cax1,orientation='horizontal')
        cb1.ax.tick_params(labelsize=8) 
        axis.set_title(title)

    axis['A'].plot(x, modelState['gt'].detach().cpu().numpy(), c = 'red')
    ll = np.logspace(1, np.log10(modelState['iterations'] * modelState['epochs'] + 1), num = 100)
    norm = LogNorm(vmin=1, vmax=modelState['iterations'] * modelState['epochs'] + 1)

    axis['A'].plot(x, modelState['predictions'][-1][:], c = 'green')
    s = scatter_sum(baseArea * kernel(torch.abs(modelState['distances'][:,0]).cpu(), particleSupport), modelState['i'].cpu(), dim = 0, dim_size = modelState['x'].shape[0])
    axis['A'].plot(x, s, c = 'blue', ls = '--')


    plot(fig, axis['B'], torch.vstack(modelState['predictions'])[:,:], title = 'prediction', norm = 'linear')
    plot(fig, axis['C'], torch.vstack(modelState['losses'])[:,:], title = 'Loss', norm = 'log')

    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = modelState['iterations'] * modelState['epochs']
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()

    axis['D'].plot(xSynthetic[:,0].detach().cpu().numpy(), modelState['model'](featuresSynthetic, iSynthetic, jSynthetic, xSynthetic).detach().cpu().numpy(),ls='-',c= 'green', alpha = 0.95)
    axis['D'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')

    axis['D'].set_title('convolution operator')
    axis['A'].set_title('Prediction')

    axis['E'].loglog(torch.mean(torch.vstack(modelState['losses']), dim = 1), ls = '-', c = 'white')
    axis['E'].loglog(torch.min(torch.vstack(modelState['losses']), dim = 1)[0], ls = '--', c = 'white')
    axis['E'].loglog(torch.max(torch.vstack(modelState['losses']), dim = 1)[0], ls = '--', c = 'white')
    axis['E'].set_title('Loss Curve')
    
    
    axis['F'].set_title('Lomb-Scargle')
    axis['F'].set_xlabel('freq')
#     x = modelState['x'][:,0].numpy()
    yMean = np.mean(modelState['gt'].detach().cpu().numpy())
    y1 = modelState['gt'].detach().cpu().numpy()
    y2 = modelState['predictions'][-1].numpy()
    
    dxmin = np.diff(x).min()
    duration = x.ptp()
    freqs = np.linspace(1/duration, x.shape[0]/duration, 1*x.shape[0])
    periodogram = lombscargle(x, y1 - yMean, freqs, normalize = False, precenter = True)
    axis['F'].semilogx(freqs, np.sqrt(4*periodogram/(1*x.shape[0])))
    periodogram = lombscargle(x, y2 - yMean, freqs, normalize = False, precenter = True)
    axis['F'].semilogx(freqs, np.sqrt(4*periodogram/(1*x.shape[0])))



    fig.suptitle('Convolution Test, basis %s, n = %2d, Window = %s [%s], params = %6d' % (modelState['basis'], modelState['n'], modelState['window'] if modelState['window'] is not None else 'None', modelState['windowNorm'] if modelState['windowNorm'] is not None else 'None', count_parameters(modelState['model'])))

    fig.tight_layout()


from sklearn.metrics import r2_score

def getMetrics(modelState, iteration = -1):
    l2 = (modelState['gt'].cpu() - modelState['predictions'][iteration])**2
    r2 = r2_score(modelState['gt'].cpu(), modelState['predictions'][iteration])
    
    x = modelState['x'][:,0].cpu().numpy()
    yMean = np.mean(modelState['gt'].cpu().numpy())
    y1 = modelState['gt'].cpu().numpy()
    y2 = modelState['predictions'][iteration].numpy()
    
    dxmin = np.diff(x).min()
    duration = x.ptp()
    freqs = np.linspace(1/duration, x.shape[0]/duration, 1*x.shape[0])
    periodogramGT = lombscargle(x, y1 - yMean, freqs, normalize = False, precenter = True)
    periodogramPred = lombscargle(x, y2 - yMean, freqs, normalize = False, precenter = True)
    lombScargle = (periodogramGT - periodogramPred) ** 2
    
    return l2.numpy(), r2, lombScargle

def plotModelsetOverfit(models, device):
    fig, axis = plt.subplot_mosaic('''ABE
    CBE
    DBF''', figsize=(16,6), sharex = False, width_ratios=[1,2,1])

    ns = range(1, 32 + 1)

    l2s = np.zeros(32)
    r2s = np.zeros(32)
    lombs = np.zeros(32)

    norm = mpl.colors.Normalize(vmin = 1, vmax = 32)

    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = models[0]['iterations'] * models[0]['epochs']
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()


    axis['E'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')
    axis['F'].plot(models[0]['x'][:,0].detach().cpu().numpy(), models[0]['gt'].detach().cpu().numpy())

    axis['B'].grid(axis = 'y', which = 'major', ls = '--', alpha = 0.6)

    for i in range(len(models)):
        modelState = models[i]
        l2, r2, scargle = getMetrics(modelState, -1)
        l2s[i] = np.mean(l2)
        r2s[i] = r2
        lombs[i] = np.mean(scargle)
        loss = [torch.mean(t).numpy().item() for t in modelState['losses']]
        c = cmap(norm(i))
        axis['B'].loglog(loss, c =c )
        axis['F'].plot(modelState['x'][:,0].detach().cpu().numpy(), modelState['predictions'][-1], c = c)
        axis['E'].plot(xSynthetic[:,0].detach().cpu().numpy(), modelState['model'](featuresSynthetic, iSynthetic, jSynthetic, xSynthetic).detach().cpu().numpy(),ls='-',c= c, alpha = 0.95)

    axis['A'].set_title('L2 loss')
    axis['A'].grid(axis = 'y', which = 'major', ls = '--', alpha = 0.6)
    axis['A'].semilogy(ns, l2s)
    axis['A'].set_xticklabels([])
    axis['C'].set_title('R2 confidence')
    axis['C'].grid(axis = 'y', which = 'both', ls = '--', alpha = 0.6)
    axis['C'].semilogy(ns, r2s)
    axis['C'].set_xticklabels([])
    axis['D'].set_title('Lomb-Scargle distance')
    axis['D'].grid(axis = 'y', which = 'both', ls = '--', alpha = 0.6)
    axis['D'].semilogy(ns, lombs)
    axis['D'].set_xlabel('n')
    axis['E'].set_title('Convolutional Operator')
    axis['F'].set_title('Prediction')
    axis['B'].set_title('Loss Curve')

    fig.suptitle('Basis: %s, Window: %s, WindowNorm: %s' % (models[0]['basis'], models[0]['window'] if models[0]['window'] is not None else 'None', models[0]['windowNorm'] if models[0]['windowNorm'] is not None else 'None'))

    fig.tight_layout()

    return fig, axis


def plotModelsetInteractive(fig, axis, models, device):
    axis['A'].cla()
    axis['B'].cla()
    axis['C'].cla()
    axis['D'].cla()
    axis['E'].cla()
    axis['F'].cla()

    ns = range(1, 32 + 1)

    l2s = np.zeros(32)
    r2s = np.zeros(32)
    lombs = np.zeros(32)

    norm = mpl.colors.Normalize(vmin = 1, vmax = 32)

    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = models[0]['iterations'] * models[0]['epochs']
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()


    axis['E'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')
    axis['F'].plot(models[0]['x'][:,0].detach().cpu().numpy(), models[0]['gt'].detach().cpu().numpy())

    axis['B'].grid(axis = 'y', which = 'major', ls = '--', alpha = 0.6)

    for i in range(len(models)):
        modelState = models[i]
        l2, r2, scargle = getMetrics(modelState, -1)
        l2s[i] = np.mean(l2)
        r2s[i] = r2
        lombs[i] = np.mean(scargle)
        loss = [torch.mean(t).numpy().item() for t in modelState['losses']]
        c = cmap(norm(i))
        axis['B'].loglog(loss, c =c )
        axis['F'].plot(modelState['x'][:,0].detach().cpu().numpy(), modelState['predictions'][-1], c = c)
        axis['E'].plot(xSynthetic[:,0].detach().cpu().numpy(), modelState['model'](featuresSynthetic, iSynthetic, jSynthetic, xSynthetic).detach().cpu().numpy(),ls='-',c= c, alpha = 0.95)

    axis['A'].set_title('L2 loss')
    axis['A'].grid(axis = 'y', which = 'major', ls = '--', alpha = 0.6)
    axis['A'].semilogy(ns, l2s)
    axis['A'].set_xticklabels([])
    axis['C'].set_title('R2 confidence')
    axis['C'].grid(axis = 'y', which = 'both', ls = '--', alpha = 0.6)
    axis['C'].semilogy(ns, r2s)
    axis['C'].set_xticklabels([])
    axis['D'].set_title('Lomb-Scargle distance')
    axis['D'].grid(axis = 'y', which = 'both', ls = '--', alpha = 0.6)
    axis['D'].semilogy(ns, lombs)
    axis['D'].set_xlabel('n')
    axis['E'].set_title('Convolutional Operator')
    axis['F'].set_title('Prediction')
    axis['B'].set_title('Loss Curve')

    fig.suptitle('Basis: %s, Window: %s, WindowNorm: %s' % (models[0]['basis'], models[0]['window'] if models[0]['window'] is not None else 'None', models[0]['windowNorm'] if models[0]['windowNorm'] is not None else 'None'))

    fig.tight_layout()



def plotModel(modelState, testData, device, baseArea, particleSupport):
    fig, axis = plt.subplot_mosaic('''ABC
    FGH
    DDD''', figsize=(12,6), sharex = False)


    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = modelState['iterations'] * modelState['epochs']
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()

    axis['F'].plot(xSynthetic[:,0].detach().cpu().numpy(), modelState['model'](featuresSynthetic, iSynthetic, jSynthetic, xSynthetic).detach().cpu().numpy(),ls='-',c= 'green', alpha = 0.95)
    axis['F'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')
    axis['F'].set_title('convolution operator')

    axis['A'].loglog(torch.mean(torch.vstack(modelState['losses']), dim = 1), ls = '-', c = 'black')
    axis['A'].loglog(torch.min(torch.vstack(modelState['losses']), dim = 1)[0], ls = '--', c = 'black')
    axis['A'].loglog(torch.max(torch.vstack(modelState['losses']), dim = 1)[0], ls = '--', c = 'black')
    axis['A'].set_title('Loss Curve')
    with torch.no_grad():
        for i, k in  enumerate(testData.keys()):
            ax = axis['B']
            if i == 0:
                ax = axis['B']
            if i == 1:
                ax = axis['C']
            if i == 2:
                ax = axis['G']
            if i == 3:
                ax = axis['H']    
            norm = mpl.colors.Normalize(vmin=0, vmax=len(testData[k][0]) - 1)
            gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
            for i, (xs, rhos) in enumerate(zip(testData[k][0], gt)):
                c = cmap(norm(i))
                ax.plot(xs.cpu().numpy(), rhos.cpu().numpy(), c = c)
            prediction = modelState['model'](testData[k][5], testData[k][6], testData[k][7], testData[k][8]).reshape(len(testData[k][0]), testData[k][0][0].shape[0]).detach().cpu().numpy()
            for i, (xs, rhos) in enumerate(zip(testData[k][0], prediction)):
                c = cmap(norm(i))
                ax.plot(xs.cpu().numpy(), rhos, ls = '--', c = c)

            ax.set_title(k)

    def plot(fig, axis, mat, title, cmap = 'viridis', norm = 'linear'):
        im = axis.imshow(mat, cmap = cmap, norm = norm)
        axis.axis('auto')
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("right", size="1%", pad="1%")
        cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
        cb1.ax.tick_params(labelsize=8) 
        axis.set_title(title)

    lMat = torch.vstack(modelState['losses']).mT
    plot(fig, axis['D'],lMat, title = 'Losses', norm = LogNorm(vmin=torch.min(lMat[lMat> 0]), vmax=torch.max(lMat)))
    fig.tight_layout()
    

def getTestcase(testingData, settings, f, frames, device):
    positions = [testingData[f]['positions'][t,:].to(device) for t in frames]
    velocities = [testingData[f]['velocity'][t,:].to(device) for t in frames]
    areas = [testingData[f]['area'][t,:].to(device) for t in frames]
    dudts = [testingData[f]['dudt'][t,:].to(device) for t in frames]
    densities = [testingData[f]['density'][t,:].to(device) for t in frames]
    setup = [settings[f] for t in frames]
    
    return positions, velocities, areas, dudts, densities, setup
def loadTestcase(testingData, settings, f, frames, device, groundTruthFn, featureFn):
    positions, velocities, areas, dudts, density, setup = getTestcase(testingData, settings, f, frames, device)

    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, rho, dudt = flatten(positions, velocities, areas, density, dudts)

    x = x[:,None].to(device)    
    groundTruth = groundTruthFn(positions, velocities, areas, density, dudts).to(device)
    distance = (distance * direction)[:,None].to(device)
    features = featureFn(x, u, v, rho, dudt).to(device)
#     print(groundTruth)
    return positions, velocities, areas, density, dudts, features, i, j, distance, groundTruth


def plotLosses(trainedModel, testData):
    fig, axis = plt.subplot_mosaic('''AA
    BC
    DE''', figsize=(12,8), sharey = True, sharex = True)

    axis['A'].semilogy(torch.mean(torch.vstack(trainedModel['losses']), dim = 1), c = 'black')
    # axis['A'].semilogy(torch.min(torch.vstack(trainedModel['losses']), dim = 1)[0], c = 'black', ls = '--')
    axis['A'].semilogy(torch.max(torch.vstack(trainedModel['losses']), dim = 1)[0], c = 'black', ls = '--')

    axis['B'].semilogy(torch.mean(torch.vstack(trainedModel['losses']), dim = 1), c = 'black', ls = '-', alpha = 0.5)
    axis['C'].semilogy(torch.mean(torch.vstack(trainedModel['losses']), dim = 1), c = 'black', ls = '-', alpha = 0.5)
    axis['D'].semilogy(torch.mean(torch.vstack(trainedModel['losses']), dim = 1), c = 'black', ls = '-', alpha = 0.5)
    axis['E'].semilogy(torch.mean(torch.vstack(trainedModel['losses']), dim = 1), c = 'black', ls = '-', alpha = 0.5)

    testFrames = list(trainedModel['testLosses'].keys())
    testLosses = trainedModel['testLosses']

    for k in testData.keys():
        frames = len(testLosses[0][k])
        norm = mpl.colors.Normalize(vmin = 0, vmax = frames - 1)

        ax = axis['B']
        if k == 'sawTooth':
            ax = axis['B']
        if k == 'square':
            ax = axis['C']
        if k == 'uniform':
            ax = axis['E']
        if k == 'sin':
            ax = axis['D']    
        for f in range(frames):
            ax.semilogy(testFrames, [np.mean(testLosses[l][k][f]) for l in testFrames], c = cmap(norm(f)))
    #         ax.plot(testFrames, [np.min(testLosses[l][k][f]) for l in testFrames], c = cmap(norm(f)), ls = '--', alpha = 0.5)
    #         ax.plot(testFrames, [np.max(testLosses[l][k][f]) for l in testFrames], c = cmap(norm(f)), ls = '--', alpha = 0.5)
    axis['B'].set_title('sawTooth')
    axis['C'].set_title('square')
    axis['D'].set_title('sin')
    axis['E'].set_title('uniform')

    fig.tight_layout()

def plotModelset(models, device, nMax = 32):
    fig, axis = plt.subplot_mosaic('''AABC
    AADE
    FFGG''', figsize=(16,6), sharex = False)
    ns = range(1, nMax + 1)

    l2s = np.zeros(nMax)
    r2s = np.zeros(nMax)
    lombs = np.zeros(nMax)

    norm = mpl.colors.Normalize(vmin = 1, vmax = nMax)

    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = models[0]['iterations'] * models[0]['epochs']
    # axis['E'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')
    # axis['F'].plot(models[0]['x'][:,0].detach().cpu().numpy(), models[0]['gt'].detach().cpu().numpy())

    # axis['B'].grid(axis = 'y', which = 'major', ls = '--', alpha = 0.6)

    trainingLosses = []
    testingLosses = []
    for i in range(len(models)):
        modelState = models[i]
        c = cmap(norm(i + 1))
        testFrames = list(modelState['testLosses'].keys())
        testLosses = modelState['testLosses']
        axis['A'].semilogy(torch.mean(torch.vstack(modelState['losses']), dim = 1), c = c)
        trainingLosses.append(torch.mean(torch.vstack(modelState['losses']), dim = 1)[-1])
        testLs = []
        for k in testData.keys():
            frames = len(testLosses[0][k])
    #         norm = mpl.colors.Normalize(vmin = 0, vmax = frames - 1)

            ax = axis['B']
            if k == 'sawTooth':
                ax = axis['B']
            if k == 'square':
                ax = axis['C']
            if k == 'uniform':
                ax = axis['E']
            if k == 'sin':
                ax = axis['D']    
            losses = []
            for f in range(frames):
                losses.append([np.mean(testLosses[l][k][f]) for l in testFrames])
            testLs.append(np.mean(losses,axis = 0)[-1])
            ax.semilogy(testFrames, np.mean(losses,axis = 0), c =c)
        testingLosses.append(np.mean(testLs))

    axis['A'].set_title('Loss Curve')
    axis['B'].set_title('sawTooth')
    axis['C'].set_title('square')
    axis['D'].set_title('sin')
    axis['E'].set_title('uniform')
    axis['F'].semilogy(np.arange(nMax) + 1, trainingLosses)
    axis['F'].set_title('Training Loss / Parameters')
    axis['G'].semilogy(np.arange(nMax) + 1, testingLosses)
    axis['G'].set_title('Testing Loss / Parameters')
    fig.suptitle('Basis: %s, Window: %s, WindowNorm: %s' % (models[0]['basis'], models[0]['window'] if models[0]['window'] is not None else 'None', models[0]['windowNorm'] if models[0]['windowNorm'] is not None else 'None'))

    fig.tight_layout()

    # return fig, axis

def plotModelset(models, testData, device, nMax = 32):
    fig, axis = plt.subplot_mosaic('''AABC
    AADE
    FFGG''', figsize=(16,6), sharex = False)
    ns = range(1, nMax + 1)

    l2s = np.zeros(nMax)
    r2s = np.zeros(nMax)
    lombs = np.zeros(nMax)

    norm = mpl.colors.Normalize(vmin = 1, vmax = nMax)

    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = models[0]['iterations'] * models[0]['epochs']
    # axis['E'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')
    # axis['F'].plot(models[0]['x'][:,0].detach().cpu().numpy(), models[0]['gt'].detach().cpu().numpy())

    # axis['B'].grid(axis = 'y', which = 'major', ls = '--', alpha = 0.6)

    trainingLosses = []
    testingLosses = []
    for i in range(len(models)):
        modelState = models[i]
        c = cmap(norm(i + 1))
        testFrames = list(modelState['testLosses'].keys())
        testLosses = modelState['testLosses']
        axis['A'].semilogy(torch.mean(torch.vstack(modelState['losses']), dim = 1), c = c)
        trainingLosses.append(torch.mean(torch.vstack(modelState['losses']), dim = 1)[-1])
        testLs = []
        for k in testData.keys():
            frames = len(testLosses[0][k])
    #         norm = mpl.colors.Normalize(vmin = 0, vmax = frames - 1)

            ax = axis['B']
            if k == 'sawTooth':
                ax = axis['B']
            if k == 'square':
                ax = axis['C']
            if k == 'uniform':
                ax = axis['E']
            if k == 'sin':
                ax = axis['D']    
            losses = []
            for f in range(frames):
                losses.append([np.mean(testLosses[l][k][f]) for l in testFrames])
            testLs.append(np.mean(losses,axis = 0)[-1])
            ax.semilogy(testFrames, np.mean(losses,axis = 0), c =c)
        testingLosses.append(np.mean(testLs))

    axis['A'].set_title('Loss Curve')
    axis['B'].set_title('sawTooth')
    axis['C'].set_title('square')
    axis['D'].set_title('sin')
    axis['E'].set_title('uniform')
    axis['F'].semilogy(np.arange(nMax) + 1, trainingLosses)
    axis['F'].set_title('Training Loss / Parameters')
    axis['G'].semilogy(np.arange(nMax) + 1, testingLosses)
    axis['G'].set_title('Testing Loss / Parameters')
    fig.suptitle('Basis: %s, Window: %s, WindowNorm: %s' % (models[0]['basis'], models[0]['window'] if models[0]['window'] is not None else 'None', models[0]['windowNorm'] if models[0]['windowNorm'] is not None else 'None'))

    fig.tight_layout()

    # return fig, axis
def plotModelset2(models, testData, device, nMax = 32):
    fig, axis = plt.subplot_mosaic('''AABCHH
    AADEHH
    FFGGHH''', figsize=(16,6), sharex = False)
    ns = range(1, nMax + 1)

    l2s = np.zeros(nMax)
    r2s = np.zeros(nMax)
    lombs = np.zeros(nMax)

    norm = mpl.colors.Normalize(vmin = 1, vmax = nMax)

    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = models[0]['iterations'] * models[0]['epochs']
    axis['H'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')
    # axis['F'].plot(models[0]['x'][:,0].detach().cpu().numpy(), models[0]['gt'].detach().cpu().numpy())

    # axis['B'].grid(axis = 'y', which = 'major', ls = '--', alpha = 0.6)

    trainingLosses = []
    testingLosses = []
    for i in range(len(models)):
        modelState = models[i]
        c = cmap(norm(i + 1))
        axis['H'].plot(xSynthetic[:,0].detach().cpu().numpy(), modelState['model'](featuresSynthetic, iSynthetic, jSynthetic, xSynthetic).detach().cpu().numpy(),ls='-',c= c, alpha = 0.95)
        testFrames = list(modelState['testLosses'].keys())
        testLosses = modelState['testLosses']
        axis['A'].semilogy(torch.mean(torch.vstack(modelState['losses']), dim = 1), c = c)
        trainingLosses.append(torch.mean(torch.vstack(modelState['losses']), dim = 1)[-1])
        testLs = []
        for k in testData.keys():
            frames = len(testLosses[0][k])
    #         norm = mpl.colors.Normalize(vmin = 0, vmax = frames - 1)

            ax = axis['B']
            if k == 'sawTooth':
                ax = axis['B']
            if k == 'square':
                ax = axis['C']
            if k == 'uniform':
                ax = axis['E']
            if k == 'sin':
                ax = axis['D']    
            losses = []
            for f in range(frames):
                losses.append([np.mean(testLosses[l][k][f]) for l in testFrames])
            testLs.append(np.mean(losses,axis = 0)[-1])
            ax.semilogy(testFrames, np.mean(losses,axis = 0), c =c)
        testingLosses.append(np.mean(testLs))

    axis['A'].set_title('Loss Curve')
    axis['B'].set_title('sawTooth')
    axis['C'].set_title('square')
    axis['D'].set_title('sin')
    axis['E'].set_title('uniform')
    axis['F'].semilogy(np.arange(nMax) + 1, trainingLosses)
    axis['F'].set_title('Training Loss / Parameters')
    axis['G'].semilogy(np.arange(nMax) + 1, testingLosses)
    axis['G'].set_title('Testing Loss / Parameters')
    axis['H'].set_title('Convolutional Operator')
    fig.suptitle('Basis: %s, Window: %s, WindowNorm: %s' % (models[0]['basis'], models[0]['window'] if models[0]['window'] is not None else 'None', models[0]['windowNorm'] if models[0]['windowNorm'] is not None else 'None'))

    fig.tight_layout()

    # return fig, axis


def plotModel(modelState, testData, device, baseArea, particleSupport):
    fig, axis = plt.subplot_mosaic('''ABC
    FGH
    DDD''', figsize=(12,6), sharex = False)


    xSynthetic, featuresSynthetic, iSynthetic, jSynthetic = generateSyntheticData(511, device)

    steps = modelState['iterations'] * modelState['epochs']
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()

    axis['F'].plot(xSynthetic[:,0].detach().cpu().numpy(), modelState['model'](featuresSynthetic, iSynthetic, jSynthetic, xSynthetic).detach().cpu().numpy(),ls='-',c= 'green', alpha = 0.95)
    axis['F'].plot(xSynthetic[:,0].detach().cpu().numpy(), kernel(torch.abs(xSynthetic), 1).detach().cpu().numpy(), c = 'red')
    axis['F'].set_title('convolution operator')

    axis['A'].loglog(torch.mean(torch.vstack(modelState['losses']), dim = 1), ls = '-', c = 'black')
    axis['A'].loglog(torch.min(torch.vstack(modelState['losses']), dim = 1)[0], ls = '--', c = 'black')
    axis['A'].loglog(torch.max(torch.vstack(modelState['losses']), dim = 1)[0], ls = '--', c = 'black')
    axis['A'].set_title('Loss Curve')
    with torch.no_grad():
        for i, k in  enumerate(testData.keys()):
            ax = axis['B']
            if i == 0:
                ax = axis['B']
            if i == 1:
                ax = axis['C']
            if i == 2:
                ax = axis['G']
            if i == 3:
                ax = axis['H']    
            norm = mpl.colors.Normalize(vmin=0, vmax=len(testData[k][0]) - 1)
            gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
            for i, (xs, rhos) in enumerate(zip(testData[k][0], gt)):
                c = cmap(norm(i))
                ax.plot(xs.cpu().numpy(), rhos.cpu().numpy(), c = c)
            prediction = modelState['model'](testData[k][5], testData[k][6], testData[k][7], testData[k][8]).reshape(len(testData[k][0]), testData[k][0][0].shape[0]).detach().cpu().numpy()
            for i, (xs, rhos) in enumerate(zip(testData[k][0], prediction)):
                c = cmap(norm(i))
                ax.plot(xs.cpu().numpy(), rhos, ls = '--', c = c)

            ax.set_title(k)

    def plot(fig, axis, mat, title, cmap = 'viridis', norm = 'linear'):
        im = axis.imshow(mat, cmap = cmap, norm = norm)
        axis.axis('auto')
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("right", size="1%", pad="1%")
        cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
        cb1.ax.tick_params(labelsize=8) 
        axis.set_title(title)

    lMat = torch.vstack(modelState['losses']).mT
    plot(fig, axis['D'],lMat, title = 'Losses', norm = LogNorm(vmin=torch.min(lMat[lMat> 0]), vmax=torch.max(lMat)))
    fig.suptitle('basis %s @ %2d terms, window: %s [%s]' % (modelState['basis'], modelState['n'], modelState['window'] if modelState['window'] is not None else 'None', modelState['windowNorm'] if modelState['windowNorm'] is not None else 'None'))
    fig.tight_layout()
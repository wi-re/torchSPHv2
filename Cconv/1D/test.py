# methods = ['rbf square', 'rbf linear', 'rbf bump', 'rbf spiky', 'rbf cubic_spline', 'rbf gaussian', 'fourier', 'chebyshev']

# import  itertools

# cmd = 'clear && python rbfnn.py --lr=0.001 --epochs=16 --cutoff=400 --n=16 --batch_size=1 --forwardBatch=1024 --backwardBatch=1024'

# for l in list(itertools.product(methods, methods)):
#     print('%s --rbf_x="%s" --rbf_y="%s"'%(cmd, l[0], l[1]))



architectures = ['2','4 2', '4 8 4 2', '4 8 8 2', '8 16 2', '8 16 16 2', '8 32 32 2', '8 16 32 32 16 2']
widths = [3, 5, 8, 9, 16, 32]

import  itertools

cmd = 'clear && python rbfnn.py --lr=0.001 --epochs=24 --cutoff=400 --batch_size=1 --forwardBatch=64 --backwardBatch=64 --rbf_x="rbf linear" --rbf_y="rbf linear"'

for l in list(itertools.product(architectures, widths)):
    print('%s --arch="%s" --n="%s"'%(cmd, l[0], l[1]))




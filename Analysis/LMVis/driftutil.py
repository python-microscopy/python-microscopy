from PYME.Analysis.exploreDrift import *

def e2d(alim=[-0.05,0.05],blim=[-0.05,0.05],size=[20,20]):
    return entlin2dr(filter['x'],filter['y'],filter['t'],alim,blim,size)

def isn(x):
    imshow(x, interpolation='nearest')


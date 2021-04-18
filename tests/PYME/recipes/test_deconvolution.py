import numpy as np

def gen_model(shape):
    sx, sy, sz = shape
    im = np.zeros(shape)
    im[int(np.floor(sx*.25)):int(np.ceil(sx*.75)),int(np.floor(sy*.25)):int(np.ceil(sy*.75)),int(np.floor(sz*.25)):int(np.ceil(sz*.75))]
    
    return im



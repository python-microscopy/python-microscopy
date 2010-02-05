import sys
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage

def calcCorrelates(data, nOrders=5, startAt=50, filtHalfWidth=25, stopAt=sys.maxint):
    '''Calculate the autocorrelations up to nOrders for SOFI inmaging'''
    d3c = np.zeros(list(data.getSliceShape()) + [1, nOrders])
    d_m = np.zeros(data.getSliceShape())
    d_mm = np.zeros(list(data.getSliceShape()) + [1])

    nm = 1./(2*filtHalfWidth)
    stopAt = min(stopAt, data.shape[2]-filtHalfWidth)

    for i in range(startAt-filtHalfWidth,startAt+filtHalfWidth):
        d_m += nm*data.getSlice(i)

    for i in range(startAt+1, stopAt):
        #print data[:,:,i-filtHalfWidth - 1].shape
        d_m = d_m - data.getSlice(i-filtHalfWidth - 1)*nm + data.getSlice(i+filtHalfWidth-1)*nm
        d_ = data.getSlice(i)
        d_mm[:,:,0] += d_
        d_ = d_ - d_m
        d_2 = np.ones(d_.shape)
        for j in range(d3c.shape[3]):
            d_2 *= d_
            if  (j % 2):
                #print d3c.shape, d_2.shape, d_m.shape
                d3c[:,:,0,j] += d_2
            else:
                d3c[:,:,0,j] += np.abs(d_2)

    return d3c/(stopAt - startAt), d_mm/(stopAt - startAt)

def calcCorrelatesI(data, nOrders=5, zoom=4, startAt=50, filtHalfWidth=25, stopAt=sys.maxint):
    '''Calculate the autocorrelations up to nOrders for SOFI inmaging'''
    slShape = list(np.array(data.getSliceShape())*zoom)
    d3c = np.zeros(slShape + [1, nOrders])
    d_m = np.zeros(slShape)
    d_mm = np.zeros(slShape + [1])

    nm = 1./(2*filtHalfWidth)
    stopAt = min(stopAt, data.shape[2]-filtHalfWidth)

    for i in range(startAt-filtHalfWidth,startAt+filtHalfWidth):
        d_m += nm*ndimage.zoom(data.getSlice(i), zoom)

    for i in range(startAt+1, stopAt):
        #print data[:,:,i-filtHalfWidth - 1].shape
        d_m = d_m - ndimage.zoom(data.getSlice(i-filtHalfWidth - 1), zoom)*nm + ndimage.zoom(data.getSlice(i+filtHalfWidth-1), zoom)*nm
        d_ = ndimage.zoom(data.getSlice(i), zoom)
        d_mm[:,:,0] += d_
        d_ = d_ - d_m
        d_2 = np.ones(d_.shape)
        for j in range(d3c.shape[3]):
            d_2 *= d_
            if  (j % 2):
                #print d3c.shape, d_2.shape, d_m.shape
                d3c[:,:,0,j] += d_2
            else:
                d3c[:,:,0,j] += np.abs(d_2)

    return d3c/(stopAt - startAt), d_mm/(stopAt - startAt)

def calcCumulants(corr):
    cum = corr.copy()

    for n in range(3,cum.shape[3]):
        for i in range(1, n-1):
            #print n+1, i+1, sp.comb(n, (i+1)), ((n+1)-(i+1), i+1), (n-i - 1, i)
            cum[:,:,:,n] -= sp.comb(n, (i+1))*cum[:,:,:,(n-i -1)]*corr[:,:,:,i]

    return cum


def calcCorrelatesZ(data, zm, nOrders=5, startAt=50, filtHalfWidth=25, stopAt=sys.maxint):
    '''Calculate the autocorrelations up to nOrders for SOFI inmaging'''

    zvals = np.array(list(set(zm.yvals)))
    zvals.sort()
    nZ = zvals.shape[0]

    d3c = np.zeros(list(data.getSliceShape()) + [nZ, nOrders])
    d_m = np.zeros(list(data.getSliceShape()))
    d_mm = np.zeros(list(data.getSliceShape()) + [nZ])

    nm = 1./(2*filtHalfWidth)
    stopAt = min(stopAt, data.shape[2]-filtHalfWidth)

    for i in range(startAt-filtHalfWidth,startAt+filtHalfWidth):
        d_m += nm*data.getSlice(i)

    for i in range(startAt+1, stopAt):
        #print zvals, zm(i)
        #print np.where((zvals - zm(i))**2 < .02)[0]
        z = int(np.where((zvals - zm(i))**2 < .02)[0])
        #print z, d3c.shape

        #print data[:,:,i-filtHalfWidth - 1].shape
        d_m = d_m - data.getSlice(i-filtHalfWidth - 1)*nm + data.getSlice(i+filtHalfWidth-1)*nm
        d_ = data.getSlice(i)
        d_mm[:,:,z] += d_
        d_ = d_ - d_m
        d_2 = np.ones(d_.shape)

        

        for j in range(d3c.shape[3]):
            d_2 *= d_
            if  (j % 2):
                #print d3c.shape, d_2.shape, d_m.shape, z, j
                #print d3c[:,:,z,j].shape

                d3c[:,:,z,j] += d_2[:,:]
            else:
                d3c[:,:,z,j] += np.abs(d_2)

    return d3c/(stopAt - startAt), d_mm/(stopAt - startAt)
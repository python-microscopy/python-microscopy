#!/usr/bin/python

##################
# twoColour.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.FileUtils.read_kdf import ReadKdfData
import numpy as np
import scipy as sp
import ofind
from PYME.Analysis.FitFactories.LatGaussFitFR import FitFactory, FitResultsDType
import MetaData
from scipy.interpolate import Rbf
from scikits import delaunay
import tables

#from PYME.Analysis.FitFactories import LatGaussFitFRTC

MetaData.TIRFDefault.tIndex = 0

def read_bead_data(filename):
    d1 = ReadKdfData(filename).squeeze()

    g = d1[:, :256]
    r = d1[:, 256:]
    r = np.fliplr(r)

    return (g,r)

def read_h5f_cols(h5f, slice):
    '''extracts colours from a h5 slice - file should be open!'''
    d1 = h5f.root.ImageData[slice]

    g = d1[:, :256]
    r = d1[:, 256:]
    r = np.fliplr(r)

    return (g,r)


def genRGBImage(g,r, gsat = 1, rsat= 1):
    g_ = g.astype('f') - g.min()
    g_ = np.minimum(gsat*g_/g_.max(), 1)

    r_ = r.astype('f') - r.min()
    r_ = np.minimum(rsat*r_/r_.max(), 1)

    b_ = np.zeros(g_.shape)

    return np.concatenate((r_.reshape(512,256,1),g_.reshape(512,256,1),b_.reshape(512,256,1)), 2)


def fitIndep(g,r,ofindThresh):
    rg = r + g #detect objects in sum image
    
    ofd = ofind.ObjectIdentifier(rg)

    ofd.FindObjects(ofindThresh, blurRadius=2)

    res_g = np.empty(len(ofd), FitResultsDType)
    res_r = np.empty(len(ofd), FitResultsDType)

    ff_g = FitFactory(g.reshape(512,256,1), MetaData.TIRFDefault)
    ff_r = FitFactory(r.reshape(512,256,1), MetaData.TIRFDefault)
    
    for i in range(len(ofd)):    
        p = ofd[i]
        res_g[i] = ff_g.FromPoint(round(p.x), round(p.y))
        res_r[i] = ff_r.FromPoint(round(p.x), round(p.y))

    return(res_g, res_r)


def dispRatio(res_g, res_r):
    Ag = res_g['fitResults']['A']
    Ar = res_r['fitResults']['A']

    
    scatter(Ag, Ar, c=Ag/Ar, cmap=cm.RdYlGn)


def genShiftVectors(res_g, res_r):
    ind1 = (res_g['fitResults']['A'] > 10)*(res_g['fitResults']['A'] < 500)*(res_g['fitResults']['sigma'] > 100)*(res_g['fitResults']['sigma'] < 400)*(res_g['fitError']['x0'] < 50)


    x = res_g['fitResults']['x0'][ind1]
    x = x + 0.5*sp.randn(len(x)) #add a bit of 'fuzz' to avoid duplicate points which could crash interpolation
    
    y = res_g['fitResults']['y0'][ind1]
    sx = res_g['fitResults']['x0'][ind1] - res_r['fitResults']['x0'][ind1]
    sy = res_g['fitResults']['y0'][ind1] - res_r['fitResults']['y0'][ind1]

    T = delaunay.Triangulation(x,y)

    nx = []
    ny = []
    nsx = []
    nsy = []


    #remove any shifts which are markedly different from their neighbours
    for i in range(len(x)):
        i1,i2 = np.where(T.edge_db == i)
        i_ = T.edge_db[i1, 1-i2]
        if (abs(sx[i] - np.median(sx[i_])) < 100) and (abs(sy[i] - np.median(sy[i_])) < 100):
            nx.append(x[i])
            ny.append(y[i])
            nsx.append(sx[i])
            nsy.append(sy[i])
        else:
            print 'point %d dropped' %i

    nx = np.array(nx)
    ny = np.array(ny)
    nsx = np.array(nsx)
    nsy = np.array(nsy)
    

    return (nx, ny, nsx, nsy)


def genShiftVectorField(nx,ny, nsx, nsy):
    '''interpolates shift vectors using radial basis functions'''
    rbx = Rbf(nx, ny, nsx, epsilon=1)
    rby = Rbf(nx, ny, nsy, epsilon=1)

    X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))


    dx = rbx(X,Y)
    dy = rby(X,Y)

    return (dx.T, dy.T, rbx, rby)


def genShiftVectorFieldMC(nx,ny, nsx, nsy, p, Nsamp):
    '''interpolates shift vectors using several monte-carlo subsampled 
    sets of the vectors and averages to give a smooth field'''

    X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))

    dx = np.zeros(X.shape)
    dy = np.zeros(X.shape)

    nIt = 0

    for i in range(Nsamp):
        r_ind = (sp.rand(len(nx)) < p)
        
        if r_ind.sum() > 2:

            rbx = Rbf(nx[r_ind], ny[r_ind], nsx[r_ind], epsilon=1)
            rby = Rbf(nx[r_ind], ny[r_ind], nsy[r_ind], epsilon=1)

            dx = dx + rbx(X,Y)
            dy = dy + rby(X,Y)
            nIt += 1

    return dx.T/nIt, dy.T/nIt
    
def getCorrection(x,y,x_sv, y_sv):
    '''looks up correction in calculated vector fields'''
    xi = np.maximum(np.minimum(sp.round_(x/100).astype('i'), x_sv.shape[0]),0)
    yi = np.maximum(np.minimum(sp.round_(y/100).astype('i'), x_sv.shape[1]),0)
    return (x_sv[xi, yi],y_sv[xi, yi])


def calcCorrections(filenames):
    gs = []
    rs = []
    
    res_gs = []
    res_rs = []
    
    dxs = []
    dys = []
    
    Ags = []
    Ars = []

    for fn in filenames:
        if fn.split('.')[-1] == 'kdf':
            g,r = read_bead_data(fn)
        else:
            h5f = tables.openFile(fn)
            g,r = read_h5f_cols(h5f, 0)
            h5f.close()

        res_g, res_r = res_g, res_r = fitIndep(g,r,6)

        nx, ny, nsx, nsy = genShiftVectors(res_g, res_r)

        dx, dy = genShiftVectorFieldMC(nx, ny, nsx, nsy, .5, 10)

        dxs.append(dx)
        dys.append(dy)
        
        res_rs.append(res_r)
        res_gs.append(res_g)
        
        Ars.append(res_r['fitResults']['A'])
        Ags.append(res_g['fitResults']['A'])

        gs.append(g)
        rs.append(r)

    dx = (np.median(np.array(dxs), 0))
    dy = (np.median(np.array(dys), 0))

    return (gs, rs, res_gs, res_rs, Ags, Ars, dx, dy)

def warpCorrectRedImage(r, dx, dy):
    X, Y = sp.meshgrid(np.arange(0, 512*70, 70), np.arange(0, 256*70,70))
    cx, cy = getCorrection(X,Y, dx, dy)


    T = delaunay.Triangulation((X + cx).ravel(),(Y+cy).ravel())
    In = delaunay.LinearInterpolator(T, r.T.ravel(), r.min())

    vals =In[0:256*70:256*1j, 0:512*70:512*1j]

    return vals.T




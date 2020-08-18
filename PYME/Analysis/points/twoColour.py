#!/usr/bin/python

##################
# twoColour.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

#from PYME.IO.FileUtils.read_kdf import ReadKdfData
import numpy as np
import scipy as sp
from PYME.localization import ofind
from PYME.Analysis import MetaData
from scipy.interpolate import Rbf, SmoothBivariateSpline
#from matplotlib import delaunay
import tables

#from PYME.localization.FitFactories import LatGaussFitFRTC

MetaData.TIRFDefault.tIndex = 0

def read_bead_data(filename):
    d1 = ReadKdfData(filename).squeeze()

    g = d1[:, :256]
    r = d1[:, 256:]
    r = np.fliplr(r)

    return (g,r)

def read_h5f_cols(h5f, slice):
    """extracts colours from a h5 slice - file should be open!"""
    d1 = h5f.root.ImageData[slice]

    g = d1[:, :256]
    r = d1[:, 256:]
    r = np.fliplr(r)

    return (g,r)
    
    


def shift_and_rot_model_eval(p, x, y):
    x0, y0, a, b, c, d = p#, e, y1 = p
    x_0 = x - x0
    y_0 = y - y0
    xp = x0 + a*(x_0) + b*(y_0) #+ e*x_0*(y_0 - y1)
    yp = y0 + c*(x_0) + d*(y_0) #+ f*x_0*y_0
    return (xp - x), (yp - y)
    
def shift_and_rot_model(p, x, y, dx, dy):
    dxm, dym = shift_and_rot_model_eval(p, x, y)
    #x0, y0, th, x1, y1, a = p
    #xp = x1 + a*np.sin(th)*(y-y0) + a*np.cos(th)*x
    #yp = y1 -a*np.sin(th)*(x-x0) + a*np.cos(th)*y
    return ((dxm -dx)**2 + (dym - dy)**2).sum()
    
    
#robust fitting of a linear shift model (magnification difference)
import scipy.stats
from scipy.optimize import fmin
from scipy import linalg

def robustLinLhood(p, x, y, var=1):
    """p is parameter vector, x and y as expected, and var the variance of the
    y value. We use a t-distribution as our likelihood as it's long tails will
    not overly weight outliers."""
    m, x0 = p
    err = (y - m*(x - x0))/var
    return -scipy.stats.t.logpdf(err, 1).sum()
    

class ShiftModel(object):
    def __init__(self , *args, **kwargs):
        """
        To recreate shiftmap from dictionary of fit results,
        call the model with a keyword argument 'dict', i.e. linModel(dict=shiftmap.__dict__)
        """
        if 'dict' in kwargs.keys():
            self.__dict__.update(kwargs['dict'])
        else:
            self.fit(*args, **kwargs)
            
    def to_JSON(self):
        import json
        
        cn = '.'.join([self.__class__.__module__, self.__class__.__name__])       
        return json.dumps({cn:self.__dict__})
    
    @classmethod
    def from_md_entry(cls, mdentry):
        import json
        import importlib
        
        cn, dict = json.loads(mdentry).items()[0]
        
        parts = cn.split('.')
        mod, cln = '.'.join(parts[:-1]), parts[-1]
        
        cl = getattr(importlib.import_module(mod), cln)
        
        return cl(dict=dict)

    
class linModel(ShiftModel):
        
    def fit(self, x, dx, var, axis):
        #do a simple linear fit to estimate start parameters
        pstart = linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, dx)[0]
        print(pstart)
        
        #now do a maximum likelihood fit with our robust lhood function
        self.m, self.x0 = fmin(robustLinLhood, [pstart[0],-pstart[1]/pstart[0]], args=(x, dx, var))
        
        self.axis = axis
        
    def ev(self, x, y):
        """Mimic a bivariate spline object. Since we're assuming it is linear
        along one axis, we use the axis that was defined when fitting the model"""
        if self.axis == 'x':
            return self.m*(x - self.x0)
        else:
            return self.m*(y - self.x0)
            
            
def robustLin2Lhood(p, x, y, dx, var=1):
    """p is parameter vector, x and y as expected, and var the variance of the
    y value. We use a t-distribution as our likelihood as it's long tails will
    not overly weight outliers."""
    mx, my, x0 = p
    err = (dx - (mx*x + my*y + x0))/var
    return -scipy.stats.t.logpdf(err, 1).sum()
    
class lin2Model(ShiftModel):
    def fit(self, x, y, dx, var=1):
        #do a simple linear fit to estimate start parameters
        pstart = linalg.lstsq(np.vstack([x, y, np.ones_like(x)]).T, dx)[0]
        print(pstart)
        
        #now do a maximum likelihood fit with our robust lhood function
        self.mx, self.my, self.x0 = fmin(robustLin2Lhood, pstart, args=(x, y,dx, var))
        
        #self.axis = axis
        
    def ev(self, x, y):
        """Mimic a bivariate spline object. Since we're assuming it is linear
        along one axis, we use the axis that was defined when fitting the model"""
        return self.mx*x +self.my*y + self.x0
            
def genShiftVectorFieldLinear(x,y, dx, dy, err_sx, err_sy):
    """interpolates shift vectors using smoothing splines"""

    spx = lin2Model(x, y, dx, err_sx**2)
    spy = lin2Model(x, y, dy, err_sy**2)

    #X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))

    #dx = spx.ev(X.ravel(),Y.ravel()).reshape(X.shape)
    #dy = spy.ev(X.ravel(),Y.ravel()).reshape(X.shape)

    return spx, spy
    
def robustLin3zLhood(p, x, y, z, dx, var=1):
    """p is parameter vector, x and y as expected, and var the variance of the
    y value. We use a t-distribution as our likelihood as it's long tails will
    not overly weight outliers."""
    mx, my, mx2, my2, mxy, mxy2, mx2y, mx3, x0, mz, mxz, myz, mxyz = p
    err = (dx - (mx*x + my*y + mx2*x*x +my2*y*y + mxy*x*y + mxy2*x*y*y + mx2y*x*x*y + mx3*x*x*x + x0 + mz*z + mxz*x*z + myz*y*z + mxyz*x*y*z))/var
    return -scipy.stats.t.logpdf(err, 1).sum()
    
class lin3zModel(ShiftModel):
    ZDEPSHIFT = True
    sc = 1./18e3
    def fit(self, x, y, z, dx, var=1):
        x = x*self.sc
        y = y*self.sc
        #do a simple linear fit to estimate start parameters
        pstart = linalg.lstsq(np.vstack([x, y, x*x, y*y, x*y, x*y*y, x*x*y,x*x*x, np.ones_like(x), z, z*x, z*y, z*x*y]).T, dx)[0]
        print(pstart)
        
        
        #now do a maximum likelihood fit with our robust lhood function
        self.mx, self.my, self.mx2, self.my2, self.mxy, self.mxy2, self.mx2y,self.mx3, self.x0, self.mz, self.mxz, self.myz, self.mxyz = fmin(robustLin3zLhood, pstart, args=(x, y, z,dx, var))
        
        self.my3 = 0
        
        #self.axis = axis
        
    def ev(self, x, y, z=0):
        """Mimic a bivariate spline object. Since we're assuming it is linear
        along one axis, we use the axis that was defined when fitting the model"""
        x = x*self.sc
        y = y*self.sc
        return self.mx*x +self.my*y + self.mx2*x*x + self.my2*y*y + self.mxy*x*y + self.mxy2*x*y*y + self.mx2y*x*x*y + self.mx3*x*x*x + self.my3*y*y*y  + self.x0 + self.mz*z + self.mxz*x*z +self.myz*y*z + self.mxyz*z*x*y
        
    def __call__(self, x, y, z=0):
        return self.ev(x, y, z)
        
def robustLin3Lhood(p, x, y, dx, var=1):
    """p is parameter vector, x and y as expected, and var the variance of the
    y value. We use a t-distribution as our likelihood as it's long tails will
    not overly weight outliers."""
    mx, my, mx2, my2, mxy, mxy2, mx2y, mx3, x0 = p
    err = (dx - (mx*x + my*y + mx2*x*x +my2*y*y + mxy*x*y + mxy2*x*y*y + mx2y*x*x*y + mx3*x*x*x + x0))/var
    return -scipy.stats.t.logpdf(err, 1).sum()
    
class lin3Model(ShiftModel):
    sc = 1./18e3
    def fit(self, x, y, dx, var=1):
        x = x*self.sc
        y = y*self.sc
        #do a simple linear fit to estimate start parameters
        pstart = linalg.lstsq(np.vstack([x, y, x*x, y*y, x*y, x*y*y, x*x*y,x*x*x, np.ones_like(x)]).T, dx)[0]
        print(pstart)
        
        
        #now do a maximum likelihood fit with our robust lhood function
        self.mx, self.my, self.mx2, self.my2, self.mxy, self.mxy2, self.mx2y,self.mx3, self.x0 = fmin(robustLin3Lhood, pstart, args=(x, y,dx, var))
        
        self.my3 = 0
        
        #self.axis = axis
        
    def ev(self, x, y):
        """Mimic a bivariate spline object. Since we're assuming it is linear
        along one axis, we use the axis that was defined when fitting the model"""
        x = x*self.sc
        y = y*self.sc
        return self.mx*x +self.my*y + self.mx2*x*x + self.my2*y*y + self.mxy*x*y + self.mxy2*x*y*y + self.mx2y*x*x*y + self.mx3*x*x*x + self.my3*y*y*y  + self.x0
        
    def __call__(self, x, y):
        return self.ev(x, y)
            
def genShiftVectorFieldQuad(x,y, dx, dy, err_sx, err_sy):
    """interpolates shift vectors using smoothing splines"""

    spx = lin3Model(x, y, dx, err_sx**2)
    spy = lin3Model(x, y, dy, err_sy**2)

    #X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))

    #dx = spx.ev(X.ravel(),Y.ravel()).reshape(X.shape)
    #dy = spy.ev(X.ravel(),Y.ravel()).reshape(X.shape)

    return spx, spy
    
def genShiftVectorFieldQ(nx,ny, nsx, nsy, err_sx, err_sy, bbox=None):
    """interpolates shift vectors using smoothing splines"""
    wonky = findWonkyVectors(nx, ny, nsx, nsy, tol=2*err_sx.mean())
    #wonky = findWonkyVectors(nx, ny, nsx, nsy, tol=100)
    good = wonky == 0

    print(('%d wonky vectors found and discarded' % wonky.sum()))
    
    #if bbox:
    #    spx = SmoothBivariateSpline(nx[good], ny[good], nsx[good], 1./err_sx[good], bbox=bbox)
    #    spy = SmoothBivariateSpline(nx[good], ny[good], nsy[good], 1./err_sy[good], bbox=bbox)
    #else:
    #    spx = SmoothBivariateSpline(nx[good], ny[good], nsx[good], 1./err_sx[good])
    #    spy = SmoothBivariateSpline(nx[good], ny[good], nsy[good], 1./err_sy[good])
    
    spx, spy = genShiftVectorFieldQuad(nx[good], ny[good], nsx[good], nsy[good], err_sx[good], err_sy[good])

    X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))

    dx = spx.ev(X.ravel(),Y.ravel()).reshape(X.shape)
    dy = spy.ev(X.ravel(),Y.ravel()).reshape(X.shape)

    return (dx.T, dy.T, spx, spy, good)
    

def genShiftVectorFieldQuadz(x,y, z, dx, dy, err_sx, err_sy):
    """interpolates shift vectors using smoothing splines"""

    spx = lin3zModel(x, y, z, dx, err_sx**2)
    spy = lin3zModel(x, y, z, dy, err_sy**2)

    #X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))

    #dx = spx.ev(X.ravel(),Y.ravel()).reshape(X.shape)
    #dy = spy.ev(X.ravel(),Y.ravel()).reshape(X.shape)

    return spx, spy
    
def genShiftVectorFieldQz(nx,ny, nz, nsx, nsy, err_sx, err_sy, bbox=None):
    """interpolates shift vectors using smoothing splines"""
    wonky = findWonkyVectors(nx, ny, nsx, nsy, tol=5*err_sx.mean())
    #wonky = findWonkyVectors(nx, ny, nsx, nsy, tol=100)
    good = wonky == 0

    print(('%d wonky vectors found and discarded' % wonky.sum()))
    
    #if bbox:
    #    spx = SmoothBivariateSpline(nx[good], ny[good], nsx[good], 1./err_sx[good], bbox=bbox)
    #    spy = SmoothBivariateSpline(nx[good], ny[good], nsy[good], 1./err_sy[good], bbox=bbox)
    #else:
    #    spx = SmoothBivariateSpline(nx[good], ny[good], nsx[good], 1./err_sx[good])
    #    spy = SmoothBivariateSpline(nx[good], ny[good], nsy[good], 1./err_sy[good])
    
    spx, spy = genShiftVectorFieldQuadz(nx[good], ny[good], nz[good], nsx[good], nsy[good], err_sx[good], err_sy[good])

    X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))

    dx = spx.ev(X.ravel(),Y.ravel()).reshape(X.shape)
    dy = spy.ev(X.ravel(),Y.ravel()).reshape(X.shape)

    return (dx.T, dy.T, spx, spy, good)


def genRGBImage(g,r, gsat = 1, rsat= 1):
    g_ = g.astype('f') - g.min()
    g_ = np.minimum(gsat*g_/g_.max(), 1)

    r_ = r.astype('f') - r.min()
    r_ = np.minimum(rsat*r_/r_.max(), 1)

    b_ = np.zeros(g_.shape)

    return np.concatenate((r_.reshape(512,256,1),g_.reshape(512,256,1),b_.reshape(512,256,1)), 2)


def fitIndep(g,r,ofindThresh):
    from PYME.localization.FitFactories.LatGaussFitFR import FitFactory, FitResultsDType
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
    from matplotlib import tri
    ind1 = (res_g['fitResults']['A'] > 10)*(res_g['fitResults']['A'] < 500)*(res_g['fitResults']['sigma'] > 100)*(res_g['fitResults']['sigma'] < 400)*(res_g['fitError']['x0'] < 50)


    x = res_g['fitResults']['x0'][ind1]
    x = x + 0.5*sp.randn(len(x)) #add a bit of 'fuzz' to avoid duplicate points which could crash interpolation
    
    y = res_g['fitResults']['y0'][ind1]
    sx = res_g['fitResults']['x0'][ind1] - res_r['fitResults']['x0'][ind1]
    sy = res_g['fitResults']['y0'][ind1] - res_r['fitResults']['y0'][ind1]

    T = tri.Triangulation(x,y)

    nx = []
    ny = []
    nsx = []
    nsy = []


    #remove any shifts which are markedly different from their neighbours
    for i in range(len(x)):
        i1,i2 = np.where(T.edges == i)
        i_ = T.edges[i1, 1-i2]
        if (abs(sx[i] - np.median(sx[i_])) < 100) and (abs(sy[i] - np.median(sy[i_])) < 100):
            nx.append(x[i])
            ny.append(y[i])
            nsx.append(sx[i])
            nsy.append(sy[i])
        else:
            print(('point %d dropped' %i))

    nx = np.array(nx)
    ny = np.array(ny)
    nsx = np.array(nsx)
    nsy = np.array(nsy)
    

    return (nx, ny, nsx, nsy)

def findWonkyVectors(x, y,dx,dy, tol=100):
    from PYME.LMVis.visHelpers import genEdgeDB
    from matplotlib import tri
    T = tri.Triangulation(x,y)

    edb = genEdgeDB(T)

    wonkyVecs = np.zeros(len(x))


    #remove any shifts which are markedly different from their neighbours
    for i in range(len(x)):
        incidentEdges = T.edges[edb[i][0]]

#        d_x = np.diff(T.x[incidentEdges])
#        d_y = np.diff(T.y[incidentEdges])
#
#        dist = (d_x**2 + d_y**2)
#
#        di = np.mean(np.sqrt(dist))

        neighb = incidentEdges.ravel()
        neighb = neighb[(neighb == i) < .5]

        if (abs(dx[i] - np.median(dx[neighb])) > tol) or (abs(dy[i] - np.median(dy[neighb])) > tol):
            wonkyVecs[i] = 1

    return wonkyVecs > .5


def genShiftVectorField(nx,ny, nsx, nsy):
    """interpolates shift vectors using radial basis functions"""
    rbx = Rbf(nx, ny, nsx, epsilon=1)
    rby = Rbf(nx, ny, nsy, epsilon=1)

    X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))


    dx = rbx(X,Y)
    dy = rby(X,Y)

    return (dx.T, dy.T, rbx, rby)

def genShiftVectorFieldSpline(nx,ny, nsx, nsy, err_sx, err_sy, bbox=None):
    """interpolates shift vectors using smoothing splines"""
    wonky = findWonkyVectors(nx, ny, nsx, nsy, tol=2*err_sx.mean())
    #wonky = findWonkyVectors(nx, ny, nsx, nsy, tol=100)
    good = wonky == 0

    print(('%d wonky vectors found and discarded' % wonky.sum()))
    
    if bbox:
        spx = SmoothBivariateSpline(nx[good], ny[good], nsx[good], 1./err_sx[good], bbox=bbox)
        spy = SmoothBivariateSpline(nx[good], ny[good], nsy[good], 1./err_sy[good], bbox=bbox)
    else:
        spx = SmoothBivariateSpline(nx[good], ny[good], nsx[good], 1./err_sx[good])
        spy = SmoothBivariateSpline(nx[good], ny[good], nsy[good], 1./err_sy[good])

    X, Y = np.meshgrid(np.arange(0, 512*70, 100), np.arange(0, 256*70, 100))

    dx = spx.ev(X.ravel(),Y.ravel()).reshape(X.shape)
    dy = spy.ev(X.ravel(),Y.ravel()).reshape(X.shape)

    return (dx.T, dy.T, spx, spy, good)


def genShiftVectorFieldMC(nx,ny, nsx, nsy, p, Nsamp):
    """interpolates shift vectors using several monte-carlo subsampled
    sets of the vectors and averages to give a smooth field"""

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
    """looks up correction in calculated vector fields"""
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
            h5f = tables.open_file(fn)
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
    import warnings
    warnings.warn('Deprecated - use IO.Datasources.AlignDataSource or IO.Datasources.UnsplitDataSource instead')
    
    from matplotlib import tri
    X, Y = sp.meshgrid(np.arange(0, 512*70, 70), np.arange(0, 256*70,70))
    cx, cy = getCorrection(X,Y, dx, dy)


    T = tri.Triangulation((X + cx).ravel(),(Y+cy).ravel())
    In = tri.LinearTriInterpolator(T, r.T.ravel(), r.min())

    vals =In[0:256*70:256*1j, 0:512*70:512*1j]

    return vals.T


class sffake(ShiftModel):
    def fit(self, val):
        self.val = val

    def ev(self, x, y):
        return self.val

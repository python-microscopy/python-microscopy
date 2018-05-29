#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import numpy as np
from scipy.optimize import leastsq, fmin
from scipy.spatial import kdtree
import multiprocessing
from PYME.util.shmarray import shmarray

def f(p, x, y, z):
    return eval('%f*x**2 + %f*x + %f*y**2 + %f*y + %f*z**2 + %f*z + %f*x*y + %f*x*z + %f*y*z + %f -1' % tuple(p))

def gp(p, x, y, z):
    dx = 2*p[0]*x + p[1] + p[6]*y + p[7]*z
    dy = 2*p[2]*y + p[3] + p[6]*x + p[8]*z
    dz = 2*p[4]*y + p[5] + p[7]*x + p[8]*z
    
    return np.vstack([dx, dy, dz])
    
def gpn(p, x, y, z):
    dx = 2*p[0]*x + p[1] + p[6]*y + p[7]*z
    dy = 2*p[2]*y + p[3] + p[6]*x + p[8]*z
    dz = 2*p[4]*y + p[5] + p[7]*x + p[8]*z
    
    return dx*dx + dy*dy + dz*dz
    
def gpnA(p, A):
    px = np.array([2*p[0], p[6], p[7], p[1]])
    py = np.array([p[6], 2*p[2], p[8], p[3]])
    pz = np.array([p[7], p[8], 2*p[4], p[5]])
    
    #dx = 2*p[0]*x + p[1] + p[6]*y + p[7]*z
    #dy = 2*p[2]*y + p[3] + p[6]*x + p[8]*z
    #dz = 2*p[4]*y + p[5] + p[7]*x + p[8]*z
    dx = np.dot(A, px)
    dy = np.dot(A, py)
    dz = np.dot(A, pz)
    
    return dx*dx + dy*dy + dz*dz

iA = np.array([[0, 6, 7, 1],
               [6, 2, 8, 3],
               [7, 8, 4, 5]], dtype='i').T
#tA = np.zeros(iA.shape)              
               
def gpnA2(p, A):
    #P = np.array([[2*p[0], p[6], p[7], p[1]],
    #              [p[6], 2*p[2], p[8], p[3]],
    #              [p[7], p[8], 2*p[4], p[5]]]).T
    tA = np.zeros(iA.shape) 
    tA[:] = p[iA]
    tA[0,0]*=2
    tA[1,1]*=2
    tA[2,2]*=2
                  
    #print P.shape
    
    g = np.dot(A, tA).T
    
    return (g*g).sum(0)

def mf(p, x, y, z):
    return f(p, x, y, z)/np.sqrt(gpn(p, x, y, z))
    
def mfA(p, A, A2):
    t = (np.dot(A, p) - 1.0)
    return t/np.sqrt(gpnA(p, A2))
    
def mfA1(p, A, x,y,z, A2):
    t = (np.dot(A, p) - 1.0)
    #t = f(p, x, y, z)
    return t/np.sqrt(gpnA2(p, A2))
    #return t/np.sqrt(gpn(p, x, y, z))

def mf2(p, x, y, z):
    return (f(p, x, y, z)**2/((gp(p, x, y, z)**2).sum(0))).sum()
    

    
def rend(p, po, r, X, Y, Z, im, xm, ym, zm, xs, ys, zs, sc=1.0):
    pox, poy, poz = po
    
    ROIx = slice((-xm - r + pox)/xs,  (-xm + r + pox)/xs, 1)
    ROIy = slice((-ym - r + poy)/ys,  (-ym + r + poy)/ys, 1)
    ROIz = slice((-zm - r + poz)/zs,  (-zm + r + poz)/zs, 1)
    
    
    Xr = X[ROIx] - pox
    Yr = Y[ROIy] - poy
    Zr = Z[ROIz] - poz
    
    Xr = Xr[:,None,None]
    Yr = Yr[None, :,None]
    Zr = Zr[None,None, :]
    
    #print Xr
    
    fv =  np.abs(mf(p, Xr, Yr, Zr))
    
    
    im[ROIx, ROIy, ROIz] += (np.abs(fv) <1.5)/sc
    
    return fv
    
def fitPts(pts, NFits = 0):
    xm, ym, zm = pts.min(0)
    xmx, ymx, zmx = pts.max(0)
    X, Y, Z = np.mgrid[xm:xmx:5, ym:ymx:5, zm:zmx:5]
    im = np.zeros(X.shape)
    
    kdt = kdtree.KDTree(pts)
    
    if NFits == 0:
        NFits = pts.shape[0]
    
    for i in np.arange(NFits):
        ptr = pts[kdt.query_ball_point(pts[i, :], 150),:]
        #po = mean(ptr, 0)
        po = pts[i, :]
        
        if ptr.shape[0] > 10:
            ptr = ptr - po[None, :]
            ps  = []
            mfs = []
            
            for i in range(5):
                op = leastsq(mf, list(1 + .1*np.random.normal(size=9)) + [-1], args=(ptr[:,0],ptr[:,1],ptr[:,2]), full_output=1, epsfcn=1)
                ps.append(op[0])
                mfs.append((op[2]['fvec']**2).sum())
                
            #print mfs
            p3 = ps[np.argmin(mfs)]
        
        fv = rend(p3, po, 100, X, Y, Z, im, xm, ym, zm, 5,5,5, 1.0/ptr.shape[0])
        
    return im
    
def fitPtA(kdt, pts, i):
    from scipy.optimize import leastsq
    #ptr = pts[kdt.query_ball_point(pts[i, :], 150),:]
    d, pi = kdt.query(pts[i, :], 50, distance_upper_bound=150)
    ptr = pts[pi[pi<pts.shape[0]],:]
    #po = mean(ptr, 0)
    po = pts[i, :]
    
    Np = ptr.shape[0]
    
    #print 'f1'
    
    if Np > 10:
        ptr = ptr - po[None, :]
                
        x, y, z = ptr[:,0],ptr[:,1],ptr[:,2]
        
        A = np.vstack([x*x, x, y*y, y, z*z, z, x*y, x*z, y*z, np.ones_like(x)]).T
        A2 = np.vstack([x, y, z, np.ones_like(x)]).T
        ps  = []
        mfs = []
        #print 'f1f'
        
        for j in range(5):
            #print 'l', j
            #op = leastsq(mfA1, list(1 + .5*np.random.normal(size=9)) + [-1], args=(A, x,y,z, A2), full_output=1, epsfcn=1)
            op = leastsq(mfA1, 5*np.random.normal(size=10), args=(A, x,y,z, A2), full_output=1, epsfcn=1)
            #op = leastsq(mf, list(0 + .1*np.random.normal(size=9)) + [1], args=(x,y,z), full_output=1, epsfcn=1)
            #print 'll'
            ps.append(op[0])
            mfs.append((op[2]['fvec']**2).sum())
            
        #print mfs
        p3 = ps[np.argmin(mfs)]
        
        #print 'ff'
    
        return p3, po, Np
    else:
        return None
    
def fitPtsA(pts, NFits = 0):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(pts)
    #kdt = kdtree.KDTree(pts)
    
    if NFits == 0:
        NFits = pts.shape[0]
        
    surfs = []
    
    for i in np.arange(NFits):
       r = fitPtA(kdt, pts, i)
       if not r == None:
           surfs.append(r)
           
    return surfs
    
def fitPtsAt(pts, kdt, ivals, res, pos, nPs):
    #kdt = kdtree.KDTree(pts)
    
    for i in ivals:
        print i
        r = fitPtA(kdt, pts, i)
        if not r == None:
            p3, po, Np = r
            res[:,i] = p3
            pos[:,i] = po
            nPs[i] = Np
        else:
            nPs[i] = 0
        print 'f', i
    
def fitPtsAP(pts, NFits = 0):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(pts)
    
    if NFits == 0:
        NFits = pts.shape[0]
        
    #surfs = []
        
    #def _task(i):
    #    return fitPtA(kdt, pts, i)
    
    nCPUs = multiprocessing.cpu_count()
    nCPUs = 1
    
    res = shmarray.zeros((10, NFits))
    pos = shmarray.zeros((3, NFits))
    nPs = shmarray.zeros(NFits)
    #rt = shmarray.zeros((2,NFits))
        
    fnums = range(NFits)
        
    processes = [multiprocessing.Process(target = fitPtsAt, args=(pts, kdt, fnums[i::nCPUs], res, pos, nPs)) for i in range(nCPUs)]

    for p in processes:
        print p
        p.start()
    

    for p in processes:
        print p
        p.join()
        

    return res, pos, nPs
    
    
def rendSurfs(surfs, pts, vs=5):
    xm, ym, zm = pts.min(0)
    xmx, ymx, zmx = pts.max(0)
    X, Y, Z = np.ogrid[xm:xmx:vs], np.ogrid[ym:ymx:vs], np.ogrid[zm:zmx:vs]
    im = np.zeros((len(X), len(Y), len(Z)))
    
    for p3, po, Np in surfs:
        rend(p3, po, 100, X, Y, Z, im, xm, ym, zm, vs,vs,vs, 1.0/Np)
        
    return im
    
def rendSurfsT(res, pos, nPs, pts):
    xm, ym, zm = pts.min(0)
    xmx, ymx, zmx = pts.max(0)
    X, Y, Z = np.mgrid[xm:xmx:5, ym:ymx:5, zm:zmx:5]
    im = np.zeros(X.shape)
    
    for p3, po, Np in zip(res, pos, nPs):
        if Np > 0:
            rend(p3, po, 100, X, Y, Z, im, xm, ym, zm, 5,5,5, 1.0/Np)
        
    return im
    
################################
# rotated surface of the form z = a*x^2 + b*y^2
    
def quad_rot_surf_misfit(p, pts, pos=None):
    #print p.shape, p.dtype
    if pos is None:
        pos = p[:3]
        theta, phi, psi, A, B = p[3:]
    else:
        theta, phi, psi, A, B = p
    
    #xs, ys, zs = x-x0, y-y0, z - z0
    pts = pts - pos[:,None]
    
    ctheta = np.cos(theta)
    cphi = np.cos(phi)
    cpsi = np.cos(psi)
    stheta = np.sin(theta)
    sphi = np.sin(phi)
    spsi = np.sin(psi)
    
    Ax = np.array([[1,0,0],[0, cphi, sphi], [0, -sphi, cphi]])
    Ay = np.array([[ctheta, 0, -stheta], [0, 1, 0], [stheta, -0, ctheta]])
    Az = np.array([[cpsi, spsi, -0], [-spsi, cpsi, 0], [0,0,1]])
    
    pts_rot = Ax.dot(Ay).dot(Az).dot(pts)
    
    #xr = ctheta*cpsi*xs + (cphi*spsi + sphi*stheta*cpsi)*ys + (stheta*spsi - cphi*stheta*cpsi)*zs
    #yr = -ctheta*spsi*xs + (cphi*cpsi - sphi*stheta*spsi)*ys + (sphi*cpsi + cphi*stheta*spsi)*zs
    #zr = stheta*xs -sphi*ctheta*ys + cphi*ctheta*zs
    
    xr, yr, zr = pts_rot
    
    return zr - (A*xr*xr + B*yr*yr)
  
def gen_quad_rot_surf(p, xr, yr):
    x0, y0, z0, theta, phi, psi, A, B = p
    ctheta = np.cos(-theta)
    cphi = np.cos(-phi)
    cpsi = np.cos(-psi)
    stheta = np.sin(-theta)
    sphi = np.sin(-phi)
    spsi = np.sin(-psi)

    Ax = np.array([[1, 0, 0], [0, cphi, sphi], [0, -sphi, cphi]])
    Ay = np.array([[ctheta, 0, -stheta], [0, 1, 0], [stheta, -0, ctheta]])
    Az = np.array([[cpsi, spsi, -0], [-spsi, cpsi, 0], [0, 0, 1]])
    
    zr = (A/1)*xr*xr + (B/1)*yr*yr
    pts_r = np.vstack([xr, yr, zr])
    
    n = np.sqrt(4*(A*A*xr*xr + B*B*yr*yr) + 1)
    N = np.vstack([2*A*xr, 2*B*yr, -1 + 0*xr])/n[None,:]
    
    R = Az.dot(Ay).dot(Ax)
    
    return R.dot(pts_r) + p[:3][:,None], R.dot(N)

def fit_quad_surf(pts, control_pt, fitPos=True):
    from PYMEnf.Analysis import arcfit
    sp = 2*np.random.randn(8).astype('f')
    #sp[-2]+= 10
    sp[:3] = control_pt
    
    if fitPos:
        #return leastsq(quad_rot_surf_misfit,sp, args=(pts,))[0]
        return leastsq(arcfit.quad_surf_mf, sp, args=(pts[0,:], pts[1,:], pts[2,:]))[0]
    else:
        res = np.copy(sp)
        #res[3:] = leastsq(quad_rot_surf_misfit,sp[3:], args=(pts,control_pt))[0]
        res[3:] = leastsq(arcfit.quad_surf_mf_fpos, sp[3:], args=(pts[0,:], pts[1,:], pts[2,:], control_pt))[0]
        return res
    

_xr = None
_yr = None
_rec_grid_settings = None

def _get_reconstr_grid(radius=50., step=10.):
    global _xr, _yr, _rec_grid_settings
    
    if not _rec_grid_settings == (radius, step):
        x, y = np.mgrid[-radius:radius:float(step), -radius:radius:float(step)]
        _xr = x.ravel()
        _yr = y.ravel()
        _rec_grid_settings = (radius, step)
        
    return _xr, _yr

def reconstruct_quad_surf(p, control_point, N, radius=50, step=10.0):
    xr, yr = _get_reconstr_grid(radius, step)

    sp, normals = gen_quad_rot_surf(p, xr, yr)
    
    d = sp - control_point[:,None]
    
    mask = (d*d).sum(0) < radius*radius
    
    return np.vstack([sp[:,mask], normals[:,mask], 0*sp[0,mask] + N])


def reconstruct_quad_surf_region_cropped(p, control_point, N, kdt, data, radius=50, step=10.0, fit_radius=100.):
    from scipy.spatial import Delaunay
    
    pts = data[kdt.query_ball_point(control_point, fit_radius),:]
    T  = Delaunay(pts)
    
    xr, yr = _get_reconstr_grid(radius, step)
    
    sp, normals = gen_quad_rot_surf(p, xr, yr)
    
    d = sp - control_point[:, None]
    
    mask = (d * d).sum(0) < radius * radius
    sp = sp[:,mask]
    
    mask = T.find_simplex(sp.T) > 0
    
    return np.vstack([sp[:, mask], normals[:,mask], 0*sp[0,mask] + N])

def fit_quad_surf_to_neighbourbood(data, kdt, i, radius=100, fitPos=True):
    pt= data[i,:]
    pts = data[kdt.query_ball_point(pt, radius),:]
    
    N = len(pts)
    
    if N < 10:
        return
    
    return (fit_quad_surf(pts.T, pt, fitPos=fitPos), pt, N)
    
def fit_quad_surfaces(data, radius, fitPos=False):
    from scipy.spatial import cKDTree
    
    kdt = cKDTree(data)
    
    fits = [fit_quad_surf_to_neighbourbood(data, kdt, i, radius, fitPos=fitPos) for i in range(len(data))]
    
    return fits


def fit_quad_surfaces_t(data, kdt, ivals, res, pos, nPs, radius=100 ,fitPos=True):
    #kdt = kdtree.KDTree(pts)
    
    for i in ivals:
        #print i
        r = fit_quad_surf_to_neighbourbood(data, kdt, i, radius, fitPos=fitPos)
        if not r == None:
            p, pt, N = r
            res[:, i] = p
            pos[:, i] = pt
            nPs[i] = N
        else:
            nPs[i] = 0
        #print 'f', i


SURF_PATCH_DTYPE_FLAT = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('theta', 'f4'), ('phi', 'f4'), ('psi', 'f4'), ('A', 'f4'), ('B', 'f4'),
                    ('x0', 'f4'), ('y0', 'f4'), ('z0', 'f4'),
                    ('N', 'i4')]

SURF_PATCH_DTYPE = [('results', [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('theta', 'f4'), ('phi', 'f4'), ('psi', 'f4'), ('A', 'f4'), ('B', 'f4')]),
                    ('pos', [('x0', 'f4'), ('y0', 'f4'), ('z0', 'f4')]),
                    ('N', 'i4')]


def fit_quad_surfaces_tr(data, kdt, ivals, results, radius=100, fitPos=True):
    #kdt = kdtree.KDTree(pts)
    res = results.view(SURF_PATCH_DTYPE)
    
    for i in ivals:
        #print i
        #res = np.zeros(1, SURF_PATCH_DTYPE)
        r = fit_quad_surf_to_neighbourbood(data, kdt, i, radius, fitPos=fitPos)
        if not r == None:
            p, pt, N = r
            res[i]['results'] = np.array(p, dtype='f4', order='C')
            res[i]['pos'] = np.array(pt, order='C')
            res[i]['N'] = N
        else:
            res[i]['N'] = 0
            #print 'f', i

def fit_quad_surfaces_P(data, radius, fitPos=False, NFits=0):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(data)
    
    if NFits == 0:
        NFits = data.shape[0]
    
    #surfs = []
    
    #def _task(i):
    #    return fitPtA(kdt, pts, i)
    
    nCPUs = multiprocessing.cpu_count()
    #nCPUs = 1
    
    #if fitPos:
    nParams = 8
    res = shmarray.zeros((nParams, NFits))
    pos = shmarray.zeros((3, NFits))
    nPs = shmarray.zeros(NFits)
    #rt = shmarray.zeros((2,NFits))
    
    fnums = range(NFits)
    
    processes = [multiprocessing.Process(target=fit_quad_surfaces_t, args=(data, kdt, fnums[i::nCPUs], res, pos, nPs)) for i in
                 range(nCPUs)]
    
    for p in processes:
        print p
        p.start()
    
    for p in processes:
        print p
        p.join()
    
    return res, pos, nPs


def fit_quad_surfaces_Pr(data, radius, fitPos=False, NFits=0):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(data)
    
    if NFits == 0:
        NFits = data.shape[0]
    
    #surfs = []
    
    #def _task(i):
    #    return fitPtA(kdt, pts, i)
    
    nCPUs = multiprocessing.cpu_count()
    #nCPUs = 1
    
    #if fitPos:
    #nParams = 8
    #res = shmarray.zeros((nParams, NFits))
    #pos = shmarray.zeros((3, NFits))
    #nPs = shmarray.zeros(NFits)
    
    results = shmarray.zeros(NFits, SURF_PATCH_DTYPE_FLAT)
    #rt = shmarray.zeros((2,NFits))
    
    fnums = range(NFits)
    
    processes = [multiprocessing.Process(target=fit_quad_surfaces_tr, args=(data, kdt, fnums[i::nCPUs], results))
                 for i in
                 range(nCPUs)]
    
    for p in processes:
        print p
        p.start()
    
    for p in processes:
        print p
        p.join()
    
    return results

def reconstruct_quad_surfaces(fits, radius):
    return np.hstack([reconstruct_quad_surf(*f, radius=radius) for f in fits if not f is None])

def reconstruct_quad_surfaces_P(fits, radius):
    res, pos, N = fits
    return np.hstack([reconstruct_quad_surf(res[:,i], pos[:,i], N[i], radius=radius) for i in range(len(N)) if N[i] >= 1])

def reconstruct_quad_surfaces_Pr(fits, radius):
    #res, pos, N = fits
    #print fits
    fits = fits.view(SURF_PATCH_DTYPE)
    return np.hstack([reconstruct_quad_surf(fits[i]['results'].view('8f4'), fits[i]['pos'].view('3f4'), fits[i]['N'], radius=radius) for i in range(len(fits)) if fits[i]['N'] >= 1])

def reconstruct_quad_surfaces_P_region_cropped(fits, radius, data, fit_radius=100.):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(data)
    
    res, pos, N = fits
    return np.hstack([reconstruct_quad_surf_region_cropped(res[:,i], pos[:,i], N[i], kdt, data, radius=radius, fit_radius=fit_radius) for i in range(len(N)) if N[i] >= 1])


def reconstruct_quad_surfaces_Pr_region_cropped(fits, radius, data, fit_radius=100.):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(data)

    fits = fits.view(SURF_PATCH_DTYPE)
    return np.hstack([reconstruct_quad_surf_region_cropped(fits[i]['results'].view('8f4'), fits[i]['pos'].view('3f4'), fits[i]['N'], kdt, data, radius=radius,
                                                           fit_radius=fit_radius) for i in range(len(fits)) if fits[i]['N'] >= 1])

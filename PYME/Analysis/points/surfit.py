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
import logging
logger = logging.getLogger(__name__)

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
        print(i)
        r = fitPtA(kdt, pts, i)
        if not r == None:
            p3, po, Np = r
            res[:,i] = p3
            pos[:,i] = po
            nPs[i] = Np
        else:
            nPs[i] = 0
        print('f', i)
    
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
        print(p)
        p.start()
    

    for p in processes:
        print(p)
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

def get_normals(p):
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

    R = Az.dot(Ay).dot(Ax)
    
    N = np.vstack([np.zeros_like(x0), np.zeros_like(y0), -np.ones_like(z0)])
    return R.dot(N)
    

def fit_quad_surf(pts, control_pt, fitPos=True):
    """
    Fit a quadratic surface to a control point and a neighbourhood of points (both supplied)
    
    Parameters
    ----------
    pts :  ndararay[N,3] coordinates of point neighbourhood. This should normally include the control point
    control_pt: ndararay[3] x,y,z coordinates of control point
    fitPos : bool, allow the fit surface to depart from the control point
    
    
    Surface model
    -------------
    
    The model which we fit has the following form. We calculate the misfit with a surface w(u,v):
    
    $w(u,v) = A \times u^2 + B \times v^2$
    
    which is defined on the rotated and offset coordinate frame
    
    $\vec{s} = [u v w] = \mathbf{Az}\dot \mathbf{Ay}\dot \mathbf{Ax}\dot (\vec{r}- \vec{r_0}) $
    
    where $\vec{r} = [x y z]$ is a localization position, $\vec{r_0}$ is a translational offset and $\mathbf{Ax}$,
     $\mathbf{Ay}$, and $\mathbf{Az}$ are one dimensional rotation matrices about the x, y and z axes respectively:
    
    
    $\mathbf{Ax}= [[1, 0, 0], [0, \cos(\phi), \sin(\phi)], [0, -\sin(\phi), \cos(\phi)]])$
    $\mathbf{Ay} = [[\cos(\theta), 0, -\sin(\theta)], [0, 1, 0], [\sin(\theta), -0, \cos(\theta)]])$
    $\mathbf{Az} = [[\cos(\psi), \sin(\psi), -0], [-\sin(\psi), \cos(\psi), 0], [0, 0, 1]])$
    
    $A$, $B$, $\phi$, $\theta$, and $\psi$ are always free parameters in the model. if fitPos=True, $\vec{r_0}$ is also
    a free parameter, otherwise it is fixed to the location of the control point.

    Returns
    -------

    """
    from PYME.Analysis.points import arcfit
    
    #randomly generate starting parameters
    sp = 2*np.random.randn(8).astype('f')
    #sp[-2]+= 10
    sp[6:] = 0
    
    #set the staring surface position to be co-incident with the control point
    sp[:3] = control_pt
    
    #choose a model based on whether or not the surface should pass through the control point
    #note that these model functions are coded in c
    if fitPos:
        #return leastsq(quad_rot_surf_misfit,sp, args=(pts,))[0]
        return leastsq(arcfit.quad_surf_mf, sp, args=(pts[0,:], pts[1,:], pts[2,:]))[0]
    else:
        res = np.copy(sp)
        print(sp[3:], len(sp[3:]))
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

def reconstruct_quad_surf(p, control_point, N,  j, radius=50, step=10.0):
    """
    Reconstruct a fitted surface by generating a number of virtual localizations on that surface
     
    Parameters
    ----------
    p : ndarray
        the fit results for this control point
    control_point : ndarray
        the control point coordinates
    N : int
        the number of points contributing to the surface (this is just copied to the output so that t can be used later
        as a quality metric)
    j : int
        surface patch ID
    radius : float
        the radius over which to reconstruct
    step : float
        the spacing at which to sample the surface (in nm).
    
    Notes
    -----
    
    Reconstruction occurs on a uniformly sampled grid in u,v space (using the terminology introduced in fit_quad_surf).
    For low surface curvatures (small A & B) this will be approximately uniformly sampled on the surface, but the
    sampling will become less uniform as curvature increases. This is not anticipated to be a significant issue,
    especially if the ensemble surfaces will be binarized (e.g. by histograming and thresholding) later, as denser
    sampling (if needed) can be achieved by decreasing the grid spacing.

    Returns
    -------
    an array of points

    """
    #generate a regular grid in u,v space on which to sample our surface
    xr, yr = _get_reconstr_grid(radius, step)

    #evaluate the fitted model on the grid, and rotate into microscope space
    sp, normals = gen_quad_rot_surf(p, xr, yr)
    
    # find the distance of each sampled point and mask out those points which are further from the control point than the
    # reconstruction radius. This limits the reconstruction to a sphere centered on the control point and
    # converts the reconstruction from a warped square to a warped disc.
    # NOTE: this introduces a non-linearity in the number of virtual points generated for each control point
    d = sp - control_point[:,None]
    mask = (d*d).sum(0) < radius*radius
    
    return np.vstack([sp[:,mask], normals[:,mask], 0*sp[0,mask] + N, 0*sp[0,mask] + j])


def reconstruct_quad_surf_region_cropped(p, control_point, N, j, kdt, data, radius=50, step=10.0, fit_radius=100.):
    """
    Like reconstruct_quad_surf, but additionally masks the reconstruction points to the convex hull of the points used
    to generate the fit.
    
    Parameters
    ----------
    p : ndarray
        the fit results for this control point
    control_point : ndarray
        the control point coordinates
    N : int
        the number of points contributing to the surface (this is just copied to the output so that t can be used later
        as a quality metric)
    j : int
        surface patch ID
    kdt : scipy.spatial.cKDTree
        k-d tree created using spatial coordinates (x, y, z)
    data : ndarray
        localization x, y, z coordinates
    radius : float
        the radius over which to reconstruct
    step : float
        the spacing at which to sample the surface (in nm).
    fit_radius: float
        The region around each localization which was queried for each surface fit [nm].

    Returns
    -------

    """
    from scipy.spatial import Delaunay
    #refind the point neighbourbood used for fitting
    pts = data[kdt.query_ball_point(control_point, fit_radius),:]
    #generate the Delaunay tesselation of these points (to allow us to extract their convex hull)
    T  = Delaunay(pts)
    
    #generate the virtual points used to reconstruct the surface
    xr, yr = _get_reconstr_grid(radius, step)
    sp, normals = gen_quad_rot_surf(p, xr, yr)
    
    #mask the virtual points to a sphere around the control point
    d = sp - control_point[:, None]
    mask = (d * d).sum(0) < radius * radius
    sp = sp[:,mask]
    normals = normals[:,mask]
    
    #mask again to the convex hull of the points used for fitting
    mask = T.find_simplex(sp.T) > 0
    
    return np.vstack([sp[:, mask], normals[:,mask], 0*sp[0,mask] + N, 0*sp[0,mask] + j])

def fit_quad_surf_to_neighbourbood(data, kdt, i, radius=100, fitPos=True):
    """
    Perform a single fit of a quadratic surface to a point and it's neighbours, extracting both the control point and
    it's neighbours from the full dataset
    
    Parameters
    ----------
    data : [N,3] ndarray of points
    kdt : scipy.spatial.cKDTree object
    i : index in points of control point to define fit support
    radius : radius in nm of neighbours to query for fit support
    fitPos : bool, allow the fit to depart from the control point

    Returns
    -------

    """
    #find the control point and it's neighbours
    pt= data[i,:]
    pts = data[kdt.query_ball_point(pt, radius),:]
    
    N = len(pts)
    
    #bail if we don't have enough points for a meaningful surface fit
    #Note: this cutoff is arbitrary
    if N < 10:
        return
    
    #do the fit
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
    """
    Fit surfaces to point data using a pre-supplied kdtree for calculating neighbourhoods and save the results into the
    supplied results array.
    
    Note: the _tr indicates that this version uses a supplied tree and saves it's results into the passed results array
    
    Parameters
    ----------
    data : [N,3] ndarray of point positions. Note that this should be all positions, not just the control points which we are responsible
        for fitting (see ivals) as neighbourhoods will likely include points which are not in our set of control points
    kdt : a scipy.spatial.cKDTree object used for fast neighbour queries
    ivals : indices of the points to use as control points in our fits
    results : a numpy array in which to save the results (pass by reference)
    radius : the support radius / neighbourhood size in nm
    fitPos : should we constrain the fit to pass through the control point

    Returns
    -------

    """
    #create a view into the results data in a way we can easily manipulate it
    res = results.view(SURF_PATCH_DTYPE)
    
    #loop over controll points
    for i in ivals:
        print('Fitting suf #: %d' % i)
        #res = np.zeros(1, SURF_PATCH_DTYPE)
        
        #do the fit
        r = fit_quad_surf_to_neighbourbood(data, kdt, i, radius, fitPos=fitPos)
        
        print('fitted surf %d' % i)
        
        #pach our results into the correct format
        #note that the fit will only be performed if there are enough points in the neighbourbood to actually constrain
        #the fit so isolated points (usually noise) will not fit and will return None instead
        if not r == None:
            p, pt, N = r
            res[i]['results'].view('8f4')[:] = np.array(p, dtype='f4', order='C')
            res[i]['pos'].view('3f4')[:] = np.array(pt, order='C')
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
        print(p)
        p.start()
    
    for p in processes:
        print(p)
        p.join()
    
    return res, pos, nPs


def fit_quad_surfaces_Pr(data, radius, fitPos=False, NFits=0):
    """
    Fits quadratic surfaces to each point in the data set using the neighbouring points within a radius r to define the
    surface. This version distributes the processing across multiple processes in order to speed things up.
    
    Parameters
    ----------
    data: [N,3] ndarray
        The point positions
        
    radius: float
        The radius in nm around each point to use as support for the surface through that point. The surface patch will
        be fit to all points within this radius. Implicitly this sets the scale of the smoothing - i.e. the scale over
        which the true object surface can be assumed to have quadratic form.
        
    fitPos: bool
        Should the surface be allowed to depart from the control point (i.e. the point which was used to define the fit
        support neighbourhood). A value of False constrains the surface such that it always passes through the control
        point, whereas True lets the surface move. True should give more accurate surfaces when the point density is high
        and surfaces are well separated, False results in a better constrained fit and deals somewhat better with the case
        when multiple surfaces are close by (forces the fit into the local minimum corresponding to the surface that the
        control point is on).
    NFits: int
        Only fit the first NFits points. Largely exists for debugging to allow faster and more interactive computation.

    Returns
    -------

    a numpy array of results with the dtype surfit.SURF_PATCH_FLAT
    """
    from scipy.spatial import cKDTree
    #generate a kdtree to allow us to rapidly find a points neighbours
    kdt = cKDTree(data)
    
    if NFits == 0:
        NFits = data.shape[0]
        
        print('NFits: %d' % NFits)
    
    nCPUs = multiprocessing.cpu_count()
    #nCPUs = 1
    
    #generate a results array in shared memory. The worker processes will each write their chunk of results into this array
    #this make the calling semantics of the the actual call below pass by reference for the results
    #there is a bit of magic going on behind the scenes for this to work - see PYME.util.shmarray
    results = shmarray.zeros(NFits, SURF_PATCH_DTYPE_FLAT)
    
    #calculate a list of points at which to fit a surface
    fnums = range(NFits)
    
    #create a process for each cpu and assign them a chunk of fits
    #note that the slicing of fnums[i::nCPUs] effectively interleaves the fit allocations - ie one process works on every
    #nCPUth point. This was done as a simple way of allocating the tasks evenly, but might not be optimal in terms of e.g.
    #cache coherency. The process creation here will be significantly more efficient on *nix platforms which use copy on
    #write forking when compared to windows which will end up copying both the data and kdt structures
    if False: #multiprocessing.current_process().name == 'MainProcess':  # avoid potentially trying to spawn children from daemon
        processes = [multiprocessing.Process(target=fit_quad_surfaces_tr,
                                             args=(data, kdt, fnums[i::nCPUs], results, radius, fitPos)) for i in range(nCPUs)]
        # launch all the processes
        logger.debug('launching quadratic surface patch fitting processes')
        for p in processes:
            p.start()
        # wait for them to complete
        for p in processes:
            p.join()
    else:
        logger.debug('fitting quadratic surface patches in main process')
        fit_quad_surfaces_tr(data, kdt, fnums, results, radius, fitPos)
    # each process should have written their results into our shared memory array, return this
    return results

def reconstruct_quad_surfaces(fits, radius):
    return np.hstack([reconstruct_quad_surf(*f, radius=radius) for f in fits if not f is None])

def reconstruct_quad_surfaces_P(fits, radius):
    res, pos, N = fits
    return np.hstack([reconstruct_quad_surf(res[:,i], pos[:,i], N[i], i, radius=radius) for i in range(len(N)) if N[i] >= 1])

def filter_quad_results(fits, data, radius=50, proj_threshold=0.85):
    fits = fits.view(SURF_PATCH_DTYPE)
    from scipy.spatial import cKDTree
    kdt = cKDTree(data)
    
    normals = np.hstack([get_normals(p['results'].view('8f4')) for p in fits]).T
    
    filtered = []
    
    print(fits[:10])
    
    
    for i in range(len(fits)):
        N = normals[i]
        neighbour_normals = normals[kdt.query_ball_point(fits[i]['pos'].view('3f4'), radius)]
        
        median_proj = np.median([np.abs(np.dot(N, n)) for n in neighbour_normals])
        #print(len(neighbour_normals), median_proj)
        if median_proj > proj_threshold: #aligned more or less the same way as the neighbours
            filtered.append(fits[i])

    print('n_fits: %d, n_filtered: %d' % (len(fits), len(filtered)))
    
    return np.hstack(filtered)
        
        

def reconstruct_quad_surfaces_Pr(fits, radius, step=10.):
    """
    Reconstruct surfaces from fit results. This is a helper function which calls reconstruct_quad_surf
    repeatedly for each fit in the data set
    
    Parameters
    ----------
    fits
    radius

    Returns
    -------

    """
    #res, pos, N = fits
    #print fits
    fits = fits.view(SURF_PATCH_DTYPE)
    return np.hstack([reconstruct_quad_surf(fits[i]['results'].view('8f4'), fits[i]['pos'].view('3f4'), fits[i]['N'], i, radius=radius, step=step) for i in range(len(fits)) if fits[i]['N'] >= 1])

def reconstruct_quad_surfaces_P_region_cropped(fits, radius, data, fit_radius=100., step=10.):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(data)
    
    res, pos, N = fits
    return np.hstack([reconstruct_quad_surf_region_cropped(res[:,i], pos[:,i], N[i], kdt, data, radius=radius, fit_radius=fit_radius, step=step) for i in range(len(N)) if N[i] >= 1])


def reconstruct_quad_surfaces_Pr_region_cropped(fits, radius, data, fit_radius=100., step=10.):
    from scipy.spatial import cKDTree
    #kdt = kdtree.KDTree(pts)
    kdt = cKDTree(data)

    fits = fits.view(SURF_PATCH_DTYPE)
    return np.hstack([reconstruct_quad_surf_region_cropped(fits[i]['results'].view('8f4'), fits[i]['pos'].view('3f4'), fits[i]['N'], i, kdt, data, radius=radius,
                                                           fit_radius=fit_radius, step=step) for i in range(len(fits)) if fits[i]['N'] >= 1])

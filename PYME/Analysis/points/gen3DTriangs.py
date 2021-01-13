#!/usr/bin/python

##################
# gen3DTriangs.py
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

#try:
#    from delaunay import Triangulation
#except:
#    print('could not import delaunay')

from scipy.spatial import Delaunay

from numpy import *
import numpy as np
#import matplotlib.delaunay

from PYME.Analysis.points import SoftRend

def calcMeanEdge(f):
    #f = array(T.facets)
    e = sqrt(((f[:,0,:] - f[:,1,:])**2).sum(1))
    e[:] = e[:] +  sqrt(((f[:,0,:] - f[:,2,:])**2).sum(1))
    e[:] = e[:] +  sqrt(((f[:,0,:] - f[:,3,:])**2).sum(1))
    e[:] = e[:] +  sqrt(((f[:,1,:] - f[:,2,:])**2).sum(1))
    e[:] = e[:] +  sqrt(((f[:,1,:] - f[:,3,:])**2).sum(1))
    e[:] = e[:] +  sqrt(((f[:,2,:] - f[:,3,:])**2).sum(1))

    return e/6.0

def calcTetVolume(f):
    f2 = f[:,:3,:] - f[:,3:,:]

    return absolute((f2[:,0,:]*cross(f2[:,1,:], f2[:,2,:])).sum(1))/6.0


def renderTetrahedra(im, y, x, z, scale = [1,1,1], pixelsize=[5,5,5]):
    T = Delaunay(array([x/scale[0],y/scale[1],z/scale[2]]).T)

    f = T.points[T.simplices]

    x_ = scale[0]*f[:,:,0]/pixelsize[0]
    y_ = scale[1]*f[:,:,1]/pixelsize[1]
    z_ = scale[2]*f[:,:,2]/pixelsize[2]

    v = 1./(calcMeanEdge(f)**3)
    #v = 1./(1 + calcTetVolume(f))

    SoftRend.drawTetrahedra(im, x_, y_, z_, v)


def testObj():
    x = 5e3*((arange(270)%27)/9 + 0.1*random.randn(270))
    y = 5e3*((arange(270)%9)/3 + 0.1*random.randn(270))
    z = 5e3*(arange(270)%3 + 0.1*random.randn(270))

    return x, y, z

def gen3DTriangs(x,y,z, sizeCutoff=inf, internalCull=True, pcut=inf):
    T = Delaunay(array([x,y,z]).T)

    return gen3DTriangsTFC(T, sizeCutoff, internalCull, pcut=sizeCutoff)[:3]

def gen3DTriangsT(T, sizeCutoff=inf):
    #T = delaunay.Triangulation(array([x,y,z]).T.ravel(),3)

    facets = T.points[T.simplices]

    P = zeros((len(facets)*3*4, 3))
    N = zeros((len(facets)*3*4, 3))
    A = zeros(len(facets)*3*4)

    pos = 0

    for f in facets:
        fa = array(f)
        s_01 = fa[0, :] - fa[1,:]
        s01 = (s_01**2).sum()
        s_12 = fa[1, :] - fa[2,:]
        s12 = (s_12**2).sum()
        s_23 = fa[2, :] - fa[3,:]
        s23 = (s_23**2).sum()
        s_02 = fa[0, :] - fa[2,:]
        s02 = (s_02**2).sum()
        s_03 = fa[0, :] - fa[3,:]
        s03 = (s_03**2).sum()
        s_13 = fa[1, :] - fa[3,:]
        s13 = (s_13**2).sum()

        #print P.shape
        #print fa[:3, :].shape

        #print 0.5*sqrt((s01*s01).sum()*(s12*s12).sum() - ((s01*s12).sum()**2))

        if mean([s01, s12, s23, s02, s03, s13]) < sizeCutoff:

            P[pos:(pos+3), :] = fa[:3, :]
            n = cross(s_01, s_02)
            #n = -n*sign((n*s_03).sum())/linalg.norm(n)
            N[pos:(pos+3), :] = array([n,n,n])
            #A[pos:(pos+3)] = 0.5*sqrt((s01*s01).sum()*(s12*s12).sum() - ((s01*s12).sum()**2))
            #A[pos:(pos+3)] = median([s01, s02, s12])

            pos+=3

            P[pos:(pos+3), :] = fa[1:, :]
            n = cross(s_12, s_13)
            #n = n*sign((n*s_03).sum())/linalg.norm(n)
            N[pos:(pos+3), :] = array([n,n,n])
            #A[pos:(pos+3)] = 0.5*sqrt((s23*s23).sum()*(s12*s12).sum() - ((s23*s12).sum()**2))
            #A[pos:(pos+3)] = median([s13, s23, s12])

            pos+=3

            P[pos:(pos+3), :] = fa[(0,2,3), :]
            n = cross(s_03, s_02)
            #n = n*sign((n*s_13).sum())/linalg.norm(n)
            N[pos:(pos+3), :] = array([n,n,n])
            #A[pos:(pos+3)] = 0.5*sqrt((s23*s23).sum()*(s02*s02).sum() - ((s23*s02).sum()**2))
            #A[pos:(pos+3)] = median([s02, s03, s23])

            pos+=3

            P[pos:(pos+3), :] = fa[(0,1,3), :]
            n = cross(s_01, s_03)
            #n = -n*sign((n*s_02).sum())/linalg.norm(n)
            N[pos:(pos+3), :] = array([n,n,n])
            #A[pos:(pos+3)] = 0.5*sqrt((s13*s13).sum()*(s01*s01).sum() - ((s13*s01).sum()**2))
            #A[pos:(pos+3)] = median([s01, s03, s13])

            pos+=3

    #truncate to actual number of faces added
    P = P[:pos,:]
    N = N[:pos,:]
    A = A[:pos]

    return (P, A, N)

class emptyListDict(dict):
    def __getitem__(self, key):
        if key in self.keys():
            return dict.__getitem__(self, key)
        else:
            return []

def gen3DTriangsTF(T, sizeCutoff = inf, internalCull=False):
    iarray = array(T.simplices)
    
    va = array(T.points)
    
    
    s_01 = va[iarray[:, 0]] - va[iarray[:, 1]]
    s01 = (s_01**2).sum(1)
    s_12 = va[iarray[:, 1]] - va[iarray[:, 2]]
    s12 = (s_12**2).sum(1)
    s_23 = va[iarray[:, 2]] - va[iarray[:, 3]]
    s23 = (s_23**2).sum(1)
    s_02 = va[iarray[:, 0]] - va[iarray[:, 2]]
    s02 = (s_02**2).sum(1)
    s_03 = va[iarray[:, 0]] - va[iarray[:, 3]]
    s03 = (s_03**2).sum(1)
    s_13 = va[iarray[:, 1]] - va[iarray[:, 3]]
    s13 = (s_13**2).sum(1)


    A = mean([s01, s12, s23, s02, s03, s13], 0)

    cutInd = A < sizeCutoff**2

    #print cutInd
    A = A[cutInd]
    A = vstack((A,A,A)).T.ravel()

    iarray = iarray[cutInd, : ]

    #print len(iarray)

    if len(iarray) == 0:
        return ([], [], [], [])

#    s_01 = s_01[cutInd]
#    s_12 = s_12[cutInd]
#    s_23 = s_23[cutInd]
#    s_02 = s_02[cutInd]
#    s_03 = s_03[cutInd]
#    s_13 = s_13[cutInd]
#
#    n1 = cross(s_01, s_02)
#    n1 = n1*(sign((n1*s_03).sum(1))/sqrt((n1**2).sum(1))).T[:,newaxis]
#
#    n2 = cross(s_12, s_13)
#    n2 = -n2*(sign((n2*s_03).sum(1))/sqrt((n2**2).sum(1))).T[:,newaxis]
#
#    n3 = cross(s_02, s_03)
#    n3 = n3*(sign((n3*s_01).sum(1))/sqrt((n3**2).sum(1))).T[:,newaxis]
#
#    n4 = cross(s_01, s_03)
#    n4 = n4*(sign((n4*s_02).sum(1))/sqrt((n4**2).sum(1))).T[:,newaxis]
#
#    N = vstack((n1, n2, n3, n4))
#
#    N = repeat(N, 3, 0)

    triInds = vstack((iarray[:,:3], iarray[:,1:], iarray[:, (0, 2, 3)], iarray[:,(0, 1, 3)]))
    triInds.sort(1)

    nInds = hstack((iarray[:,3], iarray[:,0], iarray[:, 1], iarray[:,2]))

    #print triInds.shape
    #print nInds.shape

    surfInds = triInds[:,0] > -1

#    #internal face culling

#    for triI in triInds:
#       matches = (triInds == triI).prod(1)
#       if matches.sum() > 1:
#           surfInds*=(matches == 0) #remove triangles
    if internalCull:
        fcs = {} #emptyListDict()
        for i, triI in zip(range(len(triInds)), triInds):
            t_t = tuple(triI)
            if t_t in fcs.keys():
                surfInds[fcs[t_t]] = 0
                surfInds[i] = 0
            else:
                fcs[t_t] = i

#    for triI in triInds:
#        t_t = tuple(triI)
#        if len(fcs[t_t]) > 1:
#            surfInds[fcs[t_t]] = 0

    triInds = triInds[surfInds,:]

    nInds = nInds[surfInds]
    #print nInds.shape

    s_01 = va[triInds[:,0]] - va[triInds[:,1]]
    s01 = (s_01**2).sum(1)
    s_02 = va[triInds[:,0]] - va[triInds[:,2]]
    s02 = (s_02**2).sum(1)
    s_12 = va[triInds[:,1]] - va[triInds[:,2]]
    s12 = (s_12**2).sum(1)

    sback = va[triInds[:,0]] - va[nInds]
    #print sback.shape

    N = cross(s_01, s_02)

    #print N.shape
    N = N*(sign((N*sback).sum(1))/sqrt((N**2).sum(1))).T[:,newaxis]
    N = repeat(N, 3, 0)

    P = va[triInds.ravel(), :]

    A = mean([s01, s12,s02], 0)

    A = repeat(A, 3, 0)
#    s_01 = s_01[cutInd]
#    s_12 = s_12[cutInd]
#    s_23 = s_23[cutInd]
#    s_02 = s_02[cutInd]
#    s_03 = s_03[cutInd]
#    s_13 = s_13[cutInd]

      
            
    

    #A = hstack((A,A, A, A))

    #print 'P', P.shape
    #print N.shape
    #print A.shape

    return (P, A, N, triInds.ravel())
    
def gen3DTriangsTFC(T, sizeCutoff = inf, internalCull=True, pcut = inf):
    iarray = array(T.simplices)
    
    va = array(T.points)
    
    if internalCull:
        
        #find the squared side lengths of each facet
        s_01 = va[iarray[:, 0]] - va[iarray[:, 1]]
        s01 = sqrt((s_01**2).sum(1))
        s_12 = va[iarray[:, 1]] - va[iarray[:, 2]]
        s12 = sqrt((s_12**2).sum(1))
        s_23 = va[iarray[:, 2]] - va[iarray[:, 3]]
        s23 = sqrt((s_23**2).sum(1))
        s_02 = va[iarray[:, 0]] - va[iarray[:, 2]]
        s02 = sqrt((s_02**2).sum(1))
        s_03 = va[iarray[:, 0]] - va[iarray[:, 3]]
        s03 = sqrt((s_03**2).sum(1))
        s_13 = va[iarray[:, 1]] - va[iarray[:, 3]]
        s13 = sqrt((s_13**2).sum(1))
    
        #use mean squared side length as a proxy for area
        #A = mean([s01, s12, s23, s02, s03, s13], 0)
    
        #scut = 0.4*pcut
        
        #cut triangles with an excessive perimeter or aspect ratio
        scr = 0.5
        p = (s01 + s12 + s02)
        scut = scr*p
        P012 = (p < pcut)*(s01<scut)*(s02<scut)*(s12<scut)
        
        p =(s12 + s23 + s13)
        scut = scr*p
        P123 = (p< pcut)*(s23<scut)*(s13<scut)*(s12<scut)
        
        p=(s02 + s03 + s23)
        scut = scr*p
        P023 = (p< pcut)*(s03<scut)*(s02<scut)*(s23<scut)
        
        p=(s01 + s03 + s13)
        scut = scr*p
        P013 = (p< pcut)*(s01<scut)*(s03<scut)*(s13<scut)
        
        #find all the tetrahedra with an perimeter less than cutoff    
        
        cutInd = ((1- P012*P123*P023*P013)*((P012 + P123 + P023 + P013) > 0)) > 0 #*(A < sizeCutoff)
        
        #print len(P012), P012.sum(), cutInd.sum()
    
        #cut keep the indices which pass the test    
        iarray = iarray[cutInd, : ]
        P012 = P012[cutInd]
        P123 = P123[cutInd]
        P013 = P013[cutInd]
        P023 = P023[cutInd]
    
        #print len(iarray)
        #print iarray.shape, P012.shape, P023.shape, iarray[:,(0, 1, 3)][P023,:].shape
    
        if len(iarray) == 0:
            return ([], [], [], [])
    
        #calculate the indices of the triangles to include
        triInds = vstack((iarray[P012,:][:,:3], 
                          iarray[P123,:][:,1:], 
                          iarray[P023,:][:,(0, 2, 3)], 
                          iarray[P013,:][:,(0, 1, 3)]))
    
        nInds = hstack((iarray[P012,:][:,3], iarray[P123,:][:,0], iarray[P023,:][:, 1], iarray[P013,:][:,2]))
    else:
        triInds = vstack((iarray[:,:3], 
                          iarray[:,1:], 
                          iarray[:,(0, 2, 3)], 
                          iarray[:,(0, 1, 3)]))
    
        nInds = hstack((iarray[:,3], iarray[:,0], iarray[:, 1], iarray[:,2]))

    #calculate normals    
    s_01 = va[triInds[:,0]] - va[triInds[:,1]]
    s01 = (s_01**2).sum(1)
    s_02 = va[triInds[:,0]] - va[triInds[:,2]]
    s02 = (s_02**2).sum(1)
    s_12 = va[triInds[:,1]] - va[triInds[:,2]]
    s12 = (s_12**2).sum(1)

    sback = (va[triInds[:,0]] + va[triInds[:,1]] + va[triInds[:,2]])/3.0 - va[nInds]
    #print sback.shape

    N = cross(s_01, s_02)
    
    tridir = ((N*sback).sum(1) >0)
    
    print('Num backwards triangles: %s, %s' % ((tridir.sum(), len(tridir))))

    #print N.shape
    N = -N*(sign((N*sback).sum(1))/sqrt((N**2).sum(1))).T[:,newaxis]
    Nr = repeat(N, 3, 0)
    
    #
    if internalCull:
        Na = np.zeros(va.shape)
        for i in range(len(N)):
            for j in range(3):
                i1 = triInds[i, j]
                Na[i1,:] += N[i,:]
                #Nn[i1] += 1
        
        #Nv = Na/Nn[:,None]
        
        triInds[tridir, :] = triInds[tridir, :][:,::-1] #reverse direction of offending triangles
        
        N = Na[triInds.ravel(), :] 
        N = N/(sqrt((N*N).sum(1))[:, None])
    else:
        N = Nr

    P = va[triInds.ravel(), :] #+ 1*Nr

    A = mean([s01, s12,s02], 0)

    A = repeat(A, 3, 0)

    return (P, A, N, triInds.ravel())



def cull_triangles_2D(T, max_edge_length = inf):
    """
    Cull triangles in a triangulation which have an edge length greater than max_edge_length.
    The triangle-culling is designed to walk back from the convex hull to the true object perimeter for objects with
    concave regions.
    
    Parameters
    ----------
    T :  a triangulation as produced by the matplotlib.delaunay module
    max_edge_length : the max edge length in nm.

    Returns
    -------

    """
    iarray = array(T.simplices)

    va = array(T.points)


    s_01 = va[iarray[:, 0]] - va[iarray[:, 1]]
    s01 = (s_01**2).sum(1)
    s_12 = va[iarray[:, 1]] - va[iarray[:, 2]]
    s12 = (s_12**2).sum(1)
    
    s_02 = va[iarray[:, 0]] - va[iarray[:, 2]]
    s02 = (s_02**2).sum(1)
    


    #A = median([s01, s12, s02], 0)
    max_edge_squared = maximum(s01, s12, s02)

    cutInd = max_edge_squared < max_edge_length**2

    #area
    A = 0.5*sqrt((s_01*s_01).sum(1)*(s_12*s_12).sum(1) - ((s_01*s_12).sum(1)**2))

    #print cutInd

    iarray = iarray[cutInd, : ]

    triInds = iarray[:,:]
    triInds.sort(1)

    surfInds = triInds[:,0] > -1

    #for triI in triInds:
    #   matches = (triInds == triI).prod(1)
    #   if matches.sum() > 1:
    #       surfInds*=(matches == 0) #remove triangles

    #triInds = triInds[surfInds,:]


    P = array(T.points)[triInds.ravel(), :]

    A = A[cutInd]

    A = vstack((A,A,A)).T.ravel()
    #s_01 = s_01[cutInd]
    #s_12 = s_12[cutInd]

    #s_02 = s_02[cutInd]
    

    return (P, A, triInds)

def trianglesEqual(T1, T2):
    eq = True
    i = 0
    while eq == True and i < 3:
        v = T1[i, :]
        eq = (v == T2[0,:]).all() or (v == T2[1,:]).all() or (v == T2[2,:]).all()
        i += 1

def removeInternalFaces(P, A, N):
    P_ = list(P.reshape(-1, 3, 3))
    A_ = list(A.reshape(-1, 3))
    N_ = list(N.reshape(-1, 3, 3))

    i = 0

    while i < (len(P_) - 1):
        P_i = P_[i]

        pairFound = False

        for j in range(i+1, len(P_)):
            if trianglesEqual(P_i, P_[j]):
                pairFound = True
                break

        if pairFound:
            P_.pop(i)
            P_.pop(j)
            A_.pop(i)
            A_.pop(j)
            N_.pop(i)
            N_.pop(j)
        else:
            i += 1

    P = array(P_).reshape(-1, 3)
    A = array(A_).reshape(-1)
    N = array(N_).reshape(-1, 3)

    return (P, A, N)

def getExternalEdges(triInds):
    Edges = vstack((triInds[:, :2], triInds[:, 1:], triInds[:, (0,2)]))

    Edges.sort(1)

    Edges = Edges[lexsort(Edges.T), :]

    edgeInd = ones(Edges.shape[0])

    for i, e in enumerate(Edges[:-1, :]):
        if (Edges[i+1] == e).all():
            edgeInd[i:i+2] = 0
       #matches = (Edges == e).prod(1)
       #if matches.sum() > 1:
       #    edgeInd*=(matches == 0) #remove triangles

    return Edges[edgeInd > 0, :]

def getPerimeter(extEdges, T):
    verts = array(T.points)

    edVecs = verts[extEdges[:,0], :] - verts[extEdges[:,1], :]

    return sqrt((edVecs**2).sum(1)).sum()

def averageNormals(P,N):
    i_s = range(P.shape[0])

    while len(i_s) > 0:
        i = i_s.pop(0)
        vInds = [i]

        v = P[i]

        j = 0
        while j < len(i_s):
            k = i_s[j]
            if (P[k] == v).all():
                vInds.append(k)
                i_s.pop(k)

        ns = N[vInds, :]
        N[vInds, :] = ns.mean(0)

def averageNormalsF(P,N, triI):
    #print triI.shape
    #print N.shape

    #print N

    triS = triI.argsort()
    #triS = lexsort(triI.T)
    #i_s = range(P.shape[0])

    #print triI[triS]

    #print len(triS)

    sp = 0
    v = triI[triS[0]]
    #print 'v = ', v

    for i in range(len(triS)):
        #print triI[triS[i]]
        if not (triI[triS[i]] == v).all():
            #print sp
            #print i
            N[triS[sp:i], :] = N[triS[sp:i], :].mean(0)
            sp = i
            v = triI[triS[i]]

    #print N[triS,:]



def collectConnected(T, v, verts, va, lenThresh2, objInd):
    connected = []

    for v2_ in T.neighbours[tuple(v)]: #FIXME - update for scipy.spatial.Delaunay
        v2 = array(v2_, 'd')
        #find index
#        i = 0
#        found = False
#        while i < len(verts):
#            #print v
#            #print verts[i]
#            if (v2 == verts[i]).all():
#                found = True
#            else:
#                i += 1

        if ((v - v2)**2).sum() < lenThresh2:

#            if len(v2) == 3:
#                i = int(argwhere((va[:, 0] == v2[0]) * (va[:, 1] == v2[1]) * (va[:, 2] == v2[2])))
#            else:
#                i = int(argwhere((va[:, 0] == v2[0]) * (va[:, 1] == v2[1])))

            i = objInd[v2_]
            #print i

            #if i in verts: #we haven't already done this vertex
                #print ((v - v2)**2).sum()
                #print lenThresh**2
                #if ((v - v2)**2).sum() < lenThresh**2:
                #print 'test'
            #try:
                #verts.remove(i)
            if verts[i] == 1:
                verts[i] = 0
                connected.append(v2)
                connected += collectConnected(T, v2, verts, va, lenThresh2, objInd)
            #except ValueError:
            #    pass

    return connected


def segment(T, lenThresh, minSize=None):
    objects = []
    #verts = list(range(len(T.set)))
    verts = ones(len(T.points))
    va = array(T.points)

    objInd = {}

    lenThresh2 = lenThresh**2

    #dictionary mapping vertices to indicex
    for i in range(len(T.points)):
        #print tuple(T.set[i])
        objInd[tuple(va[i, :])] = i

    if minSize is None:
        minSize = va.shape[1] + 1 #only return objects which have enough points to be a volume

    j = 0

    while verts.sum() > 0:
        while verts[j]  == 0 and j < verts.shape[0]:
            j += 1

        #print j

        verts[j] = 0
        v = va[j, :]
        obj = [v]

        con = collectConnected(T, v, verts, va, lenThresh2, objInd)

        obj += con

        if len(obj) > minSize:
            objects.append(array(obj))

    return objects

#def vertInd():

def segmentNR(T, lenThresh, minSize=None):
    #non-recursive version of segmentation
    objects = []
    verts = list(range(len(T.points)))
    va = array(T.points)

    if minSize is None:
        minSize = va.shape[1] + 1 #only return objects which have enough points to be a volume

    while len(verts) > 0:
        v = va[verts.pop(0), :]
        obj = [v]

        obj += collectConnected(T, v, verts, va, lenThresh)

        if len(obj) > minSize:
            objects.append(array(obj))

    return objects

xi = (1. + sqrt(5.))/2

xs = array([0, 0, 0, 0, 1, 1, -1, -1, xi, xi, -xi, -xi,0]).astype('f')/2.
ys = array([1, 1, -1, -1, xi, -xi, xi, -xi, 0, 0, 0, 0,0]).astype('f')/2.
zs = array([xi, -xi, xi, -xi, 0, 0, 0, 0, 1, -1, 1, -1,0]).astype('f')/2.


def blobify(objects, sizeCutoff, sm=False, sc=[10, 10, 10]):
    P_ = []
    N_ = []
    A_ = []

    for i, o in enumerate(objects):
        if sm:
            #print o.shape
            x = o[:,0][:,None] + sc[0]*xs[None, :]
            y = o[:,1][:,None] + sc[1]*ys[None, :]
            z = o[:,2][:,None] + sc[2]*zs[None, :]

            #o = o + (sc[0]/10.)*random.normal(size=o.shape)

            o = vstack((x.ravel(), y.ravel(), z.ravel())).T
            #print o.shape


        T = Delaunay(o)
        #print T.indices
        #for ti in T.indices:
        #    print len(ti)
        P, A, N, triI = gen3DTriangsTF(T, sizeCutoff, internalCull=True)

        #P, A, N = removeInternalFaces(P, A, N)
        if not P == []:
            averageNormalsF(P, N,triI)

            #triS = triI.argsort()

            #print P[triS,:]
            #print N[triS,:]

            #print P.shape
            A = ones(A.shape)*i

            P_.append(P)
            N_.append(N)
            A_.append(A)

    return (vstack(P_), hstack(A_), vstack(N_))

def blobify2D(objects, sizeCutoff):
    P_ = []
    
    A_ = []

    for o, i in zip(objects, range(len(objects))):
        T = Delaunay(o)
        #T2 = matplotlib.delaunay.Triangulation(o[:, 0], o[:,1])
        P, A, triI = cull_triangles_2D(T, sizeCutoff)

        #P, A, N = removeInternalFaces(P, A, N)

        #colour by object
        A = ones(A.shape)*i
        

        #print P.shape

        P_.append(P)
        
        A_.append(A)

    return (vstack(P_), hstack(A_))

def gen3DBlobs(x,y,z, sizeCutoff=inf, sm=False, sc=[10, 10, 10]):
    T = Delaunay(array([x,y,z]).T)

    objects = segment(T, sizeCutoff)

    return blobify(objects, sizeCutoff, sm, sc)






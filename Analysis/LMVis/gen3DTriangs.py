import delaunay
from numpy import *


def testObj():
    x = 5e3*((arange(270)%27)/9 + 0.1*random.randn(270))
    y = 5e3*((arange(270)%9)/3 + 0.1*random.randn(270))
    z = 5e3*(arange(270)%3 + 0.1*random.randn(270))

    return x, y, z

def gen3DTriangs(x,y,z, sizeCutoff=inf):
    T = delaunay.Triangulation(array([x,y,z]).T.ravel(),3)

    return gen3DTriangsTF(T, sizeCutoff)[:3]

def gen3DTriangsT(T, sizeCutoff=inf):
    #T = delaunay.Triangulation(array([x,y,z]).T.ravel(),3)

    P = zeros((len(T.facets)*3*4, 3))
    N = zeros((len(T.facets)*3*4, 3))
    A = zeros(len(T.facets)*3*4)

    pos = 0

    for f in T.facets:
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

def gen3DTriangsTF(T, sizeCutoff = inf):
    iarray = array(T.indices)
    
    va = array(T.set)
    
    
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

    for triI in triInds:
       matches = (triInds == triI).prod(1)
       if matches.sum() > 1:
           surfInds*=(matches == 0) #remove triangles

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

def gen2DTriangsTF(T, sizeCutoff = inf):
    iarray = array(T.indices)

    va = array(T.set)


    s_01 = va[iarray[:, 0]] - va[iarray[:, 1]]
    s01 = (s_01**2).sum(1)
    s_12 = va[iarray[:, 1]] - va[iarray[:, 2]]
    s12 = (s_12**2).sum(1)
    
    s_02 = va[iarray[:, 0]] - va[iarray[:, 2]]
    s02 = (s_02**2).sum(1)
    


    #A = median([s01, s12, s02], 0)
    A = maximum(s01, s12, s02)

    cutInd = A < sizeCutoff**2

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


    P = array(T.set)[triInds.ravel(), :]

    A = A[cutInd]

    A = vstack((A,A,A)).T.ravel()
    #s_01 = s_01[cutInd]
    #s_12 = s_12[cutInd]

    #s_02 = s_02[cutInd]
    

    return (P, A, triInds.ravel())

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
    verts = array(T.set)

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



def collectConnected(T, v, verts, va, lenThresh):
    connected = []

    for v2 in T.neighbours[tuple(v)]:
        v2 = array(v2)
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

        if ((v - v2)**2).sum() < lenThresh**2:

            if len(v2) == 3:
                i = int(argwhere((va[:, 0] == v2[0]) * (va[:, 1] == v2[1]) * (va[:, 2] == v2[2])))
            else:
                i = int(argwhere((va[:, 0] == v2[0]) * (va[:, 1] == v2[1])))
            #print i

            #if i in verts: #we haven't already done this vertex
                #print ((v - v2)**2).sum()
                #print lenThresh**2
                #if ((v - v2)**2).sum() < lenThresh**2:
                #print 'test'
            try:
                verts.remove(i)
                connected.append(v2)
                connected += collectConnected(T, v2, verts, va, lenThresh)
            except ValueError:
                pass

    return connected


def segment(T, lenThresh, minSize=None):
    objects = []
    verts = list(range(len(T.set)))
    va = array(T.set)

    if minSize == None:
        minSize = va.shape[1] + 1 #only return objects which have enough points to be a volume

    while len(verts) > 0:
        v = va[verts.pop(0), :]
        obj = [v]

        obj += collectConnected(T, v, verts, va, lenThresh)

        if len(obj) > minSize:
            objects.append(array(obj))

    return objects


def blobify(objects, sizeCutoff):
    P_ = []
    N_ = []
    A_ = []

    for i, o in enumerate(objects):
        T = delaunay.Triangulation(o.ravel(),3)
        P, A, N, triI = gen3DTriangsTF(T, sizeCutoff)

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
        T = delaunay.Triangulation(o.ravel(),2)
        P, A, triI = gen2DTriangsTF(T, sizeCutoff)

        #P, A, N = removeInternalFaces(P, A, N)

        #colour by object
        A = ones(A.shape)*i
        

        #print P.shape

        P_.append(P)
        
        A_.append(A)

    return (vstack(P_), hstack(A_))

def gen3DBlobs(x,y,z, sizeCutoff=inf):
    T = delaunay.Triangulation(array([x,y,z]).T.ravel(),3)

    objects = segment(T, sizeCutoff)

    return blobify(objects, sizeCutoff)






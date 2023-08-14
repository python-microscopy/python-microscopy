"""
Match fiducials between frames using rotation invariant point features,
and a probabalistic matching algorithm.
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def gen_features(points, N):
    """
    Features are distances of the N nearest neighbours to each point
    
    Parameters
    ----------
    points : array_like
        Mx2 or Mx3 array of points
    N : int
        Number of nearest neighbours to use

    """

    tree = cKDTree(points)
    dists, inds = tree.query(points, N+1)
    
    return dists[:, 1:]


def gen_features2(points, N):
    """
    Features are distances of the N nearest neighbours to each point
    
    Parameters
    ----------
    points : array_like
        Mx2 or Mx3 array of points
    N : int
        Number of nearest neighbours to use

    """

    tree = cKDTree(points)
    dists, inds = tree.query(points, N+1)

    #print(points[:,0])

    #print(inds, points[inds[:, 1:], 0], points[:, 0, None])

    d = points[inds[:, 1:], 0] - points[:, 0, None] + 1j*(points[inds[:, 1:], 1] - points[:, 1, None])
    
    return d# dists[:, 1:]



def _match_points(points0, points1, scale=1.0, p_cutoff=0.1):
    """
    Match points between two frames
    
    Parameters
    ----------
    points0 : array_like
        Mx2 or Mx3 array of points
    points1 : array_like
        Mx2 or Mx3 array of points

    Returns
    -------
    matches : array_like
        Nx2 array of indices of matched points
    """
    from scipy.spatial.distance import cdist
    from scipy.special import erf
    
    features0 = gen_features(points0, 10)
    features1 = gen_features(points1, 10)

    def _robust_dist(feat1, feat2):
        """
        Calculate a robust distance between two feature vectors,
        allowing for some points to have no matching points in the other data set.
        """

        d = np.sort(np.diff(np.sort(np.hstack([feat1, feat2]))))[:(len(feat1)-2)]
        #p = np.exp(-d/scale)
        p = 1 - erf(d/scale-1)

        return p.sum()
        
    
    plt.figure()
    plt.subplot(211)
    plt.plot(features0, 'x')
    plt.subplot(212)
    plt.plot(features1, 'x')
    
    p = cdist(features0, features1, _robust_dist)
    #p /= p.sum(0)[None, :]

    # de-weight matches where the there is another higher probability match
    #p = p*(p/p.max(1)[:,None])**2

    #matches = np.zeros(len(points0), dtype=np.int)

    plt.figure()
    plt.imshow(p)

    am = p.argmax(axis=1, keepdims=True)
    score = np.take_along_axis(p, am, axis=1)

    return am.squeeze(), score.squeeze()

def match_points(points0, points1, scale=1.0, p_cutoff=0.1):
    """
    Match points between two frames
    
    Parameters
    ----------
    points0 : array_like
        Mx2 or Mx3 array of points
    points1 : array_like
        Mx2 or Mx3 array of points

    Returns
    -------
    matches : array_like
        Nx2 array of indices of matched points
    """
    from scipy.spatial.distance import cdist
    from scipy.special import erf
    
    features0 = gen_features2(points0, 10)
    features1 = gen_features2(points1, 10)

    def _robust_dist(feat1, feat2):
        """
        Calculate a robust distance between two feature vectors,
        allowing for some points to have no matching points in the other data set.
        """

        # find which features match in each set
        a = np.abs((feat1[:,None]) - (feat2[None,:]))

        i = np.argmin(a,axis=1, keepdims=True)
        delta = np.take_along_axis(a, i, axis=1).squeeze()

        f2i = feat2[i.squeeze()]

        #print('delta:', delta)
        #print(np.angle(feat1) - np.angle(f2i))

        t = np.mod(np.angle(feat1) - np.angle(f2i), 2*np.pi)

        w = 1.0/(1.0 + delta)
        t = (t*w).sum()/w.sum()

        d = np.abs(feat1 - f2i*np.exp(1j*np.median(t)))

        #print(d)
        p = 1 - erf(d/scale-1)

        #print(p)

        return p.sum()
        
    
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(features0, 'x')
    # plt.subplot(212)
    # plt.plot(features1, 'x')

    # plt.figure()
    # plt.plot(np.real(features0)[:2, :].T, np.imag(features0)[:2, :].T, 'x')
    # plt.plot(np.real(features1)[:2, :].T, np.imag(features1)[:2, :].T, '+')

    
    
    p = cdist(features0, features1, _robust_dist)
    #p /= p.sum(0)[None, :]

    # de-weight matches where the there is another higher probability match
    #p = p*(p/p.max(1)[:,None])**2

    #matches = np.zeros(len(points0), dtype=np.int)

    print(p)

    plt.figure()
    plt.imshow(p)
    plt.colorbar()
    plt.title('Point correspondance matrix')
    plt.xlabel('Channel 1')
    plt.ylabel('Channel 0')

    am = p.argmax(axis=1, keepdims=True)
    score = np.take_along_axis(p, am, axis=1)

    am, score = am.squeeze(), score.squeeze()

    plt.figure()
    plt.plot(points0[:,0], points0[:,1], '.')
    for i, p in enumerate(points0):
        plt.text(p[0], p[1], f'{i}', color='C0')
        
    plt.plot(points1[:,0], points1[:,1], '.')
    for i, p in enumerate(points1):
        plt.text(p[0], p[1], f'{i}', color='C1', verticalalignment='top')

    for i in range(len(points0)):
        plt.plot([points0[i, 0], points1[am[i], 0]], [points0[i, 1], points1[am[i], 1]], 'k', lw=1)

    plt.title('Matched points')

    return am, score






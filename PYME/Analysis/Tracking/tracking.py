# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:41:31 2015

@author: david
"""

import numpy as np
from scipy import spatial
from six.moves import xrange

#rt = [np.vstack([x[t==i], y[t == i]]) for i in range(t.max() + 1)]
#ind_t = [index[t == i]]

class Tracker(object):
    def __init__(self, t, xvs, pNew=0.2, r0=500, linkageCuttoffProb=0.1):
        self.t = t
        self.xvs = xvs
        
        self.pNew = pNew #probability of an object not being present in previous frame
        self.r0 = r0 #mean distance an object can move
        self.linkageCuttoffProb = linkageCuttoffProb #probability below which a possible inkage is ignored

        #index of objects
        self.objIndex = np.arange(len(t))
        
        #this will be changed to reflect linkages
        self.clumpIndex = np.arange(len(t))
        
        self.xvsByT = [xvs[:,t==i] for i in xrange(t.max()+1)]
        self.indicesByT = [self.objIndex[t==i] for i in xrange(t.max()+1)]

    def calcLinkageMatrix(self, i, j, manualLinkages = []):
        """Compare this frame (i) with another frame (j) """
        #calculate distances 
        #print i, j
        if i >= len(self.xvsByT):
            return np.empty([0,0]),np.empty([0])
            
        xvs_i = self.xvsByT[i].T
        xvs_j = self.xvsByT[j].T
        if (len(xvs_i) == 0) or (len(xvs_j) == 0):
            return np.empty([0,0]),np.empty([0])
            
        dists = spatial.distance.cdist(xvs_j, xvs_i)
        
        #calculate probablility of a certain distance (given a mean jump length r0)
        pMatch = np.exp(-dists/self.r0)
        
        #add the probability that the object is new in this frame
        pMatch = np.concatenate([pMatch, self.pNew*np.ones([1,pMatch.shape[1]])], 0)
        
        #Set the probabilities for manually specified linkages:
        for iLink, jLink in manualLinkages:
            pMatch[j,:] = 0
            pMatch[:,i] = 0
            pMatch[j, i] = 1
        
        #of all possible matches for a given object, what are the relative probabilities
        lMatch = pMatch/pMatch.sum(0)[None, :]
        
        #we don't want 2 objects in this frame to match to one object in frame j
        #reduce likelihood of object according to the relative chance of another
        #object in this frame assigning to the same object
        
        lAdj = lMatch*(lMatch/lMatch.max(1)[:,None])**2
        
        #this is not, however the case for new matches - we don't care about those 
        #co-inciding - reset to previous values
        lAdj[-1,:] = lMatch[-1,:]
        
        #now repeat the normalisation above to find new relative probabilities
        lMatch = lAdj/lAdj.sum(0)[None, :]
        
        #which actual events does this information pertain to (here so we can expand this to multiple frames)
        jIndices = self.indicesByT[j]
        
        return lMatch, jIndices
    
    def getLinkageCandidates(self, lMatch, jIndices):
        linkages = {}
        for i in xrange(lMatch.shape[1]):
            pM = lMatch[:,i].squeeze()
            iM = np.argsort(-pM)
            pM = pM[iM]
            sig = pM > self.linkageCuttoffProb
            
            if sig.sum() == 0:
                pM = pM[:1]
                iM = iM[:1]
            else:
                pM = pM[sig]
                iM = iM[sig]
            
            #print jIndices.shape[0], iM            
            #if we matched to a new event, replace by -1
            iM[iM == jIndices.shape[0]] = -1
            
            #find the real object numbers which correspond to our linkages
            absIM = jIndices[iM]
            absIM[iM == -1] = -1
            
            linkages[i] = (absIM, pM)
            
        return linkages
        
    def calcLinkages(self, i, j, manualLinkages = []):
        lMatch, jIndices = self.calcLinkageMatrix(i,j, manualLinkages=manualLinkages)
        return self.getLinkageCandidates(lMatch, jIndices)
        
    def updateTrack(self, i, linkages):
        if i >= len(self.indicesByT):
            return
            
        iIndices = self.indicesByT[i]
        
        #alreadyAssigned = []

        allLinks = []        
        
        for k, links in linkages.items():
            n = iIndices[k]
            lj, lp = links
            
            for lji, lpi in zip(lj, lp):
                allLinks.append((lpi, lji, n))
                
        if len(allLinks) == 0:
            return
                
        allLinks = np.array(allLinks)
        
        #sort so that the highest probability comes first
        allLinks = allLinks[allLinks[:,0].argsort()[::-1], :]
            
        for j in range(len(allLinks)):
            lpi, lji, n = allLinks[j]
            
            if not lpi == 0:                
                allLinks[allLinks[:,2] == n, 0] = 0
                if not lji == -1:
                    #we are not a new object
                    self.clumpIndex[int(n)] = self.clumpIndex[int(lji)]
                    allLinks[allLinks[:,1] == lji, 0] = 0
                    
    

        
        
    
    
    
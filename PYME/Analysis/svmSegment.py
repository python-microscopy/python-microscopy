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
from scipy import ndimage
import sklearn
from sklearn import svm

import multiprocessing.dummy

def extractFeatures(im):
    feats = []
    
    radii = [3, 5, 7, 9, 11, 15, 19, 25, 35]#, 51, 75, 99]
    
    feats.append(im)
    
    for i in radii:
        m = ndimage.uniform_filter(im, i)
        feats.append(m)
        d = im - m
        feats.append(ndimage.uniform_filter(d*d, i))
        
        #measure displacement of ROI centre of mass
        #x, y = np.mgrid[-i:(i+1), -i:(i+1)]  
        #k = np.sqrt(1.0*x*x + 1.0*y*y)
        #k = np.abs(np.arange(-i, i + 1))
        #r = ndimage.convolve1d(ndimage.convolve1d(im, k, 0), k, 1)

        #r = ndimage.convolve(d, k)        
        
        #feats.append(r/(m + .0001))
    
    
    feats.append(np.abs(feats[0] - feats[3]))
    
    fa = np.array(feats)
    
    fv = fa.reshape(fa.shape[0], -1)
    #fv = (fv-fvm[:, None])/fvs[:,None]
    
    return fv
    
def extractFeatures2(im):
    feats = []
    
    radii = [3, 5, 7, 9, 11, 15, 19, 25, 35]#, 51, 75, 99]
    
    feats.append(im)
    
    cm  = im
    
    for i in radii:
        m = ndimage.uniform_filter(im, i)
        feats.append(m - cm)
        
        cm = m
        
        d = im - m
        feats.append(ndimage.uniform_filter(d*d, i))
        
        #measure displacement of ROI centre of mass
        #x, y = np.mgrid[-i:(i+1), -i:(i+1)]  
        #k = np.sqrt(1.0*x*x + 1.0*y*y)
        #k = np.abs(np.arange(-i, i + 1))
        #r = ndimage.convolve1d(ndimage.convolve1d(im, k, 0), k, 1)

        #r = ndimage.convolve(d, k)        
        
        #feats.append(r/(m + .0001))
    
    
    #feats.append(np.abs(feats[0] - feats[3]))
    
    fa = np.array(feats)
    
    fv = fa.reshape(fa.shape[0], -1)
    #fv = (fv-fvm[:, None])/fvs[:,None]
    
    return fv
    
def normalizeFeatures(fv, normalization= None):
    if normalization is None:
        fvm = fv.mean(1)
        fvs = fv.std(1)
    else:
        fvm, fvs = normalization
    #fv = (fv-fv.mean(1)[:, None])/fv.std(1)[:,None]
    #fv = fv/fv.std(1)[:,None]
    return (fv-fvm[:, None])/fvs[:,None], (fvm, fvs)
    
def trainClassifier(features, labels):
    clf = svm.SVC(C=100., probability=True)
    clf.fit(features.T, labels)
    
    return clf
    
def performClassification(clf, fv):
    nCPUs = multiprocessing.cpu_count()
    #nCPUs = 6
    p = multiprocessing.dummy.Pool(nCPUs)
    
    N = fv.shape[1]
    cs = int(np.ceil(float(N)/nCPUs))
    #print N
    
    chunks = [fv[:, (i*cs):min(((i+1)*cs), N)].T for i in range(nCPUs)]
    
    return np.hstack(p.map(clf.predict, chunks))


def prob_predict(clf, fv):
    nCPUs = multiprocessing.cpu_count()
    #nCPUs = 6
    p = multiprocessing.dummy.Pool(nCPUs)
    
    N = fv.shape[1]
    cs = int(np.ceil(float(N) / nCPUs))
    #print N
    
    chunks = [fv[:, (i * cs):min(((i + 1) * cs), N)].T for i in range(nCPUs)]
    
    return np.vstack(p.map(clf.predict_proba, chunks))
    
class svmClassifier(object):
    def __init__(self, clf=None, filename = None):
        #self.img = img
        
        self.featureCache = {}
        self.normalization = None
        self.clf = clf
        
        if clf is None and not filename is None:
            self.loadFile(filename)
            
    def _getNormalizedFeatures(self, im):
        print('Extracting features ...')
        features, self.normalization = normalizeFeatures(extractFeatures2(im), self.normalization)
        return features
            
    def _getAndCacheFeatures(self, im):
        #we want to cache the features during training, but not during classification:
        #use memory address of array (im.ctypes.data) as cache key
        if not im.ctypes.data in self.featureCache.keys():
            self.featureCache[im.ctypes.data] = self._getNormalizedFeatures(im)
        
        return self.featureCache[im.ctypes.data]
            
            
    def train(self, im, labels, newInstance=False):
        #im = self.img.data[:,:,0,0].squeeze()
        im = im.squeeze()
        
        features = self._getAndCacheFeatures(im)
            
        if self.clf is None or newInstance:
            self.clf = svm.SVC(C=100., probability=True)
            
        print('Training classifier ...')
        #just use those pixels which have been labeled to train the classifier
        lIND = labels > 0
        self.clf.fit(features[:,lIND.ravel()].T, labels[lIND])
            
    def trainAndClassify(self,im, labels):
        self.train(im, labels)
        
        return self.classify(im)
        
    def classify(self, im):
        im = im.squeeze()
        
        if im.ctypes.data in self.featureCache.keys():
            features = self.featureCache[im.ctypes.data]
        else:
            features = self._getNormalizedFeatures(im)
                
        print('Performing classification ...')
        c =  performClassification(self.clf, features)
        #print c.shape, im.shape, im.size
        return c.reshape(im.shape)

    def probabilities(self, im):
        im = im.squeeze()
    
        if im.ctypes.data in self.featureCache.keys():
            features = self.featureCache[im.ctypes.data]
        else:
            features = self._getNormalizedFeatures(im)
    
        print('Performing classification ...')
        c = prob_predict(self.clf, features)
        print(c.shape, im.shape, im.size)
        shape = np.ones(4, 'i')
        shape[:len(im.shape)] = im.shape
        shape[3] = c.shape[1]
        return c.reshape(shape)
        
    def save(self, filename):
        import joblib
        joblib.dump((self.clf, self.normalization), filename, compress=9)
        
    
    def loadFile(self, filename):
        import joblib
        self.clf, self.normalization = joblib.load(filename)
        
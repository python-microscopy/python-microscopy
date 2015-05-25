# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:10:02 2015

@author: david
"""
from .base import ModuleBase, register_module, Filter, Float, Enum, CStr, Bool, Int, View, Item, Group
import numpy as np
from PYME.Analysis.LMVis import inpFilt
import os

@register_module('MultifitBlobs') 
class MultifitBlobs(ModuleBase):
    inputImage = CStr('input')
    outputName = CStr('positions')
    blobSigma = Float(45.0)
    threshold = Float(2.0)
    scale = Float(1000)
    
    def execute(self, namespace):
        from PYME.Analysis.FitFactories import GaussMultifitSR
        img = namespace[self.inputImage]
        
        img.mdh['Analysis.PSFSigma'] = self.blobSigma
        
        ff = GaussMultifitSR.FitFactory(self.scale*img.data[:,:,:], img.mdh, noiseSigma=np.ones_like(img.data[:,:,:].squeeze()))
        
        res = inpFilt.fitResultsSource(ff.FindAndFit(self.threshold))
        
        namespace[self.outputName] = res#inpFilt.mappingFilter(res, x='fitResults_x0', y='fitResults_y0')

@register_module('MeanNeighbourDistances') 
class MeanNeighbourDistances(ModuleBase):
    '''Calculates mean distance to nearest neighbour in a triangulation of the 
    supplied points'''
    inputPositions = CStr('input')
    outputName = CStr('neighbourDists')
    key = CStr('neighbourDists')
    
    def execute(self, namespace):
        from matplotlib import delaunay
        from PYME.Analysis.LMVis import visHelpers
        pos = namespace[self.inputPositions]
        
        x, y = pos['x'], pos['y']
        #triangulate the data
        T = delaunay.Triangulation(x + .1*np.random.normal(size=len(x)), y + .1*np.random.normal(size=len(x)))
        #find the average edge lengths leading away from a given point
        res = np.array(visHelpers.calcNeighbourDists(T))
        
        namespace[self.outputName] = {self.key:res}

@register_module('NearestNeighbourDistances')         
class NearestNeighbourDistances(ModuleBase):
    '''Calculates the nearest neighbour distances between supplied points using
    a kdtree'''
    inputPositions = CStr('input')
    outputName = CStr('neighbourDists')
    key = CStr('neighbourDists')
    
    def execute(self, namespace):
        from scipy.spatial import cKDTree
        pos = namespace[self.inputPositions]
        
        x, y = pos['x'], pos['y']
        
        #create a kdtree
        p = np.vstack([x,y]).T
        kdt = cKDTree(p)
        
        #query the two closest entries - the closest entry will be the 
        #original point, the next closest it's nearest neighbour
        d, i = kdt.query(p, 2)
        res = d[:,1]
        
        namespace[self.outputName] = {self.key: res}

@register_module('PairwiseDistanceHistogram')         
class PairwiseDistanceHistogram(ModuleBase):
    '''Calculates a histogram of pairwise distances'''
    inputPositions = CStr('input')
    outputName = CStr('distHist')
    nbins = Int(50)
    binSize = Float(50.)
    
    def execute(self, namespace):
        from PYME.Analysis import DistHist
        
        pos = namespace[self.inputPositions]
        
        x, y = pos['x'], pos['y']
        
        res = DistHist.distanceHistogram(x, y, x, y, self.nbins, self.binsize)
        d = self.binsize*np.arange(self.nbins)
        
        namespace[self.outputName] = {'bins' : d, 'counts' : res}

@register_module('Histogram')         
class Histogram(ModuleBase):
    '''Calculates a histogram of a given measurement key'''
    inputMeasurements = CStr('input')
    outputName = CStr('distHist')
    key = CStr('key')
    nbins = Int(50)
    left = Float(0.)
    right = Float(1000)
    
    def execute(self, namespace):        
        v = namespace[self.inputMeasurements][self.key]
        
        edges = np.linspace(self.left, self.right, self.nbins)
        
        res = np.histogram(v, edges)[0]
        
        namespace[self.outputName] = {'bins' : edges, 'counts' : res}

@register_module('Measure2D') 
class Measure2D(ModuleBase):
    '''Module with one image input and one image output'''
    inputLabels = CStr('labels')
    inputIntensity = CStr('data')
    outputName = CStr('measurements')
    
    measureContour = Bool(True)    
        
    def execute(self, namespace):       
        labels = namespace[self.inputLabels]
        
        #define the measurement class, which behaves like an input filter        
        class measurements(inpFilt.inputFilter):
            _name = 'Measue 2D source'
            ps = labels.pixelSize
            
            def __init__(self):
                self.measures = []
                self.contours = []
                self.frameNos = []
                
                self._keys = []
                
            def addFrameMeasures(self, frameNo, measurements, contours = None):
                if len(self.measures) == 0:
                    #first time we've called this - determine our data type
                    self._keys = ['t', 'x', 'y'] + [r for r in dir(measurements[0]) if not r.startswith('_')]
                    
                    if not contours == None:
                        self._keys += ['contour']
                        
                self.measures.extend(measurements)
                self.frameNos.extend([frameNo for i in xrange(len(measurements))])
                
                if contours:
                    self.contours.extend(contours)
            
            def keys(self):
                return self._keys
        
            def __getitem__(self, key):
                if not key in self._keys:
                    raise RuntimeError('Key not defined')
                
                if key == 't':
                    return np.array(self.frameNos)
                elif key == 'contour':
                    return np.array(self.contours, dtype='O')
                elif key == 'x':
                    return self.ps*np.array([r.centroid[0] for r in self.measures])
                elif key == 'y':
                    return self.ps*np.array([r.centroid[1] for r in self.measures])
                else:
                    l = [getattr(r, key) for r in self.measures]
                    
                    if np.isscalar(l[0]):
                        a = np.array(l)
                        return a
                    else:
                        a = np.empty(len(l), dtype='O')
                        for i, li in enumerate(l):
                            a[i] = li
                            
                        return a
                        
#            def toDataFrame(self, keys=None):
#                import pandas as pd
#                if keys == None:
#                    keys = self._keys
#                
#                d = {k: self.__getitem__(k) for k in keys}
#                
#                return pd.DataFrame(d)
                
        
        # end measuremnt class def
        
        m = measurements()
        
        if self.inputIntensity in ['None', 'none', '']:
            #if we don't have intensity data
            intensity = None
        else:
            intensity = namespace[self.inputIntensity]
            
        for i in xrange(labels.data.shape[2]):
            m.addFrameMeasures(i, *self._measureFrame(i, labels, intensity))
                    
        namespace[self.outputName] = m#.toDataFrame()      
        
    def _measureFrame(self, frameNo, labels, intensity):
        import skimage.measure
        
        li = labels.data[:,:,frameNo].squeeze()

        if intensity:
            it = intensity.data[:,:,frameNo].squeeze()
        else:
            it = None
            
        rp = skimage.measure.regionprops(li, it)
        
        #print len(rp), li.max()
        
        if self.measureContour:
            ct = []
            for i in range(len(rp)):
                #print i, (li == (i+1)).sum()
                #c = skimage.measure.find_contours(r['image'], .5)
                c = skimage.measure.find_contours((li == (i+1)), .5)
                if len(c) == 0:
                    c = [np.zeros((2,2))]
                #ct.append(c[0] + np.array(r['bbox'])[:2][None,:])
                ct.append(c[0])
        else:
            ct = None
            
        
            
        return rp, ct

@register_module('SelectMeasurementColumns')         
class SelectMeasurementColumns(ModuleBase):
    '''Take just certain columns of a variable'''
    inputMeasurments = CStr('measurements')
    keys = CStr('')
    outputName = CStr('selectedMeasurements') 
    
    def execute(self, namespace):       
        meas = namespace[self.inputMeasurments]
        namespace[self.outputName] = {k:meas[k] for k in self.keys.split()}

@register_module('AddMetadataToMeasurements')         
class AddMetadataToMeasurements(ModuleBase):
    inputMeasurments = CStr('measurements')
    inputImage = CStr('input')
    keys = CStr('SampleNotes')
    metadataKeys = CStr('Sample.Notes')
    outputName = CStr('annotatedMeasurements')
    
    def execute(self, namespace):
        res = {}
        res.update(namespace[self.inputMeasurments])
        
        img = namespace[self.inputImage]

        nEntries = len(res.values()[0])
        for k, mdk in zip(self.keys.split(), self.metadataKeys.split()):
            if mdk == 'seriesName':
                #import os
                v = os.path.split(img.seriesName)[1]
            else:
                v = img.mdh[mdk]
            res[k] = np.array([v]*nEntries)
        
        namespace[self.outputName] = res


@register_module('AggregateMeasurements')         
class AggregateMeasurements(ModuleBase):
    '''Create a new composite measurement containing the results of multiple
    previous measurements'''
    inputMeasurements1 = CStr('meas1')
    inputMeasurements2 = CStr('')
    inputMeasurements3 = CStr('')
    inputMeasurements4 = CStr('')
    outputName = CStr('aggregatedMeasurements') 
    
    def execute(self, namespace):
        res = {}
        for mk in [getattr(self, n) for n in dir(self) if n.startswith('inputMeas')]:
            if not mk == '':
                meas = namespace[mk]
                res.update(meas)
        
        namespace[self.outputName] = res
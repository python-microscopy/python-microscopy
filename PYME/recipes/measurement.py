# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:10:02 2015

@author: david
"""
from .base import ModuleBase, register_module, Filter, Float, Enum, CStr, Bool, Int, List #, View, Item#, Group
import numpy as np
import pandas as pd
from PYME.IO import tabular
from PYME.IO import MetaDataHandler
import os

@register_module('MultifitBlobs') 
class MultifitBlobs(ModuleBase):
    inputImage = CStr('input')
    outputName = CStr('positions')
    blobSigma = Float(45.0)
    threshold = Float(2.0)
    scale = Float(1000)
    
    def execute(self, namespace):
        from PYME.localization.FitFactories import GaussMultifitSR
        img = namespace[self.inputImage]
        
        img.mdh['Analysis.PSFSigma'] = self.blobSigma
        
        res = []
        
        for i in range(img.data.shape[2]):
            md = MetaDataHandler.NestedClassMDHandler(img.mdh)
            md['tIndex'] = i
            ff = GaussMultifitSR.FitFactory(self.scale*img.data[:,:,i], img.mdh, noiseSigma=np.ones_like(img.data[:,:,i].squeeze()))
        
            res.append(tabular.fitResultsSource(ff.FindAndFit(self.threshold)))
            
        res = pd.DataFrame(np.vstack(res))
        res.mdh = img.mdh
        
        namespace[self.outputName] = res

@register_module('FitDumbells') 
class FitDumbells(ModuleBase):
    inputImage = CStr('input')
    inputPositions = CStr('objPostiions')
    outputName = CStr('fitResults')
    
    def execute(self, namespace):
        from PYME.localization.FitFactories import DumbellFitR
        from PYME.IO import MetaDataHandler
        img = namespace[self.inputImage]
        
        md = MetaDataHandler.NestedClassMDHandler()
        #set metadata entries needed for fitting to suitable defaults
        md['Camera.ADOffset'] = img.data[:,:,0].min()
        md['Camera.TrueEMGain'] = 1.0
        md['Camera.ElectronsPerCount'] = 1.0
        md['Camera.ReadNoise'] = 1.0
        md['Camera.NoiseFactor'] = 1.0
        
        #copy across the entries from the real image, replacing the defaults
        #if necessary
        md.copyEntriesFrom(img.mdh)
        
        
        inp = namespace[self.inputPositions]
    
        r = np.zeros(len(inp['x']), dtype=DumbellFitR.FitResultsDType)
        
        ff_t = -1
        
        ps = img.pixelSize
        
        for x, y, t, i in zip(inp['x'], inp['y'], inp['t'], range(len(inp['x']))):
            if not t == ff_t:
                md['tIndex'] = t
                ff = DumbellFitR.FitFactory(img.data[:,:,t], md)
                ff_t = t
            
            r[i] = ff.FromPoint(x/ps, y/ps)
            
        
        res = tabular.fitResultsSource(r)
        res.mdh = md
        
        namespace[self.outputName] = res


@register_module('FitPoints')
class FitPoints(ModuleBase):
    """ Apply one of the fit modules from PYME.localization.FitFactories to each of the points in the provided
    in inputPositions
    """
    inputImage = CStr('input')
    inputPositions = CStr('objPositions')
    outputName = CStr('fitResults')
    fitModule = CStr('LatGaussFitFR')
    roiHalfSize = Int(7)
    channel = Int(0)

    def execute(self, namespace):
        #from PYME.localization.FitFactories import DumbellFitR
        from PYME.IO import MetaDataHandler
        img = namespace[self.inputImage]

        md = MetaDataHandler.NestedClassMDHandler()
        #set metadata entries needed for fitting to suitable defaults
        md['Camera.ADOffset'] = img.data[:, :, 0].min()
        md['Camera.TrueEMGain'] = 1.0
        md['Camera.ElectronsPerCount'] = 1.0
        md['Camera.ReadNoise'] = 1.0
        md['Camera.NoiseFactor'] = 1.0

        #copy across the entries from the real image, replacing the defaults
        #if necessary
        md.copyEntriesFrom(img.mdh)

        inp = namespace[self.inputPositions]

        fitMod = __import__('PYME.localization.FitFactories.' + self.fitModule,
                            fromlist=['PYME', 'localization', 'FitFactories']) #import our fitting module

        r = np.zeros(len(inp['x']), dtype=fitMod.FitResultsDType)

        ff_t = -1

        ps = img.pixelSize
        print('pixel size: %s' % ps)

        for x, y, t, i in zip(inp['x'], inp['y'], inp['t'], range(len(inp['x']))):
            if not t == ff_t:
                md['tIndex'] = t
                ff = fitMod.FitFactory(img.data[:, :, t, self.channel], md)
                ff_t = t

            #print x/ps, y/ps
            r[i] = ff.FromPoint(x/ps, y/ps, roiHalfSize=self.roiHalfSize)

        res = tabular.fitResultsSource(r, sort=False)
        res.mdh = md

        namespace[self.outputName] = res

@register_module('IntensityAtPoints')
class IntensityAtPoints(ModuleBase):
    """ Apply one of the fit modules from PYME.localization.FitFactories to each of the points in the provided
    in inputPositions
    """
    inputImage = CStr('input')
    inputPositions = CStr('objPostiions')
    outputName = CStr('fitResults')
    radii = List([3, 5, 7, 9, 11])
    mode = Enum(['sum', 'mean'])
    #fitModule = CStr('LatGaussFitFR')

    def __init__(self, *args, **kwargs):
        self._mask_cache = {}
        ModuleBase.__init__(self, *args, **kwargs)

    def _get_mask(self, r):
        #if not '_mask_cache' in dir(self):
        #    self._mask_cache = {}

        if not r in self._mask_cache.keys():
            x_, y_ = np.mgrid[-r:(r+1.), -r:(r+1.)]
            self._mask_cache[r] = 1.0*((x_*x_ + y_*y_) < r*r)

        return self._mask_cache[r]


    def _get_mean(self, data, x, y, t, radius):
        roi = data[(x-radius):(x + radius + 1), (y-radius):(y + radius + 1), t].squeeze()
        mask = self._get_mask(radius)

        return (roi.squeeze()*mask).sum()/mask.sum()

    def _get_sum(self, data, x, y, t, radius):
        print data.shape, x, y, t
        roi = data[(x - radius):(x + radius + 1), (y - radius):(y + radius + 1), t].squeeze()
        mask = self._get_mask(radius)

        print mask.shape, roi.shape#, (roi * mask).shape

        return (roi.squeeze() * mask).sum()

    def execute(self, namespace):
        #from PYME.localization.FitFactories import DumbellFitR
        from PYME.IO import MetaDataHandler
        img = namespace[self.inputImage]

        md = MetaDataHandler.NestedClassMDHandler()
        #set metadata entries needed for fitting to suitable defaults
        md['Camera.ADOffset'] = img.data[:, :, 0].min()
        md['Camera.TrueEMGain'] = 1.0
        md['Camera.ElectronsPerCount'] = 1.0
        md['Camera.ReadNoise'] = 1.0
        md['Camera.NoiseFactor'] = 1.0

        #copy across the entries from the real image, replacing the defaults
        #if necessary
        md.copyEntriesFrom(img.mdh)

        inp = namespace[self.inputPositions]

        res = np.zeros(len(inp['x']), dtype=[('r%d' % r, 'f4') for r in self.radii])

        ff_t = -1

        aggFunc = getattr(self, '_get_%s' % self.mode)

        ps = img.pixelSize
        print('pixel size: %s' % ps)
        for x, y, t, i in zip(inp['x'], inp['y'], inp['t'], range(len(inp['x']))):
            for r in self.radii:
                res[i]['r%d' % r] = aggFunc(img.data, np.round(x / ps), np.round(y / ps), t, r)

        res = tabular.recArrayInput(res)
        res.mdh = md

        namespace[self.outputName] = res



@register_module('MeanNeighbourDistances') 
class MeanNeighbourDistances(ModuleBase):
    """Calculates mean distance to nearest neighbour in a triangulation of the
    supplied points"""
    inputPositions = CStr('input')
    outputName = CStr('neighbourDists')
    key = CStr('neighbourDists')
    
    def execute(self, namespace):
        from matplotlib import delaunay
        from PYME.LMVis import visHelpers
        pos = namespace[self.inputPositions]
        
        x, y = pos['x'], pos['y']
        #triangulate the data
        T = delaunay.Triangulation(x + .1*np.random.normal(size=len(x)), y + .1*np.random.normal(size=len(x)))
        #find the average edge lengths leading away from a given point
        res = np.array(visHelpers.calcNeighbourDists(T))
        
        res = pd.DataFrame({self.key:res})
        if 'mdh' in dir(pos):
            res.mdh = pos.mdh
        
        namespace[self.outputName] = res

@register_module('NearestNeighbourDistances')
class NearestNeighbourDistances(ModuleBase):
    """Calculates the nearest neighbour distances between supplied points using
    a kdtree"""
    inputPositions = CStr('input')
    outputName = CStr('neighbourDists')
    columns = List(['x', 'y'])
    key = CStr('neighbourDists')
    
    def execute(self, namespace):
        from scipy.spatial import cKDTree
        pos = namespace[self.inputPositions]

        
        #create a kdtree
        p = np.vstack([pos[k] for k in self.columns]).T
        kdt = cKDTree(p)
        
        #query the two closest entries - the closest entry will be the 
        #original point, the next closest it's nearest neighbour
        d, i = kdt.query(p, 2)
        res = d[:,1]
        
        res = pd.DataFrame({self.key: res})
        if 'mdh' in dir(pos):
            res.mdh = pos.mdh
        
        namespace[self.outputName] = res

@register_module('NearestNeighbourTwoSpecies')
class NearestNeighbourTwoSpecies(ModuleBase):
    """Calculates the nearest neighbour distances between supplied points using
    a kdtree"""
    inputChan0 = CStr('input')
    inputChan1 = CStr('input')
    outputName = CStr('neighbourDists')
    columns = List(['x', 'y'])
    key = CStr('neighbourDists')

    def execute(self, namespace):
        from scipy.spatial import cKDTree
        pos = namespace[self.inputChan0]
        pos1 = namespace[self.inputChan1]

        #create a kdtree
        p1 = np.vstack([pos[k] for k in self.columns]).T
        p2 = np.vstack([pos1[k] for k in self.columns]).T
        kdt = cKDTree(p1)

        d, i = kdt.query(p2, 1)

        res = pd.DataFrame({self.key: d})
        if 'mdh' in dir(pos):
            res.mdh = pos.mdh

        namespace[self.outputName] = res

@register_module('PairwiseDistanceHistogram')
class PairwiseDistanceHistogram(ModuleBase):
    """Calculates a histogram of pairwise distances"""
    inputPositions = CStr('input')
    outputName = CStr('distHist')
    nbins = Int(50)
    binSize = Float(50.)
    
    def execute(self, namespace):
        from PYME.Analysis.points import DistHist
        
        pos = namespace[self.inputPositions]
        
        x, y = pos['x'], pos['y']
        
        res = DistHist.distanceHistogram(x, y, x, y, self.nbins, self.binsize)
        d = self.binsize*np.arange(self.nbins)
        
        res = pd.DataFrame({'bins' : d, 'counts' : res})
        if 'mdh' in dir(pos):
            res.mdh = pos.mdh
        
        namespace[self.outputName] = res
        

@register_module('Histogram')         
class Histogram(ModuleBase):
    """Calculates a histogram of a given measurement key"""
    inputMeasurements = CStr('input')
    outputName = CStr('hist')
    key = CStr('key')
    nbins = Int(50)
    left = Float(0.)
    right = Float(1000)
    
    def execute(self, namespace):        
        v = namespace[self.inputMeasurements][self.key]
        
        edges = np.linspace(self.left, self.right, self.nbins)
        
        res = np.histogram(v, edges)[0]
        
        res = pd.DataFrame({'bins' : edges, 'counts' : res})
        if 'mdh' in dir(v):
            res.mdh = v.mdh
        
        namespace[self.outputName] = res
        

        
@register_module('ImageHistogram')         
class ImageHistogram(ModuleBase):
    """Calculates a histogram of a given measurement key"""
    inputImage = CStr('input')
    outputName = CStr('hist')
    inputMask = CStr('')
    nbins = Int(50)
    left = Float(0.)
    right = Float(1000)
    
    def execute(self, namespace):
        v = namespace[self.inputImage]
        vals = v.data[:,:,:].ravel()
        
        if not self.inputMask == '':
            m = namespace[self.inputMask].data[:,:,:].ravel() >0
        
            vals = vals[m]
        
        edges = np.linspace(self.left, self.right, self.nbins)
        
        res = np.histogram(vals, edges)[0]
        
        res = pd.DataFrame({'bins' : edges, 'counts' : res})
        if 'mdh' in dir(v):
            res.mdh = v.mdh
        
        namespace[self.outputName] = res
        
@register_module('ImageCumulativeHistogram')         
class ImageCumulativeHistogram(ModuleBase):
    """Calculates a histogram of a given measurement key"""
    inputImage = CStr('input')
    outputName = CStr('hist')
    inputMask = CStr('')
    #nbins = Int(50)
    #left = Float(0.)
    #right = Float(1000)
    
    def execute(self, namespace):
        v = namespace[self.inputImage]
        vals = v.data[:,:,:].ravel()
        
        if not self.inputMask == '':
            m = namespace[self.inputMask].data[:,:,:].ravel() > 0 
        
            vals = vals[m]
        
        yvals = np.linspace(0, 1.0, len(vals))
        xvals = np.sort(vals)
        
        #res = np.histogram(v, edges)[0]
        
        res = pd.DataFrame({'bins' : xvals, 'counts' : yvals})
        if 'mdh' in dir(v):
            res.mdh = v.mdh
        
        namespace[self.outputName] = res

@register_module('BinnedHistogram')
class BinnedHistogram(ModuleBase):
    """Calculates a histogram of a given measurement key, binned by a separate value"""
    inputImage = CStr('input')
    binBy = CStr('indepvar')
    outputName = CStr('hist')
    inputMask = CStr('')

    nbins = Int(50)
    left = Float(0.)
    right = Float(1000)

    def execute(self, namespace):
        from PYME.Analysis import binAvg

        v = namespace[self.inputImage]
        vals = v.data[:, :, :].ravel()

        binby = namespace[self.binBy]
        binby = binby.data[:,:,:].ravel()

        if not self.inputMask == '':
            m = namespace[self.inputMask].data[:, :, :].ravel() > 0

            vals = vals[m]
            binby = binby[m]

        #mask out NaNs
        m2 = ~np.isnan(vals)
        vals = vals[m2]
        binby = binby[m2]

        edges = np.linspace(self.left, self.right, self.nbins)

        bn, bm, bs = binAvg.binAvg(binby, vals, edges)

        res = pd.DataFrame({'bins': 0.5*(edges[:-1] + edges[1:]), 'counts': bn, 'means' : bm})
        if 'mdh' in dir(v):
            res.mdh = v.mdh

        namespace[self.outputName] = res

        

@register_module('Measure2D') 
class Measure2D(ModuleBase):
    """Module with one image input and one image output"""
    inputLabels = CStr('labels')
    inputIntensity = CStr('data')
    outputName = CStr('measurements')
    
    measureContour = Bool(True)    
        
    def execute(self, namespace):       
        labels = namespace[self.inputLabels]
        
        #define the measurement class, which behaves like an input filter        
        class measurements(tabular.TabularBase):
            _name = 'Measue 2D source'
            ps = labels.pixelSize
            
            def __init__(self):
                self.measures = []
                self.contours = []
                self.frameNos = []
                
                self._keys = []
                
            def addFrameMeasures(self, frameNo, measurements, contours = None):
                if len(measurements) == 0:
                    return
                if len(self.measures) == 0:
                    #first time we've called this - determine our data type
                    self._keys = ['t', 'x', 'y'] + [r for r in dir(measurements[0]) if not r.startswith('_')]
                    
                    self._keys.remove('euler_number') #buggy!
                    
                    if not contours is None:
                        self._keys += ['contour']
                        
                self.measures.extend(measurements)
                self.frameNos.extend([frameNo for i in xrange(len(measurements))])
                
                if contours:
                    self.contours.extend(contours)
            
            def keys(self):
                return self._keys
        
            def __getitem__(self, keys):
                if isinstance(keys, tuple) and len(keys) > 1:
                    key = keys[0]
                    sl = keys[1]
                else:
                    key = keys
                    sl = slice(None)
                
                    
                #print key, sl
                    
                if not key in self._keys:
                    raise RuntimeError('Key not defined')
                
                if key == 't':
                    return np.array(self.frameNos[sl])
                elif key == 'contour':
                    return np.array(self.contours[sl], dtype='O')
                elif key == 'x':
                    return self.ps*np.array([r.centroid[0] for r in self.measures[sl]])
                elif key == 'y':
                    return self.ps*np.array([r.centroid[1] for r in self.measures[sl]])
                else:
                    l = [getattr(r, key) for r in self.measures[sl]]
                    
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
            
        m.mdh = labels.mdh
                    
        namespace[self.outputName] = m#.toDataFrame()      
        
    def _measureFrame(self, frameNo, labels, intensity):
        import skimage.measure
        
        li = labels.data[:,:,frameNo].squeeze().astype('i')

        if intensity:
            it = intensity.data[:,:,frameNo].squeeze()
        else:
            it = None
            
        rp = skimage.measure.regionprops(li, it, cache=True)
        
        #print len(rp), li.max()
        
        if self.measureContour:
            ct = []
            for r in rp:
                #print i, (li == (i+1)).sum()
                #c = skimage.measure.find_contours(r['image'], .5)
                c = skimage.measure.find_contours((li == r.label), .5)
                if len(c) == 0:
                    c = [np.zeros((2,2))]
                #ct.append(c[0] + np.array(r['bbox'])[:2][None,:])
                ct.append(c[0])
        else:
            ct = None
            
        
            
        return rp, ct

@register_module('SelectMeasurementColumns')         
class SelectMeasurementColumns(ModuleBase):
    """Take just certain columns of a variable"""
    inputMeasurements = CStr('measurements')
    keys = CStr('')
    outputName = CStr('selectedMeasurements') 
    
    def execute(self, namespace):       
        meas = namespace[self.inputMeasurements]
        out = pd.DataFrame({k:meas[k] for k in self.keys.split()})
        if 'mdh' in dir(meas):
            #propagate metadata
            out.mdh = meas.mdh
            
        namespace[self.outputName] = out
        
@register_module('Plot')         
class Plot(ModuleBase):
    """Take just certain columns of a variable"""
    input0 = CStr('measurements')
    input1 = CStr('')
    input2 = CStr('')
    input3 = CStr('')
    xkey = CStr('')
    ykey = CStr('')
    outputName = CStr('outGraph') 
    
    def execute(self, namespace):
        ms = []
        labs = []
        if not self.input0 == '':
            ms.append(namespace[self.input0])
            labs.append(self.input0)
        if not self.input1 == '':
            ms.append(namespace[self.input1])
            labs.append(self.input1)
        if not self.input2 == '':
            ms.append(namespace[self.input2])
            labs.append(self.input2)
        if not self.input3 == '':
            ms.append(namespace[self.input3])
            labs.append(self.input3)

        import pylab
        
        pylab.figure()
        for meas in ms:
            pylab.plot(meas[self.xkey], meas[self.ykey])
        
        pylab.grid()
        pylab.legend(labs)
        pylab.xlabel(self.xkey)
        pylab.ylabel(self.ykey)
        #namespace[self.outputName] = out

@register_module('AddMetadataToMeasurements')         
class AddMetadataToMeasurements(ModuleBase):
    """Adds metadata entries as extra column(s) to the output"""
    inputMeasurements = CStr('measurements')
    inputImage = CStr('input')
    keys = CStr('SampleNotes')
    metadataKeys = CStr('Sample.Notes')
    outputName = CStr('annotatedMeasurements')
    
    def execute(self, namespace):
        res = {}
        meas = namespace[self.inputMeasurements]
        res.update(meas)
        
        img = namespace[self.inputImage]

        nEntries = len(res.values()[0])
        for k, mdk in zip(self.keys.split(), self.metadataKeys.split()):
            if mdk == 'seriesName':
                #import os
                v = os.path.split(img.seriesName)[1]
            else:
                v = img.mdh[mdk]
            res[k] = np.array([v]*nEntries)
        
        res = pd.DataFrame(res)
        #if 'mdh' in dir(meas):
        res.mdh = img.mdh
        
        namespace[self.outputName] = res


@register_module('AggregateMeasurements')         
class AggregateMeasurements(ModuleBase):
    """Create a new composite measurement containing the results of multiple
    previous measurements"""
    inputMeasurements1 = CStr('meas1')
    suffix1 = CStr('')
    inputMeasurements2 = CStr('')
    suffix2 = CStr('')
    inputMeasurements3 = CStr('')
    suffix3 = CStr('')
    inputMeasurements4 = CStr('')
    suffix4 = CStr('')
    outputName = CStr('aggregatedMeasurements') 
    
    def execute(self, namespace):
        res = {}
        for mk, suffix in [(getattr(self, n), getattr(self, 'suffix' + n[-1])) for n in dir(self) if n.startswith('inputMeas')]:
            if not mk == '':
                meas = namespace[mk]
                
                #res.update(meas)
                for k in meas.keys():
                    res[k + suffix] = meas[k]
                
        
        meas1 = namespace[self.inputMeasurements1]
        #res = pd.DataFrame(res)
        res = tabular.cloneSource(res)
        if 'mdh' in dir(meas1):
            res.mdh = meas1.mdh
            
        namespace[self.outputName] = res
        


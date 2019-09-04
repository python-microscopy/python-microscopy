# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Mon May 25 17:10:02 2015

@author: david
"""
from .base import ModuleBase, register_module, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List
import numpy as np
import pandas as pd
from PYME.IO import tabular
from PYME.IO import MetaDataHandler
import os
from six.moves import xrange
import logging

logger = logging.getLogger(__name__)

@register_module('MultifitBlobs') 
class MultifitBlobs(ModuleBase):
    inputImage = Output('input')
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
    inputImage = Input('input')
    inputPositions = Input('objPostiions')
    outputName = Output('fitResults')
    
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

@register_module('DetectPoints2D')
class DetectPoints2D(ModuleBase):
    """

    Parameters
    ----------
    input_name : Input
        PYME.IO.ImageStack
    snr_threshold : Bool
        How should we interpret the threshold? If True, the signal-to-noise (SNR) is estimated at each pixel, and the threshold 
        applied at each pixel is this estimate multiplied by the 'threshold' parameter. If False, the threshold parameter is used
        directly.
    threshold : Float
        The intensity threshold applied during detection if 'snr_threshold' is False, otherwise this scalar is first
        multiplied by the SNR estimate at each pixel before the threshold is applied
    debounce_radius : Int
        Radius is pixels to check for other detected points. If multiple points are found within this radius the brightest 
        will be preserved and the other(s) will be removed
    edge_mask_width : Int
        Thickness of border region to exclude detecting points from.

    Returns
    -------
    output_name : Output
        PYME.IO.tabular containing x and y coordinates of each point, as well as the frame index they were detected on

    Notes
    -----

    Input image series should already be camera corrected (see Processing.FlatfieldAndDarkCorrect)

    """

    input_name = Input('input')

    threshold = Float(1.)
    debounce_radius = Int(4)
    snr_threshold = Bool(True)
    edge_mask_width = Int(5)

    output_name = Output('candidate_points')

    def execute(self, namespace):
        from PYME.localization.ofind import ObjectIdentifier
        from PYME.localization.remFitBuf import fitTask

        im_stack = namespace[self.input_name]

        x, y, t = [], [], []
        # note that ObjectIdentifier is only 2D-aware
        for ti in range(im_stack.data.shape[2]):
            frame = im_stack.data.getSlice(ti)
            finder = ObjectIdentifier(frame * (frame > 0))

            if self.snr_threshold:  # calculate a per-pixel threshold based on an estimate of the SNR
                sigma = fitTask.calcSigma(im_stack.mdh, frame).squeeze()
                threshold = sigma * self.threshold
            else:
                threshold = self.threshold

            finder.FindObjects(threshold, 0, debounceRadius=self.debounce_radius, maskEdgeWidth=self.edge_mask_width)

            x.append(finder.x[:])
            y.append(finder.y[:])
            t.append(ti * np.ones_like(finder.x[:]))

        # FIXME - make a dict source so we don't abuse the mapping filter for everything
        out = tabular.mappingFilter({'x': np.concatenate(x, axis=0), 'y': np.concatenate(y, axis=0),
                                     't': np.concatenate(t, axis=0)})

        out.mdh = MetaDataHandler.NestedClassMDHandler()
        out.mdh.copyEntriesFrom(im_stack.mdh)

        out.mdh['Processing.DetectPoints2D.SNRThreshold'] = self.snr_threshold
        out.mdh['Processing.DetectPoints2D.DetectionThreshold'] = self.threshold
        out.mdh['Processing.DetectPoints2D.DebounceRadius'] = self.debounce_radius
        out.mdh['Processing.DetectPoints2D.MaskEdgeWidth'] = self.edge_mask_width

        namespace[self.output_name] = out

@register_module('FitPoints')
class FitPoints(ModuleBase):
    """ Apply one of the fit modules from PYME.localization.FitFactories to each of the points in the provided
    in inputPositions
    """
    inputImage = Input('input')
    inputPositions = Input('objPositions')
    outputName = Output('fitResults')
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

        md['voxelsize.x'] = .001
        md['voxelsize.y'] = .001

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
    inputImage = Input('input')
    inputPositions = Input('objPostiions')
    outputName = Output('fitResults')
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
        print(data.shape, x, y, t)
        roi = data[(x - radius):(x + radius + 1), (y - radius):(y + radius + 1), t].squeeze()
        mask = self._get_mask(radius)

        print(mask.shape, roi.shape)#, (roi * mask).shape

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
    inputPositions = Input('input')
    outputName = Output('neighbourDists')
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
    inputChan0 = Input('input')
    inputChan1 = Input('')
    outputName = Output('neighbourDists')
    columns = List(['x', 'y'])
    key = CStr('neighbourDists')

    def execute(self, namespace):
        from scipy.spatial import cKDTree
        pos = namespace[self.inputChan0]
        
        if self.inputChan1 == '':
            pos1 = pos
            singleChan = True  # flag to not pair molecules with themselves
        else:
            pos1 = namespace[self.inputChan1]
            singleChan = False

        #create a kdtree
        p1 = np.vstack([pos[k] for k in self.columns]).T
        p2 = np.vstack([pos1[k] for k in self.columns]).T
        kdt = cKDTree(p1)

        if singleChan:
            #query the two closest entries - the closest entry will be the orig point paired with itself, so ignore it
            d, i = kdt.query(p2, 2)
            d = d[:, 1]
        else:
            d, i = kdt.query(p2, 1)

        res = pd.DataFrame({self.key: d})
        if 'mdh' in dir(pos):
            res.mdh = pos.mdh

        namespace[self.outputName] = res

@register_module('PairwiseDistanceHistogram')
class PairwiseDistanceHistogram(ModuleBase):
    """Calculates a histogram of pairwise distances"""
    inputPositions = Input('input')
    inputPositions2 = Input('')
    outputName = Output('distHist')
    nbins = Int(50)
    binSize = Float(50.)
    
    def execute(self, namespace):
        from PYME.Analysis.points import DistHist
        
        pos0 = namespace[self.inputPositions]
        pos1 = namespace[self.inputPositions2 if self.inputPositions2 is not '' else self.inputPositions]
        if np.count_nonzero(pos0['z']) == 0 and np.count_nonzero(pos1['z']) == 0:
            res = DistHist.distanceHistogram(pos0['x'], pos0['y'], pos1['x'], pos1['y'], self.nbins, self.binSize)
        else:
            res = DistHist.distanceHistogram3D(pos0['x'], pos0['y'], pos0['z'],
                                               pos1['x'], pos1['y'], pos1['z'], self.nbins, self.binSize)

        d = self.binSize*np.arange(self.nbins)

        res = pd.DataFrame({'bins': d, 'counts': res})

        # propagate metadata, if present
        try:
            res.mdh = pos0.mdh
        except AttributeError:
            try:
                res.mdh = pos1.mdh
            except AttributeError:
                pass
        
        namespace[self.outputName] = res
        

@register_module('Histogram')         
class Histogram(ModuleBase):
    """Calculates a histogram of a given measurement key"""
    inputMeasurements = Input('input')
    outputName = Output('hist')
    key = CStr('key')
    nbins = Int(50)
    left = Float(0.)
    right = Float(1000)
    normalize = Bool(False)
    
    def execute(self, namespace):        
        v = namespace[self.inputMeasurements][self.key]
        
        edges = np.linspace(self.left, self.right, self.nbins)
        
        res = np.histogram(v, edges, normed=self.normalize)[0]
        
        res = pd.DataFrame({'bins' : 0.5*(edges[:-1] + edges[1:]), 'counts' : res})
        if 'mdh' in dir(v):
            res.mdh = v.mdh
        
        namespace[self.outputName] = res
        

        
@register_module('ImageHistogram')         
class ImageHistogram(ModuleBase):
    """Calculates a histogram of a given measurement key"""
    inputImage = Input('input')
    outputName = Output('hist')
    inputMask = Input('')
    nbins = Int(50)
    left = Float(0.)
    right = Float(1000)
    normalize = Bool(False)
    
    def execute(self, namespace):
        v = namespace[self.inputImage]
        vals = v.data[:,:,:].ravel()
        
        if not self.inputMask == '':
            m = namespace[self.inputMask].data[:,:,:].ravel() >0
        
            vals = vals[m]
        
        edges = np.linspace(self.left, self.right, self.nbins)
        
        res = np.histogram(vals, edges, normed=self.normalize)[0]
        
        res = pd.DataFrame({'bins' : 0.5*(edges[:-1] + edges[1:]), 'counts' : res})
        if 'mdh' in dir(v):
            res.mdh = v.mdh
        
        namespace[self.outputName] = res
        
@register_module('ImageCumulativeHistogram')         
class ImageCumulativeHistogram(ModuleBase):
    """Calculates a histogram of a given measurement key"""
    inputImage = Input('input')
    outputName = Output('hist')
    inputMask = Input('')
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
    inputImage = Input('input')
    binBy = CStr('indepvar')
    outputName = Output('hist')
    inputMask = Input('')

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
    inputLabels = Input('labels')
    inputIntensity = Input('data')
    outputName = Output('measurements')
    
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
                    
                    if not (('image' in key) or (key == 'coords')):#np.isscalar(l[0]):
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


        
@register_module('Plot')         
class Plot(ModuleBase):
    """Take just certain columns of a variable"""
    input0 = Input('measurements')
    input1 = Input('')
    input2 = Input('')
    input3 = Input('')
    xkey = CStr('')
    ykey = CStr('')
    type = Enum(['line', 'bar'])
    outputName = Output('outGraph')
    
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
            if self.type == 'bar':
                xv = meas[self.xkey]
                pylab.bar(xv, meas[self.ykey], align='center', width=(xv[1] - xv[0]))
            else:
                pylab.plot(meas[self.xkey], meas[self.ykey])
        
        pylab.grid()
        pylab.legend(labs)
        pylab.xlabel(self.xkey)
        pylab.ylabel(self.ykey)
        #namespace[self.outputName] = out

@register_module('AddMetadataToMeasurements')         
class AddMetadataToMeasurements(ModuleBase):
    """Adds metadata entries as extra column(s) to the output

    This was written to allow key parameters or experimental variables to be tracked prior to aggregating / concatenating
    measurements from multiple different images. By adding, e.g. the sample labelling, or a particular experimental
    parameter to an output table as an extra column we can then aggregate and group that data in processing.
    """
    inputMeasurements = Input('measurements')
    inputImage = Input('input')
    keys = CStr('SampleNotes')
    metadataKeys = CStr('Sample.Notes')
    outputName = Output('annotatedMeasurements')
    
    def execute(self, namespace):
        res = {}
        meas = namespace[self.inputMeasurements]
        res.update(meas)
        
        img = namespace[self.inputImage]

        nEntries = len(list(res.values())[0])
        for k, mdk in zip(self.keys.split(), self.metadataKeys.split()):
            if mdk == 'seriesName':
                #import os
                v = os.path.split(img.seriesName)[1]
            else:
                v = img.mdh[mdk]
            res[k] = np.array([v]*nEntries)
        
        #res = pd.DataFrame(res)
        res = tabular.mappingFilter(res)
        #if 'mdh' in dir(meas):
        res.mdh = img.mdh
        
        namespace[self.outputName] = res
        
        
@register_module('TilePhysicalCoords')
class TilePhysicalCoords(ModuleBase):
    """
    Adds x_um and y_um columns to the results of Measure2D, performed on an Supertile image sequence, mapping the x and y
    values (which are in pixels with respect to the current frame) to physical co-ordinates
    
    NOTE: inputImage must be a SupertileDatasource instance
    
    TODO: Does this belong here??
    """

    inputMeasurements = Input('measurements')
    inputImage = Input('input')
    
    outputName = Output('meas_physical_coords')
    
    def execute(self, namespace):
        meas = namespace[self.inputMeasurements]
        img = namespace[self.inputImage]
        
        out = tabular.mappingFilter(meas)
        
        x_frame_um, y_frame_um =img.data.tile_coords_um[meas['t']].T
        x_frame_px, y_frame_px = img.data.tile_coords[meas['t']].T
        
        out.addColumn('x_um', x_frame_um + 1e-3*meas['x'])
        out.addColumn('y_um', y_frame_um + 1e-3*meas['y'])

        out.addColumn('x_px', x_frame_px + meas['x'])
        out.addColumn('y_px', y_frame_px + meas['y'])
        
        out.mdh = meas.mdh
        
        namespace[self.outputName] = out


@register_module('FilterOverlappingROIs')
class FilterOverlappingROIs(ModuleBase):
    """

    Filter input ROI positions such that ROIs of a given size will not overlap. Output maintains all points, but with
    the addition of a column indicating whether the point has been rejected due to overlap or not.

    Parameters
    ----------
    input : Input
        PYME.IO.tabular containing x and y coordinates. Compatible with measurement output for Supertile coordinates,
        e.g. 'x_um'
    roi_size_pixels: Int
        Size of ROI to be used in calculating overlap.
    reject_key: CStr
        Name of column to add to output datasource encoding whether a point has been rejected due to overlap (1) or
        kept (0).
    output: Output
        PYME.IO.tabular

    Notes
    -----
    Currently roi overlap is defined as being within sqrt(2) * roi_size. Obviously two square ROIs can be roi_size + 1
    away from each other and not overlap if they are arranged correctly, so the current check is a little more happy-to-
    toss points then it could be.

    """
    input = Input('input')
    roi_size_pixels = Int(256)
    reject_key = CStr('rejected')
    output = Output('cluster_metrics')

    def execute(self, namespace):
        from scipy.spatial import KDTree

        points = namespace[self.input]

        try:
            positions = np.stack([points['x_um'], points['y_um']], axis=1)
        except KeyError:
            positions = np.stack([points['x'], points['y']], axis=1) / 1e3  # assume x and y were in [nanometers]

        tree = KDTree(positions)

        far_flung = np.sqrt(2) * points.mdh['voxelsize.x'] * self.roi_size_pixels  # [micrometers]
        neighbors = tree.query_ball_tree(tree, r=far_flung, p=2)

        tossing = set()
        for ind, close in enumerate(neighbors):
            # ignore points we've already decided to reject
            if ind not in tossing and len(close) > 1:
                # reject points too close to our current indexed point
                # TODO - don't reject points inside of this cirdular radius if their square ROIs don't actually overlap
                close.remove(ind)
                tossing.update(close)

        out = tabular.mappingFilter(points)
        reject = np.zeros(tree.n, dtype=int)
        reject[list(tossing)] = 1
        out.addColumn(self.reject_key, reject.astype(int))

        try:
            out.mdh = points.mdh
        except AttributeError:
            pass

        namespace[self.output] = out

def two_opt_section(positions, start_section, counts, n_tasks, epsilon, master_route):
    """
    Perform two-opt TSP on (potentially) multiple sections.

    Parameters
    ----------
    positions: ndarray
        positions, shape (n_points, 2), where n_points are just the positions for the tasks this call is responsible for
    start_section: int
        index denoting where the first position in the input positions belongs in the master_route array
    counts: ndarray
        array of counts corresponding to the number of positions in each sorting task
    n_tasks: int
        number of sorting tasks to execute in this function call. Each task will be sorted independently and shoved into
        the master_route array in the order the tasks are executed.
    epsilon: float
        relative improvement exit criteria for sorting
    master_route: shmarray
        output array for the sorted tasks

    """
    from scipy.spatial import distance_matrix
    start_pos = 0
    start_route = start_section
    for ti in range(n_tasks):
        pos = positions[start_pos: start_pos + counts[ti]]
        distances = distance_matrix(pos, pos)

        # start on a corner, rather than center
        route = np.argsort(pos[:, 0] + pos[:, 1])
        # print('route %s' % (route,))

        best_route, best_distance, og_distance = two_opt(distances, epsilon, route)
        # print(best_route)
        master_route[start_route:start_route + counts[ti]] = start_route + best_route
        start_route += counts[ti]
        start_pos += counts[ti]




@register_module('ChunkedTravelingSalesperson')
class ChunkedTravelingSalesperson(ModuleBase):
    """

    Optimize route visiting each position in an input dataset exactly once, starting from the last point in the input.
    2-opt algorithm is used, and points are chunked (very) roughly into a grid based on an assumption of uniform density
    and the points_per_chunk argument. After the chunks are individually optimized, a modified two-opt algorithm is run
    on the section end-points, so section order can be optimized as well as whether sections are traversed forwards or
    backwards.

    Parameters
    ----------
    input : Input
        PYME.IO.tabular containing x and y coordinates. Compatible with measurement output for Supertile coordinates,
        e.g. 'x_um'
    epsilon: Float
        Relative improvement threshold used to stop algorithm when gains become negligible
    points_per_chunk: Int
        Number of points desired to be in each chunk that a two-opt algorithm is run on. Larger chunks tend toward more
        ideal paths, but much larger computational complexity.
    output: Output
        PYME.IO.tabular

    """
    input = Input('input')
    epsilon = Float(0.001)
    points_per_chunk = Int(500)
    output = Output('sorted')

    def execute(self, namespace):
        import multiprocessing
        from PYME.util.shmarray import shmarray
        import time

        points = namespace[self.input]

        try:
            positions = np.stack([points['x_um'], points['y_um']], axis=1)
        except KeyError:
            positions = np.stack([points['x'], points['y']], axis=1) / 1e3

        # assume density is uniform
        x_min, y_min = positions.min(axis=0)
        x_max, y_max = positions.max(axis=0)

        sections_per_side = int(np.sqrt((positions.shape[0] /  self.points_per_chunk)))
        size_x = (x_max - x_min) / sections_per_side
        size_y = (y_max - y_min) / sections_per_side

        # bin points into our "pixels"
        X = np.round(positions[:, 0] / size_x).astype(int)
        Y = np.round(positions[:, 1] / size_y).astype(int)

        # number the sections
        section = X + Y * (Y.max() + 1)
        # keep all section numbers positive, starting at zero
        section -= section.min()
        n_sections = int(section.max() + 1)
        I = np.argsort(section)
        section = section[I]
        positions = positions[I, :]

        # import matplotlib.pyplot as plt
        # from matplotlib import cm
        # colors = cm.get_cmap('hsv', n_sections)
        # plt.figure()
        # for pi in range(len(section)):
        #     plt.scatter(positions[pi, 0], positions[pi, 1], marker='$' + str(section[pi]) + '$', color=colors(section[pi]))
        # # plt.plot(positions[:, 0], positions[:, 1])
        # plt.show()

        # split out points
        n_cpu = multiprocessing.cpu_count()
        tasks = int(n_sections / n_cpu) * np.ones(n_cpu, 'i')
        tasks[:int(n_sections % n_cpu)] += 1

        route = shmarray.zeros(positions.shape[0], dtype='i')

        uni, counts = np.unique(section, return_counts=True)
        logger.debug('%d points total, section counts: %s' % (counts.sum(), (counts,)))
        if (counts > 1000).any():
            logger.warning('%d counts in a bin, traveling salesperson algorithm may be very slow' % counts.max())

        ind_task_start = 0
        ind_pos_start = 0
        processes = []

        cumcount = counts.cumsum()
        cumtasks = tasks.cumsum()
        t = time.time()
        for ci in range(n_cpu):
            ind_task_end = cumtasks[ci]
            ind_pos_end = cumcount[ind_task_end -1]

            subcounts = counts[ind_task_start: ind_task_end]

            p = multiprocessing.Process(target=two_opt_section,
                                        args=(positions[ind_pos_start:ind_pos_end, :],
                                              ind_pos_start,
                                              subcounts,
                                              tasks[ci], self.epsilon, route))
            p.start()
            processes.append(p)
            ind_task_start = ind_task_end
            ind_pos_start = ind_pos_end

        # next we need to join our sections. Prepare for this while the other processes are executing
        pivot_indices = np.sort(np.concatenate([[0], cumcount[:-1], cumcount - 1]))  # get start/stop indices for each

        [p.join() for p in processes]
        print('Chunked TSPs finished after ~%.2f s, connecting chunks' % (time.time() - t))

        # do a two-opt on just the section start/ends, with ability to reverse the section
        # pivot positions won't be correct unless they're already sorted. No need to sort section because its the same
        pivot_positions = positions[route, :][pivot_indices]
        # spike the exit criteria low since the cost is cheap and the gains are high
        section_order, reversals = reversal_two_opt(section[pivot_indices], pivot_positions, self.epsilon/1e3)

        final_route = np.copy(route)
        start = cumcount[0]
        new_pivot_inds = []
        for sind in range(1, n_sections):  # we got section 0 for free with the copy
            cur_section = section_order[sind]
            section_count = counts[cur_section]
            if reversals[sind]:
                final_route[start: start + section_count] = route[cumcount[cur_section - 1]:cumcount[cur_section]][::-1]
            else:
                final_route[start: start + section_count] = route[cumcount[cur_section - 1]:cumcount[cur_section]]
            new_pivot_inds.append(start)
            new_pivot_inds.append(start + section_count - 1)
            start += section_count

        # import matplotlib.pyplot as plt
        # from matplotlib import cm
        # colors = cm.get_cmap('prism', n_sections)
        # plt.figure()
        # sorted_pos = positions[route, :]
        # plt.plot(positions[final_route, 0], positions[final_route, 1], color='k')
        # plt.scatter(positions[final_route, 0][new_pivot_inds], positions[final_route, 1][new_pivot_inds], color='k')
        # for pi in range(len(section)):
        #     plt.scatter(sorted_pos[pi, 0], sorted_pos[pi, 1], marker='$' + str(section[pi]) + '$',
        #                 color=colors(section[pi]))
        # plt.show()

        # note that we sorted the positions / sections once before, need to propagate that through before sorting
        out = tabular.mappingFilter({k: points[k][I][final_route] for k in points.keys()})
        out.mdh = MetaDataHandler.NestedClassMDHandler()
        try:
            out.mdh.copyEntriesFrom(points.mdh)
        except AttributeError:
            pass

        # use the already sorted output to get the final distance
        try:
            og_distance = np.sqrt((points['x_um'][1:] - points['x_um'][:-1]) ** 2 + (points['y_um'][1:] - points['y_um'][:-1]) ** 2).sum()
            final_distance = np.sqrt((out['x_um'][1:] - out['x_um'][:-1]) ** 2 + (out['y_um'][1:] - out['y_um'][:-1]) ** 2).sum()
        except KeyError:
            og_distance = np.sqrt((points['x'][1:] - points['x'][:-1]) ** 2 + (points['y'][1:] - points['y'][:-1]) ** 2).sum() / 1e3
            final_distance = np.sqrt((out['x'][1:] - out['x'][:-1]) ** 2 + (out['y'][1:] - out['y'][:-1]) ** 2).sum() / 1e3

        out.mdh['TravelingSalesperson.OriginalDistance'] = og_distance
        out.mdh['TravelingSalesperson.Distance'] = final_distance

        namespace[self.output] = out

def reversal_two_opt(section_ids, pivot_positions, epsilon):
    """

    Parameters
    ----------
    pivot_indices: ndarray
        sorted indices (into original position array) of start and end points for each section
    section_ids: ndarray
        same shape as pivot indices, encoding which section each pivot is in
    epsilon: float
        relative improvement exit criteria

    Returns
    -------

    """
    sections = np.unique(section_ids)
    n_sections = len(sections)
    section_order = sections
    reversals = np.zeros_like(sections, dtype=bool)
    improvement = 1
    best_distance = calculate_length_with_reversal(section_order, reversals, pivot_positions)
    while improvement > epsilon:
        last_distance = best_distance
        for i in range(1, n_sections - 1):  # already know section 0 is first, and it is forward
            for k in range(i + 1, n_sections):
                new_section_order = two_opt_swap(section_order, i, k)
                for rev_i in range(i, n_sections):
                    new_reversals = reversal_swap(reversals, rev_i)
                    new_distance = calculate_length_with_reversal(new_section_order, new_reversals, pivot_positions)
                    if new_distance < best_distance:
                        section_order = new_section_order
                        reversals = new_reversals
                        best_distance = new_distance
        improvement = (last_distance - best_distance) / last_distance
    return section_order, reversals

def calc_dist(p0, p1):
    # todo - sqrt is monotonic, so can we skip it?
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def calculate_length_with_reversal(order, reversals, positions):
    """

    Parameters
    ----------
    order: ndarray
        section order
    reversals: ndarray
        array of bool, forward (False) and reverse (True)
    positions: ndarray
        pivot positions, sorted such that [::2] yields the positions of one end of each section.

    Returns
    -------
    length: float
        cumulative length _between_ sections

    """
    length = 0
    porder = np.repeat(2 * order, 2) + np.concatenate([[1, 0] if rev else [0, 1] for rev in reversals])
    ordered_positions = positions[porder]
    # just do the links, need to offset by 1. Note len always even since we add sections with pairs of points
    for ind in range(1, len(positions) - 1, 2):
        length += calc_dist(ordered_positions[ind, :], ordered_positions[ind + 1, :])
    return length

def reversal_swap(reversals, ind):
    """
    Flag a section as being reversed, returning a copy.
    Parameters
    ----------
    reversals: ndarray
        array of boolean flags denoting whether a segment is reversed (True) or not
    ind: int
        index to swap

    Returns
    -------
    new_reversals: ndarray
        copy of input reversals but with ind "not"-ed

    """
    new_reversals = np.copy(reversals)
    new_reversals[ind] = ~new_reversals[ind]
    return new_reversals

@register_module('TravelingSalesperson')
class TravelingSalesperson(ModuleBase):
    """

    Optimize route visiting each position in an input dataset exactly once, starting from the last point in the input.
    2-opt algorithm is used.

    Parameters
    ----------
    input : Input
        PYME.IO.tabular containing x and y coordinates. Compatible with measurement output for Supertile coordinates,
        e.g. 'x_um'
    epsilon: Float
        Relative improvement threshold used to stop algorithm when gains become negligible
    output: Output
        PYME.IO.tabular


    """
    input = Input('input')
    epsilon = Float(0.001)
    output = Output('sorted')

    def execute(self, namespace):
        from scipy.spatial import distance_matrix

        points = namespace[self.input]

        try:
            positions = np.stack([points['x_um'], points['y_um']], axis=1)
        except KeyError:
            # units don't matter for these calculations, but we want to preserve them on the other side
            positions = np.stack([points['x'], points['y']], axis=1) / 1e3

        distances = distance_matrix(positions, positions)


        route, best_distance, og_distance = two_opt(distances, self.epsilon)

        # plot_path(positions, route)
        out = tabular.mappingFilter({'x_um': positions[:, 0][route],
                                     'y_um': positions[:, 1][route]})
        out.mdh = MetaDataHandler.NestedClassMDHandler()
        try:
            out.mdh.copyEntriesFrom(points.mdh)
        except AttributeError:
            pass
        out.mdh['TravelingSalesperson.Distance'] = best_distance
        out.mdh['TravelingSalesperson.OriginalDistance'] = og_distance

        namespace[self.output] = out

def plot_path(positions, route):
    import matplotlib.pyplot as plt
    ordered = positions[route]
    plt.scatter(positions[:, 0], positions[:, 1])
    plt.plot(ordered[:, 0], ordered[:, 1])
    plt.show()

def calculate_path_length(distances, route):
    return distances[route[:-1], route[1:]].sum()

def two_opt_swap(route, i, k):
    """
    Take everything the same up to i, then reverse i:k, then take k: normally.
    Parameters
    ----------
    route: ndarray
        Path to swap postions in
    i: int
        first swap index
    k: int
        second swap index
    Returns
    -------
    two-opt swapped route

    Notes
    -----
    Returns a copy. Temping to do something in place, e.g. route[i:k + 1] = route[k:i - 1: -1], but the algorithm
    seems to require a copy somewhere anyway, so might as well do it here.

    """
    return np.concatenate([route[:i],  # [start up to first swap position)
                           route[k:i - 1: -1],  # [from first swap to second], reversed
                           route[k + 1:]])  # (everything else]



def two_opt(distances, epsilon, initial_route=None):
    """

    Parameters
    ----------
    distances: ndarray
        distance array, which distances[i, j] is the distance from the ith to the jth point
    epsilon: float
        exit tolerence on relative improvement. 0.01 corresponds to 1%
    initial_route: ndarray
        [optional] route to initialize search with. Note that the first position in the route is fixed, but all others
        may vary. If no route is provided, the initial route is the same order the distances array was constructed with.

    Returns
    -------
    route: ndarray
        "solved" route
    best_distance: float
        distance of the route
    og_distance: float
        distance of the initial route.

    Notes
    -----
    see https://en.wikipedia.org/wiki/2-opt for pseudo code

    """
    # start route backwards. Starting point will be fixed, and we want LIFO for fast microscope acquisition
    route = initial_route if initial_route is not None else np.arange(distances.shape[0] - 1, -1, -1)

    og_distance = calculate_path_length(distances, route)
    # initialize values we'll be updating
    improvement = 1
    best_distance = og_distance
    while improvement > epsilon:
        last_distance = best_distance
        for i in range(1, distances.shape[0] - 2):  # don't swap the first position
            for k in range(i + 1, distances.shape[0]):  # allow the last position in the route to vary
                new_route = two_opt_swap(route, i, k)
                new_distance = calculate_path_length(distances, new_route)

                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
        improvement = (last_distance - best_distance) / last_distance

    return route, best_distance, og_distance

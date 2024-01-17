# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Mon May 25 17:10:02 2015

@author: david
"""
from .base import ModuleBase, register_module, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrAny, observe
import numpy as np
#import pandas as pd
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
    
    def run(self, inputImage):
        from PYME.localization.FitFactories import GaussMultifitSR
        import pandas as pd
        
       # mdh = MetaDataHandler.NestedClassMDHandler(inputImage.mdh)
        #mdh['Analysis.PSFSigma'] = self.blobSigma
        
        res = []
        
        for i in range(inputImage.data.shape[2]):
            md = MetaDataHandler.NestedClassMDHandler(mdh)
            md['tIndex'] = i
            ff = GaussMultifitSR.FitFactory(self.scale*inputImage.data[:,:,i], md, noiseSigma=np.ones_like(inputImage.data[:,:,i].squeeze()))
        
            res.append(tabular.FitResultsSource(ff.FindAndFit(self.threshold)))
            
        #FIXME - this shouldn't be a DataFrame
        res = pd.DataFrame(np.vstack(res))
        #res.mdh = mdh
        
        return res

@register_module('FitDumbells') 
class FitDumbells(ModuleBase):
    inputImage = Input('input')
    inputPositions = Input('objPostiions')
    outputName = Output('fitResults')
    
    def run(self, inputImage, inputPositions):
        from PYME.localization.FitFactories import DumbellFitR
        from PYME.IO import MetaDataHandler
        
        md = MetaDataHandler.NestedClassMDHandler()
        #set metadata entries needed for fitting to suitable defaults
        md['Camera.ADOffset'] = inputImage.data[:,:,0].min()
        md['Camera.TrueEMGain'] = 1.0
        md['Camera.ElectronsPerCount'] = 1.0
        md['Camera.ReadNoise'] = 1.0
        md['Camera.NoiseFactor'] = 1.0
        
        #copy across the entries from the real image, replacing the defaults
        #if necessary
        md.copyEntriesFrom(inputImage.mdh)
    
        r = np.zeros(len(inputPositions['x']), dtype=DumbellFitR.FitResultsDType)
        
        ff_t = -1
        
        ps = inputImage.pixelSize
        
        for x, y, t, i in zip(inputPositions['x'], inputPositions['y'], inputPositions['t'], range(len(inputPositions['x']))):
            if not t == ff_t:
                md['tIndex'] = t
                ff = DumbellFitR.FitFactory(inputImage.data[:,:,t], md)
                ff_t = t
            
            r[i] = ff.FromPoint(x/ps, y/ps)
            
        
        res = tabular.FitResultsSource(r)
        res.mdh = md
        
        return res

@register_module('DetectPoints2D')
class DetectPoints2D(ModuleBase):
    """
    Detect point-like objects using a sombrero (or Difference of Gaussians) filter

    Parameters
    ----------
    input_name : Input
        PYME.IO.ImageStack
    snr_threshold : Bool
        How should we interpret the threshold? If True, the signal-to-noise (SNR) is estimated at each pixel, and the threshold 
        applied at each pixel is this estimate multiplied by the 'threshold' parameter. If False, the threshold parameter is used
        directly.
        NB - this cannot be set in conjunction with multithreshold.
    multithreshold : Bool
        If true, detect objects at multiple decreasing thresholds, removing detected objects before progressing to the next step.
        This is similar to the CLEAN algorithm from astromy and useful for detecting weak point-like objects
        in the immediate vicinity of strong ones. 
        NB - this cannot be set together with snr_threshold. 
    threshold : Float
        The intensity threshold applied during detection if 'snr_threshold' is False, otherwise this scalar is first
        multiplied by the SNR estimate at each pixel before the threshold is applied. When multithreshold is set, the
        highest threshold is max_intensity/2, the lowest is threshold*max_intensity.
    filter_radius_lowpass : Float
        The std. deviation (in pixels) of the lowpass part of the sombrero. Should match the expected object size.
    filter_radius_highpass : Float
        The std deviation (in pixels) of the highpass part of the sombrero. Ideally > 2x the object size, but less than the
        spacing between objects.
    blur_radius : Float
        Used with multithrehold. Std deviation in pixels of a Gaussian filter used to blur the segmented image with at each threshold step
        prior to subtraction. Should be set so the Gaussian width approximates the PSF width. Most critical if you want to
        reliably detect points on the slope of a brighter point, as making this too large will lead to a "halo of exclusion" around bright points.
        Conversely, setting it to small will lead to spurious detections due to incomplete removal of the bright point.
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
    muiltithreshold = Bool(False)
    edge_mask_width = Int(5)
    filter_radius_lowpass = Float(1.0)
    filter_radius_highpass = Float(3.0)
    blur_radius = Float(1.5)

    output_name = Output('candidate_points')

    def run(self, input_name):
        from PYME.localization.ofind import ObjectIdentifier
        from PYME.localization.remFitBuf import fitTask

        if self.snr_threshold and self.muiltithreshold:
            raise RuntimeError('Cannot specify both snr_threshold and multithreshold simultaeneously')

        x, y, t = [], [], []
        # note that ObjectIdentifier is only 2D-aware
        for ti in range(input_name.data.shape[2]):
            frame = input_name.data.getSlice(ti)
            finder = ObjectIdentifier(frame * (frame > 0), filterRadiusLowpass=self.filter_radius_lowpass, filterRadiusHighpass=self.filter_radius_highpass)

            if self.snr_threshold:  # calculate a per-pixel threshold based on an estimate of the SNR
                sigma = fitTask.calcSigma(input_name.mdh, frame).squeeze()
                threshold = sigma * self.threshold
            else:
                threshold = self.threshold

            if self.muiltithreshold:
                thresholdSteps = "default"
            else:
                thresholdSteps = 0

            finder.FindObjects(threshold, numThresholdSteps=thresholdSteps, debounceRadius=self.debounce_radius, maskEdgeWidth=self.edge_mask_width)

            x.append(finder.x[:])
            y.append(finder.y[:])
            t.append(ti * np.ones_like(finder.x[:]))

        return tabular.DictSource({'x': np.concatenate(x, axis=0), 'y': np.concatenate(y, axis=0),
                                     't': np.concatenate(t, axis=0)})



@register_module('FitPoints')
class FitPoints(ModuleBase):
    """ 
    Apply one of the fit modules from PYME.localization.FitFactories to each of
    the points in the provided in inputPositions. 

    Parameters
    ----------
    inputImage: PYME.IO.image.ImageStack
        FitPoints does not do the camera correction normally done during 
        localization analysis in PYME. To accomplish this using recipe modules,
        run your ImageStack through `processing.FlatfieldAndDarkCorrect` first.
    inputPositions: PYME.IO.tabular
        positions to fit in units of nanometers. If inputImage is missing voxelsize
        metadata, a pixel size of 1 nm is assumed.
    outputName: PYME.IO.tabular
        see selected fit module datatype for fit result and fit error parameters

    """
    inputImage = Input('input')
    inputPositions = Input('objPositions')
    outputName = Output('fitResults')
    fitModule = CStr('LatGaussFitFR')
    roiHalfSize = Int(7)
    channel = Int(0)
    parameters = DictStrAny() #fit parameters ('Analysis' metadata entries)

    @observe('fitModule')
    def _on_fit_module_change(self, event=None):
        # populate parameters for the given fit module
        from PYME.localization.FitFactories import import_fit_factory
        fitMod = import_fit_factory(self.fitModule)

        try:
            params = fitMod.PARAMETERS

            # add default value for any parameters which are absent
            for p in params:
                if not p.paramName in self.parameters:
                    self.parameters[p.paramName] = p.default

            # remove any parameters which don't belong to the selected fit factory
            pnames = [p.paramName for p in params]
            for k in self.parameters.keys():
                if not k in pnames:
                    self.parameters.pop(k)

        except AttributeError:
            pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._on_fit_module_change()
    
    def run(self, inputImage, inputPositions):
        from PYME.localization.FitFactories import import_fit_factory
        from PYME.IO import MetaDataHandler

        md = MetaDataHandler.NestedClassMDHandler()
        #set metadata entries needed for fitting to suitable defaults
        md['Camera.ADOffset'] = inputImage.data[:, :, 0].min()
        md['Camera.TrueEMGain'] = 1.0
        md['Camera.ElectronsPerCount'] = 1.0
        md['Camera.ReadNoise'] = 1.0
        md['Camera.NoiseFactor'] = 1.0

        md['voxelsize.x'] = .001
        md['voxelsize.y'] = .001

        #copy across the entries from the real image, replacing the defaults
        #if necessary
        md.copyEntriesFrom(inputImage.mdh)

        #copy fitting parameters into metadata
        md.update(self.parameters)

        fitMod = import_fit_factory(self.fitModule) #import our fitting module

        r = np.zeros(len(inputPositions['x']), dtype=fitMod.FitResultsDType)

        ff_t = -1

        ps = inputImage.pixelSize
        print('pixel size: %s' % ps)

        for x, y, t, i in zip(inputPositions['x'], inputPositions['y'], inputPositions['t'].astype(int), range(len(inputPositions['x']))):
            if not t == ff_t:
                md['tIndex'] = t
                ff = fitMod.FitFactory(np.atleast_3d(inputImage.data[:, :, t, self.channel]), md)
                ff_t = t

            #print x/ps, y/ps
            r[i] = ff.FromPoint(x/ps, y/ps, roiHalfSize=self.roiHalfSize)

        res = tabular.FitResultsSource(r, sort=False)
        res.mdh = md

        return res

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
        roi = data[int(x - radius):int(x + radius + 1), int(y - radius):int(y + radius + 1), int(t)].squeeze()
        mask = self._get_mask(radius)

        print(mask.shape, roi.shape)#, (roi * mask).shape

        return (roi.squeeze() * mask).sum()

    def run(self, inputImage, inputPositions):
        #from PYME.localization.FitFactories import DumbellFitR
        from PYME.IO import MetaDataHandler

        md = MetaDataHandler.NestedClassMDHandler()
        #set metadata entries needed for fitting to suitable defaults
        md['Camera.ADOffset'] = inputImage.data[:, :, 0].min()
        md['Camera.TrueEMGain'] = 1.0
        md['Camera.ElectronsPerCount'] = 1.0
        md['Camera.ReadNoise'] = 1.0
        md['Camera.NoiseFactor'] = 1.0

        #copy across the entries from the real image, replacing the defaults
        #if necessary
        md.copyEntriesFrom(inputImage.mdh)

        res = np.zeros(len(inputPositions['x']), dtype=[('r%d' % r, 'f4') for r in self.radii])
        ff_t = -1

        aggFunc = getattr(self, '_get_%s' % self.mode)

        ps = inputImage.pixelSize
        print('pixel size: %s' % ps)
        for x, y, t, i in zip(inputPositions['x'], inputPositions['y'], inputPositions['t'], range(len(inputPositions['x']))):
            for r in self.radii:
                res[i]['r%d' % r] = aggFunc(inputImage.data, np.round(x / ps), np.round(y / ps), t, r)

        res = tabular.RecArraySource(res)
        res.mdh = md

        return res


class ExtractROIs(ModuleBase):
    """
    Extract ROIs around input positions and return them as an ImageStack. Points
    too close to the edge of the input image will be filtered. The ROI-relative
    positions are also determined (distance from the upper left hand corner of
    each ROI to the point about which it was cropped). 

    Parameters
    ----------
    input_series : PYME.IO.image.ImageStack
        series to crop ROIs from
    input_positions : PYME.IO.tabular.TabularBase
        positions to extract
    roi_half_size : int
        size of ROIs to extract, as a half size, so final ROI size will be
        `2 * roi_half_size + 1`.
    output_relative_positions : PYME.IO.tabular.TabularBase
        distances [nm] from ROI upper left hand corner (x, y = 0, 0) to 
        corresponding extraction position.
    output_rois : PYME.IO.image.ImageStack
        extracted ROIs as an image stack. Dimension 2 (typically z/t)
        corresponds to position index.

    Notes
    -----
    - Positions too close to the frame edge, which will result in ROIs cropped
    being smaller than specified, will be removed and not included in the module
    outputs.
    - Multiview positions are not supported, and must be 'unmapped' to the
    original camera positions first.
    """
    input_series = Input('series')
    input_positions = Input('positions')

    roi_half_size = Int(7)
    
    output_relative_positions = Output('relative_positions')
    output_rois = Output('roi_stack')

    def run(self, input_series, input_positions):
        from PYME.IO.image import ImageStack

        ox_nm, oy_nm, oz = input_series.origin
        vs_nm = np.array([input_series.voxelsize_nm.x, input_series.voxelsize_nm.y])
        roi_half_nm = (self.roi_half_size + 1) * vs_nm
        x_max, y_max = (input_series.data.shape[:2] * vs_nm) - roi_half_nm
        
        edge_filtered = tabular.ResultsFilter(input_positions, 
                                              x=[ox_nm + roi_half_nm[0], x_max],
                                              y=[oy_nm + roi_half_nm[1], y_max])
        n_filtered = len(input_positions) - len(edge_filtered)
        if n_filtered > 0:
            logger.error('%d positions too close to edge, filtering.' % n_filtered)

        extracted = FitPoints(fitModule='ROIExtractNR', 
                              roiHalfSize=self.roi_half_size, 
                              channel=0).apply_simple(inputImage=input_series,
                                                      inputPositions=edge_filtered)
        
        # get roi edge position. FFBase._get_roi rounds before extracting
        relative_positions = edge_filtered.to_recarray()
        x_edge = np.round(edge_filtered['x'] / input_series.voxelsize_nm.x) - self.roi_half_size
        y_edge = np.round(edge_filtered['y'] / input_series.voxelsize_nm.y) - self.roi_half_size
        # subtract off ofset to ROI edge 
        relative_positions['x'] = edge_filtered['x'] - (x_edge * input_series.voxelsize_nm.x)
        relative_positions['y'] = edge_filtered['y'] - (y_edge * input_series.voxelsize_nm.y)
        relative_positions['fitResults_x0'] = relative_positions['x']
        relative_positions['fitResults_y0'] = relative_positions['y']
        relative_positions = tabular.RecArraySource(relative_positions)
        relative_positions.mdh = MetaDataHandler.NestedClassMDHandler(input_positions.mdh)
        relative_positions.mdh['ExtractROIs.RoiHalfSize'] = self.roi_half_size


        return {'output_rois' : ImageStack(data=np.moveaxis(extracted['data'], 0, 2), 
                                                 mdh=input_series.mdh),
                'output_relative_positions' : relative_positions}


@register_module('MeanNeighbourDistances') 
class MeanNeighbourDistances(ModuleBase):
    """Calculates mean distance to nearest neighbour in a triangulation of the
    supplied points"""
    inputPositions = Input('input')
    outputName = Output('neighbourDists')
    key = CStr('neighbourDists')
    
    def run(self, inputPositions):
        from matplotlib import delaunay
        from PYME.LMVis import visHelpers
        
        x, y = inputPositions['x'], inputPositions['y']
        #triangulate the data
        T = delaunay.Triangulation(x + .1*np.random.normal(size=len(x)), y + .1*np.random.normal(size=len(x)))
        #find the average edge lengths leading away from a given point
        res = np.array(visHelpers.calcNeighbourDists(T))
        
        return tabular.DictSource({self.key:res})

@register_module('NearestNeighbourDistances')
class NearestNeighbourDistances(ModuleBase):
    """Calculates the nearest neighbour distances between supplied points using
    a kdtree"""
    inputChan0 = Input('input')
    inputChan1 = Input('')
    outputName = Output('neighbourDists')
    columns = List(['x', 'y'])
    key = CStr('neighbourDists')

    def run(self, inputChan0, inputChan1 = 0):
        from scipy.spatial import cKDTree
        
        if not inputChan1:
            inputChan1 = inputChan0
            singleChan = True  # flag to not pair molecules with themselves
        else:
            singleChan = False

        #create a kdtree
        p1 = np.vstack([inputChan0[k] for k in self.columns]).T
        p2 = np.vstack([inputChan1[k] for k in self.columns]).T
        kdt = cKDTree(p1)

        if singleChan:
            #query the two closest entries - the closest entry will be the orig point paired with itself, so ignore it
            d, i = kdt.query(p2, 2)
            d = d[:, 1]
        else:
            d, i = kdt.query(p2, 1)

        return tabular.DictSource({self.key: d})

@register_module('PairwiseDistanceHistogram')
class PairwiseDistanceHistogram(ModuleBase):
    """Calculates a histogram of pairwise distances"""
    inputPositions = Input('input')
    inputPositions2 = Input('')
    outputName = Output('distHist')
    nbins = Int(50)
    binSize = Float(50.)
    threaded = Bool(False)
    
    def run(self, inputPositions, inputPositions2):
        from PYME.Analysis.points import DistHist
        
        pos1 = inputPositions2 if inputPositions2 else inputPositions

        if np.count_nonzero(inputPositions['z']) == 0 and np.count_nonzero(pos1['z']) == 0:
            if self.threaded:
                res = DistHist.distanceHistogramThreaded(inputPositions['x'], inputPositions['y'], pos1['x'], pos1['y'], self.nbins, self.binSize)
            else:
                res = DistHist.distanceHistogram(inputPositions['x'], inputPositions['y'], pos1['x'], pos1['y'], self.nbins, self.binSize)
        else:
            if self.threaded:
                res = DistHist.distanceHistogram3DThreaded(inputPositions['x'], inputPositions['y'], inputPositions['z'],
                                                           pos1['x'], pos1['y'], pos1['z'], self.nbins, self.binSize)
            else:
                res = DistHist.distanceHistogram3D(inputPositions['x'], inputPositions['y'], inputPositions['z'],
                                                   pos1['x'], pos1['y'], pos1['z'], self.nbins, self.binSize)

        d = self.binSize*np.arange(self.nbins)

        return tabular.DictSource({'bins': d, 'counts': res})

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
    
    def run(self, inputMeasurements):        
        v = inputMeasurements[self.key]
        
        edges = np.linspace(self.left, self.right, self.nbins)
        res = np.histogram(v, edges, normed=self.normalize)[0]
        
        return tabular.DictSource({'bins' : 0.5*(edges[:-1] + edges[1:]), 'counts' : res})
    
        
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
    
    def run(self, inputImage, inputMask=None):
        vals = inputImage.data[:,:,:].ravel()
        
        if inputMask:
            m = inputMask.data[:,:,:].ravel() >0
            vals = vals[m]
        
        edges = np.linspace(self.left, self.right, self.nbins)
        res = np.histogram(vals, edges, normed=self.normalize)[0]
        
        return tabular.DictSource({'bins' : 0.5*(edges[:-1] + edges[1:]), 'counts' : res})
        
@register_module('ImageCumulativeHistogram')         
class ImageCumulativeHistogram(ModuleBase):
    """Calculates a histogram of a given measurement key"""
    inputImage = Input('input')
    outputName = Output('hist')
    inputMask = Input('')
    #nbins = Int(50)
    #left = Float(0.)
    #right = Float(1000)
    
    def run(self, inputImage, inputMask=None):
        vals = inputImage.data[:,:,:].ravel()
        
        if inputMask:
            m = inputMask.data[:,:,:].ravel() > 0 
            vals = vals[m]
        
        yvals = np.linspace(0, 1.0, len(vals))
        xvals = np.sort(vals)
        
        return tabular.DictSource({'bins' : xvals, 'counts' : yvals})

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

    def run(self, inputImage, binBy, inputMask=None):
        from PYME.Analysis import binAvg
        vals = inputImage.data[:, :, :].ravel()
        binby = binBy.data[:,:,:].ravel()

        if inputMask:
            m = inputMask.data[:, :, :].ravel() > 0

            vals = vals[m]
            binby = binby[m]

        #mask out NaNs
        m2 = ~np.isnan(vals)
        vals = vals[m2]
        binby = binby[m2]

        edges = np.linspace(self.left, self.right, self.nbins)

        bn, bm, bs = binAvg.binAvg(binby, vals, edges)

        return tabular.DictSource({'bins': 0.5*(edges[:-1] + edges[1:]), 'counts': bn, 'means' : bm})
        

        
# there are some measurements we don't want / are not particlarly useful (i.e. the ones which just return the image)
# some of these changed in recent versions of skimage (2022), so try both new and old versions
_MEASURE2D_KEYS_TO_IGNORE = ['convex_image',
'filled_image',
'image_convex',
'image_filled',
'image',
'intensity_image',
'image_intensity']

# euler_number calculation can be buggy and cause crashes
_MEASURE2D_KEYS_TO_IGNORE.append('euler_number')   

@register_module('Measure2D') 
class Measure2D(ModuleBase):
    """Perform 2D morphological measurements based on an image mask and optional intensity image. 

    **Note:** To perform mesurements on multi-colour images, split channels first and measure each channel
    separately. 
    
    **Note:** Measure2D currently works on *flattened* data where z and t dimensions have been collapsed
    to a single "frame" dimension. TODO - fix this (or at least record the corresponding z and t indices for
    each measurement).
    """
    inputLabels = Input('labels')
    inputIntensity = Input('data')
    outputName = Output('measurements')
    
    measureContour = Bool(True)    
        
    def run(self, inputLabels, inputIntensity):       
        
        #define the measurement class, which behaves like an input filter        
        class measurements(tabular.TabularBase):
            _name = 'Measue 2D source'
            ps = inputLabels.pixelSize
            
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
                    self._keys = ['t', 'x', 'y'] + [r for r in dir(measurements[0]) if not (r.startswith('_') or r in _MEASURE2D_KEYS_TO_IGNORE)]
                    
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
                    raise KeyError("'%s' key not defined" % key)
                
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
        
        assert (inputLabels.data_xyztc.shape[4 ==1]), 'Measure2D labels must have a single colour channel'

        m = measurements()
        
        if inputIntensity:
            assert (inputLabels.data_xyztc.shape == inputIntensity.data_xyztc.shape), 'Measure2D labels and intensity must be the same shape'
            
        for i in xrange(inputLabels.data.shape[2]):
            m.addFrameMeasures(i, *self._measureFrame(i, inputLabels, inputIntensity))
            
        #m.mdh = inputLabels.mdh
        
        return m      
        
    def _measureFrame(self, frameNo, labels, intensity):
        import skimage.measure
        
        li = labels.data_xytc[:,:,frameNo, 0].squeeze().astype('i')

        if intensity:
            it = intensity.data_xytc[:,:,frameNo, 0].squeeze()
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

        # import pylab
        import matplotlib.pyplot as plt
        
        plt.figure()
        for meas in ms:
            if self.type == 'bar':
                xv = meas[self.xkey]
                plt.bar(xv, meas[self.ykey], align='center', width=(xv[1] - xv[0]))
            else:
                plt.plot(meas[self.xkey], meas[self.ykey])
        
        plt.grid()
        plt.legend(labs)
        plt.xlabel(self.xkey)
        plt.ylabel(self.ykey)
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
    string_length = Int(128, desc='Ammount of storage (characters) to allocate for string values')
    
    def run(self, inputMeasurements, inputImage):
        from PYME.IO import MetaDataHandler
        res = {}
        res.update(inputMeasurements)

        nEntries = len(list(res.values())[0])
        for k, mdk in zip(self.keys.split(), self.metadataKeys.split()):
            if mdk == 'seriesName':
                #import os
                v = os.path.split(inputImage.seriesName)[1]
            else:
                v = inputImage.mdh[mdk]

            if isinstance(v, str):
                # use bytes/cstring dtype (rather than U) so that 
                # pytables doesn't bork on us
                res[k] = np.array([v]*nEntries, dtype='S%d'%self.string_length)
            else:
                res[k] = np.array([v]*nEntries)
        
        #res = pd.DataFrame(res)
        res = tabular.DictSource(res)
        #if 'mdh' in dir(meas):
        res.mdh = MetaDataHandler.DictMDHandler(inputImage.mdh)
        
        return res
        
        
@register_module('IdentifyOverlappingROIs')
class IdentifyOverlappingROIs(ModuleBase):
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

    def run(self, input):
        from scipy.spatial import KDTree

        try:
            positions = np.stack([input['x_um'], input['y_um']], axis=1)
            far_flung = np.sqrt(2) * input.mdh['Pyramid.PixelSize'] * self.roi_size_pixels  # [micrometers]
        except KeyError:
            positions = np.stack([input['x'], input['y']], axis=1) / 1e3  # assume x and y were in [nanometers]
            far_flung = np.sqrt(2) * input.mdh['voxelsize.x'] * self.roi_size_pixels  # [micrometers]

        tree = KDTree(positions)


        neighbors = tree.query_ball_tree(tree, r=far_flung, p=2)

        tossing = set()
        for ind, close in enumerate(neighbors):
            if ind not in tossing and len(close) > 1:  # ignore points we've already decided to reject
                # reject points too close to our current indexed point
                # TODO - don't reject points inside of this circular radius if their square ROIs don't actually overlap
                close.remove(ind)
                tossing.update(close)

        out = tabular.MappingFilter(input)
        reject = np.zeros(tree.n, dtype=int)
        reject[list(tossing)] = 1
        out.addColumn(self.reject_key, reject)

        return out


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
    n_processes: Int
        Number of processes to split the chunk tasks among. Setting of zero defaults to using one process per CPU core.
    output: Output
        PYME.IO.tabular

    """
    input = Input('input')
    epsilon = Float(0.001)
    points_per_chunk = Int(500)
    n_processes = Int(0)
    output = Output('sorted')

    def execute(self, namespace):
        from PYME.Analysis.points.traveling_salesperson import sectioned_two_opt

        points = namespace[self.input]

        try:
            positions = np.stack([points['x_um'], points['y_um']], axis=1)
        except KeyError:
            positions = np.stack([points['x'], points['y']], axis=1) / 1e3

        final_route = sectioned_two_opt.tsp_chunk_two_opt_multiproc(positions, self.epsilon, self.points_per_chunk,
                                                                        self.n_processes)

        # note that we sorted the positions / sections once before, need to propagate that through before sorting
        out = tabular.DictSource({k: points[k][final_route] for k in points.keys()})
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




@register_module('TravelingSalesperson')
class TravelingSalesperson(ModuleBase):
    """

    Optimize route visiting each position in an input dataset exactly once, starting from the 0th point or
    the minimum x + y corner. A greedy sort is done first because it is quite fast and should for average case reduce
    number of moves from O(nlog(n)) to O(n).

    Parameters
    ----------
    input : Input
        PYME.IO.tabular containing x and y coordinates. Compatible with measurement output for Supertile coordinates,
        e.g. 'x_um'
    epsilon: Float
        Relative improvement threshold used to stop algorithm when gains become negligible
    start_from_corner: Bool
        Flag to toggle starting from the min(x + y) point [True] or the 0th point in the input positions.
    output: Output
        PYME.IO.tabular


    """
    input = Input('input')
    epsilon = Float(0.001)
    start_from_corner = Bool(True)
    output = Output('sorted')

    def execute(self, namespace):
        from PYME.Analysis.points.traveling_salesperson import sort

        points = namespace[self.input]

        try:
            positions = np.stack([points['x_um'], points['y_um']], axis=1)
        except KeyError:
            # units don't matter for these calculations, but we want to preserve them on the other side
            positions = np.stack([points['x'], points['y']], axis=1) / 1e3

        start_index = 0 if not self.start_from_corner else np.argmin(positions.sum(axis=1))

        positions, ogd, final_distance = sort.tsp_sort(positions, start_index, self.epsilon, return_path_length=True)

        out = tabular.DictSource({'x_um': positions[:, 0],
                                     'y_um': positions[:, 1]})
        out.mdh = MetaDataHandler.NestedClassMDHandler()
        try:
            out.mdh.copyEntriesFrom(points.mdh)
        except AttributeError:
            pass
        out.mdh['TravelingSalesperson.Distance'] = final_distance
        out.mdh['TravelingSalesperson.OriginalDistance'] = ogd

        namespace[self.output] = out


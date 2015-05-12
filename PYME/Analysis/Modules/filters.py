# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:11:05 2015

@author: david
"""
try:
    from enthought.traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance
    from enthought.traits.ui.api import View, Item, EnumEditor, InstanceEditor
except ImportError:
    from traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance
    from traitsui.api import View, Item, EnumEditor, InstanceEditor

from PYME.DSView.image import ImageStack
from scipy import ndimage
import numpy as np
import os
       

class ModuleCollection(HasTraits):
    modules = List()
    
    def __init__(self, *args, **kwargs):
        HasTraits.__init__(self, *args, **kwargs)
        
        self.namespace = {}
        
    def dependancyGraph(self):
        dg = {}
        
        for mod in self.modules:
            #print mod
            s = mod.inputs
            try:
                s.update(dg[mod])
            except KeyError:
                pass
            
            dg[mod] = s
            
            for op in mod.outputs:
                dg[op] = {mod,}
                
        return dg
        
    def resolveDependencies(self):
        import toposort
        #build dependancy graph
                    
        dg = self.dependancyGraph()
        
        #solve the dependency tree        
        return toposort.toposort_flatten(dg)
        
    def execute(self, **kwargs):
        self.namespace.update(kwargs)
        
        exec_order = self.resolveDependencies()

        for m in exec_order:
            if isinstance(m, ModuleBase):
                m.execute(self.namespace)
        
        if 'output' in self.namespace.keys():
            return self.namespace['output']
            
    @classmethod
    def fromMD(cls, md):
        moduleNames = set([s.split('.')[0] for s in md.keys()])
        
        mc = []
        
        for mn in moduleNames:
            mod = eval(mn)()
            mod.set(**md[mn])
            mc.append(mod)
            
        return cls(modules=mc)
        
    def toYAML(self):
        import yaml
        l = []
        for mod in self.modules:
            l.append({mod.__class__.__name__: mod.get()})
            
        return yaml.dump(l, default_flow_style=False)
    
    @classmethod
    def fromYAML(cls, data):
        import yaml
        
        l = yaml.load(data)
        
        mc = []
        
        if l == None:
            l = []
        
        for mdd in l:
            mn, md = mdd.items()[0]
            mod = eval(mn)()
            mod.set(**md)
            mc.append(mod)
            
        return cls(modules=mc)
        
    @property
    def inputs(self):
        ip = set()
        for mod in self.modules:
            ip.update({k for k in mod.inputs if k.startswith('in')})
        return ip
        
    @property
    def outputs(self):
        op = set()
        for mod in self.modules:
            op.update({k for k in mod.outputs if k.startswith('out')})
        return op
        
            
    

class ModuleBase(HasTraits):
    def execute(namespace):
        '''prototype function - should be over-ridden in derived classes
        
        takes a namespace (a dictionary like object) from which it reads its inputs and 
        into which it writes outputs
        '''
        pass
    
    @property
    def inputs(self):
        return {v for k,v in self.get().items() if k.startswith('input')}
        
    @property
    def outputs(self):
        return {v for k,v in self.get().items() if k.startswith('output')}
    
class ExtractChannel(ModuleBase):
    '''extract one channel from an image'''
    inputName = CStr('input')
    outputName = CStr('filtered_image')     
    
    channelToExtract = Int(0)
    
    def _pickChannel(self, image):
        chan = image.data[:,:,:,self.channelToExtract]
        
        im = ImageStack(chan, titleStub = 'Filtered Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
    
    def execute(self, namespace):
        namespace[self.outputName] = self._pickChannel(namespace[self.inputName])
    
    
class Filter(ModuleBase):
    '''Module with one image input and one image output'''
    inputName = CStr('input')
    outputName = CStr('filtered_image')
    
    processFramesIndividually = Bool(False)
    
    def filter(self, image):
        if self.processFramesIndividually:
            filt_ims = []
            for chanNum in range(image.data.shape[3]):
                filt_ims.append(np.concatenate([np.atleast_3d(self.applyFilter(image.data[:,:,i,chanNum].squeeze().astype('f'), chanNum, i, image)) for i in range(image.data.shape[2])], 2))
        else:
            filt_ims = [np.atleast_3d(self.applyFilter(image.data[:,:,:,chanNum].squeeze().astype('f'), chanNum, 0, image)) for chanNum in range(image.data.shape[3])]
            
        im = ImageStack(filt_ims, titleStub = self.outputName)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        self.completeMetadata(im)
        
        return im
        
    def execute(self, namespace):
        namespace[self.outputName] = self.filter(namespace[self.inputName])
        
    def completeMetadata(self, im):
        pass  
    
class GaussianFilter(Filter):
    sigmaY = Float(1.0)
    sigmaX = Float(1.0)
    sigmaZ = Float(1.0)
    #def __init__(self, **kwargs):
    #    pass
    @property
    def sigmas(self):
        return [self.sigmaX, self.sigmaY, self.sigmaZ]
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.gaussian_filter(data, self.sigmas[:len(data.shape)])
    
    def completeMetadata(self, im):
        im.mdh['Processing.GaussianFilter'] = self.sigmas
        
class Zoom(Filter):
    zoom = Float(1.0)
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.zoom(data, self.zoom)
    
    def completeMetadata(self, im):
        im.mdh['Processing.Zoom'] = self.zoom
        im.mdh['voxelsize.x'] = im.mdh['voxelsize.x']/self.zoom
        im.mdh['voxelsize.y'] = im.mdh['voxelsize.y']/self.zoom
        
        if not self.processFramesIndividually:
            im.mdh['voxelsize.z'] = im.mdh['voxelsize.z']/self.zoom
            
        
class DoGFilter(Filter):
    '''Difference of Gaussians'''
    sigmaY = Float(1.0)
    sigmaX = Float(1.0)
    sigmaZ = Float(1.0)
    
    sigma2Y = Float(1.0)
    sigma2X = Float(1.0)
    sigma2Z = Float(1.0)

    #def __init__(self, **kwargs):
    #    pass
    @property
    def sigmas(self):
        return [self.sigmaX, self.sigmaY, self.sigmaZ]
        
    @property
    def sigma2s(self):
        return [self.sigma2X, self.sigma2Y, self.sigma2Z]
    
    def applyFilter(self, data, chanNum, frNum, im):
        return ndimage.gaussian_filter(data, self.sigmas[:len(data.shape)]) - ndimage.gaussian_filter(data, self.sigma2s[:len(data.shape)])
    
    def completeMetadata(self, im):
        im.mdh['Processing.GaussianFilter'] = self.sigmas

class SimpleThreshold(Filter):
    threshold = Float(0.5)
    
    def applyFilter(self, data, chanNum, frNum, im):
        mask = data > self.threshold
        return mask

    def completeMetadata(self, im):
        im.mdh['Processing.SimpleThreshold'] = self.threshold
        
class Label(Filter):
    minRegionPixels = Int(10)
    
    def applyFilter(self, data, chanNum, frNum, im):
        mask = data > 0.5
        labs, nlabs = ndimage.label(mask)
        
        rSize = self.minRegionPixels
        
        if rSize > 1:
            m2 = 0*mask
            objs = ndimage.find_objects(labs)
            for i, o in enumerate(objs):
                r = labs[o] == i+1
                #print r.shape
                if r.sum() > rSize:
                    m2[o] = r
                                
            labs, nlabs = ndimage.label(m2)
            
        return labs

    def completeMetadata(self, im):
        im.mdh['Labelling.MinSize'] = self.minRegionPixels
        
class LocalMaxima(Filter):
    threshold = Float(.3)
    minDistance = Int(10)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.feature
        im = data.astype('f')/data.max()
        return skimage.feature.peak_local_max(im, threshold_abs = self.threshold, min_distance = self.minDistance, indices=False)

    def completeMetadata(self, im):
        im.mdh['LocalMaxima.threshold'] = self.threshold
        im.mdh['LocalMaxima.minDistance'] = self.minDistance
        
class Deconvolve(Filter):
    offset = Float(0)
    method = Enum('Richardson-Lucy', 'ICTM') 
    iterations = Int(10)
    psfType = Enum('file', 'bead', 'Lorentzian')
    psfFilename = CStr('') #only used for psfType == 'file'
    lorentzianFWHM = Float(50.) #only used for psfType == 'Lorentzian'
    beadDiameter = Float(200.) #only used for psfType == 'bead'
    regularisationLambda = Float(0.1)
    
    _psfCache = {}
    
    def GetPSF(self, vshint = None):
        psfKey = (self.psfType, self.psfFilename, self.lorentzianFWHM, self.beadDiameter)
        
        if not psfKey in self._psfCache.keys():
            if self.psfType == 'file':
                psf, vs = np.load(self.psfFilename)
                psf = np.atleast_3d(psf)
                
                self._psfCache[psfKey] = (psf, vs)        
            elif (self.psfType == 'Laplace'):
                from scipy import stats
                sc = self.lorentzianFWHM/2.0
                X, Y = np.mgrid[-30.:31., -30.:31.]
                R = np.sqrt(X*X + Y*Y)
                
                if not vshint is None:
                    vx = vshint[0]
                else:
                    vx = sc/2.
                
                vs = type('vs', (object,), dict(x=vx/1e3, y=vx/1e3))
                
                psf = np.atleast_3d(stats.cauchy.pdf(vx*R, scale=sc))
                    
                self._psfCache[psfKey] = (psf/psf.sum(), vs)
            elif (self.psfType == 'bead'):
                from PYME.Deconv import beadGen
                psf = beadGen.genBeadImage(self.beadDiameter/2, vshint)
                
                vs = type('vs', (object,), dict(x=vshint[0]/1e3, y=vshint[1]/1e3))
                
                self._psfCache[psfKey] = (psf/psf.sum(), vs)
                
                
        return self._psfCache[psfKey]
            
    
    def applyFilter(self, data, chanNum, frNum, im):
        d = data.astype('f') - self.offset
        vx, vy, vz = np.array(im.voxelsize)*1e-3
        
        psf, vs = self.GetPSF(im.voxelsize)
        if not (vs.x == vx and vs.y == vy and vs.z ==vz):
            #rescale psf to match data voxel size
            psf = ndimage.zoom(psf, [vs.x/vx, vs.y/vy, vs.z/vz])
            
        
        
        return 

    def completeMetadata(self, im):
        im.mdh['Deconvolution.Offset'] = self.offset
        im.mdh['Deconvolution.Method'] = self.method
        im.mdh['Deconvolution.Iterations'] = self.iterations
        im.mdh['Deconvolution.PsfType'] = self.psfType
        im.mdh['Deconvolution.PSFFilename'] = self.psfFilename
        im.mdh['Deconvolution.LorentzianFWHM'] = self.lorentzianFWHM
        im.mdh['Deconvolution.BeadDiameter'] = self.beadDiameter
        im.mdh['Deconvolution.RegularisationLambda'] = self.regularisationLambda
        
class DistanceTransform(Filter):    
    def applyFilter(self, data, chanNum, frNum, im):
        mask = 1.0*(data > 0.5)
        voxelsize = np.array(im.voxelsize)[:mask.ndim]
        dt = -ndimage.distance_transform_edt(data, sampling=voxelsize)
        dt = dt + ndimage.distance_transform_edt(ndimage.binary_dilation(1-mask), sampling=voxelsize)
        return dt
     
class BinaryDilation(Filter):
    iterations = Int(1)
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.binary_dilation(data, selem)
        
class BinaryErosion(Filter):
    iterations = Int(1)
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.binary_erosion(data, selem)
        
class BinaryFillHoles(Filter):
    iterations = Int(1)
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.binary_fill_holes(data, selem)
        
class Watershed(ModuleBase):
    '''Module with one image input and one image output'''
    inputImage = CStr('input')
    inputMarkers = CStr('markers')
    inputMask = CStr('')
    outputName = CStr('watershed')
    
    processFramesIndividually = Bool(False)
    
    def filter(self, image, markers, mask=None):
        if self.processFramesIndividually:
            filt_ims = []
            for chanNum in range(image.data.shape[3]):
                if not mask == None:
                    filt_ims.append(np.concatenate([np.atleast_3d(self.applyFilter(image.data[:,:,i,chanNum].squeeze(), markers.data[:,:,i,chanNum].squeeze(), mask.data[:,:,i,chanNum].squeeze())) for i in range(image.data.shape[2])], 2))
                else:
                    filt_ims.append(np.concatenate([np.atleast_3d(self.applyFilter(image.data[:,:,i,chanNum].squeeze(), markers.data[:,:,i,chanNum].squeeze())) for i in range(image.data.shape[2])], 2))
        else:
            if not mask == None:
                filt_ims = [np.atleast_3d(self.applyFilter(image.data[:,:,:,chanNum].squeeze(), markers.data[:,:,:,chanNum].squeeze(), mask.data[:,:,:,chanNum].squeeze())) for chanNum in range(image.data.shape[3])]
            else:
                filt_ims = [np.atleast_3d(self.applyFilter(image.data[:,:,:,chanNum].squeeze(), mask.data[:,:,:,chanNum].squeeze())) for chanNum in range(image.data.shape[3])]
            
        im = ImageStack(filt_ims, titleStub = self.outputName)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        #self.completeMetadata(im)
        
        return im
        
    def applyFilter(self, image,markers, mask=None):
        import skimage.morphology

        img = ((image/image.max())*2**15).astype('int16')         
        
        if not mask is None:
            return skimage.morphology.watershed(img, markers.astype('int16'), mask = mask.astype('int16'))
        else:
            return skimage.morphology.watershed(img, markers.astype('int16'))
        
    def execute(self, namespace):
        image = namespace[self.inputImage]
        markers =  namespace[self.inputMarkers]
        if self.inputMask in ['', 'none', 'None']:
            namespace[self.outputName] = self.filter(image, markers)
        else:
            mask = namespace[self.inputMask]
            namespace[self.outputName] = self.filter(image, markers, mask)



from PYME.Analysis.LMVis import inpFilt

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
        
class SelectMeasurementColumns(ModuleBase):
    '''Take just certain columns of a variable'''
    inputMeasurments = CStr('measurements')
    keys = CStr('')
    outputName = CStr('selectedMeasurements') 
    
    def execute(self, namespace):       
        meas = namespace[self.inputMeasurments]
        namespace[self.outputName] = {k:meas[k] for k in self.keys.split()}
        
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
        

def _issubclass(cl, c):
    try:
        return issubclass(cl, c)
    except TypeError:
        return False
 
#d = {}
#d.update(locals())
#moduleList = [c for c in d if _issubclass(c, ModuleBase) and not c == ModuleBase]       
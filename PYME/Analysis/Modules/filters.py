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
                filt_ims.append(np.concatenate([np.atleast_3d(self.applyFilter(image.data[:,:,i,chanNum].squeeze(), chanNum, i, image)) for i in range(image.data.shape[2])], 2))
        else:
            filt_ims = [np.atleast_3d(self.applyFilter(image.data[:,:,:,chanNum].squeeze(), chanNum, 0, image)) for chanNum in range(image.data.shape[3])]
            
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
        
        if not mask == None:
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


class Measure2D(ModuleBase):
    '''Module with one image input and one image output'''
    inputLabels = CStr('labels')
    inputIntensity = CStr('data')
    outputName = CStr('measurements')
    
    measureContour = Bool(True)    
        
    def execute(self, namespace):       
        labels = namespace[self.inputLabels]
        
        #define the measurement class, which behaves like an input filter        
        class measurements(object):
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
                    return np.array(self.contours)
                elif key == 'x':
                    return self.ps*np.array([r.centroid[0] for r in self.measures])
                elif key == 'y':
                    return self.ps*np.array([r.centroid[1] for r in self.measures])
                else:
                    a = np.array([getattr(r, key) for r in self.measures])
                    if a.ndim == 2:
                        return a.T
                    else:
                        return a
        
        # end measuremnt class def
        
        m = measurements()
        
        if self.inputIntensity in ['None', 'none', '']:
            #if we don't have intensity data
            intensity = None
        else:
            intensity = namespace[self.inputIntensity]
            
        for i in xrange(labels.data.shape[2]):
            m.addFrameMeasures(i, *self._measureFrame(i, labels, intensity))
                    
        namespace[self.outputName] = m      
        
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

def _issubclass(cl, c):
    try:
        return issubclass(cl, c)
    except TypeError:
        return False
 
#d = {}
#d.update(locals())
#moduleList = [c for c in d if _issubclass(c, ModuleBase) and not c == ModuleBase]       
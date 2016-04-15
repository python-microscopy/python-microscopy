# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:02:04 2015

@author: david
"""

try:
    from enthought.traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance, on_trait_change
    from enthought.traits.ui.api import View, Item, EnumEditor, InstanceEditor, Group
except ImportError:
    from traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance, on_trait_change
    from traitsui.api import View, Item, EnumEditor, InstanceEditor, Group

from PYME.DSView.image import ImageStack
#from scipy import ndimage
import numpy as np
#import os

all_modules = {}
module_names = {}

def register_module(moduleName):
    def c_decorate(cls):
        all_modules[moduleName] = cls
        module_names[cls] = moduleName
        return cls
        
    return c_decorate
    
    
#def register_module(cls):
#    all_modules[cls.__class__.__name__] = cls
#    return cls


class ModuleCollection(HasTraits):
    modules = List()
    
    def __init__(self, *args, **kwargs):
        HasTraits.__init__(self, *args, **kwargs)
        
        self.namespace = {}
        
    def dependancyGraph(self):
        dg = {}
        
        #only add items to dependancy graph if they are not already in the namespace
        #calculated_objects = namespace.keys()
        
        for mod in self.modules:
            #print mod
            s = mod.inputs
            
            try:
                s.update(dg[mod])
            except KeyError:
                pass
            
            dg[mod] = s
            
            for op in mod.outputs:
                #if not op in calculated_objects:
                dg[op] = {mod,}
                
        return dg
        
    def reverseDependancyGraph(self):
        dg = self.dependancyGraph()
        
        rdg = {}
        
        for k, vs in dg.items():
            for v in vs:                
                vdeps = set()
                try: 
                    vdeps = rdg[v]
                except KeyError:
                    pass
                
                vdeps.add(k)
                rdg[v] = vdeps
                
        return rdg
        
    def _getAllDownstream(self, rdg, keys):
        '''get all the downstream items which depend on the given key'''
        
        downstream = set()
        
        next_level = set()
        
        for k in keys:
            try:        
                next_level.update(rdg[k])
            except KeyError:
                pass
            
        if len(list(next_level)) > 0:
        
            downstream.update(next_level)
            
            downstream.update(self._getAllDownstream(rdg, list(next_level)))
        
        return downstream
        
        
    def pruneDependanciesFromNamespace(self, keys_to_prune):
        rdg = self.reverseDependancyGraph()
        
        downstream = list(keys_to_prune) + list(self._getAllDownstream(rdg, list(keys_to_prune)))
        
        print downstream
        
        for dsi in downstream:
            try:
                self.namespace.pop(dsi)
            except KeyError:
                pass
        
        
    def resolveDependencies(self):
        import toposort
        #build dependancy graph
                    
        dg = self.dependancyGraph()
        
        #solve the dependency tree        
        return toposort.toposort_flatten(dg)
        
    def execute(self, **kwargs):
        #remove anything which is downstream from changed inputs
        print self.namespace.keys()
        for k, v in kwargs.items():
            print k, v
            try:
                if not (self.namespace[k] == v):
                    #input has changed
                    print 'pruning: ', k
                    self.pruneDependanciesFromNamespace([k])
            except KeyError:
                #key wasn't in namespace previously
                print 'KeyError'
                pass
    
        self.namespace.update(kwargs)
        
        exec_order = self.resolveDependencies()

        for m in exec_order:
            if isinstance(m, ModuleBase) and not m.outputs_in_namespace(self.namespace):
                m.execute(self.namespace)
        
        if 'output' in self.namespace.keys():
            return self.namespace['output']
            
    @classmethod
    def fromMD(cls, md):
        c = cls()
        
        moduleNames = set([s.split('.')[0] for s in md.keys()])
        
        mc = []
        
        for mn in moduleNames:
            mod = all_modules[mn]()
            mod.set(**md[mn])
            mc.append(mod)
            
        #return cls(modules=mc)
        c.modules = mc
            
        return c
        
    def toYAML(self):
        import yaml
        l = []
        for mod in self.modules:
            #l.append({mod.__class__.__name__: mod.get()})
            l.append({module_names[mod.__class__]: mod.get()})
            
        return yaml.dump(l, default_flow_style=False)
    
    @classmethod
    def fromYAML(cls, data):
        import yaml
        
        c = cls()
        
        l = yaml.load(data)
        
        mc = []
        
        if l == None:
            l = []
        
        for mdd in l:
            mn, md = mdd.items()[0]
            mod = all_modules[mn](c)
            mod.set(**md)
            mc.append(mod)
            
        c.modules = mc
            
        return c#cls(modules=mc)
        
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
    def __init__(self, parent=None):
        HasTraits.__init__(self)
        
        self.__dict__['_parent'] = parent
        
    @on_trait_change('anytrait')
    def remove_outputs(self):
        if not self._parent == None:
            self._parent.pruneDependanciesFromNamespace(self.outputs)
            
    def outputs_in_namespace(self, namespace):
        keys = namespace.keys()
        return np.all([op in keys for op in self.outputs])
        
    def execute(self, namespace):
        '''prototype function - should be over-ridden in derived classes
        
        takes a namespace (a dictionary like object) from which it reads its inputs and 
        into which it writes outputs
        '''
        pass
    
    @property
    def inputs(self):
        return {v for k,v in self.get().items() if k.startswith('input') and not v == ""}
        
    @property
    def outputs(self):
        return {v for k,v in self.get().items() if k.startswith('output')}
        
class Filter(ModuleBase):
    '''Module with one image input and one image output'''
    inputName = CStr('input')
    outputName = CStr('filtered_image')
    
    processFramesIndividually = Bool(True)
    
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
    
class ArithmaticFilter(ModuleBase):
    '''Module with one image input and one image output'''
    inputName0 = CStr('input')
    inputName1 = CStr('input')
    outputName = CStr('filtered_image')
    
    processFramesIndividually = Bool(False)
    
    def filter(self, image0, image1):
        if self.processFramesIndividually:
            filt_ims = []
            for chanNum in range(image0.data.shape[3]):
                out = []
                for i in range(image0.data.shape[2]):
                    d0 = image0.data[:,:,i,chanNum].squeeze().astype('f')
                    d1 = image1.data[:,:,i,chanNum].squeeze().astype('f')
                    out.append(np.atleast_3d(self.applyFilter(d0, d1, chanNum, i, image0)))
                filt_ims.append(np.concatenate(out, 2))
        else:
            filt_ims = []
            for chanNum in range(image0.data.shape[3]):
                d0 = image0.data[:,:,:,chanNum].squeeze().astype('f')
                d1 = image1.data[:,:,:,chanNum].squeeze().astype('f')
                filt_ims.append(np.atleast_3d(self.applyFilter(d0, d1, chanNum, 0, image0)))
            
        im = ImageStack(filt_ims, titleStub = self.outputName)
        im.mdh.copyEntriesFrom(image0.mdh)
        im.mdh['Parents'] = '%s, %s' % (image0.filename, image1.filename)
        
        self.completeMetadata(im)
        
        return im
        
    def execute(self, namespace):
        namespace[self.outputName] = self.filter(namespace[self.inputName0], namespace[self.inputName1])
        
    def completeMetadata(self, im):
        pass  

@register_module('ExtractChannel')    
class ExtractChannel(ModuleBase):
    '''extract one channel from an image'''
    inputName = CStr('input')
    outputName = CStr('filtered_image')     
    
    channelToExtract = Int(0)
    
    def _pickChannel(self, image):
        chan = image.data[:,:,:,self.channelToExtract]
        
        im = ImageStack(chan, titleStub = 'Filtered Image')
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        return im
    
    def execute(self, namespace):
        namespace[self.outputName] = self._pickChannel(namespace[self.inputName])
        
@register_module('JoinChannels')    
class JoinChannels(ModuleBase):
    '''extract one channel from an image'''
    inputChan0 = CStr('input0')
    inputChan1 = CStr('')
    inputChan2 = CStr('')
    inputChan3 = CStr('')
    outputName = CStr('output')     
    
    #channelToExtract = Int(0)
    
    def _joinChannels(self, namespace):
        chans = []

        image = namespace[self.inputChan0]        
        
        chans.append(image.data[:,:,:,0])
        
        if not self.inputChan1 == '':
            chans.append(namespace[self.inputChan1].data[:,:,:,0])
        if not self.inputChan2 == '':
            chans.append(namespace[self.inputChan2].data[:,:,:,0])
        if not self.inputChan3 == '':
            chans.append(namespace[self.inputChan3].data[:,:,:,0])
        
        im = ImageStack(chans, titleStub = 'Composite Image')
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        return im
    
    def execute(self, namespace):
        namespace[self.outputName] = self._joinChannels(namespace)
        
@register_module('Add')    
class Add(ArithmaticFilter):
    '''Add two images'''
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0 + data1
        
@register_module('Subtract')    
class Subtract(ArithmaticFilter):
    '''Add two images'''
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0 - data1
        
@register_module('Multiply')    
class Multiply(ArithmaticFilter):
    '''Add two images'''
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0*data1
    
@register_module('Divide')    
class Divide(ArithmaticFilter):
    '''Add two images'''
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0/data1  
        
@register_module('Scale')    
class Scale(Filter):
    '''Add two images'''
    
    scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        return self.scale*data
        
@register_module('Normalize')    
class Normalize(Filter):
    '''Normalize an image so that the maximum is 1'''
    
    #scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        return data/float(data.max())
        
        
@register_module('Invert')    
class Invert(Filter):
    '''Invert image'''
    
    #scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        return 1 - data
        
@register_module('BinaryOr')    
class BinaryOr(ArithmaticFilter):
    '''Add two images'''
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return (data0 + data1) > .5
        
def _issubclass(cl, c):
    try:
        return issubclass(cl, c)
    except TypeError:
        return False
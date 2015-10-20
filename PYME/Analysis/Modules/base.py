# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:02:04 2015

@author: david
"""

try:
    from enthought.traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance
    from enthought.traits.ui.api import View, Item, EnumEditor, InstanceEditor, Group
except ImportError:
    from traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance
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
            mod = all_modules[mn]()
            mod.set(**md[mn])
            mc.append(mod)
            
        return cls(modules=mc)
        
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
        
        l = yaml.load(data)
        
        mc = []
        
        if l == None:
            l = []
        
        for mdd in l:
            mn, md = mdd.items()[0]
            mod = all_modules[mn]()
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
        return {v for k,v in self.get().items() if k.startswith('input') and not v == ""}
        
    @property
    def outputs(self):
        return {v for k,v in self.get().items() if k.startswith('output')}
        
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
        
        
def _issubclass(cl, c):
    try:
        return issubclass(cl, c)
    except TypeError:
        return False
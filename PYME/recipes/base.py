# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:02:04 2015

@author: david
"""
#import wx

from PYME.recipes.traits import HasTraits, Float, List, Bool, Int, on_trait_change, Input, Output
    
    #for some reason traitsui raises SystemExit when called from sphinx on OSX
    #This is due to the framework build problem of anaconda on OSX, and also
    #creates a problem whenever there is no GUI available.
    #as we want to be able to use recipes without a GUI (presumably the reason for this problem)
    #it's prudent to catch this and spoof the View and Item functions which are not going to be used anyway
    #try:
    #from traitsui.api import View, Item, Group# EnumEditor, InstanceEditor, Group
    #except SystemExit:
    #   print('Got stupid OSX SystemExit exception - using dummy traitsui')
    #   from PYME.misc.mock_traitsui import *

from PYME.IO.image import ImageStack
import numpy as np

all_modules = {}
_legacy_modules = {}
module_names = {}

def register_module(moduleName):
    def c_decorate(cls):
        py_module = cls.__module__.split('.')[-1]
        full_module_name = '.'.join([py_module, moduleName])

        all_modules[full_module_name] = cls
        _legacy_modules[moduleName] = cls #allow acces by non-hierarchical names for backwards compatibility

        module_names[cls] = full_module_name
        return cls
        
    return c_decorate

class ModuleBase(HasTraits):
    def __init__(self, parent=None, **kwargs):
        self._parent = parent

        HasTraits.__init__(self)
        self.set(**kwargs)

    @on_trait_change('anytrait')
    def remove_outputs(self):
        if not self.__dict__.get('_parent', None) is None:
            self._parent.pruneDependanciesFromNamespace(self.outputs)

    def outputs_in_namespace(self, namespace):
        keys = namespace.keys()
        return np.all([op in keys for op in self.outputs])

    def execute(self, namespace):
        """prototype function - should be over-ridden in derived classes

        takes a namespace (a dictionary like object) from which it reads its inputs and
        into which it writes outputs
        """
        pass

    @property
    def inputs(self):
        return {v for k, v in self.get().items() if k.startswith('input') and not v == ""}

    @property
    def outputs(self):
        return {v for k, v in self.get().items() if k.startswith('output')}

    def trait_view(self, name=None, view_element=None):
        import traitsui.api as tui

        if view_element is None and isinstance(name, basestring):
            try:
                tc = getattr(self, name)

                if isinstance(tc, tui.View):
                    return tc
            except AttributeError:
                pass

        return HasTraits.trait_view(self, name, view_element)

    @property
    def hide_in_overview(self):
        return []

    @property
    def pipeline_view(self):
        import traitsui.api as tui

        modname = ','.join(self.inputs) + ' -> ' + self.__class__.__name__ + ' -> ' + ','.join(self.outputs)

        hidden = self.hide_in_overview

        params = [tn for tn in self.class_editable_traits() if not (tn.startswith('input') or tn.startswith('output') or tn in hidden)]

        return tui.View(tui.Group([tui.Item(tn) for tn in params],label=modname))


    @property
    def _namespace_keys(self):
        try:
            namespace_keys = {'input', } | set(self._parent.namespace.keys())
            namespace_keys.update(self._parent.module_outputs)
            return list(namespace_keys)
        except:
            return []

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group
        from PYME.ui.custom_traits_editors import CBEditor

        editable = self.class_editable_traits()
        inputs = [tn for tn in editable if tn.startswith('input')]
        outputs = [tn for tn in editable if tn.startswith('output')]
        params = [tn for tn in editable if not (tn in inputs or tn in outputs or tn.startswith('_'))]

        return View([Item(tn, editor=CBEditor(choices=self._namespace_keys)) for tn in inputs] + [Item('_'),] +
                    [Item(tn) for tn in params] + [Item('_'),] +
                    [Item(tn) for tn in outputs], buttons=['OK', 'Cancel'])



    def default_traits_view( self ):
        return self.default_view


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
        """get all the downstream items which depend on the given key"""
        
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
        
        #print downstream
        
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
        #print self.namespace.keys()
        for k, v in kwargs.items():
            #print k, v
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

            mod_traits_cleaned = {}
            for k, v in mod.get().items():
                if not k.startswith('_'): #don't save private data - this is usually used for caching etc ..,
                    if isinstance(v, dict) and not type(v) == dict:
                        v = dict(v)
                    elif isinstance(v, list) and not type(v) == list:
                        v = list(v)
                    elif isinstance(v, set) and not type(v) == set:
                        v = set(v)

                    mod_traits_cleaned[k] = v

            l.append({module_names[mod.__class__]: mod_traits_cleaned})
            
        return yaml.safe_dump(l, default_flow_style=False)
        
    def toJSON(self):
        import json
        l = []
        for mod in self.modules:
            #l.append({mod.__class__.__name__: mod.get()})
            l.append({module_names[mod.__class__]: mod.get()})
            
        return json.dumps(l)
    
    @classmethod
    def fromYAML(cls, data):
        import yaml
        
        c = cls()
        
        l = yaml.load(data)
        
        mc = []
        
        if l is None:
            l = []
        
        for mdd in l:
            mn, md = mdd.items()[0]
            try:
                mod = all_modules[mn](c)
            except KeyError:
                # still support loading old recipes which do not use hierarchical names
                # also try and support modules which might have moved
                mod = _legacy_modules[mn.split('.')[-1]](c)

            mod.set(**md)
            mc.append(mod)
            
        c.modules = mc
            
        return c#cls(modules=mc)

    def add_module(self, module):
        self.modules.append(module)
        
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

    @property
    def module_outputs(self):
        op = set()
        for mod in self.modules:
            op.update(set(mod.outputs))
        return op


    @property
    def pipeline_view(self):
        from traitsui.api import View, ListEditor, InstanceEditor, Item
        #v = tu.View(tu.Item('modules', editor=tu.ListEditor(use_notebook=True, view='pipeline_view'), style='custom', show_label=False),
        #            buttons=['OK', 'Cancel'])

        return View(Item('modules', editor=ListEditor(style='custom', editor=InstanceEditor(view='pipeline_view'),
                                                      mutable=False),
                         style='custom', show_label=False),
                    buttons=['OK', 'Cancel'])

        
class Filter(ModuleBase):
    """Module with one image input and one image output"""
    inputName = Input('input')
    outputName = Output('filtered_image')
    
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

    @classmethod
    def dsviewer_plugin_callback(cls, dsviewer, showGUI=True, **kwargs):
        """Implements a callback which allows this module to be used as a plugin for dsviewer.

        Parameters
        ----------

        dsviewer : :class:`PYME.DSView.dsviewer.DSViewFrame` instance
            This is the current :class:`~PYME.DSView.dsviewer.DSViewFrame` instance. The filter will be run with the
            associated ``.image`` as input and display the output in a new window.

        showGUI : bool
            Should we show a GUI to set parameters (generated by calling configure_traits()), or just run with default
            parameters.

        **kwargs : dict
            Optionally, provide default values for parameters. Makes most sense when used with showGUI = False

        """
        from PYME.DSView import ViewIm3D

        mod = cls(inputName='input', outputName='output', **kwargs)
        if (not showGUI) or mod.configure_traits(kind='modal'):
            namespace = {'input' : dsviewer.image}
            mod.execute(namespace)

            ViewIm3D(mod['output'], parent=dsviewer, glCanvas=dsviewer.glCanvas)

    
class ArithmaticFilter(ModuleBase):
    """Module with one image input and one image output"""
    inputName0 = Input('input')
    inputName1 = Input('input')
    outputName = Output('filtered_image')
    
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
    """extract one channel from an image"""
    inputName = Input('input')
    outputName = Output('filtered_image')
    
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
    """extract one channel from an image"""
    inputChan0 = Input('input0')
    inputChan1 = Input('')
    inputChan2 = Input('')
    inputChan3 = Input('')
    outputName = Output('output')
    
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
    """Add two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0 + data1
        
@register_module('Subtract')    
class Subtract(ArithmaticFilter):
    """Add two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0 - data1
        
@register_module('Multiply')    
class Multiply(ArithmaticFilter):
    """Add two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0*data1
    
@register_module('Divide')    
class Divide(ArithmaticFilter):
    """Add two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0/data1  
        
@register_module('Scale')    
class Scale(Filter):
    """Add two images"""
    
    scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        return self.scale*data
        
@register_module('Normalize')    
class Normalize(Filter):
    """Normalize an image so that the maximum is 1"""
    
    #scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        return data/float(data.max())
        
@register_module('NormalizeMean')    
class NormalizeMean(Filter):
    """Normalize an image so that the mean is 1"""
    
    offset = Float(0)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        data = data - self.offset
        
        return data/float(data.mean())
        
        
@register_module('Invert')    
class Invert(Filter):
    """Invert image"""
    
    #scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        return 1 - data
        
@register_module('BinaryOr')    
class BinaryOr(ArithmaticFilter):
    """Add two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return (data0 + data1) > .5
        
def _issubclass(cl, c):
    try:
        return issubclass(cl, c)
    except TypeError:
        return False

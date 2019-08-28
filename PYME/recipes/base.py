# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Mon May 25 17:02:04 2015

@author: david
"""
#import wx
import six

from PYME.recipes.traits import HasTraits, Float, List, Bool, Int, CStr, Enum, File, on_trait_change, Input, Output
    
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

import logging
logger = logging.getLogger(__name__)

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


def register_legacy_module(moduleName, py_module=None):
    """Permits a module to be accessed by an old name"""
    
    def c_decorate(cls):
        if py_module is None:
            py_module_ = cls.__module__.split('.')[-1]
        else:
            py_module_ = py_module
            
        full_module_name = '.'.join([py_module_, moduleName])

        _legacy_modules[full_module_name] = cls
        _legacy_modules[moduleName] = cls #allow access by non-hierarchical names for backwards compatibility

        #module_names[cls] = full_module_name
        return cls

    return c_decorate

class ModuleBase(HasTraits):
    def __init__(self, parent=None, **kwargs):
        self._parent = parent
        self._invalidate_parent = True

        HasTraits.__init__(self)
        self.set(**kwargs)

    @on_trait_change('anytrait')
    def invalidate_parent(self):
        if self._invalidate_parent and not self.__dict__.get('_parent', None) is None:
            self._parent.prune_dependencies_from_namespace(self.outputs)
            
            self._parent.invalidate_data()

    def outputs_in_namespace(self, namespace):
        keys = namespace.keys()
        return np.all([op in keys for op in self.outputs])

    def execute(self, namespace):
        """prototype function - should be over-ridden in derived classes

        takes a namespace (a dictionary like object) from which it reads its inputs and
        into which it writes outputs
        """
        pass
    
    def apply(self, *args, **kwargs):
        """
        Execute this module on the given input without the need for creating a recipe. Creates a namespace, populates it,
        runs the module, and returns a dictionary of outputs.
        
        If a module has a single input, you can provide the input directly as the first and only argument. If the module
        supports multiple inputs, they must be specified using keyword arguments.
        
        NOTE: Use the trait names as keys.
        
        If the module has only one output, using apply_simple() will automatically pull it out of the namespace and return it.
        
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        namespace = {}
        
        if (len(self.inputs) == 1) and (len(args) == 1):
            namespace[self.inputs[0]] = args[0]
        elif (len(args) > 0):
            raise RuntimeError('This module has multiple inputs, please use keyword arguments')
        else:
            for k, v in kwargs.items():
                namespace[getattr(self, k)] = v
                
        self.execute(namespace)
        
        return {k : namespace[k] for k in self.outputs}
    
    def apply_simple(self, *args, **kwargs):
        """
        See documentaion for apply above - this allows single output modules to be used as though they were functions.
        
        
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        if (len(self.outputs) > 1):
            raise RuntimeError('Module has multiple outputs - use apply instead')
        
        return self.apply(*args, **kwargs)[next(iter(self.outputs))]
            

    @property
    def inputs(self):
        return {v for k, v in self.get().items() if k.startswith('input') and not v == ""}

    @property
    def outputs(self):
        return {v for k, v in self.get().items() if k.startswith('output')}
    
    @property
    def file_inputs(self):
        """
        Allows templated file names which will be substituted when a user runs the recipe
        
        Any key of the form {USERFILEXXXXXX} where XXXXXX is a unique name for this input will then give rise to a
        file input box and will be replaced by the file path / URI in the recipe.
        
        Returns
        -------
        
        any inputs which should be substituted

        """
        #print(self.get().items())
        return [v.lstrip('{').rstrip('}') for k, v in self.get().items() if isinstance(v, six.string_types) and v.startswith('{USERFILE')]
    
    def get_name(self):
        return module_names[self.__class__]

    def trait_view(self, name=None, view_element=None):
        import traitsui.api as tui
        from six import string_types

        if view_element is None and isinstance(name, string_types):
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
    
    def get_params(self):
        editable = self.class_editable_traits()
        inputs = [tn for tn in editable if tn.startswith('input')]
        outputs = [tn for tn in editable if tn.startswith('output')]
        params = [tn for tn in editable if not (tn in inputs or tn in outputs or tn.startswith('_'))]
        
        return inputs, outputs, params
        
    def _pipeline_view(self, show_label=True):
        import wx
        if wx.GetApp() is None:
            return None
        
        import traitsui.api as tui

        modname = ','.join(self.inputs) + ' -> ' + self.__class__.__name__ + ' -> ' + ','.join(self.outputs)

        hidden = self.hide_in_overview
        
        inputs, outputs, params = self.get_params()

        #params = [tn for tn in self.class_editable_traits() if not (tn.startswith('input') or tn.startswith('output') or tn in hidden)]

        if show_label:
            return tui.View(tui.Group([tui.Item(tn) for tn in params],label=modname))
        else:
            return tui.View([tui.Item(tn) for tn in params])

    @property
    def pipeline_view(self):
        return self._pipeline_view()

    @property
    def pipeline_view_min(self):
        return self._pipeline_view(False)


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
        import wx
        if wx.GetApp() is None:
            return None
        
        from traitsui.api import View, Item, Group
        from PYME.ui.custom_traits_editors import CBEditor

        #editable = self.class_editable_traits()
        #inputs = [tn for tn in editable if tn.startswith('input')]
        #outputs = [tn for tn in editable if tn.startswith('output')]
        #params = [tn for tn in editable if not (tn in inputs or tn in outputs or tn.startswith('_'))]
        inputs, outputs, params = self.get_params()

        return View([Item(tn, editor=CBEditor(choices=self._namespace_keys)) for tn in inputs] + [Item('_'),] +
                    [Item(tn) for tn in params] + [Item('_'),] +
                    [Item(tn) for tn in outputs], buttons=['OK', 'Cancel'])



    def default_traits_view( self ):
        return self.default_view


class OutputModule(ModuleBase):
    filePattern = CStr('{output_dir}/{file_stub}.csv')
    scheme = Enum('File', 'pyme-cluster://', 'pyme-cluster:// - aggregate')

    def _schemafy_filename(self, out_filename):
        if self.scheme == 'File':
            return out_filename
        elif self.scheme == 'pyme-cluster://':
            from PYME.IO import clusterIO
            import os
            return os.path.join(clusterIO.local_dataroot, out_filename.lstrip('/'))
        elif self.scheme == 'pyme-cluster:// - aggregate':
            raise RuntimeError('Aggregation not suported')
        
    def generate(self, namespace, recipe_context={}):
        """
        Function to be called from within dh5view (rather than batch processing). Some outputs are ignored, in which
        case this function returns None.
        
        Parameters
        ----------
        namespace

        Returns
        -------

        """
        return None


    def execute(self, namespace):
        """
        Output modules be definition do nothing when executed - they act as a sink and implement a save method instead.

        """
        pass

import dispatch
class ModuleCollection(HasTraits):
    modules = List()
    execute_on_invalidation = Bool(False)
    
    def __init__(self, *args, **kwargs):
        HasTraits.__init__(self, *args, **kwargs)
        
        self.namespace = {}
        
        # we open hdf files and don't necessarily read their contents into memory - these need to be closed when we
        # either delete the recipe, or clear the namespace
        self._open_input_files = []
        
        self.recipe_changed = dispatch.Signal()
        self.recipe_executed = dispatch.Signal()
        
    def invalidate_data(self):
        if self.execute_on_invalidation:
            self.execute()
            
    def clear(self):
        self.namespace.clear()
        
    def new_output_name(self, stub):
        count = len([k.startswith(stub) for k in self.namespace.keys()])
        
        if count == 0:
            return stub
        else:
            return '%s_%d' % (stub, count)
        
        
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
        
        
    def prune_dependencies_from_namespace(self, keys_to_prune, keep_passed_keys = False):
        rdg = self.reverseDependancyGraph()
        
        if keep_passed_keys:
            downstream = list(self._getAllDownstream(rdg, list(keys_to_prune)))
        else:
            downstream = list(keys_to_prune) + list(self._getAllDownstream(rdg, list(keys_to_prune)))
        
        #print downstream
        
        for dsi in downstream:
            try:
                self.namespace.pop(dsi)
            except KeyError:
                #the output is not in our namespace, no need to prune
                pass
            except AttributeError:
                #we might not have our namespace defined yet
                pass
        
        
    def resolveDependencies(self):
        import toposort
        #build dependancy graph
                    
        dg = self.dependancyGraph()
        
        #solve the dependency tree        
        return toposort.toposort_flatten(dg, sort=False)
        
    def execute(self, **kwargs):
        #remove anything which is downstream from changed inputs
        #print self.namespace.keys()
        for k, v in kwargs.items():
            #print k, v
            try:
                if not (self.namespace[k] == v):
                    #input has changed
                    print('pruning: ', k)
                    self.prune_dependencies_from_namespace([k])
            except KeyError:
                #key wasn't in namespace previously
                print('KeyError')
                pass
    
        self.namespace.update(kwargs)
        
        exec_order = self.resolveDependencies()

        for m in exec_order:
            if isinstance(m, ModuleBase) and not m.outputs_in_namespace(self.namespace):
                try:
                    m.execute(self.namespace)
                except:
                    logger.exception("Error in recipe module: %s" % m)
                    raise
        
        self.recipe_executed.send_robust(self)
        
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
        
    def get_cleaned_module_list(self):
        l = []
        for mod in self.modules:
            #l.append({mod.__class__.__name__: mod.get()})
            
            ct = mod.class_traits()

            mod_traits_cleaned = {}
            for k, v in mod.get().items():
                if not k.startswith('_'): #don't save private data - this is usually used for caching etc ..,
                    try:
                        if (not (v == ct[k].default)) or (k.startswith('input')) or (k.startswith('output')):
                            #don't save defaults
                            if isinstance(v, dict) and not type(v) == dict:
                                v = dict(v)
                            elif isinstance(v, list) and not type(v) == list:
                                v = list(v)
                            elif isinstance(v, set) and not type(v) == set:
                                v = set(v)
        
                            mod_traits_cleaned[k] = v
                    except KeyError:
                        # for some reason we have a trait that shouldn't be here
                        pass

            l.append({module_names[mod.__class__]: mod_traits_cleaned})

        return l


    def toYAML(self):
        import yaml
        return yaml.safe_dump(self.get_cleaned_module_list(), default_flow_style=False)
        
    def toJSON(self):
        import json
        return json.dumps(self.get_cleaned_module_list())
    
    def _update_from_module_list(self, l):
        """
        Update from a parsed yaml or json list of modules
        
        It probably makes no sense to call this directly as the format is pretty wack - a
        list of dictionarys each with a single entry, but that is how the yaml parses

        Parameters
        ----------
        l: list
            List of modules as obtained from parsing a yaml recipe,
            Each module is a dictionary mapping with a single e.g.
            [{'Filtering.Filter': {'filters': {'probe': [-0.5, 0.5]}, 'input': 'localizations', 'output': 'filtered'}}]

        Returns
        -------

        """
        mc = []
    
        if l is None:
            l = []
    
        for mdd in l:
            mn, md = list(mdd.items())[0]
            try:
                mod = all_modules[mn](self)
            except KeyError:
                # still support loading old recipes which do not use hierarchical names
                # also try and support modules which might have moved
                mod = _legacy_modules[mn.split('.')[-1]](self)
        
            mod.set(**md)
            mc.append(mod)
    
        self.modules = mc
        
        self.recipe_changed.send_robust(self)
        self.invalidate_data()
    
    @classmethod
    def _from_module_list(cls, l):
        """ A factory method which contains the common logic for loading/creating from either
        yaml or json. Do not call directly"""
        c = cls()
        c._update_from_module_list(l)
                
        return c

    @classmethod
    def fromYAML(cls, data):
        import yaml

        l = yaml.load(data)
        return cls._from_module_list(l)
    
    def update_from_yaml(self, data):
        """
        Update from a yaml formatted recipe description

        Parameters
        ----------
        data: str
            either yaml formatted text, or the path to a yaml file.

        Returns
        -------
        None

        """
        import os
        import yaml

        if os.path.isfile(data):
            with open(data) as f:
                data = f.read()
    
        l = yaml.load(data)
        return self._update_from_module_list(l)

    @classmethod
    def fromJSON(cls, data):
        import json
        return cls._from_module_list(json.loads(data))


    def add_module(self, module):
        self.modules.append(module)
        self.recipe_changed.send_robust(self)
        
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
    def file_inputs(self):
        out = []
        for mod in self.modules:
            out += mod.file_inputs
            
        return out

    def save(self, context={}):
        """
        Find all OutputModule instances and call their save methods with the recipe context

        Parameters
        ----------
        context : dict
            A context dictionary used to substitute and create variable names.

        """
        for mod in self.modules:
            if isinstance(mod, OutputModule):
                mod.save(self.namespace, context)
                
    def gather_outputs(self, context={}):
        """
        Find all OutputModule instances and call their generate methods with the recipe context

        Parameters
        ----------
        context : dict
            A context dictionary used to substitute and create variable names.

        """
        
        outputs = []
        
        for mod in self.modules:
            if isinstance(mod, OutputModule):
                out = mod.generate(self.namespace, context)
                
                if not out is None:
                    outputs.append(out)
                    
        return outputs

    def loadInput(self, filename, key='input'):
        """Load input data from a file and inject into namespace

        Currently only handles images (anything you can open in dh5view). TODO -
        extend to other types.
        """
        #modify this to allow for different file types - currently only supports images
        from PYME.IO import unifiedIO
        import os
        extension = os.path.splitext(filename)[1]
        if extension in ['.h5r', '.h5', '.hdf']:
            import tables
            from PYME.IO import MetaDataHandler
            from PYME.IO import tabular

            with unifiedIO.local_or_temp_filename(filename) as fn:
                with tables.open_file(fn, mode='r') as h5f:
                    #make sure our hdf file gets closed
                    
                    key_prefix = '' if key == 'input' else key + '_'
    
                    try:
                        mdh = MetaDataHandler.NestedClassMDHandler(MetaDataHandler.HDFMDHandler(h5f))
                    except tables.FileModeError:  # Occurs if no metadata is found, since we opened the table in read-mode
                        logger.warning('No metadata found, proceeding with empty metadata')
                        mdh = MetaDataHandler.NestedClassMDHandler()
                    
                    for t in h5f.list_nodes('/'):
                        # FIXME - The following isinstance tests are not very safe (and badly broken in some cases e.g.
                        # PZF formatted image data, Image data which is not in an EArray, etc ...)
                        # Note that EArray is only used for streaming data!
                        # They should ideally be replaced with more comprehensive tests (potentially based on array or dataset
                        # dimensionality and/or data type) - i.e. duck typing. Our strategy for images in HDF should probably
                        # also be improved / clarified - can we use hdf attributes to hint at the data intent? How do we support
                        # > 3D data?
                        
                        if isinstance(t, tables.VLArray):
                            from PYME.IO.ragged import RaggedVLArray
                            
                            rag = RaggedVLArray(h5f, t.name, copy=True) #force an in-memory copy so we can close the hdf file properly
                            rag.mdh = mdh
    
                            self.namespace[key_prefix + t.name] = rag
    
                        elif isinstance(t, tables.table.Table):
                            #  pipe our table into h5r or hdf source depending on the extension
                            tab = tabular.h5rSource(h5f, t.name) if extension == '.h5r' else tabular.hdfSource(h5f, t.name)
                            tab.mdh = mdh
    
                            self.namespace[key_prefix + t.name] = tab
    
                        elif isinstance(t, tables.EArray):
                            # load using ImageStack._loadh5, which finds metdata
                            im = ImageStack(filename=filename, haveGUI=False)
                            # assume image is the main table in the file and give it the named key
                            self.namespace[key] = im
                        
        elif extension == '.csv':
            logger.error('loading .csv not supported yet')
            raise NotImplementedError
        elif extension in ['.xls', '.xlsx']:
            logger.error('loading .xls not supported yet')
            raise NotImplementedError
        else:
            self.namespace[key] = ImageStack(filename=filename, haveGUI=False)


    @property
    def pipeline_view(self):
        import wx
        if wx.GetApp() is None:
            return None
        else:
            from traitsui.api import View, ListEditor, InstanceEditor, Item
            #v = tu.View(tu.Item('modules', editor=tu.ListEditor(use_notebook=True, view='pipeline_view'), style='custom', show_label=False),
            #            buttons=['OK', 'Cancel'])
    
            return View(Item('modules', editor=ListEditor(style='custom', editor=InstanceEditor(view='pipeline_view'),
                                                          mutable=False),
                             style='custom', show_label=False),
                        buttons=['OK', 'Cancel'])
        
    def to_svg(self):
        from . import recipeLayout
        return recipeLayout.to_svg(self.dependancyGraph())
        

        
class Filter(ModuleBase):
    """Module with one image input and one image output"""
    inputName = Input('input')
    outputName = Output('filtered_image')
    
    processFramesIndividually = Bool(True)
    
    def filter(self, image):
        #from PYME.util.shmarray import shmarray
        #import multiprocessing
        
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
    """Module with two image inputs and one image output"""
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
    """Extract one channel from an image"""
    inputName = Input('input')
    outputName = Output('filtered_image')
    
    channelToExtract = Int(0)
    
    def _pickChannel(self, image):
        chan = image.data[:,:,:,self.channelToExtract]
        
        im = ImageStack(chan, titleStub = 'Filtered Image')
        im.mdh.copyEntriesFrom(image.mdh)
        try:
            im.mdh['ChannelNames'] = [image.names[self.channelToExtract],]
        except (KeyError, AttributeError):
            logger.warn("Error setting channel name")

        im.mdh['Parent'] = image.filename
        
        return im
    
    def execute(self, namespace):
        namespace[self.outputName] = self._pickChannel(namespace[self.inputName])
        
@register_module('JoinChannels')    
class JoinChannels(ModuleBase):
    """Join multiple channels to form a composite image"""
    inputChan0 = Input('input0')
    inputChan1 = Input('')
    inputChan2 = Input('')
    inputChan3 = Input('')
    outputName = Output('output')
    
    #channelToExtract = Int(0)
    
    def _joinChannels(self, namespace):
        chans = []

        image = namespace[self.inputChan0]        
        
        chans.append(np.atleast_3d(image.data[:,:,:,0]))
        
        channel_names = [self.inputChan0,]
        
        if not self.inputChan1 == '':
            chans.append(namespace[self.inputChan1].data[:,:,:,0])
            channel_names.append(self.inputChan1)
        if not self.inputChan2 == '':
            chans.append(namespace[self.inputChan2].data[:,:,:,0])
            channel_names.append(self.inputChan2)
        if not self.inputChan3 == '':
            chans.append(namespace[self.inputChan3].data[:,:,:,0])
            channel_names.append(self.inputChan3)
        
        im = ImageStack(chans, titleStub = 'Composite Image')
        im.mdh.copyEntriesFrom(image.mdh)
        im.names = channel_names
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
    """Subtract two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0 - data1
        
@register_module('Multiply')    
class Multiply(ArithmaticFilter):
    """Multiply two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0*data1
    
@register_module('Divide')    
class Divide(ArithmaticFilter):
    """Divide two images"""
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return data0/data1
    
@register_module('Pow')
class Pow(Filter):
    "Raise an image to a given power (can be fractional for sqrt)"
    power = Float(2)
    
    def applyFilter(self, data, chanNum, i, image0):
        return np.power(data, self.power)
        
@register_module('Scale')    
class Scale(Filter):
    """Scale an image intensities by a constant"""
    
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
    """Invert image

    This is implemented as :math:`B = (1-A)`. As such the results only really make sense for binary images / masks and
    for images which have been normalized such that the maximum value is 1.
    """
    
    #scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        
        return 1 - data
        
@register_module('BinaryOr')    
class BinaryOr(ArithmaticFilter):
    """Perform a bitwise OR on images

    Notes
    -----

    This is actually implemented as :math:`(A + B) > .5`
    """
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        
        return (data0 + data1) > .5
        
def _issubclass(cl, c):
    try:
        return issubclass(cl, c)
    except TypeError:
        return False

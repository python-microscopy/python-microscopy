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

class MissingInputError(Exception):
    pass

class ModuleBase(HasTraits):
    """
    Recipe modules represent a "functional" processing block, the effects of which depend solely on its
    inputs and parameters. They read a number of named inputs from the recipe namespace and write the
    results of their computation back to one or more named output variables in the recipe namespace.
    
    They must not modify anything other than their named output variables, and should not maintain state
    between executions (no side effects). This is critical in ensuring that repeat executions of the same
    recipe are reproducible and in allowing the incremental update (without re-running every module) when
    the parameters of one module change.
    
    If you want side effects - e.g. saving something to disk, look at the OutputModule class.
    """
    _invalidate_parent = True
    
    def __init__(self, parent=None, invalidate_parent = True, **kwargs):
        self._parent = parent
        self._invalidate_parent = invalidate_parent

        HasTraits.__init__(self)
        self.trait_set(**kwargs)
        
        self._check_outputs()

    @on_trait_change('anytrait')
    def invalidate_parent(self, name='', new=None):
        #print('invalidate_parent', name, new)
        if (name == 'trait_added') or name.startswith('_'):
            # don't trigger on private variables
            return
        
        if self._invalidate_parent and not self.__dict__.get('_parent', None) is None:
            #print('invalidating')
            self._parent.prune_dependencies_from_namespace(self.outputs)
            
            self._parent.invalidate_data()
            
    def _check_outputs(self):
        """
        This function exists to help with debugging when writing a new recipe module. It generates an
        exception if no outputs have been defined (a module with no outputs will never execute).
        
        Over-ridden in the special case IO modules derived from OutputModule as these are permitted to
        have side-effects, but not permitted to have classical outputs to the namespace and will execute
        (when appropriate) regardless.
        """
        if len(self.outputs) == 0:
            raise RuntimeError('Module should define at least one output.')
        
    def check_inputs(self, namespace):
        """
        Checks that module inputs are present in namespace, raising an exception if they are missing. Existing to simplify
        debugging, this function is called in ModuleCollection.execute prior to executing the module so that an
        informative error message is generated if the inputs are missing (rather than the somewhat cryptic KeyError
        you would get when a module tries to access a missing input.
        
        
        Parameters
        ----------
        namespace

        """
        keys = list(namespace.keys())
        
        for input in self.inputs:
            if not input in keys:
                raise MissingInputError('Input "%s" is missing from namespace; keys: %s' % (input, keys))
        

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
        namespace = dict()
        
        ips = list(self.inputs)
        
        if (len(ips) == 1) and (len(args) == 1):
            namespace[ips[0]] = args[0]
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
        return {v for k, v in self.trait_get().items() if (k.startswith('input') or isinstance(k, Input)) and not v == ''}

    @property
    def outputs(self):
        return {v for k, v in self.trait_get().items() if (k.startswith('output') or isinstance(k, Output)) and not v ==''}
    
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
        return [v.lstrip('{').rstrip('}') for k, v in self.trait_get().items() if isinstance(v, six.string_types) and v.startswith('{USERFILE')]
    
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
    
    def edit_no_invalidate(self, *args, **kwargs):
        inv_mode = self._invalidate_parent
        
        try:
            #print('turning off invalidation')
            self._invalidate_parent = False
            old_traits = self.trait_get()
            #print('edit_traits')
            self.edit_traits(*args, kind='modal', **kwargs)
            self._invalidate_parent = inv_mode
            if not self.trait_get() == old_traits:
                #print('invalidating ...')
                self.invalidate_parent()
        finally:
            self._invalidate_parent = inv_mode
        

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

        if show_label:
            return tui.View(tui.Group(self._view_items(),label=modname))
        else:
            return tui.View(self._view_items())
        
    def _view_items(self, params=None):
        """
        Used to customize module views, bye.g.  specifying custom editors for a particular parameter, by grouping
        parameters in a more functionally meaningful manner, or even by hiding certain parameters based on e.g. method
        selection.
         
        See `https://docs.enthought.com/traitsui/traitsui_user_manual/view.html`_ and the following 2 topics for details
        on defining a view. Critically, this function should not return an entire view, but rather a list of items
        (which could potentially be Groups). default_view, pipeline_view, and pipeline_view_min will then augment this
        list as appropriate. The list should include items for all parameters of the module, but not for the input and
        outputs as these will be added separately by default_view (and don't appear in the pipeline views).
          
        Returns
        -------
        
        a list of traitsui view Items
        
        See Also
        --------
        
        default_view
        pipeline_view
        pipeline_view_min
        
        """
        import traitsui.api as tui
        if not params:
            inputs, outputs, params = self.get_params()
            
        return [tui.Item(tn) for tn in params]

    @property
    def pipeline_view(self):
        """
        A condensed view of the module suitable for display in the 'pipeline' section of VisGUI (or anywhere where you
        might want to edit all of the parameters of a recipe at once). It differs from the default_view in that it doesn't
        display input and output variables - i.e. it lets you edit the recipe parameters but not connectivity.
        
        The difference between pipeline_view and pipeline_view_min is that pipeline_view puts all the parameters in a
        named group which encodes the module name and its connectivity, whereas pipeline_view_min omits this header.
         
        To customise the view, you should over-ride `_view_items` instead of this function as this will customise both
        pipeline views and the default view in one hit, reducing duplicate code.
        
        Returns
        -------
        
        A traitsui `View` object - see `https://docs.enthought.com/traitsui/traitsui_user_manual/view.html`_
        
        See Also
        --------
        
        _view_items
        pipeline_view_min
        default_view

        """
        return self._pipeline_view()

    @property
    def pipeline_view_min(self):
        """ See docs for pipeline_view - this is the same, but lacks the module header"""
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
        """ The default traits view - displayed when clicking on a recipe module in the full recipe view to edit it
        
        This can be over-ridden in a derived class to customise how the recipe module is edited and, e.g. specify custom
        traits editors. NOTE: Editing this property will NOT change how the module is displayed in the compact recipe
        overview displayed in VisGUI. **In most cases, it is preferable to over-ride the  `_view_items()` method** which
        just constructs the part of the view associated with the module parameters, leaving the base module to
        auto-generate the input and output sections.
        
        In general a view should have the inputs, a separator, the module parameters, another separator, and finally an OK button.
        
        Returns
        -------
        
        A traitsui `View` object, see `https://docs.enthought.com/traitsui/traitsui_user_manual/view.html`_
        
        See Also
        --------
        
        pipeline_view
        _view_items
            
        """
        import wx
        if wx.GetApp() is None:
            return None
        
        from traitsui.api import View, Item, Group
        from PYME.ui.custom_traits_editors import CBEditor

        inputs, outputs, params = self.get_params()

        return View([Item(tn, editor=CBEditor(choices=self._namespace_keys)) for tn in inputs] +
                    [Item('_'),] +
                    self._view_items(params) +
                    [Item('_'),] +
                    [Item(tn) for tn in outputs], buttons=['OK', 'Cancel']) #TODO - should we have cancel? Traits update whilst being edited and cancel doesn't roll back



    def default_traits_view( self ):
        """ This is the traits stock method to specify the default view"""
        return self.default_view


class OutputModule(ModuleBase):
    """
    Output modules are the one exception to the recipe-module functional (no side effects) programming
    paradigm and are used to perform IO and save or display designated outputs/endpoints from the recipe.
    
    As such, they should act solely as a sink, and should not do any processing or write anything back
    to the namespace.
    """
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

    def _check_outputs(self):
        """
        This function exists to help with debugging when writing a new recipe module.

        Over-ridden here for the special case IO modules derived from OutputModule as these are permitted to
        have side-effects, but not permitted to have classical outputs to the namespace and will execute
        (when appropriate) regardless.
        """
        if len(self.outputs) != 0:
            raise RuntimeError('Output modules should not write anything to the namespace')
    
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

from PYME.contrib import dispatch
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
        self.recipe_failed = dispatch.Signal()
        
        self.failed = False
        
        self._dg_sig = None
        
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
    
    def upstream_inputs(self, keys):
        dg = self.dependancyGraph()
        
        def walk_upstream(keys):
            upstream = set()
            for k in keys:
                u = dg.get(k, None)
                if u is not None:
                    upstream.update(walk_upstream(list(u)))
                    
            return upstream
                
        return list(walk_upstream(keys))
            
    
    def downstream_outputs(self, keys):
        rdg = self.reverseDependancyGraph()
        return list(self._getAllDownstream(rdg, list(keys)))
        
        
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
        
    def execute(self, progress_callback=None, **kwargs):
        """
        Execute the recipe. Recipe execution is lazy / incremental in that modules will only be executed if their outputs
        are no longer in the namespace or their inputs have changed.
        
        Parameters
        ----------
        progress_callback : a function, progress_callback(recipe, module)
                This function is called after the successful execution of each module in the recipe. To abort the recipe,
                an exception may be raised in progress_callback.
        kwargs : any values to set in the recipe namespace prior to executing.

        Returns
        -------
        
        output : the value of the 'output' variable in the namespace if present, otherwise none. output is only used in
            dh5view and ignored in recipe batch processing. It might well disappear completely in the future.

        """
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

        #mark all modules which should execute as not having executed
        for m in exec_order:
            if isinstance(m, ModuleBase) and not m.outputs_in_namespace(self.namespace):
                m._success = False
        
        for m in exec_order:
            if isinstance(m, ModuleBase) and not getattr(m, '_success', False):
                try:
                    m.check_inputs(self.namespace)
                    m.execute(self.namespace)
                    m._last_error = None
                    m._success = True
                    if progress_callback:
                        progress_callback(self, m)
                except:
                    import traceback
                    logger.exception("Error in recipe module: %s" % m)
                    
                    #record our error so that we can associate it with a module
                    m._last_error = traceback.format_exc()
                    self.failed = True
                    
                    # make sure we didn't leave any partial results
                    logger.debug('removing failed module dependencies')
                    self.prune_dependencies_from_namespace(m.outputs)
                    logger.debug('notifying failure')
                    self.recipe_failed.send_robust(self)
                    raise
        
        if self.failed:
            # make sure we update the GUI if we've fixed a broken recipe
            # TODO - make make this a bit lighter weight - we shouldn't need to redraw the whole recipe just to change
            # the shading of the module caption
            self.recipe_changed.send_robust(self)
            
        self.failed = False
        self.recipe_executed.send_robust(self)

        # detect changes in recipe wiring
        dg_sig = str(self.dependancyGraph())
        if not self._dg_sig == dg_sig:
            #print(dg_sig)
            #print(self._dg_sig)
            self._dg_sig = dg_sig
            self.recipe_changed.send_robust(self)
        
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
        class MyDumper(yaml.SafeDumper):
            def represent_mapping(self, tag, value, flow_style=None):
                return super(MyDumper, self).represent_mapping(tag, value, False)
            
        return yaml.dump(self.get_cleaned_module_list(), Dumper=MyDumper)
    
    def save_yaml(self, uri):
        """
        Save the recipe text to .yaml using the cluster-aware unified IO library
        
        WARNING: Experimental - this was added as a quick hack to get web-based recipe editing working, and WILL LIKELY
        BE REMOVED without deprecation once that moves to using a recipe-manager. Whilst arguably the least contentious
        of the web-editor additions, it is unclear whether the saving logic (past dumping to YAML in the toYAML()
        method) should reside within the recipe itself. As the recipe class is proxied into the browser, there are also
        potential security implications here, particularly as this accepts both filenames and clusterURIs. As a
        consequence, saving should probably be factored out into something which can operate in an appropriate sandbox
        (the other option is to sandbox unifiedIO).
        
        Parameters
        ----------
        uri: str
            A filename or PYME-CLUSTER:// URI


        """
        from PYME.IO import unifiedIO
        
        unifiedIO.write(uri, self.toYAML().encode())
        
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
        # delete all outputs from the previous set of recipe modules
        self.prune_dependencies_from_namespace(self.module_outputs)
        
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

        l = yaml.safe_load(data)
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
    
        l = yaml.safe_load(data)

        return self._update_from_module_list(l)
    
    def update_from_file(self, filename):
        """
        Update the contents of the recipe from a .yaml file
        
        WARNING: This function will likely be REMOVED WITHOUT NOTICE. It is a quick hack to get the prototype web-based
        recipe editor working, but will be surplus to requirements once we have a proper recipe manager in the web based
        editor. It's logically obtuse to consider something the same recipe once you've completely replaced it with a
        recipe that has been loaded from file. It is much more sensible to create a new recipe instance when loading
        a recipe from file, and this is the recommended approach.
        
        Parameters
        ----------
        filename: str
            filename or PYME-CLUSTER:// URI

        """
        from PYME.IO import unifiedIO
        
        self.update_from_yaml(unifiedIO.read(filename).decode())

    @classmethod
    def fromJSON(cls, data):
        import json
        return cls._from_module_list(json.loads(data))


    def add_module(self, module):
        self.modules.append(module)
        self.recipe_changed.send_robust(self)
        
    def add_modules_and_execute(self, modules, rollback_on_failure=True):
        """
        Adds modules to the recipe and then execute the recipe. Added to make UI interaction in PYMEVis a bit nicer when
        modules added from the menu fail, this function gives the option (enabled by default) of rolling back the
        additions should execute fail.
        
        Parameters
        ----------
        modules : list
            a list of modules to add
        rollback_on_failure : bool
            rollback and remove modules (and their outputs) if .execute() fails

        """
        
        try:
            for m in modules:
                self.modules.append(m)
            
            self.execute()
        except:
            if rollback_on_failure:
                #do cleanup
                
                #remove any outputs
                for m in modules:
                    self.prune_dependencies_from_namespace(m.outputs)
                    
                #remove the modules themselves
                for m in modules:
                    self.modules.remove(m)
                
            raise
        finally:
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

    def _inject_tables_from_hdf5(self, key, h5f, filename, extension):
        """
        Search through hdf5 file nodes and add them to the recipe namespace

        Parameters
        ----------
        key : str
            base key name for loaded file components, if key is not the default 'input', each file node will be loaded into
            recipe namespace with `key`_`node_name`.
        h5f : file
            open hdf5 file
        filename : str
            full filename
        extension : str
            file extension, used here mainly to toggle which PYME.IO.tabular source is used for table nodes.
        """
        import tables
        from PYME.IO import MetaDataHandler, tabular

        key_prefix = '' if key == 'input' else key + '_'

        # Handle a 'MetaData' group as a special case
        # TODO - find/implement a more portable way of handling metadata in HDF (e.g. as .json in a blob) so that
        # non-python exporters have a chance of adding metadata
        if 'MetaData' in h5f.root:
            mdh = MetaDataHandler.NestedClassMDHandler(MetaDataHandler.HDFMDHandler(h5f))
        else:
            logger.warning('No metadata found, proceeding with empty metadata')
            mdh = MetaDataHandler.NestedClassMDHandler()
        
        events = None
        # handle an 'Events' table as a special case (so that it can be attached to subsequently loaded tables)
        # FIXME - this relies on a special /reserved table name and format and could raise name collision issues
        # when importing 3rd party / generic HDF
        # FIXME - do we really want to attach events (which will not get propagated through recipe modules)
        if ('Events' in h5f.root):
            if 'EventName' in h5f.root.Events.description._v_names:
                # check that the event table is formatted as we expect
                if ('StartTime' in mdh.keys()):
                    events = h5f.root.Events[:]
                else:
                    logger.warning('Acquisition events found in .hdf, but no "StartTime" in metadata')
            else:
                logger.warning(
                    'Table called "Events" found in .hdf does not match the signature for acquisition events, ignoring')
        
        for t in h5f.list_nodes('/'):
            # FIXME - The following isinstance tests are not very safe (and badly broken in some cases e.g.
            # PZF formatted image data, Image data which is not in an EArray, etc ...)
            # Note that EArray is only used for streaming data!
            # They should ideally be replaced with more comprehensive tests (potentially based on array or dataset
            # dimensionality and/or data type) - i.e. duck typing. Our strategy for images in HDF should probably
            # also be improved / clarified - can we use hdf attributes to hint at the data intent? How do we support
            # > 3D data?

            if getattr(t, 'name', None) == 'Events':
                # NB: This assumes we've handled this in the special case earlier, and blocks anything in a 3rd party
                # HDF events table from being seen.
                # TODO - do we really want to have so much special case stuff in our generic hdf handling? Are we sure
                # that events shouldn't be injected into the namespace (given that events do not propagate through recipe modules)?
                continue
            
            elif isinstance(t, tables.VLArray):
                from PYME.IO.ragged import RaggedVLArray
                
                rag = RaggedVLArray(h5f, t.name, copy=True) #force an in-memory copy so we can close the hdf file properly
                rag.mdh = mdh
                if events is not None:
                    rag.events = events

                self.namespace[key_prefix + t.name] = rag

            elif isinstance(t, tables.table.Table):
                #  pipe our table into h5r or hdf source depending on the extension
                tab = tabular.H5RSource(h5f, t.name) if extension == '.h5r' else tabular.HDFSource(h5f, t.name)
                tab.mdh = mdh
                if events is not None:
                    tab.events = events

                self.namespace[key_prefix + t.name] = tab

            elif isinstance(t, tables.EArray):
                # load using ImageStack._loadh5
                # FIXME - ._loadh5 will load events lazily, which isn't great if we got here after
                # sending file over clusterIO inside of a context manager -> force it through since we already found it
                im = ImageStack(filename=filename, mdh=mdh, events=events, haveGUI=False)
                # assume image is the main table in the file and give it the named key
                self.namespace[key] = im

    def loadInput(self, filename, key='input'):
        """
        Load input data from a file and inject into namespace
        """
        from PYME.IO import unifiedIO
        import os

        extension = os.path.splitext(filename)[1]
        if extension in ['.h5r', '.hdf']:
            import tables
            from PYME.IO import h5rFile
            try:
                with unifiedIO.local_or_temp_filename(filename) as fn, h5rFile.openH5R(fn, mode='r')._h5file as h5f:
                        self._inject_tables_from_hdf5(key, h5f, fn, extension)
            except tables.exceptions.HDF5ExtError:  # access issue likely due to multiple processes
                if unifiedIO.is_cluster_uri(filename):
                    # try again, this time forcing access through the dataserver
                    # NOTE: it is unclear why this should work when local_or_temp_filename() doesn't
                    # as this still opens / copies the file independently, albeit in the same process as is doing the writing.
                    # The fact that this works is relying on one of a quirk of the GIL, a quirk in HDF5 locking, or the fact
                    # that copying the file to a stream is much faster than opening it with pytables. The copy vs pytables open
                    # scenario would match what has been observed with old style spooling analysis where copying a file
                    # prior to opening in VisGUI would work more reliably than opening directly. This retains, however,
                    # an inherent race condition so we risk replacing a predictable failure with a less frequent one.
                    # TODO - consider whether h5r_part might be a better choice.
                    # FIXME: (DB) I'm not comfortable with having this kind of special case retry logic here, and would
                    # much prefer if we could find an alternative workaround, refactor into something like h5rFile.open_robust(),
                    # or just let this fail). Leaving it for the meantime to get chained recipes working, but we should revisit.
                    from PYME.IO import clusterIO
                    relative_filename, server_filter = unifiedIO.split_cluster_url(filename)
                    file_as_bytes = clusterIO.get_file(relative_filename, serverfilter=server_filter, local_short_circuit=False)
                    with tables.open_file('in-memory.h5', driver='H5FD_CORE', driver_core_image=file_as_bytes, driver_core_backing_store=0) as h5f:
                        self._inject_tables_from_hdf5(key, h5f, filename, extension)
                else:
                    #not a cluster file, doesn't make sense to retry with cluster. Propagate exception to user.
                    raise
                        
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
    
    def layout(self):
        """ Added as a visualisation aid for the web-based recipe editor. Very much a work in progress and not
        guaranteed to remain in it's current form.
        
        TODO - does this logic belong here????
        TODO - rename?? (potentially make it webui specific)???
        TODO - potential issues on Py3 with how jigna treats namedtuple?
        """
        from . import recipeLayout
        #from collections import namedtuple
        #layout_info = namedtuple('layout', ['node_positions', 'connecting_lines'])
        #node = namedtuple('node', ['key', 'pos'])
        class layout_info(object):
            def __init__(self, node_positions, connecting_lines):
                self.node_positions = node_positions
                self.connecting_lines = connecting_lines

        class node(object):
            def __init__(self, key, pos):
                self.key = key
                self.pos = pos

        node_positions, connecting_lines = recipeLayout.layout(self.dependancyGraph())
        ret =   layout_info([node(k, v) for k, v in node_positions.items()], [(a.tolist(), b.tolist(), c) for a,b,c in connecting_lines])
        #print (ret)
        return ret
    
    def _repr_svg_(self):
        """ Make us look pretty in Jupyter"""
        return self.to_svg()
        

        
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


@register_module('LogicalAnd')
class LogicalAnd(ArithmaticFilter):
    """Perform a logical AND on images"""

    def applyFilter(self, data0, data1, chanNum, i, image0):
        return np.logical_and(data0, data1)


@register_module('IsClose')
class IsClose(ArithmaticFilter):
    """
    Wrapper for numpy.isclose

    Parameters:
    -----------
    abs_tolerance: Float
        absolute tolerance
    rel_tolerance: Float
        relative tolerance

    Notes:
    ------
    from numpy docs, tolerances are combined as: absolute(a - b) <= (atol + rtol * absolute(b))

    """
    abs_tolerance = Float(1e-8)
    rel_tolerance = Float(1e-5)
    def applyFilter(self, data0, data1, chanNum, i, image0):
        return np.isclose(data0, data1, atol=self.abs_tolerance, rtol=self.rel_tolerance)
        
def _issubclass(cl, c):
    try:
        return issubclass(cl, c)
    except TypeError:
        return False

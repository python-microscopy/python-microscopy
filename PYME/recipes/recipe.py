from PYME.recipes.traits import HasTraits, List, Bool
from PYME.contrib import dispatch
from .base import ModuleBase, OutputModule
from PYME.recipes import base
from PYME.IO.image import ImageStack

import logging
logger = logging.getLogger(__name__)

#custom error for typos in recipe module names
class RecipeModuleNotFound(Exception):
    pass

class Recipe(HasTraits):
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
        count = len([k for k in self.namespace.keys() if k.startswith(stub)])
        
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
                dg[op] = {mod, }
        
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
    
    def prune_dependencies_from_namespace(self, keys_to_prune, keep_passed_keys=False):
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
        
        print('recipe.execute()')
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
                    print('Executing %s' % m)
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
                    self._failing_module = m
                    
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
            mod = base.all_modules[mn]()
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
            
            l.append({base.module_names[mod.__class__]: mod_traits_cleaned})
        
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
                mod = base.all_modules[mn](self, **md)
            except KeyError:
                # still support loading old recipes which do not use hierarchical names
                # also try and support modules which might have moved
                try:
                    mod = base._legacy_modules[mn.split('.')[-1]](self, **md)
                except KeyError:
                    raise RecipeModuleNotFound('No recipe module found with name "%s"\nThis could either be caused by a typo in the module name or a missing 3rd party plugin.' % mn) from None # Full traceback is probably unhelpful for people using modules 
            
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
                
                rag = RaggedVLArray(h5f, t.name,
                                    copy=True) #force an in-memory copy so we can close the hdf file properly
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
    
    def loadInput(self, filename, key='input', metadata_defaults={}):
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
                    file_as_bytes = clusterIO.get_file(relative_filename, serverfilter=server_filter,
                                                       local_short_circuit=False)
                    with tables.open_file('in-memory.h5', driver='H5FD_CORE', driver_core_image=file_as_bytes,
                                          driver_core_backing_store=0) as h5f:
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
            im = ImageStack(filename=filename, haveGUI=False)
            if im.mdh.get('voxelize.x', None) is None:
                logger.error('No voxelsize metadata found for image %s, using defaults' % filename)
            im.mdh.mergeEntriesFrom(metadata_defaults)
            self.namespace[key] = im
    
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
                        buttons=['OK'])
    
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
        ret = layout_info([node(k, v) for k, v in node_positions.items()],
                          [(a.tolist(), b.tolist(), c) for a, b, c in connecting_lines])
        #print (ret)
        return ret
    
    def _repr_mimebundle_(self):
        """ Make us look pretty in Jupyter"""
        try:
            return {'text/svg+xml' : self.to_svg()}
        except ModuleNotFoundError:
            return {'text/plain' : repr(self) + '\nTo enable pretty formatting of recipes in jupyter, install the `svgwrite` module'}
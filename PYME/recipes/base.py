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

# TODO - move to recipe.py?
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

        
        if (parent is not None):
            # make sure that the default output name does not collide with any outputs
            # already in the recipe
            for k, v in self._output_traits.items():
                if v in parent.module_outputs:
                    duplicate_num = 0
                    val = v
            
                    while (val in parent.module_outputs):
                        # we already have an output of that name in the recipe
                        # increase the subscript until we get a unique value
                        duplicate_num += 1
                        val = v + '_%d' % duplicate_num
                        
                    self.trait_set(**{k:val})
                

        # if an input matches the default value for an output, our circular reference check will fail, even if we are
        # setting both values to good values in the kwargs (see issue #695). To mitigate, we first set without validation
        # to overwrite any default values which may be modified, and then re-set with validation turned on to catch any
        # circular references in the final values.
        self._initial_set = False
        self.trait_set(trait_change_notify=False, **kwargs) #don't notify here - next set will do notification.
        self._initial_set = True
        
        # validate input and outputs now that output names have been set.
        # for now, just set all the values again to re-trigger validation
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
        See documentaion for `self.apply` above - this allows single output modules to be used as though they were functions.
        
        
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
        """
        Get module inputs

        Returns
        -------
        set of input names
        """
        return {v for k, v in self.trait_get().items() if (k.startswith('input') or isinstance(k, Input)) and not v == ''}

    @property
    def _output_traits(self):
        return {k:v for k, v in self.trait_get().items() if (k.startswith('output') or isinstance(k, Output)) and not v ==''}
    
    @property
    def outputs(self):
        """
        Get module outputs

        Returns
        -------
        set of output names
        """
        return set(self._output_traits.values())
    
    @property
    def file_inputs(self):
        """
        Allows templated file names which will be substituted when a user runs the recipe
        
        Any key of the form {USERFILEXXXXXX} where XXXXXX is a unique name for this input will then give rise to a
        file input box and will be replaced by the file path / URI in the recipe.
        
        Returns
        -------
        inputs: list
            keys for any inputs which should be substituted

        """
        return [v.lstrip('{').rstrip('}') for k, v in self.trait_get().items() if isinstance(v, six.string_types) and v.startswith('{USERFILE')]
    
    def get_name(self):
        """

        Returns
        -------
        registered_name : str
            returns the name of this recipe module as it is registered and will be displayed in e.g. 'Add Module' menus.
        """
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
            old_traits = self.trait_get()
            #print('turning off invalidation')
            self._invalidate_parent = False
            #print('edit_traits')
            self.edit_traits(*args, kind='modal', **kwargs)
            self._invalidate_parent = inv_mode
            if not self.trait_get() == old_traits:
                #print(self.trait_get(), old_traits)
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
        view_list: list
            a list of traitsui.View Items
        
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
        view : traitsui.View
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
                    [Item(tn) for tn in outputs], buttons=['OK']) #TODO - should we have cancel? Traits update whilst being edited and cancel doesn't roll back

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
        namespace : dict
            The recipe namespace
        recipe_context : dict
            Information about the source file(s) to allow pattern substitution and generate the output name. 
            At least 'basedir' (which is the fully resolved directory name in which the input file resides) 
            and 'filestub' (which is the filename without any extension) should be resolved.

        Returns
        -------

        """
        return None

    def execute(self, namespace):
        """
        Output modules by definition do nothing when executed - they act as a sink and implement a save method instead.

        Parameters
        ----------
        namespace : dict
            The recipe namespace
        
        Returns
        -------
        None

        """
        pass

    def save(self, namespace, context={}):
        """

        Parameters
        ----------
        namespace : dict
            The recipe namespace
        context : dict
            Information about the source file to allow pattern substitution to
            generate the output name. At least 'basedir' (which is the fully-
            resolved directory name in which the input file resides) and 
            'file_stub' (which is the filename without any extension) should be
            resolved.

        Returns
        -------

        """
        raise NotImplementedError

def ModuleCollection(*args, **kwargs):
    """ Backwards compatible factory stub for Recipe class"""
    from .recipe import Recipe
    import warnings
    
    warnings.warn(DeprecationWarning('recipes.base.ModuleCollection is deprecated, use recipes.recipe.Recipe instead'))
    return Recipe(*args, **kwargs)
        
class ImageModuleBase(ModuleBase):
    # NOTE - if a derived class only supports, e.g. XY analysis, it should redefine this trait ro only include the dimensions
    # it supports
    dimensionality = Enum('XY', 'XYZ', 'XYZT', desc='Which image dimensions should the filter be applied to?')
    
    #processFramesIndividually = Bool(True)
    
    @property
    def processFramesIndividually(self):
        import warnings
        warnings.warn(
            'Use dimensionality =="XY" instead to check which dimensions a filter should be applied to, chunking '
            'hints for computational optimisation', DeprecationWarning, stacklevel=2)
        
        logger.warning(
            'Use dimensionality =="XY" instead to check which dimensions a filter should be applied to, chunking '
            'hints for computational optimisation')
        return self.dimensionality == 'XY'
    
    @processFramesIndividually.setter
    def processFramesIndividually(self, value):
        import warnings
        warnings.warn(
            'Use dimensionality ="XY" instead to check which dimensions a filter should be applied to, chunking '
            'hints for computational optimisation', DeprecationWarning, stacklevel=2)
    
        logger.warning(
            'Use dimensionality ="XY" instead to check which dimensions a filter should be applied to, chunking '
            'hints for computational optimisation')
        
        if value:
            self.dimensionality = 'XY'
        else:
            self.dimensionality = 'XYZ'

class Filter(ImageModuleBase):
    """Module with one image input and one image output"""
    inputName = Input('input')
    outputName = Output('filtered_image')
    
    def output_shapes(self, input_shapes):
        """What shape is the output (without running any computation)
        
        Filters which modify image size (e.g. zoom) should over-write this. Will be used in the future for smart chunking.
        """
        return {self.outputName : input_shapes[self.inputName]}
    
    def filter(self, image):
        from PYME.IO.dataWrap import ListWrapper
        out = []
        for c in range(image.data_xyztc.shape[4]):
            if self.dimensionality == 'XYZT':
                data = image.data_xyztc[:, :, :, :,c].squeeze().astype('f')
                xyzt = np.atleast_3d(self._apply_filter(data, image, c=c))
            else: # XYZ or XY
                xyzt = []
                for t in range(image.data_xyztc.shape[3]):
                    if self.dimensionality == 'XYZ':
                        data = image.data_xyztc[:, :, :, t, c].squeeze().astype('f')
                        xyz = np.atleast_3d(self._apply_filter(data, image, c=c, t=t))
                    else: #XY
                        xyz = []
                        for z in range(image.data_xyztc.shape[2]):
                            data = image.data_xyztc[:, :, z, t, c].squeeze().astype('f')
                            
                            xyz.append(np.atleast_2d(self._apply_filter(data, image, c=c, t=t, z=z).squeeze()))
                    
                    xyzt.append(xyz)
            out.append(xyzt)        
        
        im = ImageStack(ListWrapper(out, strict_dims=True), titleStub = self.outputName)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        self.completeMetadata(im)
        
        return im
    
    def _apply_filter(self, data, image, z=None, t=None, c=None):
        if hasattr(self, 'applyFilter'):
            import warnings
            warnings.warn('applyFilter() is deprecated, derived classes should implement `apply_filter()` instead')
            return  self.applyFilter(data, c, z, image)
        else:
            return self.apply_filter(data, voxelsize=image.voxelsize)
        
    def apply_filter(self, data, voxelsize):
        raise NotImplementedError('Should be over-ridden in derived class')
        
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
    """
    Module with two image inputs and one image output
    
    Parameters
    ----------
    inputName0: PYME.IO.image.ImageStack
    inputName1: PYME.IO.image.ImageStack
    outputName: PYME.IO.image.ImageStack
    
    """
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
    """
    Add two images
    
    Parameters
    ----------
    inputName0: PYME.IO.image.ImageStack
    inputName1: PYME.IO.image.ImageStack
    outputName: PYME.IO.image.ImageStack
    
    """

    def applyFilter(self, data0, data1, chanNum, i, image0):
        return data0 + data1


@register_module('Subtract')    
class Subtract(ArithmaticFilter):
    """
    Subtract two images. inputName1 is subtracted from inputName0.

    Parameters
    ----------
    inputName0: PYME.IO.image.ImageStack
        image to be subtracted from
    inputName1: PYME.IO.image.ImageStack
        image being subtracted
    outputName: PYME.IO.image.ImageStack
        difference image
    
    """

    def applyFilter(self, data0, data1, chanNum, i, image0):
        return data0 - data1


@register_module('Multiply')    
class Multiply(ArithmaticFilter):
    """
    Multiply two images

    Parameters
    ----------
    inputName0: PYME.IO.image.ImageStack
    inputName1: PYME.IO.image.ImageStack
    outputName: PYME.IO.image.ImageStack
    
    """
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        return data0 * data1


@register_module('Divide')    
class Divide(ArithmaticFilter):
    """
    Divide two images. inputName0 is divided by inputName1.

    Parameters
    ----------
    inputName0: PYME.IO.image.ImageStack
        numerator
    inputName1: PYME.IO.image.ImageStack
        denominator
    outputName: PYME.IO.image.ImageStack
    
    """
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        return data0 / data1


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
        return self.scale * data


@register_module('Normalize')    
class Normalize(Filter):
    """Normalize an image so that the maximum is 1"""
    
    #scale = Float(1)
    
    def applyFilter(self, data, chanNum, i, image0):
        return data / float(data.max())


@register_module('NormalizeMean')    
class NormalizeMean(Filter):
    """Normalize an image so that the mean is 1"""
    
    offset = Float(0)
    
    def applyFilter(self, data, chanNum, i, image0):
        data = data - self.offset
        return data / float(data.mean())
        
        
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
    """
    Perform a bitwise OR on images

    Parameters
    ----------
    inputName0: PYME.IO.image.ImageStack
    inputName1: PYME.IO.image.ImageStack
    outputName: PYME.IO.image.ImageStack
    
    Notes
    -----

    This is actually implemented as :math:`(A + B) > .5`
    """
    
    def applyFilter(self, data0, data1, chanNum, i, image0):
        return (data0 + data1) > .5


@register_module('LogicalAnd')
class LogicalAnd(ArithmaticFilter):
    """
    Perform a logical AND on images

    Parameters
    ----------
    inputName0: PYME.IO.image.ImageStack
    inputName1: PYME.IO.image.ImageStack
    outputName: PYME.IO.image.ImageStack
    """

    def applyFilter(self, data0, data1, chanNum, i, image0):
        return np.logical_and(data0, data1)


@register_module('IsClose')
class IsClose(ArithmaticFilter):
    """
    Wrapper for numpy.isclose

    Parameters
    ----------
    abs_tolerance: Float
        absolute tolerance
    rel_tolerance: Float
        relative tolerance

    Notes
    -----
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

@register_module('Crop')
class Crop(ModuleBase):
    # adapted from PR #831
    input = Input('input')
    x_range = List(Int)([0, -1])
    y_range = List(Int)([0, -1])
    z_range = List(Int)([0, -1])
    t_range = List(Int)([0, -1])
    output = Output('cropped')

    def execute(self, namespace):
        from PYME.IO.DataSources.CropDataSource import crop_image

        im = namespace[self.input]
        cropped = crop_image(im, xrange=slice(*self.x_range), yrange=slice(*self.y_range), zrange=slice(*self.z_range), trange=slice(*self.t_range))
        namespace[self.output] = cropped


@register_module('Redimension')
class Redimension(ModuleBase):
    """
    Interprete input image as one having a different dimension order / slicing

    Used for, e.g. recovering colour dimensions in data saved without metadata correctly indicating dimensionality

    Treats the input data as a flat series of XY slices (using default ordering of the input data) and then interprets into 5D using
    the given dimention ordering.

    Parameters
    -----------

    dim_order : enum, the order of dimensions in the image sequence
    size_z : int, number of z slices
    size_t : int, number of t slices
    size_c : int, number of c slices

    Notes
    -----

    The size_ parameters accept two special values:

    0 : this dimension should be the same size as it is in the input
    -1 : set this dimension automatically based on the total number of slices / the product of the other two dimentsions. Only one of the dimensions can have a size
         of -1 
    """
    input = Input('imput')
    output = Output('redimensioned')

    dim_order = Enum(values=['XYZTC', 'XYTZC', 'XYCZT', 'XYCTZ', 'XYZCT', 'XYTCZ'])
    size_z = Int(-1)
    size_t = Int(1)
    size_c = Int(1)

    def execute(self, namespace):
        from PYME.IO.image import ImageStack
        from PYME.IO.DataSources.BaseDataSource import XYZTCWrapper

        im = namespace[self.input]

        d = XYZTCWrapper(im.data_xyztc)
        d.set_dim_order_and_size(self.dim_order, size_z=self.size_z,size_t=self.size_t, size_c=self.size_c)
        im = ImageStack(data=d, titleStub='Redimensioned')
        
        im.mdh.copyEntriesFrom(getattr(im, 'mdh', {})) 

        namespace[self.output] = d
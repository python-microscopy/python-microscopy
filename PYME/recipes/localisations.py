from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Str, Bool, Int, List, DictStrStr, DictStrFloat, DictStrList, ListFloat, ListStr

import numpy as np
from PYME.IO import tabular
from PYME.LMVis import renderers
import logging
logger = logging.getLogger(__name__)


@register_module('ExtractTableChannel')
class ExtractTableChannel(ModuleBase):
    """Create and return a ColourFilter which has filtered out one colour channel from a table of localizations."""
    inputName = Input('measurements')
    channel = CStr('everything')
    outputName = Output('filtered')

    # def execute(self, namespace):
    #     inp = namespace[self.inputName]

    #     map = tabular.ColourFilter(inp, currentColour=self.channel)

    #     if 'mdh' in dir(inp):
    #         map.mdh = inp.mdh

    #     namespace[self.outputName] = map

    def run(self, inputName):
        return tabular.ColourFilter(inputName, currentColour=self.channel)

    @property
    def _colour_choices(self):
        #try and find the available column names
        try:
            return tabular.ColourFilter.get_colour_chans(self._parent.namespace[self.inputName])
        except:
            return []

    def _view_items(self, params=None):
        from traitsui.api import Item
        from PYME.ui.custom_traits_editors import CBEditor
        return [Item('channel', editor=CBEditor(choices=self._colour_choices)),]
    


@register_module('DensityMapping')
class DensityMapping(ModuleBase):
    """ Use density estimation methods to generate an image from localizations

     """
    inputLocalizations = Input('localizations')
    outputImage = Output('output')
    
    available_renderers = sorted(renderers.RENDERERS.keys())
    
    renderingModule = Enum(available_renderers)#, default_value='Jittered Triangulation')

    pixelSize = Float(5)
    jitterVariable = CStr('1.0')
    jitterScale = Float(1.0)
    jitterVariableZ = CStr('1.0')
    jitterScaleZ = Float(1.0)
    MCProbability = Float(1.0)
    numSamples = Int(10)
    colours = List(Str, ['all',])
    zBoundsMode = Enum(['manual', 'min-max'])
    zBounds = List(Float, [-500, 500])
    zSliceThickness = Float(50.0)
    softRender = Bool(True)
    xyBoundsMode = Enum(['estimate', 'inherit', 'metadata', 'manual'])
    manualXYBounds = List(Float, [0,0,5e3, 5e3])
    

    # def execute(self, namespace):
    #     from PYME.IO.image import ImageBounds
    #     inp = namespace[self.inputLocalizations]
    #     if not isinstance(inp, tabular.ColourFilter):
    #         cf = tabular.ColourFilter(inp, None)
            
    #         print('Created colour filter with chans: %s' % cf.getColourChans())
    #         cf.mdh = inp.mdh
    #     else:
    #         cf = inp
            
    #     #default to taking min and max localizations as image bounds
    #     imb = ImageBounds.estimateFromSource(inp)
        
    #     if self.zBoundsMode == 'min-max':
    #         self.zBounds[0], self.zBounds[1] = float(imb.z0), float(imb.z1)
        
    #     if (self.xyBoundsMode == 'inherit') and not (getattr(inp, 'imageBounds', None) is None):
    #         imb = inp.imageBounds
    #     elif self.xyBoundsMode == 'metadata':
    #         imb = ImageBounds.extractFromMetadata(inp.mdh)
    #     elif self.xyBoundsMode == 'manual':
    #         imb.x0, imb.y0, imb.x1, imb.y1 = self.manualXYBounds
            
    #     cf.imageBounds = imb
        

    #     renderer = renderers.RENDERERS[str(self.renderingModule)](None, cf)

    #     logger.debug('about to generate')

    #     out = renderer.Generate(self.trait_get())

    #     logger.debug(f'generated:{out.data_xyztc.shape}')

    #     namespace[self.outputImage] = out

    def run(self, inputLocalizations):
        from PYME.IO.image import ImageBounds
        inp = inputLocalizations
        if not isinstance(inp, tabular.ColourFilter):
            cf = tabular.ColourFilter(inp, None)
            
            print('Created colour filter with chans: %s' % cf.getColourChans())
            cf.mdh = inp.mdh
        else:
            cf = inp
            
        #default to taking min and max localizations as image bounds
        imb = ImageBounds.estimateFromSource(inp)
        
        if self.zBoundsMode == 'min-max':
            self.zBounds[0], self.zBounds[1] = float(imb.z0), float(imb.z1)
        
        if (self.xyBoundsMode == 'inherit') and not (getattr(inp, 'imageBounds', None) is None):
            imb = inp.imageBounds
        elif self.xyBoundsMode == 'metadata':
            imb = ImageBounds.extractFromMetadata(inp.mdh)
        elif self.xyBoundsMode == 'manual':
            imb.x0, imb.y0, imb.x1, imb.y1 = self.manualXYBounds
            
        cf.imageBounds = imb
        

        renderer = renderers.RENDERERS[str(self.renderingModule)](None, cf)

        logger.debug('about to generate')

        out = renderer.Generate(self.trait_get())

        logger.debug(f'generated:{out.data_xyztc.shape}')

        return out
        
    def _view_items(self, params=None):
        from traitsui.api import Group, Item, CSVListEditor
        return [Item('renderingModule'),
                    Item('pixelSize'),
                    Item('colours', style='text',editor=CSVListEditor(auto_set=False, enter_set=True)),
                    #Item('softRender'),
                    Group(
                        Item('jitterVariable'),
                        Item('jitterScale'),
                        Item('jitterVariableZ', visible_when='"3D" in renderingModule'),
                        Item('jitterScaleZ', visible_when='"3D" in renderingModule'),
                        Item('numSamples', visible_when='"Triangulation" in renderingModule'),
                        Item('MCProbability', visible_when='"Triangulation" in renderingModule'),
                        label='Jittering/Gaussian Size', visible_when='not (("Histogram" in renderingModule) or (renderingModule=="Current"))'),
                    Group(
                        Item('zSliceThickness'),
                        Item('zBoundsMode'),
                        Item('zBounds', visible_when='zBoundsMode=="manual"',editor=CSVListEditor(auto_set=False, enter_set=True)),
                        label='3D', visible_when='"3D" in renderingModule'),
                    Group(
                        Item('xyBoundsMode'),
                        Item('manualXYBounds', visible_when='xyBoundsMode=="manual"',editor=CSVListEditor(auto_set=False, enter_set=True)),
                        label='Output Image Size',
                    ),
                ]

@register_module('AddPipelineDerivedVars')
class Pipelineify(ModuleBase):
    """
    Perform standard mappings, including those derived from acquisition events.

    Parameters
    ----------
    inputFitResults : string - the name of a tabular.TabularBase object
        Typically the FitResults table of an h5r file
    inputEvents : string - name of a tabular.TabularBase object containing acquisition events [optional]
        This is not usually required as the IO methods attach `.events` as a datasource attribute. Use when events come
        from a separate file or when there are intervening processing steps between IO and this module (the `.events`
        attribute does not propagate through recipe modules).
        TODO - do we really want to be attaching events as an attribute or should they be there own entry in the recipe namespace
        TODO - should we change this to the processed events???
    pixelSizeNM : float
        Scaling factor to get 'x' and 'y' into units of nanometers. Useful if handling external data input in pixel units. Defaults to 1.
    
    Returns
    -------
    outputLocalizations : tabular.MappingFilter
    
    """
    inputFitResults = Input('FitResults')
    inputEvents = Input('')
    
    # Fiducial table input
    # inputDriftResults = Input('')
    # TODO - to replicate the pipeline input processing, we should take inputFitResults and inputDriftResults and output
    # 'Localisations' and 'Fiducials' (the fiducials get some, but not all of the manipulations and extra columns). Should
    # we expand this module, or pass the fiducials though in the same way as the fit results, living with the fact that
    # there will be extra columns?

    pixelSizeNM = Float(1, label='nanometer units',
                        desc="scaling factor to get 'x' and 'y' into units of nanometers. Useful if handling external data input in pixel units")

    outputLocalizations = Output('Localizations')
    
    # def execute(self, namespace):
    #     from PYME.LMVis import pipeline
    #     fitResults = namespace[self.inputFitResults]
    #     mdh = fitResults.mdh

    #     mapped_ds = tabular.MappingFilter(fitResults)


    #     if not self.pixelSizeNM == 1: # TODO - check close instead?
    #         mapped_ds.addVariable('pixelSize', self.pixelSizeNM)
    #         mapped_ds.setMapping('x', 'x*pixelSize')
    #         mapped_ds.setMapping('y', 'y*pixelSize')

    #     #extract information from any events
    #     if self.inputEvents != '':
    #         # Use specified table for events if given (otherwise look for a `.events` attribute on the input data
    #         # TODO: resolve how best to handle events (i.e. should they be a separate table, or should they be attached to data tables)
    #         events = namespace.get(self.inputEvents, None)
    #     else:
    #         try:
    #             events = fitResults.events
    #         except AttributeError:
    #             logger.debug('no events found')
    #             events = None
        
    #     if isinstance(events, tabular.TabularBase):
    #         events = events.to_recarray()

    #     ev_maps, ev_charts = pipeline._processEvents(mapped_ds, events, mdh)
    #     pipeline._add_missing_ds_keys(mapped_ds, ev_maps)

    #     #Fit module specific filter settings
    #     if 'Analysis.FitModule' in mdh.getEntryNames():
    #         fitModule = mdh['Analysis.FitModule']

    #         if 'LatGaussFitFR' in fitModule:
    #             # TODO - move getPhotonNums() out of pipeline
    #             mapped_ds.addColumn('nPhotons', pipeline.getPhotonNums(mapped_ds, mdh))
            
    #         if 'SplitterFitFNR' in fitModule:
    #             mapped_ds.addColumn('nPhotonsg', pipeline.getPhotonNums({'A': mapped_ds['fitResults_Ag'], 'sig': mapped_ds['fitResults_sigma']}, mdh))
    #             mapped_ds.addColumn('nPhotonsr', pipeline.getPhotonNums({'A': mapped_ds['fitResults_Ar'], 'sig': mapped_ds['fitResults_sigma']}, mdh))
    #             mapped_ds.setMapping('nPhotons', 'nPhotonsg+nPhotonsr')

    #     mapped_ds.mdh = mdh

    #     namespace[self.outputLocalizations] = mapped_ds

    def run(self, inputFitResults, inputEvents=None):
        from PYME.LMVis import pipeline
        from PYME.IO import MetaDataHandler
        fitResults = inputFitResults
        mdh = MetaDataHandler.DictMDHandler(fitResults.mdh)

        mapped_ds = tabular.MappingFilter(fitResults)


        if not self.pixelSizeNM == 1: # TODO - check close instead?
            mapped_ds.addVariable('pixelSize', self.pixelSizeNM)
            mapped_ds.setMapping('x', 'x*pixelSize')
            mapped_ds.setMapping('y', 'y*pixelSize')

        #extract information from any events
        if inputEvents:
            events = inputEvents
        else:
            try:
                events = fitResults.events
            except AttributeError:
                logger.debug('no events found')
                events = None
        
        if isinstance(events, tabular.TabularBase):
            events = events.to_recarray()

        ev_maps, ev_charts = pipeline._processEvents(mapped_ds, events, mdh)
        pipeline._add_missing_ds_keys(mapped_ds, ev_maps)

        #Fit module specific filter settings
        if 'Analysis.FitModule' in mdh.getEntryNames():
            fitModule = mdh['Analysis.FitModule']

            if 'LatGaussFitFR' in fitModule:
                # TODO - move getPhotonNums() out of pipeline
                mapped_ds.addColumn('nPhotons', pipeline.getPhotonNums(mapped_ds, mdh))
            
            if 'SplitterFitFNR' in fitModule:
                mapped_ds.addColumn('nPhotonsg', pipeline.getPhotonNums({'A': mapped_ds['fitResults_Ag'], 'sig': mapped_ds['fitResults_sigma']}, mdh))
                mapped_ds.addColumn('nPhotonsr', pipeline.getPhotonNums({'A': mapped_ds['fitResults_Ar'], 'sig': mapped_ds['fitResults_sigma']}, mdh))
                mapped_ds.setMapping('nPhotons', 'nPhotonsg+nPhotonsr')

        mapped_ds.mdh = mdh

        return mapped_ds
        
@register_module("ProcessColour")
class ProcessColour(ModuleBase):
    input = Input('localizations')
    output = Output('colour_mapped')
    
    # ratios  & dyes for ratiometric colour - maps species name to floating point ratio
    # Note that these get filled in automatically from the metadata but will get saved with the recipe when the recipe
    # is saved. If you want a generic recipe, you will need to manually remove the dye entries from the .yaml file
    # TODO - do this automatically somehow?
    # TODO - default is to override saved values with those from metadata. Change this?
    species_ratios = DictStrFloat()
    species_dyes = DictStrStr()
    
    ratios_from_metadata = Bool(True)

    def _get_dye_ratios_from_metadata(self, mdh):
        from PYME.LMVis import dyeRatios
        
        labels = mdh.getOrDefault('Sample.Labelling', [])
        seen_structures = []
    
        for structure, dye in labels:
            #info might be unicode - convert to a standard string to keep traits happy
            structure = str(structure)
            dye = str(dye)
            
            if structure in seen_structures:
                strucname = structure + '_1'
            else:
                strucname = structure
            seen_structures.append(structure)
        
            ratio = dyeRatios.getRatio(dye, mdh)
        
            if not ratio is None:
                self.species_ratios[strucname] = ratio
                self.species_dyes[strucname] = dye
    
    # def execute(self, namespace):
    #     input = namespace[self.input]
    #     mdh = input.mdh
        
    #     if self.ratios_from_metadata:
    #         # turn off invalidation so we don't get a recursive loop. TODO - fix this properly as it's gross to be changing
    #         # our parameters here
    #         invalidate = self._invalidate_parent
    #         self._invalidate_parent = False
    #         self._get_dye_ratios_from_metadata(mdh)
    #         self._invalidate_parent = invalidate
        
    #     output = tabular.MappingFilter(input)
    #     output.mdh = mdh
    
    #     if 'gFrac' in output.keys():
    #         #ratiometric
    #         #raise NotImplementedError('Ratiometric processing in recipes not implemented yet')
    #         for structure, ratio in self.species_ratios.items():
    #             if not ratio is None:
    #                 output.setMapping('p_%s' % structure,
    #                                         'exp(-(%f - gFrac)**2/(2*error_gFrac**2))/(error_gFrac*sqrt(2*numpy.pi))' % ratio)
    #     else:
    #         if 'probe' in output.keys():
    #             #non-ratiometric (i.e. sequential) colour
    #             #color channel is given in 'probe' column
    #             output.setMapping('ColourNorm', '1.0 + 0*probe')
                
    #             for i in range(int(output['probe'].min()), int(output['probe'].max() + 1)):
    #                 output.setMapping('p_chan%d' % i, '1.0*(probe == %d)' % i)
            
    #         nSeqCols = mdh.getOrDefault('Protocol.NumberSequentialColors', 1)
    #         if nSeqCols > 1:
    #             for i in range(nSeqCols):
    #                 output.setMapping('ColourNorm', '1.0 + 0*t')
    #                 cr = mdh['Protocol.ColorRange%d' % i]
    #                 output.setMapping('p_chan%d' % i, '(t>= %d)*(t<%d)' % cr)
                    
    #     cached_output = tabular.CachingResultsFilter(output)
    #     #cached_output.mdh = output.mdh
    #     namespace[self.output] = cached_output

    def run(self, input):
        mdh = input.mdh
        
        if self.ratios_from_metadata:
            # turn off invalidation so we don't get a recursive loop. TODO - fix this properly as it's gross to be changing
            # our parameters here
            invalidate = self._invalidate_parent
            self._invalidate_parent = False
            self._get_dye_ratios_from_metadata(mdh)
            self._invalidate_parent = invalidate
        
        output = tabular.MappingFilter(input)
        output.mdh = mdh
    
        if 'gFrac' in output.keys():
            #ratiometric
            #raise NotImplementedError('Ratiometric processing in recipes not implemented yet')
            for structure, ratio in self.species_ratios.items():
                if not ratio is None:
                    output.setMapping('p_%s' % structure,
                                            'exp(-(%f - gFrac)**2/(2*error_gFrac**2))/(error_gFrac*sqrt(2*numpy.pi))' % ratio)
        else:
            if 'probe' in output.keys():
                #non-ratiometric (i.e. sequential) colour
                #color channel is given in 'probe' column
                output.setMapping('ColourNorm', '1.0 + 0*probe')
                
                for i in range(int(output['probe'].min()), int(output['probe'].max() + 1)):
                    output.setMapping('p_chan%d' % i, '1.0*(probe == %d)' % i)
            
            nSeqCols = mdh.getOrDefault('Protocol.NumberSequentialColors', 1)
            if nSeqCols > 1:
                for i in range(nSeqCols):
                    output.setMapping('ColourNorm', '1.0 + 0*t')
                    cr = mdh['Protocol.ColorRange%d' % i]
                    output.setMapping('p_chan%d' % i, '(t>= %d)*(t<%d)' % cr)
                    
        cached_output = tabular.CachingResultsFilter(output)
        #cached_output.mdh = output.mdh
        return cached_output


@register_module('TimeBlocks')
class TimeBlocks(ModuleBase):
    """

    Divides series into alternating time blocks to generate 2 fake colour channels for Fourier Ring / Fourier shell correlation.
     
    This is probably a better approach than taking random subsets as the later will tend to generate unrealistically high
    correlation values for repeated localizations.
    
    Adapted from Christian Soeller's 'splitRender' implementation.
    """
    input = Input('localizations')
    output = Output('time_blocks')
    
    block_size = Int(100)
    
    # def execute(self, namespace):
    #     input = namespace[self.input]
    #     mdh = input.mdh
    
    #     output = tabular.MappingFilter(input)
    #     output.mdh = mdh
        
    #     output.addColumn('block_id', np.mod((output['t']/self.block_size).astype('int'),2))

    #     channel_names = [k for k in input.keys() if k.startswith('p_')]
        
    #     print(channel_names)
    #     print(input.keys())
        
    #     if len(channel_names) == 0:
    #         #single channel data - no channels defined.
    #         output.setMapping('ColourNorm', '1.0 + 0*t')
    #         output.setMapping('p_block0', '1.0*block_id')
    #         output.setMapping('p_block1', '1.0 - block_id')
    #     else:
    #         #have colour channels - subdivide them
    #         for k in channel_names:
    #             output.setMapping('%s_block0' % k, '%s*block_id' % k)
    #             output.setMapping('%s_block1' % k, '%s*(1.0 - block_id)' % k)
            
    #         #hide original channel names
    #         output.hidden_columns.extend(channel_names)
        

    #     namespace[self.output] = output

    def run(self, input):
        output = tabular.MappingFilter(input)
        
        output.addColumn('block_id', np.mod((output['t']/self.block_size).astype('int'),2))

        channel_names = [k for k in input.keys() if k.startswith('p_')]
        
        #print(channel_names)
        #print(input.keys())
        
        if len(channel_names) == 0:
            #single channel data - no channels defined.
            output.setMapping('ColourNorm', '1.0 + 0*t')
            output.setMapping('p_block0', '1.0*block_id')
            output.setMapping('p_block1', '1.0 - block_id')
        else:
            #have colour channels - subdivide them
            for k in channel_names:
                output.setMapping('%s_block0' % k, '%s*block_id' % k)
                output.setMapping('%s_block1' % k, '%s*(1.0 - block_id)' % k)
            
            #hide original channel names
            output.hidden_columns.extend(channel_names)

        return output
    

@register_module('MergeClumps')
class MergeClumps(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('clumped')
    outputName = Output('merged')
    labelKey = CStr('clumpIndex')
    discardTrivial = Bool(False)

    # def execute(self, namespace):
    #     from PYME.Analysis.points.DeClump import pyDeClump

    #     inp = namespace[self.inputName]
    #     grouped = pyDeClump.mergeClumps(inp, labelKey=self.labelKey)
    #     try:
    #         grouped.mdh = inp.mdh
    #     except AttributeError:
    #         pass
    #    grouped = pyDeClump.mergeClumps(inp, labelKey=self.labelKey, discard_trivial=self.discardTrivial)
    #    try:
    #        grouped.mdh = inp.mdh
    #    except AttributeError:
    #        pass

    #     namespace[self.outputName] = grouped

    def run(self, inputName):
        from PYME.Analysis.points.DeClump import pyDeClump
        return pyDeClump.mergeClumps(inp, labelKey=self.labelKey, discard_trivial=self.discardTrivial)


@register_module('IDTransientFrames')
class IDTransientFrames(ModuleBase): #FIXME - move to multi-view specific module and potentially rename (depending on whether we introduce scoping)
    """
    Adds an 'isTransient' column to the input datasource so that one can filter localizations that are from frames
    acquired during z-translation
    """
    inputName = Input('zmapped')
    inputEvents = Input('Events')
    framesPerStep = Float()
    outputName = Output('transientFiltered')

    # def execute(self, namespace):
    #     from PYME.experimental import zMotionArtifactUtils

    #     inp = namespace[self.inputName]

    #     mapped = tabular.MappingFilter(inp)

    #     if 'mdh' not in dir(inp):
    #         if self.framesPerStep <= 0:
    #             raise RuntimeError('idTransientFrames needs metadata')
    #         else:
    #             fps = self.framesPerStep
    #     else:
    #         fps = inp.mdh['StackSettings.FramesPerStep']

    #     mask = zMotionArtifactUtils.flagMotionArtifacts(mapped, namespace[self.inputEvents], fps)
    #     mapped.addColumn('piezoUnstable', mask)

    #     mapped.mdh = inp.mdh

    #     namespace[self.outputName] = mapped

    def run(self, inputName, inputEvents):
        from PYME.experimental import zMotionArtifactUtils

        mapped = tabular.MappingFilter(inputName)

        if 'mdh' not in dir(inputName):
            if self.framesPerStep <= 0:
                raise RuntimeError('idTransientFrames needs metadata')
            else:
                fps = self.framesPerStep
        else:
            fps = inputName.mdh['StackSettings.FramesPerStep']

        mask = zMotionArtifactUtils.flagMotionArtifacts(mapped, inputEvents, fps)
        mapped.addColumn('piezoUnstable', mask)

        return mapped

@register_module('DBSCANClustering')
class DBSCANClustering(ModuleBase):
    """
    Performs DBSCAN clustering on input dictionary

    Parameters
    ----------
    searchRadius : float
        search radius for clustering [nm]
    minPtsForCore : int
        number of points within SearchRadius required for a given point to be 
        considered a core point

    Notes
    -----

    See `sklearn.cluster.dbscan` for more details about the underlying 
    algorithm and parameter meanings.

    """
    import multiprocessing
    inputName = Input('filtered')

    columns = ListStr(['x', 'y', 'z'])
    searchRadius = Float(10)
    minClumpSize = Int(1)
    
    #exposes sklearn parallelism. Recipe modules are generally assumed
    #to be single-threaded. Enable at your own risk
    multithreaded = Bool(False)
    numberOfJobs = Int(max(multiprocessing.cpu_count()-1,1))
    
    clumpColumnName = CStr('dbscanClumpID')

    outputName = Output('dbscanClustered')

    # def execute(self, namespace):
    #     from sklearn.cluster import dbscan

    #     inp = namespace[self.inputName]
    #     mapped = tabular.MappingFilter(inp)

    #     # Note that sklearn gives unclustered points label of -1, and first value starts at 0.
    #     if self.multithreaded:
    #         core_samp, dbLabels = dbscan(np.vstack([inp[k] for k in self.columns]).T,
    #                                      eps=self.searchRadius, min_samples=self.minClumpSize, n_jobs=self.numberOfJobs)
    #     else:
    #         #NB try-catch from Christians multithreaded example removed as I think we should see failure here
    #         core_samp, dbLabels = dbscan(np.vstack([inp[k] for k in self.columns]).T,
    #                                  eps=self.searchRadius, min_samples=self.minClumpSize)

    #     # shift dbscan labels up by one to match existing convention that a clumpID of 0 corresponds to unclumped
    #     mapped.addColumn(str(self.clumpColumnName), dbLabels + 1)

    #     # propogate metadata, if present
    #     try:
    #         mapped.mdh = inp.mdh
    #     except AttributeError:
    #         pass

    #     namespace[self.outputName] = mapped

    def run(self, inputName):
        from sklearn.cluster import dbscan

        inp = inputName
        mapped = tabular.MappingFilter(inp)

        # Note that sklearn gives unclustered points label of -1, and first value starts at 0.
        if self.multithreaded:
            core_samp, dbLabels = dbscan(np.vstack([inp[k] for k in self.columns]).T,
                                         eps=self.searchRadius, min_samples=self.minClumpSize, n_jobs=self.numberOfJobs)
        else:
            #NB try-catch from Christians multithreaded example removed as I think we should see failure here
            core_samp, dbLabels = dbscan(np.vstack([inp[k] for k in self.columns]).T,
                                     eps=self.searchRadius, min_samples=self.minClumpSize)

        # shift dbscan labels up by one to match existing convention that a clumpID of 0 corresponds to unclumped
        mapped.addColumn(str(self.clumpColumnName), dbLabels + 1)

        return mapped

    @property
    def hide_in_overview(self):
        return ['columns']
        
    def _view_items(self, params=None):
        from traitsui.api import Item, TextEditor
        return [Item('columns', editor=TextEditor(auto_set=False, enter_set=True, evaluate=ListStr)),
                    Item('searchRadius'),
                    Item('minClumpSize'),
                    Item('multithreaded'),
                    Item('numberOfJobs'),
                    Item('clumpColumnName'),]



#TODO - this is very specialized and probably doesn't belong here - at least not in this form
@register_module('ClusterCountVsImagingTime')
class ClusterCountVsImagingTime(ModuleBase):
    """
    WARNING: This module will likely move, dissapear, or be refactored

    ClusterCountVsImagingTime iteratively filters a dictionary-like object on t, and at each step counts the number of
    labeled objects (e.g. DBSCAN clusters) which contain at least N-points. It does this for two N-points, so one can be
    set according to density with all frames included, and the other can be set for one of the earlier frame-counts.

    args:
        stepSize: number of frames to add in on each iteration
        labelsKey: key containing labels for each localization
        lowerMinPtsPerCluster:
        higherMinPtsPerCluster:

    returns:
        dictionary-like object with the following keys:
            t: upper bound on frame number included in calculations on each iteration.
            N_labelsWithLowMinPoints:
            N_labelsWithHighMinPoints:

    From wikipedia: "While minPts intuitively is the minimum cluster size, in some cases DBSCAN can produce smaller
    clusters. A DBSCAN cluster consists of at least one core point. As other points may be border points to more than
    one cluster, there is no guarantee that at least minPts points are included in every cluster."
    """
    inputName = Input('input')

    labelsKey = CStr('dbscanClumpID')
    lowerMinPtsPerCluster = Int(3)
    higherMinPtsPerCluster = Int(6)
    stepSize = Int(3000)

    outputName = Output('incremented')

    # def execute(self, namespace):
    #     from PYME.IO import tabular

    #     if self.lowerMinPtsPerCluster > self.higherMinPtsPerCluster:
    #         print('Swapping low and high MinPtsPerCluster - input was reversed')
    #         temp = self.lowerMinPtsPerCluster
    #         self.lowerMinPtsPerCluster = self.higherMinPtsPerCluster
    #         self.higherMinPtsPerCluster = temp

    #     iters = (int(np.max(namespace[self.inputName]['t']))/int(self.stepSize)) + 2

    #     # other counts
    #     lowDensMinPtsClumps = np.empty(iters)
    #     lowDensMinPtsClumps[0] = 0
    #     hiDensMinPtsClumps = np.empty(iters)
    #     hiDensMinPtsClumps[0] = 0
    #     t = np.empty(iters)
    #     t[0] = 0

    #     inp = tabular.MappingFilter(namespace[self.inputName])

    #     for ind in range(1, iters):  # start from 1 since t=[0,0] will yield no clumps
    #         # filter time
    #         inc = tabular.ResultsFilter(inp, t=[0, self.stepSize * ind])
    #         t[ind] = np.max(inc['t'])

    #         cid, counts = np.unique(inc[self.labelsKey], return_counts=True)
    #         # cmask = np.in1d(inc['DBSCAN_allFrames'], cid)

    #         cidL = cid[counts >= self.lowerMinPtsPerCluster]
    #         lowDensMinPtsClumps[ind] = np.sum(cidL != -1)  # ignore unclumped in count
    #         cid = cid[counts >= self.higherMinPtsPerCluster]
    #         hiDensMinPtsClumps[ind] = np.sum(cid != -1)  # ignore unclumped in count


    #     res = tabular.MappingFilter({'t': t,
    #                                  'N_labelsWithLowMinPoints': lowDensMinPtsClumps,
    #                                  'N_labelsWithHighMinPoints': hiDensMinPtsClumps})

    #     # propagate metadata, if present
    #     try:
    #         res.mdh = namespace[self.inputName].mdh
    #     except AttributeError:
    #         pass

    #     namespace[self.outputName] = res

    def run(self, inputName):
        from PYME.IO import tabular

        inp = tabular.MappingFilter(inputName)
        
        if self.lowerMinPtsPerCluster > self.higherMinPtsPerCluster:
            print('Swapping low and high MinPtsPerCluster - input was reversed')
            temp = self.lowerMinPtsPerCluster
            self.lowerMinPtsPerCluster = self.higherMinPtsPerCluster
            self.higherMinPtsPerCluster = temp

        iters = (int(np.max(inp['t']))/int(self.stepSize)) + 2

        # other counts
        lowDensMinPtsClumps = np.empty(iters)
        lowDensMinPtsClumps[0] = 0
        hiDensMinPtsClumps = np.empty(iters)
        hiDensMinPtsClumps[0] = 0
        t = np.empty(iters)
        t[0] = 0


        for ind in range(1, iters):  # start from 1 since t=[0,0] will yield no clumps
            # filter time
            inc = tabular.ResultsFilter(inp, t=[0, self.stepSize * ind])
            t[ind] = np.max(inc['t'])

            cid, counts = np.unique(inc[self.labelsKey], return_counts=True)
            # cmask = np.in1d(inc['DBSCAN_allFrames'], cid)

            cidL = cid[counts >= self.lowerMinPtsPerCluster]
            lowDensMinPtsClumps[ind] = np.sum(cidL != -1)  # ignore unclumped in count
            cid = cid[counts >= self.higherMinPtsPerCluster]
            hiDensMinPtsClumps[ind] = np.sum(cid != -1)  # ignore unclumped in count


        return tabular.MappingFilter({'t': t,
                                     'N_labelsWithLowMinPoints': lowDensMinPtsClumps,
                                     'N_labelsWithHighMinPoints': hiDensMinPtsClumps})


@register_module('LabelsFromImage')
class LabelsFromImage(ModuleBase):
    """
    Maps each point in the input table to a pixel in a labelled image, and extracts the pixel value at that location to
    use as a label for the point data. 

    Inputs
    ------
    inputName: Input
        name of tabular input containing positions ('x', 'y', and optionally 'z' columns should be present)
    inputImage: Input
        name of image input containing labels

    Outputs
    -------
    outputName: Output
        name of tabular output. A mapped version of the tabular input with 2 extra columns
    label_key_name : CStr
        name of new column which will contain the label number from image, mapped to each localization within that label
    label_count_key_name : CStr
        name of new column which will contain the number of localizations within the label that a given localization
        belongs to
    minimum_localizations: Int
        threshold for the number of localizations required to propagate a label through to localizations

    """
    inputName = Input('input')
    inputImage = Input('labeled')

    label_key_name = CStr('objectID')
    label_count_key_name = CStr('NEvents')

    minimum_localizations = Int(1)

    outputName = Output('labeled_points')

    # def execute(self, namespace):
    #     from PYME.IO import tabular
    #     from PYME.Analysis.points import cluster_morphology

    #     inp = namespace[self.inputName]
    #     img = namespace[self.inputImage]

    #     ids, numPerObject = cluster_morphology.get_labels_from_image(img, inp, minimum_localizations=self.minimum_localizations)

    #     labeled = tabular.MappingFilter(inp)
    #     labeled.addColumn(self.label_key_name, ids)
    #     labeled.addColumn(self.label_count_key_name, numPerObject[ids - 1])

    #     # propagate metadata, if present
    #     try:
    #         labeled.mdh = namespace[self.inputName].mdh
    #     except AttributeError:
    #         pass

    #     namespace[self.outputName] = labeled

    def run(self, inputName, inputImage):
        from PYME.IO import tabular
        from PYME.Analysis.points import cluster_morphology

        ids, numPerObject = cluster_morphology.get_labels_from_image(inputImage, inputName, minimum_localizations=self.minimum_localizations)

        labeled = tabular.MappingFilter(inputName)
        labeled.addColumn(self.label_key_name, ids)
        labeled.addColumn(self.label_count_key_name, numPerObject[ids - 1])

        return labeled


@register_module('MeasureClusters3D')
class MeasureClusters3D(ModuleBase):
    """
    Measures the 3D morphology of clusters of points

    Inputs
    ------

    inputName : name of tabular data containing x, y, and z columns and labels identifying which cluster each point
                belongs to.

    Outputs
    -------

    outputName: a new tabular data source containing measurements of the clusters
    
    Parameters
    ----------
        labelKey: name of column to use as a label identifying clusters

    Notes
    -----

    Measures calculated (to be expanded)
    --------------------------------------
        count : int
            Number of localizations (points) in the cluster
        x : float
            x center of mass
        y : float
            y center of mass
        z : float
            z center of mass
        gyrationRadius : float
            root mean square displacement to center of cluster, a measure of compaction or spatial extent see also
            supplemental text of DOI: 10.1038/nature16496
        axis0 : ndarray, shape (3,)
            principle axis which accounts for the largest variance of the cluster, i.e. corresponds to the largest
            eigenvalue
        axis1 : ndarray, shape (3,)
            next principle axis
        axis2 : ndarray, shape (3,)
            principle axis corresponding to the smallest eigenvalue
        sigma0 : float
            standard deviation along axis0
        sigma1 : float
            standard deviation along axis1
        sigma2 : float
            standard deviation along axis2
        anisotropy : float
            metric of anisotropy based on the spread along principle axes. Standard deviations of alpha * [1, 0, 0],
            where alpha is a scalar, will result in an 'anisotropy' value of 1, i.e. maximally anisotropic. Completely
            isotropic clusters will have equal standard deviations, i.e. alpha * [1, 1, 1], which corresponds to an
            'anisotropy' value of 0. Intermediate cases result in values between 0 and 1.
        theta : float
            Azimuthal angle, in radians, along which the principle axis (axis0) points
        phi : float
            Zenith angle, in radians, along which the principle axis (axis0) points

    """
    inputName = Input('input')
    labelKey = CStr('clumpIndex')

    outputName = Output('clusterMeasures')

    # def execute(self, namespace):
    #     from PYME.Analysis.points import cluster_morphology as cmorph
    #     import numpy as np

    #     inp = namespace[self.inputName]

    #     labels = inp[self.labelKey].astype(np.int)
    #     # make sure labeling scheme is consistent with what pyme conventions
    #     if (len(labels) > 0) and  (labels.min() < 0):
    #         raise UserWarning('This module expects 0-label for unclustered points, and no negative labels')
        
    #     I = np.argsort(labels)
    #     I = I[labels[I] > 0]
        
    #     x_vals, y_vals, z_vals = inp['x'][I], inp['y'][I], inp['z'][I]
    #     labels = labels[I]
    #     maxLabel = labels[-1] if (len(labels) > 0) else 0
        
    #     #find the unique labels, and their separation in the sorted list of points
    #     unique_labels, counts = np.unique(labels, return_counts=True)
        
    #     #allocate memory to store results in
    #     measurements = np.zeros(maxLabel, cmorph.measurement_dtype)

    #     # loop over labels, recalling that input is now sorted, and we know how many points are in each label.
    #     # Note that missing labels result in zeroed entries (i.e. the initial values are not changed).
    #     # Missing values can be filtered out later, if desired, by filtering on the 'counts' column, but having a dense
    #     # array where index == label number makes any postprocessing in which we might want to find the data
    #     # corresponding to a particular label MUCH easier and faster.
    #     indi = 0
    #     for label_num, ct in zip(unique_labels, counts):
    #         indf = indi + ct

    #         # create x,y,z arrays for this cluster, and calculate center of mass
    #         x, y, z = x_vals[indi:indf], y_vals[indi:indf], z_vals[indi:indf]

    #         cluster_index = label_num - 1  # we ignore the unclustered points, and start labeling at 1
    #         cmorph.measure_3d(x, y, z, output=measurements[cluster_index])

    #         indi = indf

    #     meas = tabular.RecArraySource(measurements)

    #     try:
    #         meas.mdh = namespace[self.inputName].mdh
    #     except AttributeError:
    #         pass

    #     namespace[self.outputName] = meas


    def run(self, inputName):
        from PYME.Analysis.points import cluster_morphology as cmorph
        import numpy as np

        inp = inputName

        labels = inp[self.labelKey].astype(np.int)
        # make sure labeling scheme is consistent with what pyme conventions
        if (len(labels) > 0) and  (labels.min() < 0):
            raise UserWarning('This module expects 0-label for unclustered points, and no negative labels')
        
        I = np.argsort(labels)
        I = I[labels[I] > 0]
        
        x_vals, y_vals, z_vals = inp['x'][I], inp['y'][I], inp['z'][I]
        labels = labels[I]
        maxLabel = labels[-1] if (len(labels) > 0) else 0
        
        #find the unique labels, and their separation in the sorted list of points
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        #allocate memory to store results in
        measurements = np.zeros(maxLabel, cmorph.measurement_dtype)

        # loop over labels, recalling that input is now sorted, and we know how many points are in each label.
        # Note that missing labels result in zeroed entries (i.e. the initial values are not changed).
        # Missing values can be filtered out later, if desired, by filtering on the 'counts' column, but having a dense
        # array where index == label number makes any postprocessing in which we might want to find the data
        # corresponding to a particular label MUCH easier and faster.
        indi = 0
        for label_num, ct in zip(unique_labels, counts):
            indf = indi + ct

            # create x,y,z arrays for this cluster, and calculate center of mass
            x, y, z = x_vals[indi:indf], y_vals[indi:indf], z_vals[indi:indf]

            cluster_index = label_num - 1  # we ignore the unclustered points, and start labeling at 1
            cmorph.measure_3d(x, y, z, output=measurements[cluster_index])

            indi = indf

        return tabular.RecArraySource(measurements)


@register_module('FiducialCorrection')
class FiducialCorrection(ModuleBase):
    """
    Maps each point in the input table to a pixel in a labelled image, and extracts the pixel value at that location to
    use as a label for the point data.

    Inputs
    ------
    inputName: name of tabular input containing positions ('x', 'y', and optionally 'z' columns should be present)
    inputImage: name of image input containing labels

    Outputs
    -------
    outputName: name of tabular output. A mapped version of the tabular input with 2 extra columns
        objectID: Label number from image, mapped to each localization within that label
        NEvents: Number of localizations within the label that a given localization belongs to

    """
    inputLocalizations = Input('Localizations')
    inputFiducials = Input('Fiducials')

    clumpRadiusVar = CStr('error_x')
    clumpRadiusMultiplier = Float(5.0)
    timeWindow = Int(25)
    
    temporalFilter = Enum(['Gaussian', 'Uniform', 'Median'])
    temporalFilterScale = Float(10.0)

    outputName = Output('corrected_localizations')
    outputFiducials = Output('corrected_fiducials')

    # def execute(self, namespace):
    #     from PYME.IO import tabular
    #     from PYME.Analysis.points import fiducials

    #     locs = namespace[self.inputLocalizations]
    #     fids = namespace[self.inputFiducials]
        
    #     t_fid, fid_trajectory, clump_index = fiducials.extractAverageTrajectory(fids, clumpRadiusVar=self.clumpRadiusVar,
    #                                                     clumpRadiusMultiplier=float(self.clumpRadiusMultiplier),
    #                                                     timeWindow=int(self.timeWindow),
    #                                                     filter=self.temporalFilter, filterScale=float(self.temporalFilterScale))
        
    #     out = tabular.MappingFilter(locs)
    #     t_out = out['t']

    #     out_f = tabular.MappingFilter(fids)
    #     out_f.addColumn('clumpIndex', clump_index)
    #     t_out_f = out_f['t']

    #     for dim in fid_trajectory.keys():
    #         print(dim)
    #         out.addColumn('fiducial_{0}'.format(dim), np.interp(t_out, t_fid, fid_trajectory[dim]))
    #         out.setMapping(dim, '{0} - fiducial_{0}'.format(dim))

    #         out_f.addColumn('fiducial_{0}'.format(dim), np.interp(t_out_f, t_fid, fid_trajectory[dim]))
    #         out_f.setMapping(dim, '{0} - fiducial_{0}'.format(dim))

    #     # propagate metadata, if present
    #     try:
    #         out.mdh = locs.mdh
    #     except AttributeError:
    #         pass

    #     namespace[self.outputName] = out
    #     namespace[self.outputFiducials] = out_f

    def run(self, inputLocalizations, inputFiducials):
        from PYME.IO import tabular
        from PYME.Analysis.points import fiducials
        from PYME.IO import MetaDataHandler
        
        t_fid, fid_trajectory, clump_index = fiducials.extractAverageTrajectory(inputFiducials, clumpRadiusVar=self.clumpRadiusVar,
                                                        clumpRadiusMultiplier=float(self.clumpRadiusMultiplier),
                                                        timeWindow=int(self.timeWindow),
                                                        filter=self.temporalFilter, filterScale=float(self.temporalFilterScale))
        
        out = tabular.MappingFilter(inputLocalizations)
        t_out = out['t']

        out_f = tabular.MappingFilter(inputFiducials)
        out_f.addColumn('clumpIndex', clump_index)
        t_out_f = out_f['t']

        for dim in fid_trajectory.keys():
            print(dim)
            out.addColumn('fiducial_{0}'.format(dim), np.interp(t_out, t_fid, fid_trajectory[dim]))
            out.setMapping(dim, '{0} - fiducial_{0}'.format(dim))

            out_f.addColumn('fiducial_{0}'.format(dim), np.interp(t_out_f, t_fid, fid_trajectory[dim]))
            out_f.setMapping(dim, '{0} - fiducial_{0}'.format(dim))

        # propagate metadata, if present
        try:
            out.mdh = MetaDataHandler.DictMDHandler(inputLocalizations.mdh)
        except AttributeError:
            pass

        return {'outputName': out, 'outputFiducials' : out_f}


@register_module('AutocorrelationDriftCorrection')
class AutocorrelationDriftCorrection(ModuleBase):
    """
    Perform drift correction using autocorrelation between subsets of the point data

    Inputs
    ------
    inputName: name of tabular input containing positions ('x', 'y', and 't' columns should be present)
    step : time step (in frames) with which to traverse the series
    window: size of time window (in frames). A series of images will be generated from
            multiple overlapping windows, spaced by `step` frames.
    binsize: size of histogram bins in nm

    Outputs
    -------
    outputName: name of tabular output. A mapped version of the tabular input with 2 extra columns
        
    """
    inputName = Input('Localizations')
    step = Int(200)
    window = Int(500)
    binsize = Float(30)
    
    outputName = Output('corrected_localizations')

    def calcCorrDrift(self, x, y, t):
        from scipy import ndimage
    
        tMax = int(t.max())
    
        bx = np.arange(x.min(), x.max() + self.binsize, self.binsize)
        by = np.arange(y.min(), y.max() + self.binsize, self.binsize)
    
        tInd = t < self.window
    
        h1 = np.histogram2d(x[tInd], y[tInd], [bx, by])[0]
        H1 = np.fft.fftn(h1)
    
        shifts = []
        tis = []
    
        for ti in range(0, tMax + 1, self.step):
            tInd = (t >= ti) * (t < (ti + self.window))
            h2 = np.histogram2d(x[tInd], y[tInd], [bx, by])[0]
        
            xc = abs(np.fft.ifftshift(np.fft.ifftn(H1 * np.fft.ifftn(h2))))
        
            xct = (xc - xc.max() / 3) * (xc > xc.max() / 3)
        
            shifts.append(ndimage.measurements.center_of_mass(xct))
            tis.append(ti + self.window / 2.)
    
        sha = np.array(shifts)
    
        return np.array(tis), self.binsize * (sha - sha[0])

    # def execute(self, namespace):
    #     from PYME.IO import tabular
    #     locs = namespace[self.inputName]
        
    #     t_shift, shifts = self.calcCorrDrift(locs['x'], locs['y'], locs['t'])
    #     shx = shifts[:, 0]
    #     shy = shifts[:, 1]

    #     out = tabular.MappingFilter(locs)
    #     t_out = out['t']
    #     dx = np.interp(t_out, t_shift, shx)
    #     dy = np.interp(t_out, t_shift, shy)
        
        
    #     out.addColumn('dx', dx)
    #     out.addColumn('dy', dy)
    #     out.setMapping('x', 'x + dx')
    #     out.setMapping('y', 'y + dy')
        
    #     # propagate metadata, if present
    #     try:
    #         out.mdh = locs.mdh
    #     except AttributeError:
    #         pass
        
    #     namespace[self.outputName] = out

    def run(self, inputName):
        from PYME.IO import tabular
        locs = inputName
        
        t_shift, shifts = self.calcCorrDrift(locs['x'], locs['y'], locs['t'])
        shx = shifts[:, 0]
        shy = shifts[:, 1]

        out = tabular.MappingFilter(locs)
        t_out = out['t']
        dx = np.interp(t_out, t_shift, shx)
        dy = np.interp(t_out, t_shift, shy)
        
        
        out.addColumn('dx', dx)
        out.addColumn('dy', dy)
        out.setMapping('x', 'x + dx')
        out.setMapping('y', 'y + dy')
        
        return out


#!/usr/bin/python

###############
# pipeline.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################
from PYME.IO import tabular
from PYME.IO.image import ImageBounds
from PYME.LMVis import dyeRatios
from PYME.LMVis import statusLog

try:
    # make sure pipeline works when wx is not avalable
    # TODO - move blob stuff into a recipe module and remove from pipeline.
    from PYME.LMVis.triBlobs import BlobSettings
except ImportError:
    # create a dummy class
    class BlobSettings:
        pass

from PYME.Analysis import piecewiseMapping
from PYME.IO import MetaDataHandler

import warnings

#from traits.api import HasTraits
#from traitsui.api import View
from PYME.recipes.base import ModuleCollection

import numpy as np
import scipy.special
import os

from PYME.contrib import dispatch

from PYME.Analysis.BleachProfile.kinModels import getPhotonNums

import logging
logger = logging.getLogger(__name__)

def _processPriSplit(ds):
    """set mappings ascociated with the use of a splitter"""

    ds.setMapping('gFrac', 'fitResults_ratio')
    ds.setMapping('error_gFrac', 'fitError_ratio')

    ds.setMapping('fitResults_Ag', 'gFrac*A')
    ds.setMapping('fitResults_Ar', '(1.0 - gFrac)*A + error_gFrac*A')
    ds.setMapping('fitError_Ag', 'gFrac*fitError_A + error_gFrac*A')
    ds.setMapping('fitError_Ar', '(1.0 - gFrac)*fitError_A')
    #ds.setMapping('fitError_Ag', '1*sqrt(fitResults_Ag/1e-3)')
    #ds.setMapping('fitError_Ar', '1*sqrt(fitResults_Ar/1e-3)')

    sg = ds['fitError_Ag']
    sr = ds['fitError_Ar']
    g = ds['fitResults_Ag']
    r = ds['fitResults_Ar']
    I = ds['A']

    colNorm = np.sqrt(2 * np.pi) * sg * sr / (2 * np.sqrt(sg ** 2 + sr ** 2) * I) * (
        scipy.special.erf((sg ** 2 * r + sr ** 2 * (I - g)) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2)))
        - scipy.special.erf((sg ** 2 * (r - I) - sr ** 2 * g) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2))))

    colNorm /= (sg * sr)

    ds.addColumn('ColourNorm', colNorm)


def _processSplitter(ds):
    """set mappings ascociated with the use of a splitter"""

    #ds.gF_zcorr = 0
    ds.setMapping('A', 'fitResults_Ag + fitResults_Ar')
    if 'fitResults_z0' in ds.keys():
        ds.addVariable('gF_zcorr', 0)
        ds.setMapping('gFrac', 'fitResults_Ag/(fitResults_Ag + fitResults_Ar) + gF_zcorr*fitResults_z0')
    else:
        ds.setMapping('gFrac', 'fitResults_Ag/(fitResults_Ag + fitResults_Ar)')

    if 'fitError_Ag' in ds.keys():
        ds.setMapping('error_gFrac',
                      'sqrt((fitError_Ag/fitResults_Ag)**2 + (fitError_Ag**2 + fitError_Ar**2)/(fitResults_Ag + fitResults_Ar)**2)*fitResults_Ag/(fitResults_Ag + fitResults_Ar)')
    else:
        ds.setMapping('error_gFrac', '0*x + 0.01')
        ds.setMapping('fitError_Ag', '1*sqrt(fitResults_Ag/1)')
        ds.setMapping('fitError_Ar', '1*sqrt(fitResults_Ar/1)')

    sg = ds['fitError_Ag']
    sr = ds['fitError_Ar']
    g = ds['fitResults_Ag']
    r = ds['fitResults_Ar']
    I = ds['A']

    colNorm = np.sqrt(2 * np.pi) * sg * sr / (2 * np.sqrt(sg ** 2 + sr ** 2) * I) * (
        scipy.special.erf((sg ** 2 * r + sr ** 2 * (I - g)) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2)))
        - scipy.special.erf((sg ** 2 * (r - I) - sr ** 2 * g) / (np.sqrt(2) * sg * sr * np.sqrt(sg ** 2 + sr ** 2))))

    ds.addColumn('ColourNorm', colNorm)


def _add_eventvars_to_ds(ds, ev_mappings):
    zm = ev_mappings.get('zm', None)
    if zm:
        z_focus = 1.e3 * zm(ds['t'])
        ds.addColumn('focus', z_focus)

        piezo_moving = ev_mappings.get('piezo_moving')
        if piezo_moving:
            ds.addColumn('piezoMoving', piezo_moving(ds['t']))

    xm = ev_mappings.get('xm', None)
    if xm:
        # todo - comment on why the -0.01??
        scan_x = 1.e3 * xm(ds['t'] - .01)
        ds.addColumn('scanx', scan_x)
        
        # scanned acquisition re-mapping temporarily disabled because we abuse the scanner for taking shift-fields (where
        # we don't want re-mapping to occur) TODO - either detect shift-fields here, or implement remapping later.
        #ds.setMapping('x', 'x + scanx')

    ym = ev_mappings.get('ym', None)
    if ym:
        scan_y = 1.e3 * ym(ds['t'] - .01)
        ds.addColumn('scany', scan_y)
        
        # temporarily disabled - see scanx
        #ds.setMapping('y', 'y + scany')

    driftx = ev_mappings.get('driftx', None)
    drifty = ev_mappings.get('drifty', None)
    driftz = ev_mappings.get('driftz', None)
    if driftx:
        ds.addColumn('driftx', driftx(ds['t'] - .01))
        ds.addColumn('drifty', drifty(ds['t'] - .01))
        ds.addColumn('driftz', driftz(ds['t'] - .01))


def _add_missing_ds_keys(mapped_ds, ev_mappings={}):
    """
    VisGUI, and various rendering and postprocessing commands rely on having certain parameters defined or re-mapped
    within a data source. Take care of these here.

    Parameters
    ----------
    mapped_ds

    """
    _add_eventvars_to_ds(mapped_ds, ev_mappings)

    #handle special cases which get detected by looking for the presence or
    #absence of certain variables in the data.
    if 'fitResults_Ag' in mapped_ds.keys():
        #if we used the splitter set up a number of mappings e.g. total amplitude and ratio
        _processSplitter(mapped_ds)

    if 'fitResults_ratio' in mapped_ds.keys():
        #if we used the splitter set up a number of mappings e.g. total amplitude and ratio
        _processPriSplit(mapped_ds)

    if 'fitResults_sigxl' in mapped_ds.keys():
        #fast, quickpalm like astigmatic fitting
        mapped_ds.setMapping('sig', 'fitResults_sigxl + fitResults_sigyu')
        mapped_ds.setMapping('sig_d', 'fitResults_sigxl - fitResults_sigyu')

        mapped_ds.addVariable('dsigd_dz', -30.)
        mapped_ds.setMapping('fitResults_z0', 'dsigd_dz*sig_d')
    if not 'y' in mapped_ds.keys():
        mapped_ds.setMapping('y', '10*t')

    # set up correction for foreshortening and z focus stepping
    if not 'foreShort' in dir(mapped_ds):
        mapped_ds.addVariable('foreShort', 1.)
        
    if not 'focus' in mapped_ds.keys():
        # set up a dummy focus variable if not already present
        mapped_ds.setMapping('focus', '0*x')

    if not 'z' in mapped_ds.keys():
        if 'fitResults_z0' in mapped_ds.keys():
            mapped_ds.setMapping('z', 'fitResults_z0 + foreShort*focus')
        elif 'astigmatic_z' in mapped_ds.keys():
            mapped_ds.setMapping('z', 'astigmatic_z + foreShort*focus')
        elif 'astigZ' in mapped_ds.keys():  # legacy handling
            mapped_ds.setMapping('z', 'astigZ + foreShort*focus')
        else:
            mapped_ds.setMapping('z', 'foreShort*focus')

    if not 'A' in mapped_ds.keys() and 'fitResults_photons' in mapped_ds.keys():
        mapped_ds.setMapping('A', 'fitResults_photons')



def _processEvents(ds, events, mdh):
    """Read data from events table and translate it into mappings for,
    e.g. z position"""

    eventCharts = []
    ev_mappings = {}

    if not events is None:
        evKeyNames = set()
        for e in events:
            evKeyNames.add(e['EventName'])

        if b'ProtocolFocus' in evKeyNames:
            zm = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], mdh['Protocol.PiezoStartPos'], eventName=b'ProtocolFocus')
            ev_mappings['z_command'] = zm
            

            if b'PiezoOnTarget' in evKeyNames:
                # Sometimes we also emit PiezoOnTarget events with the actual piezo position, rather than where we
                # told it to go, use these preferentially
                # TODO - deprecate in favour of, e.g. 'FocusOnTarget' events which are offset-corrected - see issue 766
                from PYME.Analysis import piezo_movement_correction
                spoofed_evts =  piezo_movement_correction.spoof_focus_events_from_ontarget(events, mdh)
                zm = piecewiseMapping.GeneratePMFromEventList(spoofed_evts, mdh, mdh['StartTime'], mdh['Protocol.PiezoStartPos'], eventName=b'ProtocolFocus')
                ev_mappings['z_ontarget'] = zm
                ev_mappings['piezo_moving'] = piecewiseMapping.bool_map_between_events(events, mdh, b'ProtocolFocus', b'PiezoOnTarget',default=False)
                
            # the z position we use for localizations gets the ontarget info if present
            ev_mappings['zm'] = zm
            eventCharts.append(('Focus [um]', zm, b'ProtocolFocus'))
            
        if b'ScannerXPos' in evKeyNames:
            x0 = 0
            if 'Positioning.Stage_X' in mdh.getEntryNames():
                x0 = mdh.getEntry('Positioning.Stage_X')
            xm = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], x0, b'ScannerXPos', 0)
            ev_mappings['xm'] = xm
            eventCharts.append(('XPos [um]', xm, 'ScannerXPos'))

        if b'ScannerYPos' in evKeyNames:
            y0 = 0
            if 'Positioning.Stage_Y' in mdh.getEntryNames():
                y0 = mdh.getEntry('Positioning.Stage_Y')
            ym = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), y0, b'ScannerYPos', 0)
            ev_mappings['ym'] = ym
            eventCharts.append(('YPos [um]', ym, 'ScannerYPos'))

        if b'ShiftMeasure' in evKeyNames:
            driftx = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), 0, b'ShiftMeasure',
                                                              0)
            drifty = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), 0, b'ShiftMeasure',
                                                              1)
            driftz = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh.getEntry('StartTime'), 0, b'ShiftMeasure',
                                                              2)

            ev_mappings['driftx'] = driftx
            ev_mappings['drifty'] = drifty
            ev_mappings['driftz'] = driftz

            eventCharts.append(('X Drift [px]', driftx, 'ShiftMeasure'))
            eventCharts.append(('Y Drift [px]', drifty, 'ShiftMeasure'))
            eventCharts.append(('Z Drift [px]', driftz, 'ShiftMeasure'))

            # self.eventCharts = eventCharts
            # self.ev_mappings = ev_mappings
    elif all(k in mdh.keys() for k in ['StackSettings.FramesPerStep', 'StackSettings.StepSize',
                                       'StackSettings.NumSteps', 'StackSettings.NumCycles']):
        # TODO - Remove this code - anytime we get here it's generally the result of an error in the input data
        # This should be handled upstream when spooling
        logger.warning('Spoofing focus from metadata: this usually implies an error in the input data (missing events) and results might vary')
        try:
            # if we dont have events file, see if we can use metadata to spoof focus
            from PYME.experimental import labview_spooling_hacks

            position, frames = labview_spooling_hacks.spoof_focus_from_metadata(mdh)
            zm = piecewiseMapping.piecewiseMap(0, frames, position, mdh['Camera.CycleTime'], xIsSecs=False)
            ev_mappings['z_command'] = zm
            eventCharts.append(('Focus [um]', zm, b'ProtocolFocus'))

        except:
            # It doesn't really matter if this fails, print our traceback anyway
            logger.exception('Error trying to fudge focus positions')

    return ev_mappings, eventCharts

class Pipeline:
    def __init__(self, filename=None, visFr=None):
        self.filter = None
        self.mapping = None
        self.colourFilter = None
        self.events = None
        
        self.recipe = ModuleCollection(execute_on_invalidation=True)
        self.recipe.recipe_executed.connect(self.Rebuild)

        self.selectedDataSourceKey = None
        self.filterKeys = {'error_x': (0,30), 'error_y':(0,30),'A':(5,20000), 'sig' : (95, 200)}

        self.blobSettings = BlobSettings()
        self.objects = None

        self.imageBounds = ImageBounds(0,0,0,0)
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        
        self.Triangles = None
        self.edb = None
        self.Quads = None
        self.GeneratedMeasures = {}
        
        self.QTGoalPixelSize = 5
        
        self._extra_chan_num = 0
        
        self.filesToClose = []

        self.ev_mappings = {}

        #define a signal which a GUI can hook if the pipeline is rebuilt (i.e. the output changes)
        self.onRebuild = dispatch.Signal()

        #a cached list of our keys to be used to decide whether to fire a keys changed signal
        self._keys = None
        #define a signal which can be hooked if the pipeline keys have changed
        self.onKeysChanged = dispatch.Signal()
        
        self.ready = False
        #self.visFr = visFr

        if not filename is None:
            self.OpenFile(filename)
            
        #renderers.renderMetadataProviders.append(self.SaveMetadata)
            
    @property
    def output(self):
        return self.colourFilter
    
    def __getitem__(self, keys):
        """gets values from the 'tail' of the pipeline (ie the colourFilter)"""
        
        return self.output[keys]

    def keys(self):
        return self.output.keys()
    
    def __getattr__(self, item):
        try:
            #if 'colourFilter in '
            if self.output is None:
                raise AttributeError('colourFilter not yet created')
            
            return self.output[item]
        except KeyError:
            raise AttributeError("'%s' has not attribute '%s'" % (self.__class__, item))

    def __dir__(self):
        if self.output is None:
            return list(self.__dict__.keys()) + list(dir(type(self)))
        else:
            return list(self.output.keys()) + list(self.__dict__.keys()) + list(dir(type(self)))
    
    #compatibility redirects
    @property
    def fluorSpecies(self):
        #warnings.warn(DeprecationWarning('Use colour_mapper.species_ratios instead'))
        raise DeprecationWarning('Use colour_mapper.species_ratios instead')
        return self.colour_mapper.species_ratios
    
    @property
    def fluorSpeciesDyes(self):
        #warnings.warn(DeprecationWarning('Use colour_mapper.species_dyes instead'))
        raise DeprecationWarning('Use colour_mapper.species_dyes instead')
        return self.colour_mapper.species_dyes
        

    @property
    def chromaticShifts(self):
        return self.colourFilter.chromaticShifts

    #end compatibility redirects
    
    @property
    def dataSources(self):
        return self.recipe.namespace
    
    @property
    def layer_datasources(self):
        lds = {'output':self.colourFilter}
        lds.update(self.dataSources)
        return lds

    @property
    def layer_data_source_names(self):
        """
        Return a list of names of datasources we can use with dotted channel selection

        There is a little bit of magic here as we augment the names with dotted names for colour channel selection
        """
        names = []#'']
        for k, v in self.layer_datasources.items():
            names.append(k)
            if isinstance(v, tabular.ColourFilter):
                for c in v.getColourChans():
                    names.append('.'.join([k, c]))
    
        return names
    
    def get_layer_data(self, dsname):
        """
        Returns layer data for a given name. The principle difference to just accessing self.dataSources directly is that
        we do some magic relating to allow colour channels to be accessed with the dot notation e.g. dsname.colour_channel

        """
        if dsname == '':
            return self
    
        parts = dsname.split('.')
        if len(parts) == 2:
            # special case - permit access to channels using dot notation
            # NB: only works if our underlying datasource is a ColourFilter
            ds, channel = parts
            if ds == 'output':
                return self.colourFilter.get_channel_ds(channel)
            else:
                return self.dataSources.get(ds, None).get_channel_ds(channel)
        else:
            if dsname =='output':
                return self.colourFilter
            else:
                return self.dataSources.get(dsname, None)

    @property
    def selectedDataSource(self):
        """

        The currently selected data source (an instance of tabular.inputFilter derived class)

        """
        if self.selectedDataSourceKey is None:
            return None
        else:
            return self.dataSources[self.selectedDataSourceKey]

    def selectDataSource(self, dskey):
        """
        Set the currently selected data source

        Parameters
        ----------
        dskey : string
            The data source name

        """
        if not dskey in self.dataSources.keys():
            raise KeyError('Data Source "%s" not found' % dskey)

        self.selectedDataSourceKey = dskey

        #remove any keys from the filter which are not present in the data
        for k in list(self.filterKeys.keys()):
            if not k in self.selectedDataSource.keys():
                self.filterKeys.pop(k)

        self.Rebuild()
        
    def new_ds_name(self, stub, return_count=False):
        """
        Generate a name for a new, unused, pipeline step output based on a stub
        
        FIXME - should this be in ModuleCollection instead?
        FIXME - should this use recipe.outputs as well?
        
        Parameters
        ----------
        stub - string to start the name with

        Returns
        -------

        """
        count = 0
        pattern = stub + '%d'
        
        name = pattern % count
        while name in self.dataSources.keys():
            count += 1
            name = pattern % count
            
        if return_count:
            return name, count
            
        return name

    def addColumn(self, name, values, default = 0):
        """
        Adds a column to the currently selected data source. Attempts to guess whether the size matches the input or
        the output, and adds padding values appropriately if it matches the output.

        Parameters
        ----------
        name : str
            The column name
        values : array like
            The values
        default : float
            The default value to pad with if we've given an output-sized array

        """
        import warnings
        warnings.warn('Deprecated. You should not add columns to the pipeline as this injects data and is not captured by the recipe', DeprecationWarning)

        ds_len = len(self.selectedDataSource[self.selectedDataSource.keys()[0]])
        val_len = len(values)

        if val_len == ds_len:
            #length matches the length of our input data source - do a simple add
            self.selectedDataSource.addColumn(name, values)
        elif val_len == len(self[self.keys()[0]]):

            col_index = self.colourFilter.index

            idx = np.copy(self.filter.Index)
            idx[self.filter.Index] = col_index

            ds_vals = np.zeros(ds_len) + default
            ds_vals[idx] = np.array(values)

            self.selectedDataSource.addColumn(name, ds_vals)
        else:
            raise RuntimeError("Length of new column doesn't match either the input or output lengths")

    def addDataSource(self, dskey, ds, add_missing_vars=True):
        """
        Add a new data source

        Parameters
        ----------
        dskey : str
            The name of the new data source
        ds : an tabular.inputFilter derived class
            The new data source

        """
        #check that we have a suitable object - note that this could potentially be relaxed
        assert isinstance(ds, tabular.TabularBase)

        if not isinstance(ds, tabular.MappingFilter):
            #wrap with a mapping filter
            ds = tabular.MappingFilter(ds)

        #add keys which might not already be defined
        if add_missing_vars:
            _add_missing_ds_keys(ds,self.ev_mappings)

        if getattr(ds, 'mdh', None) is None:
            try:
                ds.mdh = self.mdh
            except AttributeError:
                logger.error('No metadata defined in pipeline')
                pass

        self.dataSources[dskey] = ds



    def Rebuild(self, **kwargs):
        """
        Rebuild the pipeline. Called when the selected data source is changed/modified and/or the filter is changed.

        """
        for s in self.dataSources.values():
            if 'setMapping' in dir(s):
                #keep raw measurements available
                s.setMapping('x_raw', 'x')
                s.setMapping('y_raw', 'y')
                
                if 'z' in  s.keys():
                    s.setMapping('z_raw', 'z')
                
        if not self.selectedDataSource is None:
            if not self.mapping is None:
                # copy any mapping we might have made across to the new mapping filter (should fix drift correction)
                # TODO - make drift correction a recipe module so that we don't need this code. Long term we should be
                # ditching the mapping filter here.
                old_mapping = self.mapping
                self.mapping = tabular.MappingFilter(self.selectedDataSource)
                self.mapping.mappings.update(old_mapping.mappings)
            else:
                self.mapping = tabular.MappingFilter(self.selectedDataSource)

            #the filter, however needs to be re-generated with new keys and or data source
            self.filter = tabular.ResultsFilter(self.mapping, **self.filterKeys)

            #we can also recycle the colour filter
            if self.colourFilter is None:
                self.colourFilter = tabular.ColourFilter(self.filter)
            else:
                self.colourFilter.resultsSource = self.filter

            #self._process_colour()
            
            self.ready = True

        self.ClearGenerated()
        
    def ClearGenerated(self):
        self.Triangles = None
        self.edb = None
        self.GeneratedMeasures = {}
        self.Quads = None

        self.onRebuild.send_robust(sender=self)

        #check to see if any of the keys have changed - if so, fire a keys changed event so the GUI can update
        newKeys = self.keys()
        if not newKeys == self._keys:
            self.onKeysChanged.send_robust(sender=self)
        
    def CloseFiles(self):
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()

    def _ds_from_file(self, filename, **kwargs):
        """
        loads a data set from a file

        Parameters
        ----------
        filename : str
        kwargs : any additional arguments (see OpenFile)

        Returns
        -------
        ds : tabular.TabularBase
            the datasource, complete with metadatahandler and events if found.

        """
        mdh = MetaDataHandler.NestedClassMDHandler()
        events = None
        if os.path.splitext(filename)[1] == '.h5r':
            import tables
            h5f = tables.open_file(filename)
            self.filesToClose.append(h5f)
            
            try:
                ds = tabular.H5RSource(h5f)

                if 'DriftResults' in h5f.root:
                    driftDS = tabular.H5RDSource(h5f)
                    self.driftInputMapping = tabular.MappingFilter(driftDS)
                    #self.dataSources['Fiducials'] = self.driftInputMapping
                    self.addDataSource('Fiducials', self.driftInputMapping)

                    if len(ds['x']) == 0:
                        self.selectDataSource('Fiducials')

            except: #fallback to catch series that only have drift data
                logger.exception('No fitResults table found')
                ds = tabular.H5RDSource(h5f)

                self.driftInputMapping = tabular.MappingFilter(ds)
                #self.dataSources['Fiducials'] = self.driftInputMapping
                self.addDataSource('Fiducials', self.driftInputMapping)
                #self.selectDataSource('Fiducials')

            # really old files might not have metadata, so test for it before assuming
            if 'MetaData' in h5f.root:
                mdh = MetaDataHandler.HDFMDHandler(h5f)
            
            if ('Events' in h5f.root) and ('StartTime' in mdh.keys()):
                events = h5f.root.Events[:]

        elif filename.endswith('.hdf'):
            #recipe output - handles generically formatted .h5
            import tables
            h5f = tables.open_file(filename)
            self.filesToClose.append(h5f)
            
            #defer our IO to the recipe IO method - TODO - do this for other file types as well
            self.recipe._inject_tables_from_hdf5('', h5f, filename, '.hdf')

            for dsname, ds_ in self.dataSources.items():
                #loop through tables until we get one which defines x. If no table defines x, take the last table to be added
                #TODO make this logic better.
                ds = ds_
                if 'x' in ds.keys():
                    # TODO - get rid of some of the grossness here
                    mdh = getattr(ds, 'mdh', mdh)
                    events = getattr(ds, 'events', events)
                    break
                    

        elif os.path.splitext(filename)[1] == '.mat': #matlab file
            if 'VarName' in kwargs.keys():
                #old style matlab import
                ds = tabular.MatfileSource(filename, kwargs['FieldNames'], kwargs['VarName'])
            else:
                if 'Multichannel' in kwargs.keys():
                    ds = tabular.MatfileMultiColumnSource(filename)
                else:
                    ds = tabular.MatfileColumnSource(filename)
                
                # check for column name mapping
                field_names = kwargs.get('FieldNames', None)
                if field_names:
                    if 'Multichannel' in kwargs.keys():
                        field_names.append('probe')  # don't forget to copy this field over
                    ds = tabular.MappingFilter(ds, **{new_field : old_field for new_field, old_field in zip(field_names, ds.keys())})

        elif os.path.splitext(filename)[1] == '.csv':
            #special case for csv files - tell np.loadtxt to use a comma rather than whitespace as a delimeter
            if 'SkipRows' in kwargs.keys():
                ds = tabular.TextfileSource(filename, kwargs['FieldNames'], delimiter=',', skiprows=kwargs['SkipRows'])
            else:
                ds = tabular.TextfileSource(filename, kwargs['FieldNames'], delimiter=',')

        else: #assume it's a tab (or other whitespace) delimited text file
            if 'SkipRows' in kwargs.keys():
                ds = tabular.TextfileSource(filename, kwargs['FieldNames'], skiprows=kwargs['SkipRows'])
            else:
                ds = tabular.TextfileSource(filename, kwargs['FieldNames'])
        
        # make sure mdh is writable (file-based might not be)
        ds.mdh = MetaDataHandler.NestedClassMDHandler(mdToCopy=mdh)
        if events is not None:
            # only set the .events attribute if we actually have events.
            ds.events = events
            
        return ds

    def OpenFile(self, filename= '', ds = None, clobber_recipe=True, **kwargs):
        """Open a file - accepts optional keyword arguments for use with files
        saved as .txt and .mat. These are:
            
            FieldNames: a list of names for the fields in the text file or
                        matlab variable.
            VarName:    the name of the variable in the .mat file which 
                        contains the data.
            SkipRows:   Number of header rows to skip for txt file data
            
            PixelSize:  Pixel size if not in nm
            
        """
        

        #close any files we had open previously
        while len(self.filesToClose) > 0:
            self.filesToClose.pop().close()
        
        # clear our state
        # nb - equivalent to clearing recipe namespace
        self.dataSources.clear()
        
        if clobber_recipe:
            # clear any processing modules from the pipeline
            # call with clobber_recipe = False in a 'Open a new file with the processing pipeline I've set up' use case
            # TODO: Add an "File-->Open [preserving recipe]" menu option or similar
            self.recipe.modules = []
        
        if 'zm' in dir(self):
            del self.zm
        self.filter = None
        self.mapping = None
        self.colourFilter = None
        self.events = None
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        
        self.filename = filename
        
        if ds is None:
            from PYME.IO import unifiedIO # TODO - what is the launch time penalty here for importing clusterUI and finding a nameserver?
            
            # load from file(/cluster, downloading a copy of the file if needed)
            with unifiedIO.local_or_temp_filename(filename) as fn:
                # TODO - check that loading isn't lazy (i.e. we need to make a copy of data in memory whilst in the
                # context manager in order to be safe with unifiedIO and cluster data). From a quick look, it would seem
                # that _ds_from_file() copies the data, but potentially keeps the file open which could be problematic.
                # This won't effect local file loading even if loading is lazy (i.e. shouldn't cause a regression)
                ds = self._ds_from_file(fn, **kwargs)
                self.events = getattr(ds, 'events', None)
                self.mdh.copyEntriesFrom(ds.mdh)

        # skip the MappingFilter wrapping, etc. in self.addDataSource and add this datasource as-is
        self.dataSources['FitResults'] = ds

        # Fit module specific filter settings
        # TODO - put all the defaults here and use a local variable rather than in __init__ (self.filterKeys is largely an artifact of pre-recipe based pipeline)
        if 'Analysis.FitModule' in self.mdh.getEntryNames():
            fitModule = self.mdh['Analysis.FitModule']
            if 'Interp' in fitModule:
                self.filterKeys['A'] = (5, 100000)
            if fitModule == 'SplitterShiftEstFR':
                self.filterKeys['fitError_dx'] = (0, 10)
                self.filterKeys['fitError_dy'] = (0, 10)

        if clobber_recipe:
            from PYME.recipes.localisations import ProcessColour, Pipelineify
            from PYME.recipes.tablefilters import FilterTable
            
            add_pipeline_variables = Pipelineify(self.recipe,
                inputFitResults='FitResults',
                pixelSizeNM=kwargs.get('PixelSize', 1.),
                outputLocalizations='Localizations')
            self.recipe.add_module(add_pipeline_variables)
          
            #self._get_dye_ratios_from_metadata()
                   
            colour_mapper = ProcessColour(self.recipe, input='Localizations', output='colour_mapped')
            self.recipe.add_module(colour_mapper)
            self.recipe.add_module(FilterTable(self.recipe, inputName='colour_mapped', outputName='filtered_localizations', filters={k:list(v) for k, v in self.filterKeys.items() if k in ds.keys()}))
        else:
            logger.warn('Opening file without clobbering recipe, filter and ratiometric colour settings might not be handled properly')
            # FIXME - should we update filter keys and/or make the filter more robust
            # FIXME - do we need to do anything about colour settings?
        
        self.recipe.execute()
        self.filterKeys = {}
        if 'filtered_localizations' in self.dataSources.keys():
            self.selectDataSource('filtered_localizations') #NB - this rebuilds the pipeline
        else:
            self.selectDataSource(self.dataSources.keys()[0])

        # FIXME - we do this already in pipelinify, maybe we can avoid doubling up?
        self.ev_mappings, self.eventCharts = _processEvents(ds, self.events,
                                                            self.mdh)  # extract information from any events
        # Retrieve or estimate image bounds
        if False:  # 'imgBounds' in kwargs.keys():
            # TODO - why is this disabled? Current usage would appear to be when opening from LMAnalysis
            # during real-time localization, to force image bounds to match raw data, but also potentially useful
            # for other scenarios where metadata is not fully present.
            self.imageBounds = kwargs['imgBounds']
        elif ('scanx' not in self.selectedDataSource.keys() or 'scany' not in self.selectedDataSource.keys()) and 'Camera.ROIWidth' in self.mdh.getEntryNames():
            self.imageBounds = ImageBounds.extractFromMetadata(self.mdh)
        else:
            self.imageBounds = ImageBounds.estimateFromSource(self.selectedDataSource)
        
        #self._process_colour()
        
    @property
    def colour_mapper(self):
        """ Search for a colour mapper rather than use a hard coded reference - allows loading of saved pipelines with colour mapping"""
        from PYME.recipes.localisations import ProcessColour
        
        # find ProcessColour instance(s) in the pipeline
        mappers = [m for m in self.recipe.modules if isinstance(m, ProcessColour)]
        
        if len(mappers) > 0:
            #return the first mapper we find
            return mappers[0]
        
        else:
            return None

    def OpenChannel(self, filename='', ds=None, channel_name='', **kwargs):
        """Open a file - accepts optional keyword arguments for use with files
        saved as .txt and .mat. These are:

            FieldNames: a list of names for the fields in the text file or
                        matlab variable.
            VarName:    the name of the variable in the .mat file which
                        contains the data.
            SkipRows:   Number of header rows to skip for txt file data

            PixelSize:  Pixel size if not in nm

        """
        if channel_name == '' or channel_name is None:
            #select a channel name automatically
            channel_name = 'Channel%d' % self._extra_chan_num
            self._extra_chan_num += 1
                
        if ds is None:
            #load from file
            ds = self._ds_from_file(filename, **kwargs)
    
        #wrap the data source with a mapping so we can fiddle with things
        #e.g. combining z position and focus
        mapped_ds = tabular.MappingFilter(ds)
    
        if 'PixelSize' in kwargs.keys():
            mapped_ds.addVariable('pixelSize', kwargs['PixelSize'])
            mapped_ds.setMapping('x', 'x*pixelSize')
            mapped_ds.setMapping('y', 'y*pixelSize')
    
    
        self.addDataSource(channel_name, mapped_ds)



    def _process_colour(self):
        """
        Locate any colour / channel information and munge it into a format that the colourFilter understands.

        We currently accept 3 ways of specifying channels:

         - ratiometric colour, where 'gFrac' is defined to be the ratio between our observation channels
         - defining a 'probe' column in the input data which gives a channel index for each point
         - specifying colour ranges in the metadata

         All of these get munged into the p_dye type entries that the colour filter needs.

        """
        #clear out old colour keys
        warnings.warn(DeprecationWarning('This should not be called (colour now handled by the ProcessColour recipe module)'))
        for k in self.mapping.mappings.keys():
            if k.startswith('p_'):
                self.mapping.mappings.pop(k)
        
        if 'gFrac' in self.selectedDataSource.keys():
            #ratiometric
            for structure, ratio in self.fluorSpecies.items():
                if not ratio is None:
                    self.mapping.setMapping('p_%s' % structure, 'exp(-(%f - gFrac)**2/(2*error_gFrac**2))/(error_gFrac*sqrt(2*numpy.pi))' % ratio)
        else:
            if 'probe' in self.mapping.keys():
                #non-ratiometric (i.e. sequential) colour
                #color channel is given in 'probe' column
                self.mapping.setMapping('ColourNorm', '1.0 + 0*probe')
    
                for i in range(int(self.mapping['probe'].min()), int(self.mapping['probe'].max() + 1)):
                    self.mapping.setMapping('p_chan%d' % i, '1.0*(probe == %d)' % i)
    
            nSeqCols = self.mdh.getOrDefault('Protocol.NumberSequentialColors', 1)
            if nSeqCols > 1:
                for i in range(nSeqCols):
                    self.mapping.setMapping('ColourNorm', '1.0 + 0*t')
                    cr = self.mdh['Protocol.ColorRange%d' % i]
                    self.mapping.setMapping('p_chan%d' % i, '(t>= %d)*(t<%d)' % cr)
                
        #self.ClearGenerated()


    def _get_dye_ratios_from_metadata(self):
        warnings.warn(DeprecationWarning('This should not be called (colour now handled by the ProcessColour recipe module)'))
        labels = self.mdh.getOrDefault('Sample.Labelling', [])
        seen_structures = []

        for structure, dye in labels:
            if structure in seen_structures:
                strucname = structure + '_1'
            else:
                strucname = structure
            seen_structures.append(structure)
            
            ratio = dyeRatios.getRatio(dye, self.mdh)

            if not ratio is None:
                self.fluorSpecies[strucname] = ratio
                self.fluorSpeciesDyes[strucname] = dye
                #self.mapping.setMapping('p_%s' % structure, '(1.0/(ColourNorm*2*numpy.pi*fitError_Ag*fitError_Ar))*exp(-(fitResults_Ag - %f*A)**2/(2*fitError_Ag**2) - (fitResults_Ar - %f*A)**2/(2*fitError_Ar**2))' % (ratio, 1-ratio))
                #self.mapping.setMapping('p_%s' % structure, 'exp(-(%f - gFrac)**2/(2*error_gFrac**2))/(error_gFrac*sqrt(2*numpy.pi))' % ratio)
                

    def getNeighbourDists(self, forceRetriang = False):
        from PYME.LMVis import visHelpers
        
        if forceRetriang or not 'neighbourDistances' in self.GeneratedMeasures.keys():
            statNeigh = statusLog.StatusLogger("Calculating mean neighbour distances ...")
            self.GeneratedMeasures['neighbourDistances'] = np.array(visHelpers.calcNeighbourDists(self.getTriangles(forceRetriang)))
            
        return self.GeneratedMeasures['neighbourDistances']
        
    def getTriangles(self, recalc = False):
        from matplotlib import tri
        
        if self.Triangles is None or recalc:
            statTri = statusLog.StatusLogger("Generating Triangulation ...")
            self.Triangles = tri.Triangulation(self.colourFilter['x'] + .1*np.random.normal(size=len(self.colourFilter['x'])), self.colourFilter['y']+ .1*np.random.normal(size=len(self.colourFilter['x'])))
            
            #reset things which will have changed
            self.edb = None
            try:
                self.GeneratedMeasures.pop('neighbourDistances')
            except KeyError:
                pass
            
        return self.Triangles
        
    def getEdb(self):
        from PYME.Analysis.points.EdgeDB import edges
        if self.edb is None:
            self.edb = edges.EdgeDB(self.getTriangles())
            
        return self.edb
            
    def getBlobs(self):
        from PYME.Analysis.points.EdgeDB import edges
        
        tri = self.getTriangles()        
        edb = self.getEdb()
        
        
        if self.blobSettings.jittering == 0:
            self.objIndices = edges.objectIndices(edb.segment(self.blobSettings.distThreshold), self.blobSettings.minSize)
            self.objects = [np.vstack((tri.x[oi], tri.y[oi])).T for oi in self.objIndices]
        else:
            from matplotlib import tri
            
            ndists = self.getNeighbourDists()
            
            x_ = np.hstack([self['x'] + 0.5*ndists*np.random.normal(size=ndists.size) for i in range(self.blobSettings.jittering)])
            y_ = np.hstack([self['y'] + 0.5*ndists*np.random.normal(size=ndists.size) for i in range(self.blobSettings.jittering)])

            T = tri.Triangulation(x_, y_)
            edb = edges.EdgeDB(T)
            
            objIndices = edges.objectIndices(edb.segment(self.blobSettings.distThreshold), self.blobSettings.minSize)
            self.objects = [np.vstack((T.x[oi], T.y[oi])).T for oi in objIndices]
            
        return self.objects, self.blobSettings.distThreshold
        
    def GenQuads(self, max_leaf_size=10):
        from PYME.Analysis.points.QuadTree import pointQT
        
        di = max(self.imageBounds.x1 - self.imageBounds.x0, 
                 self.imageBounds.y1 - self.imageBounds.y0)

        numPixels = di/self.QTGoalPixelSize

        di = self.QTGoalPixelSize*2**np.ceil(np.log2(numPixels))

        
        self.Quads = pointQT.qtRoot(self.imageBounds.x0, self.imageBounds.x0+di, 
                                    self.imageBounds.y0, self.imageBounds.y0 + di)

        for xi, yi in zip(self['x'],self['y']):
            self.Quads.insert(pointQT.qtRec(xi,yi, None), max_leaf_size)
            
    def measureObjects(self):
        from PYME.Analysis.points import objectMeasure
        
        self.objectMeasures = objectMeasure.measureObjects(self.objects, self.objThreshold)
        
        return self.objectMeasures
        
    def save_txt(self, outFile, keys=None):
        if outFile.endswith('.csv'):
            delim = ', '
        else:
            delim = '\t'
            
        if keys is None:
            keys = self.keys()

        #nRecords = len(ds[keys[0]])
    
        of = open(outFile, 'w')
    
        of.write('#' + delim.join(['%s' % k for k in keys]) + '\n')
    
        for row in zip(*[self[k] for k in keys]):
            of.write(delim.join(['%e' % c for c in row]) + '\n')
    
        of.close()
        
    def save_hdf(self, filename):
        self.colourFilter.to_hdf(filename, tablename='Localizations', metadata=self.mdh)
        
    def to_recarray(self, keys=None):
        return self.colourFilter.to_recarray(keys=keys)
        
    def toDataFrame(self, keys=None):
        import pandas as pd
        if keys is None:
            keys = self.keys()
        
        d = {k: self[k] for k in keys}
        
        return pd.DataFrame(d)
        
    @property
    def dtypes(self):
        return {k: str(self[k, :2].dtype) for k in self.keys()}
    
    def _repr_html_(self):
        import jinja2
        TEMPLATE = """
        <h3> LMVis.pipeline.Pipeline viewing {{ pipe.filename }} </h3>
        <br>
        {{ recipe_svg }}
        <b> Data Sources: </b> {% for k in  pipe.dataSources.keys() %} {% if k != pipe.selectedDataSourceKey %} {{ k }} - [{{ pipe.dataSources[k]|length }} evts], {% endif %} {% endfor %} <b> {{ pipe.selectedDataSourceKey }} - [{{ pipe.dataSources[pipe.selectedDataSourceKey]|length }} evts]</b>
        <br>
        <b> Columns: </b> {{ grouped_keys }}
        """
        
        try:
            svg = self.recipe.to_svg()
        except:
            svg = None
            
        fr_keys = []
        fe_keys = []
        sl_keys = []
        st_keys = []
        
        for k in self.keys():
            if k.startswith('fitResults'):
                fr_keys.append(k)
            elif k.startswith('fitError'):
                fe_keys.append(k)
            elif k.startswith('slicesUsed'):
                sl_keys.append(k)
            else:
                st_keys.append(k)
                
        grouped_keys = sorted(st_keys) + sorted(fr_keys) + sorted(fe_keys) + sorted(sl_keys)
        
        return jinja2.Template(TEMPLATE).render(pipe=self, recipe_svg = svg, grouped_keys=', '.join(grouped_keys))

        
    







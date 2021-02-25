from .base import register_module, register_legacy_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr, ListInt

import numpy as np
#import pandas as pd
from PYME.IO import tabular, MetaDataHandler
#from PYME.LMVis import renderers
import logging
logger = logging.getLogger(__name__)

@register_module('Mapping')
class Mapping(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('measurements')
    mappings = DictStrStr()
    outputName = Output('mapped')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        mapped = tabular.MappingFilter(inp, **self.mappings)

        if 'mdh' in dir(inp):
            mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('FilterTable')
@register_legacy_module('Filter') #Deprecated - use FilterTable in new code / recipes
class FilterTable(ModuleBase):
    """Filter a table by specifying valid ranges for table columns"""
    inputName = Input('measurements')
    filters = DictStrList()
    outputName = Output('filtered')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        filtered = tabular.ResultsFilter(inp, **self.filters)

        if 'mdh' in dir(inp):
            filtered.mdh = inp.mdh

        namespace[self.outputName] = filtered

    @property
    def _ds(self):
        try:
            return self._parent.namespace[self.inputName]
        except:
            return None

    def _view_items(self, params=None):
        from traitsui.api import Item
        from PYME.ui.custom_traits_editors import FilterEditor
        
        return [Item('filters', editor=FilterEditor(datasource=self._ds), show_label=False),]


@register_module('FilterTableByIds')
class FilterTableByIDs(ModuleBase):
    """Filter a table by specifying valid ranges for table columns"""
    inputName = Input('measurements')
    idColumnName = CStr('clumpID')
    ids = ListInt()
    outputName = Output('filtered')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        filtered = tabular.IdFilter(inp, id_column=self.idColumnName, valid_ids=self.ids)

        if 'mdh' in dir(inp):
            filtered.mdh = inp.mdh

        namespace[self.outputName] = filtered

    @property
    def _ds(self):
        try:
            return self._parent.namespace[self.inputName]
        except:
            return None
        
    @property
    def _possible_ids(self):
        ids = [int(id) for id in set(self._ds[self.idColumnName]) if id > 0]
        
        return ids

    def _view_items(self, params=None):
        from traitsui.api import Item, TextEditor, SetEditor
        return [Item('idColumnName'),
                Item('ids', editor=SetEditor(values=self._possible_ids)),
                ]
    


@register_module('ConcatenateTables')
class ConcatenateTables(ModuleBase):
    """
    Concatenate two tabular objects

    The output will be a tabular object with a length equal to the sum of the input lengths and with those columns
    which are present in both inputs.
    """
    inputName0 = Input('chan0')
    inputName1 = Input('chan1')
    outputName = Output('output')

    def execute(self, namespace):
        inp0 = namespace[self.inputName0]
        inp1 = namespace[self.inputName1]

        concatenated = tabular.ConcatenateFilter(inp0, inp1)

        if 'mdh' in dir(inp0):
            concatenated.mdh = inp0.mdh

        namespace[self.outputName] = concatenated


@register_legacy_module('AggregateMeasurements')
@register_module('AggregateTables')
class AggregateMeasurements(ModuleBase):
    """Create a new composite measurement containing the results of multiple
    previous measurements"""
    inputMeasurements1 = Input('meas1')
    suffix1 = CStr('')
    inputMeasurements2 = Input('')
    suffix2 = CStr('')
    inputMeasurements3 = Input('')
    suffix3 = CStr('')
    inputMeasurements4 = Input('')
    suffix4 = CStr('')
    outputName = Output('aggregatedMeasurements')

    def execute(self, namespace):
        import collections
        res = collections.OrderedDict()
        for mk, suffix in [(getattr(self, n), getattr(self, 'suffix' + n[-1])) for n in dir(self) if
                           n.startswith('inputMeas')]:
            if not mk == '':
                meas = namespace[mk]

                #res.update(meas)
                for k in meas.keys():
                    res[k + suffix] = meas[k]

        meas1 = namespace[self.inputMeasurements1]
        #res = pd.DataFrame(res)
        res = tabular.DictSource(res)
        if 'mdh' in dir(meas1):
            res.mdh = meas1.mdh

        namespace[self.outputName] = res


@register_legacy_module('SelectMeasurementColumns') #deprecated - do not use
@register_module('SelectTableColumns')
class SelectTableColumns(ModuleBase):
    """Take just certain columns of a variable"""
    inputMeasurements = Input('measurements')
    keys = CStr('')
    outputName = Output('selectedMeasurements')

    def execute(self, namespace):
        meas = namespace[self.inputMeasurements]

        out = tabular.CloneSource(meas, keys=self.keys.split())
        # propagate metadata, if present
        try:
            out.mdh = meas.mdh
        except AttributeError:
            pass

        namespace[self.outputName] = out


@register_module('RandomSubset')
class RandomSubset(ModuleBase):
    """Select a random subset of rows from a table
    
    Parameters:
    -----------
    
    input :
    output :
    num_to_select : int
        The number of samples rows to return
    strict: bool
        How strict are we about the number of rows - governs what happens when the number of rows in the original data
        set is less than or equal to the number we want. If strict=False, the number of rows is truncated to the length
        of the original data. If strict=True trying to take a number of samples > the number of rows will generate an
        error. A warning will also be displayed if num_to_select is > n_rows/2. Monte-Carlo (or similar) applications
        should use strict=True.
    
    """
    
    input = Input('input')
    output = Output('output')
    num_to_select = Int(100)
    strict = Bool(False)
    
    def execute(self, namespace):
        data = namespace[self.input]
        
        n_rows = len(data)
        
        if n_rows < self.num_to_select:
            if self.strict:
                raise IndexError('Trying to select %d from data with only %d rows. To allow truncation, use strict=False' % (self.num_to_select, n_rows))
            else:
                logger.info('RandomSubset: Truncating from %d to %d rows as data only has %d rows. To make this an error, use strict=True' % (self.num_to_select, n_rows, n_rows)) 
        
        if self.strict and (self.num_to_select > 0.5*n_rows):
            logger.warning('RandomSubset: Selecting %d from %d rows will not be very random' % (self.num_to_select, n_rows))
        
        out = tabular.RandomSelectionFilter(data, num_Samples=min(n_rows, self.num_to_select))
        
        try:
            out.mdh = MetaDataHandler.DictMDHandler(data.mdh)
        except AttributeError:
            pass

        namespace[self.output] = out

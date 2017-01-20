from .base import register_module, register_legacy_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
import pandas as pd
from PYME.IO import tabular
from PYME.LMVis import renderers

@register_module('Mapping')
class Mapping(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('measurements')
    mappings = DictStrStr()
    outputName = Output('mapped')

    def execute(self, namespace):
        inp = namespace[self.inputName]

        mapped = tabular.mappingFilter(inp, **self.mappings)

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

        mapped = tabular.resultsFilter(inp, **self.filters)

        if 'mdh' in dir(inp):
            mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped

    @property
    def _ds(self):
        try:
            return self._parent.namespace[self.inputName]
        except:
            return None

    @property
    def pipeline_view(self):
        from traitsui.api import View, Group, Item
        from PYME.ui.custom_traits_editors import FilterEditor

        modname = ','.join(self.inputs) + ' -> ' + self.__class__.__name__ + ' -> ' + ','.join(self.outputs)

        return View(Group(Item('filters', editor=FilterEditor(datasource=self._ds)), label=modname))

    @property
    def default_view(self):
        from traitsui.api import View, Group, Item
        from PYME.ui.custom_traits_editors import CBEditor, FilterEditor

        return View(Item('inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item('_'),
                    Item('filters', editor=FilterEditor(datasource=self._ds)),
                    Item('_'),
                    Item('outputName'), buttons=['OK'])


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

        mapped = tabular.concatenateFilter(inp0, inp1)

        if 'mdh' in dir(inp0):
            mapped.mdh = inp0.mdh

        namespace[self.outputName] = mapped


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
        res = {}
        for mk, suffix in [(getattr(self, n), getattr(self, 'suffix' + n[-1])) for n in dir(self) if
                           n.startswith('inputMeas')]:
            if not mk == '':
                meas = namespace[mk]

                #res.update(meas)
                for k in meas.keys():
                    res[k + suffix] = meas[k]

        meas1 = namespace[self.inputMeasurements1]
        #res = pd.DataFrame(res)
        res = tabular.cloneSource(res)
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
        # out = pd.DataFrame({k: meas[k] for k in self.keys.split()})
        out = tabular.cloneSource({{k: meas[k] for k in self.keys.split()}})
        # propagate metadata, if present
        try:
            out.mdh = meas.mdh
        except AttributeError:
            pass

        namespace[self.outputName] = out






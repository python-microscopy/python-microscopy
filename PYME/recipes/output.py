from .base import register_module, ModuleBase, OutputModule
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, DictStrStr

import numpy as np
import pandas as pd
from PYME.IO import tabular

@register_module('CSVOutput')
class CSVOutput(OutputModule):
    inputName = Input('output')

    def save(self, namespace, context={}):
        """
        Save recipes output(s) to CSV

        Parameters
        ----------
        namespace : dict
            The recipe namespace
        context : dict
            Information about the source file to allow pattern substitution to generate the output name. At least
            'basedir' (which is the fully resolved directory name in which the input file resides) and
            'filestub' (which is the filename without any extension) should be resolved.

        Returns
        -------

        """

        out_filename = self.filePattern.format(context)

        v = namespace[self.inputName]
        v.toDataFrame.to_csv(out_filename)

@register_module('XLSXOutput')
class XLSOutput(OutputModule):
    inputName = Input('output')

    def save(self, namespace, context={}):
        """
        Save recipes output(s) to CSV

        Parameters
        ----------
        namespace : dict
            The recipe namespace
        context : dict
            Information about the source file to allow pattern substitution to generate the output name. At least
            'basedir' (which is the fully resolved directory name in which the input file resides) and
            'filestub' (which is the filename without any extension) should be resolved.

        Returns
        -------

        """

        out_filename = self.filePattern.format(context)

        v = namespace[self.inputName]
        v.toDataFrame.to_excel(out_filename)

@register_module('HDFOutput')
class HDFOutput(OutputModule):
    inputVariables = DictStrStr()

    def inputs(self):
        return set(self.inputVariables.keys())

    def save(self, namespace, context={}):
        """
        Save recipes output(s) to HDF5

        Parameters
        ----------
        namespace : dict
            The recipe namespace
        context : dict
            Information about the source file to allow pattern substitution to generate the output name. At least
            'basedir' (which is the fully resolved directory name in which the input file resides) and
            'filestub' (which is the filename without any extension) should be resolved.

        Returns
        -------

        """

        out_filename = self.filePattern.format(context)

        for name, h5_name in self.inputVariables.items():
            v = namespace[name]
            v.to_hdf(out_filename, tablename=h5_name, metadata=getattr(v, 'mdh', None))

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group
        from PYME.ui.custom_traits_editors import CBEditor

        editable = self.class_editable_traits()
        inputs = [tn for tn in editable if tn.startswith('input')]
        outputs = [tn for tn in editable if tn.startswith('output')]
        params = [tn for tn in editable if not (tn in inputs or tn in outputs or tn.startswith('_'))]

        return View([Item(tn) for tn in inputs] + [Item('_'), ] +
                    [Item(tn) for tn in params] + [Item('_'), ] +
                    [Item(tn) for tn in outputs], buttons=['OK', 'Cancel'])




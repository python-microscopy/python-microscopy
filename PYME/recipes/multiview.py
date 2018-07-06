from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
from PYME.IO import tabular


@register_module('MultiviewFold')
class MultiviewFold(ModuleBase):
    """
    Fold localizations from images which have been taken with an image splitting device but analysed without channel
    awareness. Images taken in this fashion will have the channels side by side. This module folds the x co-ordinate to
    overlay the different channels, using the image metadata to determine the appropriate ROI boundaries.

    The current implementation is somewhat limited as it only handles folding along the x axis, and assumes that ROI
    sizes and spacings are completely uniform.
    """
    inputName = Input('localizations')
    outputName = Output('folded')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('Unfold needs metadata')

        mapped = multiview.foldX(inp, inp.mdh)
        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('MultiviewShiftCorrect')
class MultiviewShiftCorrect(ModuleBase):
    """
    Applies chromatic shift correction to folded localization data that was acquired with an image splitting device,
    but localized without splitter awareness.

    Parameters
    ----------

    shift_map_path : str
        file path of shift map to be applied. Can also be a URL for shiftmaps stored remotely
    """
    inputName = Input('folded')
    shift_map_path = CStr('')
    outputName = Output('registered')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview
        from PYME.IO import unifiedIO
        import json

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('ShiftCorrect needs metadata')

        if self.shift_map_path == '':  # grab shftmap from the metadata
            s = unifiedIO.read(inp.mdh['Shiftmap'])
        else:
            s = unifiedIO.read(self.shift_map_path)

        shiftMaps = json.loads(s)

        mapped = tabular.mappingFilter(inp)

        dx, dy = multiview.calcShifts(mapped, shiftMaps)
        mapped.addColumn('chromadx', dx)
        mapped.addColumn('chromady', dy)

        mapped.setMapping('x', 'x + chromadx')
        mapped.setMapping('y', 'y + chromady')

        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('MultiviewFindClumps')
class MultiviewFindClumps(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('registered')
    gapTolerance = Int(1, desc='Number of off-frames allowed to still be a single clump')
    radiusScale = Float(2.0,
                        desc='Factor by which error_x is multiplied to detect clumps. The default of 2-sigma means we link ~95% of the points which should be linked')
    radius_offset = Float(0.,
                          desc='Extra offset (in nm) for cases where we want to link despite poor channel alignment')
    probeAware = Bool(False, desc='''Use probe-aware clumping. NB this option does not work with standard methods of colour
                                             specification, and splitting by channel and clumping separately is preferred''')
    outputName = Output('clumped')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        if self.probeAware and 'probe' in inp.keys():  # special case for using probe aware clumping NB this is a temporary fudge for non-standard colour handling
            mapped = multiview.probeAwareFindClumps(inp, self.gapTolerance, self.radiusScale, self.radius_offset)
        else:  # default
            mapped = multiview.findClumps(inp, self.gapTolerance, self.radiusScale, self.radius_offset)

        if 'mdh' in dir(inp):
            mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped


@register_module('MultiviewMergeClumps')
class MultiviewMergeClumps(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('clumped')
    outputName = Output('merged')
    labelKey = CStr('clumpIndex')

    def execute(self, namespace):
        from PYME.Analysis.points import multiview

        inp = namespace[self.inputName]

        try:
            grouped = multiview.mergeClumps(inp, inp.mdh.getOrDefault('Multiview.NumROIs', 0), labelKey=self.labelKey)
            grouped.mdh = inp.mdh
        except AttributeError:
            grouped = multiview.mergeClumps(inp, numChan=0, labelKey=self.labelKey)

        namespace[self.outputName] = grouped


@register_module('MapAstigZ')
class MapAstigZ(ModuleBase):
    """Create a new mapping object which derives mapped keys from original ones"""
    inputName = Input('merged')

    astigmatismMapLocation = CStr('')  # FIXME - rename and possibly change type
    rough_knot_spacing = Float(50.)

    outputName = Output('zmapped')

    def execute(self, namespace):
        from PYME.Analysis.points.astigmatism import astigTools
        from PYME.IO import unifiedIO
        import json

        inp = namespace[self.inputName]

        if 'mdh' not in dir(inp):
            raise RuntimeError('MapAstigZ needs metadata')

        if self.astigmatismMapLocation == '':  # grab calibration from the metadata
            s = unifiedIO.read(inp.mdh['Analysis.AstigmatismMapID'])
        else:
            s = unifiedIO.read(self.astigmatismMapLocation)

        astig_calibrations = json.loads(s)

        mapped = tabular.mappingFilter(inp)

        z, zerr = astigTools.lookup_astig_z(mapped, astig_calibrations, self.rough_knot_spacing, plot=False)

        mapped.addColumn('astigZ', z)
        mapped.addColumn('zLookupError', zerr)
        mapped.setMapping('z', 'astigZ + z')

        mapped.mdh = inp.mdh

        namespace[self.outputName] = mapped

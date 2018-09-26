import PYME.Analysis.points.spherical_harmonics as spharm
from PYME.recipes.localisations import SphericalHarmonicShell, AddShellMappedCoordinates
from PYME.recipes.base import ModuleCollection
from mayavi import mlab
import logging

logger=logging.getLogger(__name__)


class ShellManager(object):
    def __init__(self, vis_frame):
        self.vis_frame = vis_frame
        self.pipeline = vis_frame.pipeline

        self._shells = []

        logging.debug('Adding menu items for spherical harmonic (shell) fitting')

        vis_frame.AddMenuItem('Analysis>Spherical Harmonic Fitting', itemType='separator')

        vis_frame.AddMenuItem('Analysis>Spherical Harmonic Fitting', 'Fit Spherical Harmonic Shell',
                              self.OnCalcHarmonicRepresentation)

    def OnCalcHarmonicRepresentation(self, wx_event):
        recipe = ModuleCollection()
        recipe.namespace['input'] = self.pipeline

        shell_fitter = SphericalHarmonicShell(recipe, input_name='input', output_name='harmonic_shell')
        recipe.add_module(shell_fitter)

        if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
            return

        recipe.execute()
        shell = recipe.namespace['harmonic_shell']
        self._shells.append(shell)

        center = shell.mdh['Processing.SphericalHarmonicShell.Centre']
        z_scale = shell.mdh['Processing.SphericalHarmonicShell.ZScale']
        spharm.visualize_reconstruction(shell['modes'], shell['coefficients'], zscale=1. / z_scale)
        mlab.points3d(self.pipeline['x'] - center[0], self.pipeline['y'] - center[1],
                      self.pipeline['z'] - center[2]/z_scale, mode='point')

def Plug(vis_frame):
    vis_frame.shell_manager= ShellManager(vis_frame)
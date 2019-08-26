import PYME.Analysis.points.spherical_harmonics as spharm
from PYME.recipes.localisations import SphericalHarmonicShell, AddShellMappedCoordinates
from PYME.recipes.base import ModuleCollection

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
        recipe = self.pipeline.recipe
        recipe.trait_set(execute_on_invalidation=False)

        shell_fitter = SphericalHarmonicShell(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                              output_name='harmonic_shell')
        distance_mapper = AddShellMappedCoordinates(recipe, inputName=self.pipeline.selectedDataSourceKey,
                                                    inputSphericalHarmonics='harmonic_shell', outputName='shell_mapped')
        recipe.add_module(shell_fitter)
        recipe.add_module(distance_mapper)

        if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
            return

        try:
            recipe.execute()
        finally:  # make sure we leave things as we found them
            recipe.trait_set(execute_on_invalidation=True)
        shell = recipe.namespace['harmonic_shell']
        shell_mapped = recipe.namespace['shell_mapped']
        self._shells.append(shell)

        center = shell.mdh['Processing.SphericalHarmonicShell.Centre']
        z_scale = shell.mdh['Processing.SphericalHarmonicShell.ZScale']
        spharm.visualize_reconstruction(shell['modes'], shell['coefficients'], zscale=1. / z_scale)
        
        try:
            from mayavi import mlab
            mlab.points3d(self.pipeline['x'] - center[0], self.pipeline['y'] - center[1],
                      self.pipeline['z'] - center[2] / z_scale, mode='point')
        except ImportError:
            logger.exception('Could not import Mayavi, 3D shell display disabled')

        self.pipeline.addDataSource('shell_mapped', shell_mapped)
        self.pipeline.selectDataSource('shell_mapped')


        self.vis_frame.RefreshView()
        #self.vis_frame.CreateFoldPanel()

def Plug(vis_frame):
    vis_frame.shell_manager= ShellManager(vis_frame)
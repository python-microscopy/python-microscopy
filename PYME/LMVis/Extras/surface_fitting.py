
import logging
logger=logging.getLogger(__name__)

class SurfaceFitter(object):
    def __init__(self, vis_frame):
        self.vis_frame = vis_frame

        self.vis_frame.AddMenuItem('Analysis>Surface Fitting', "Fit Surface With Quadratic Patches",
                                   self.OnFitQuadraticPatches)
        
    
    def OnFitQuadraticPatches(self, event):
        """

        Parameters
        ----------
        event: wx.Event

        Returns
        -------

        """
        from PYME.recipes.surface_fitting import FitSurfaceWithPatches
        pipeline = self.vis_frame.pipeline

        raw_key = 'quadratic_surface_fits_raw'
        filtered_key = 'quadratic_surface_fits_filtered'
        reconstruction_key = 'quadratic_surface'

        fit_module = FitSurfaceWithPatches(pipeline.recipe, input=pipeline.selectedDataSourceKey, output_fits_raw=raw_key,
                                        output_fits_filtered=filtered_key,
                                        output_surface_reconstruction=reconstruction_key)

        if not fit_module.configure_traits(kind='modal'):
            return

        pipeline.recipe.add_module(fit_module)
        pipeline.recipe.execute()
        pipeline.addDataSource(raw_key, pipeline.recipe.namespace[raw_key], False)
        pipeline.addDataSource(filtered_key, pipeline.recipe.namespace[filtered_key], False)
        pipeline.addDataSource(reconstruction_key, pipeline.recipe.namespace[reconstruction_key], False)

        # refresh our view
        pipeline.selectDataSource(reconstruction_key)
        pipeline.Rebuild()
        
        self.vis_frame.Refresh()


class SphericalHarmonicShellManager(object):
    def __init__(self, vis_frame):
        self.vis_frame = vis_frame
        self.pipeline = vis_frame.pipeline

        self._shells = []
        self.d_angle = 0.1

        logging.debug('Adding menu items for spherical harmonic (shell) fitting')

        vis_frame.AddMenuItem('Analysis>Surface Fitting>Spherical Harmonic Fitting', itemType='separator')

        vis_frame.AddMenuItem('Analysis>Surface Fitting>Spherical Harmonic Fitting', 'Fit Spherical Harmonic Shell',
                              self.OnCalcHarmonicRepresentation)
        vis_frame.AddMenuItem('Analysis>Surface Fitting>Spherical Harmonic Fitting', 'Load Spherical Harmonic Shell',
                              self.OnLoadHarmonicRepresentation)

    def OnCalcHarmonicRepresentation(self, wx_event):
        from PYME.recipes import spherical_harmonics as spharm
        recipe = self.pipeline.recipe
        recipe.trait_set(execute_on_invalidation=False)

        shell_maker = spharm.SphericalHarmonicShell(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                                    output_name='harmonic_shell')
        recipe.add_module(shell_maker)

        if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
            return

        try:
            recipe.execute()
        finally:  # make sure we leave things as we found them
            recipe.trait_set(execute_on_invalidation=True)
        shell = recipe.namespace['harmonic_shell']

        shell_mapped = recipe.namespace['shell_mapped']
        self._shells.append(shell)

        self.pipeline.addDataSource('shell_mapped', shell_mapped)
        self.pipeline.selectDataSource('shell_mapped')

        self.vis_frame.RefreshView()
        self.vis_frame.CreateFoldPanel()

        shell._visualize_shell()

    def OnLoadHarmonicRepresentation(self, wx_event):
        import wx
        from PYME.IO import tabular, FileUtils
        from PYME.Analysis.points.spherical_harmonics.fitting import scaled_shell_from_hdf
        fdialog = wx.FileDialog(None, 'Load Spherical Harmonic Representation', wildcard='Harmonic shell (*.hdf)|*.hdf',
                                style=wx.OPEN, defaultDir=FileUtils.nameUtils.genShiftFieldDirectoryPath())
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            path = fdialog.GetPath()
            fdialog.Destroy()
        else:
            fdialog.Destroy()
            return

        shell = scaled_shell_from_hdf(path)

        points = tabular.MappingFilter(self.pipeline.selectedDataSource)
        separations, closest_points = shell.distance_to_shell((points['x'], points['y'], points['z']),
                                                              d_zenith=self.d_angle)

        self._shells.append(shell)
        shell_number = len(self._shells)
        points.addColumn('distance_to_loaded_shell%d' % shell_number, separations)
        points.addColumn('inside_shell%d' % shell_number, shell.check_inside(points['x'], points['y'], points['z']))

        self.pipeline.addDataSource('shell%d_mapped' % shell_number, points)
        self.pipeline.selectDataSource('shell%d_mapped' % shell_number)

        self.vis_frame.RefreshView()
        self.vis_frame.CreateFoldPanel()

        shell._visualize_shell(self.d_angle, (points['x'], points['y'], points['z']))


def Plug(vis_frame):
    '''Plugs this module into the gui'''
    SurfaceFitter(vis_frame)
    vis_frame.spherical_harmonic_shell_manager = SphericalHarmonicShellManager(vis_frame)

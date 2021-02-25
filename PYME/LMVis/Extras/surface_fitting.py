
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

        pipeline.recipe.add_modules_and_execute([fit_module,])
        
        pipeline.addDataSource(raw_key, pipeline.recipe.namespace[raw_key], False)
        pipeline.addDataSource(filtered_key, pipeline.recipe.namespace[filtered_key], False)
        pipeline.addDataSource(reconstruction_key, pipeline.recipe.namespace[reconstruction_key], False)

        # refresh our view
        pipeline.selectDataSource(reconstruction_key)
        pipeline.Rebuild()
        
        self.vis_frame.Refresh()

def Plug(visFr):
    '''Plugs this module into the gui'''
    SurfaceFitter(visFr)

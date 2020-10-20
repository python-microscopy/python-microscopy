
import logging

logger=logging.getLogger(__name__)


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
        from PYME.recipes import surface_fitting
        import PYME.experimental._triangle_mesh as triangle_mesh
        from PYME.LMVis.layers.mesh import TriangleRenderLayer
        recipe = self.pipeline.recipe

        shell_maker = surface_fitting.SphericalHarmonicShell(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                                             output_name='harmonic_shell')
        
        if shell_maker.configure_traits(view=shell_maker.pipeline_view_min, kind='modal'):
            recipe.add_modules_and_execute([shell_maker,])
        
            shell = recipe.namespace['harmonic_shell']
    
            shell_mapped = recipe.namespace['shell_mapped']
            self._shells.append(shell)
    
            self.pipeline.addDataSource('shell_mapped', shell_mapped)
            self.pipeline.selectDataSource('shell_mapped')
    
            # Add a surface rendering
            v, f = shell.get_mesh_vertices_faces(self.d_angle)
            surf = triangle_mesh.TriangleMesh(v, f)
            self.pipeline.dataSources['shell_surface'] = surf
    
            layer = TriangleRenderLayer(self.pipeline, dsname='shell_surface', method='shaded', cmap = 'C')
            self.vis_frame.add_layer(layer)
    
            self.vis_frame.RefreshView()

    def OnLoadHarmonicRepresentation(self, wx_event):
        import wx
        from PYME.IO import tabular, FileUtils
        from PYME.Analysis.points.spherical_harmonics import scaled_shell_from_hdf
        import PYME.experimental._triangle_mesh as triangle_mesh
        from PYME.LMVis.layers.mesh import TriangleRenderLayer

        fdialog = wx.FileDialog(None, 'Load Spherical Harmonic Representation', wildcard='Harmonic shell (*.hdf)|*.hdf',
                                style=wx.FD_OPEN, defaultDir=FileUtils.nameUtils.genShiftFieldDirectoryPath())
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
                                                              d_angles=self.d_angle)

        self._shells.append(shell)
        shell_number = len(self._shells)
        points.addColumn('distance_to_loaded_shell%d' % shell_number, separations)
        points.addColumn('inside_shell%d' % shell_number, shell.check_inside(points['x'], points['y'], points['z']))

        self.pipeline.addDataSource('shell%d_mapped' % shell_number, points)
        self.pipeline.selectDataSource('shell%d_mapped' % shell_number)

        v, f = shell.get_mesh_vertices_faces(self.d_angle)        
        surf = triangle_mesh.TriangleMesh(v, f)
        self.pipeline.dataSources['shell_surface'] = surf

        layer = TriangleRenderLayer(self.pipeline, dsname='shell_surface', method='shaded', cmap = 'C')
        self.vis_frame.add_layer(layer)

        self.vis_frame.RefreshView()
        # self.vis_frame.CreateFoldPanel()

        # shell._visualize_shell(self.d_angle, (points['x'], points['y'], points['z']))

def Plug(vis_frame):
    vis_frame.shell_manager= SphericalHarmonicShellManager(vis_frame)
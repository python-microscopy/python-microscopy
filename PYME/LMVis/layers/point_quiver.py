
from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram
from PYME.LMVis.shader_programs.WireFrameShaderProgram import WireFrameShaderProgram
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram, OITGouraudShaderProgram #, OITCompositorProgram
from PYME.LMVis.shader_programs.TesselShaderProgram import TesselShaderProgram

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List, Bool
# from pylab import cm
from PYME.misc.colormaps import cm
import numpy as np
from PYME.contrib import dispatch

from OpenGL.GL import *

class PointQuiverEngine(BaseEngine):
    """
    Engine for rendering oriented points, e.g. from single-molecule
    orientation microscopy datasets
    """
    _outlines = True
    def __init__(self):
        BaseEngine.__init__(self)
        #self.set_shader_program(WireFrameShaderProgram)
        self.set_shader_program(DefaultShaderProgram)
    
    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)

        with self.get_shader_program(gl_canvas):            
            vertices = layer.get_vertices()
            n_vertices = vertices.shape[0]
            vecs = layer.get_vecs()
            
            vec_buffer = np.empty((vertices.shape[0]+vecs.shape[0],3), dtype=vertices.dtype)
            vec_buffer[0::2,:] = vertices - layer.scaling*vecs
            vec_buffer[1::2,:] = vertices + layer.scaling*vecs
            
            glVertexPointerf(vec_buffer)
            colors = np.repeat(layer.get_colors(), 2, axis=0)

            glNormalPointerf(np.ones((vec_buffer.shape[0],3),dtype='f4'))
            glColorPointerf(colors)
            glLineWidth(3)  # slightly thick
            glDrawArrays(GL_LINES, 0, 2*n_vertices)

ENGINES = {
    'shader_quiver' : PointQuiverEngine,
}

class PointQuiverRenderLayer(EngineLayer):
    """
    Layer for viewing oriented 3D points (e.g. from single-molecule orientation microscopy).
    """
    vector_property = CStr('', desc='Name of variable used to colour our points')
    # method = Enum(*ENGINES.keys(), desc='Method used to display vectors')
    method = Enum('shader_quiver', desc='Method used to display vectors')
    scaling = Float(100.0, desc='length to draw lines for points, in nanometers')
    dsname = CStr('output', desc='Name of the datasource within the pipeline to use as source (should be a OrientedPoints object)')
    cmap = Enum(*cm.cmapnames, default='gist_rainbow', desc='Name of colourmap used to colour points')
    clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')
    alpha = Float(1.0, desc='Point tranparency')
    _datasource_choices = List()
    _datasource_keys = List()

    def __init__(self, pipeline, method='shader_quiver', dsname='', **kwargs):
        EngineLayer.__init__(self, **kwargs)
        self._pipeline = pipeline
        self.engine = None
        self.cmap = 'gist_rainbow'

        self.x_key = 'x'  # TODO - make these traits?
        self.y_key = 'y'
        self.z_key = 'z'

        self._bbox = None

        # define a signal so that people can be notified when we are updated (currently used to force a redraw when
        # parameters change)
        self.on_update = dispatch.Signal()

        # define responses to changes in various traits
        self.on_trait_change(self._update, 'vector_property')#, 'scaling', 'cmap', 'clim', 'alpha')
        self.on_trait_change(lambda: self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'dsname, scaling, cmap, clim, alpha')
        self.on_trait_change(self._set_method, 'method')

        # update any of our traits which were passed as command line arguments
        self.set(**kwargs)

        # update datasource and method
        self.dsname = dsname
        if self.method == method:
            #make sure we still call _set_method even if we start with the default method
            self._set_method()
        else:
            self.method = method

        # if we were given a pipeline, connect ourselves to the onRebuild signal so that we can automatically update
        # ourselves
        if not self._pipeline is None:
            try:
                self._pipeline.onRebuild.connect(self.update)
            except AttributeError:
                pass
    
    @property
    def datasource(self):
        """
        Return the datasource we are connected to (through our dsname property).
        """
        # return self._pipeline.get_layer_data(self.dsname)
        try:
            return self._pipeline.get_layer_data(self.dsname)
        except AttributeError:
            try:
                return self._pipeline[self.dsname]
            except AttributeError:
                return None
    
    @property
    def _ds_class(self):  # FIXME - just temporary hack, definitely wrong
        # from PYME.experimental import triangle_mesh
        from PYME.experimental import _triangle_mesh as triangle_mesh
        return triangle_mesh.TrianglesBase
    
    def update_from_datasource(self, ds):
        """
        

        Parameters
        ----------
        ds :
            PYME.IO.tabular.TabularBase

        Returns
        -------
        None
        """
        # from PYME.Analysis.points import coordinate_tools
        
        x, y, z = ds['x'], ds['y'], ds['z']

        try:
            vec_data = getattr(ds, 'orientation')
        except AttributeError:
            # pipeline often nests the datasource as a mappingfilter
            vec_data = getattr(ds.resultsSource, 'orientation')
        xv, yv, zv = vec_data
        
        if self.vector_property != '':
            colors = ds[self.vector_property]
        else:
            colors = np.ones(len(x), dtype=float)

        clim = self.clim

        cmap = cm[self.cmap]
        self._color_map = cmap
        alpha = float(self.alpha)

        cs_ = ((colors - clim[0]) / (clim[1] - clim[0]))
        cs = cmap(cs_)
        cs[:, 3] = alpha
        cs = cs.ravel().reshape(len(colors), 4)

        self._colors = cs.astype('f4')

        # Do we have coordinates? Concatenate into vertices.
        if x is not None and y is not None and z is not None:
            vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            self._vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)
            self._vecs = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T.ravel().reshape(len(x.ravel()), 3)

            self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
        else:
            self._bbox = None
    
    def _set_method(self):
        self.engine = ENGINES[self.method]()
        self.update()
    
    def _update(self, *args, **kwargs):
        #pass
        cdata = self._get_cdata()
        self.clim = [float(np.nanmin(cdata)), float(np.nanmax(cdata))]
    
    @property
    def bbox(self):
        return self._bbox
    
    def get_colors(self):
        return self._colors
    
    @property
    def colour_map(self):
        return self._color_map    
    
    def update(self, *args, **kwargs):
        try:
            self._datasource_choices = [k for k, v in self._pipeline.dataSources.items() if hasattr(v, 'orientation')]
        except AttributeError:
            pass
        
        if not self.datasource is None:
            # dks = sorted(getattr(self.datasource, 'vertex_vector_properties', []))
            
            # self._datasource_keys = dks
            self._datasource_keys = sorted(self.datasource.keys())
        
        if not (self.engine is None or self.datasource is None):
            print('lw update')
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)
    
    def _get_cdata(self):
        try:
            cdata = self.datasource[self.vector_property]
        except (KeyError, TypeError):
            cdata = np.array([0, 1])
        return cdata
    
    def get_vertices(self):
        return self._vertices
    
    def get_vecs(self):
        return self._vecs
    
    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor
        # vis_when = 'cmap not in %s' % cm.solid_cmaps
        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices'), visible_when='_datasource_choices')]),
                     #Item('method'),
                     Item('scaling'),
                     #Item('normal_mode', visible_when='method=="shaded"'),
                    #  Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour', visible_when=vis_when),
                     Item('vector_property', editor=EnumEditor(name='_datasource_keys'), label='Property'),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False)]),#, visible_when='vertexColour != "constant"'),
                     Group([Item('cmap', label='LUT'),
                           Item('alpha')#, visible_when='method in ["flat", "tessel", "shaded"]')
                           ])
                     ], )
        # buttons=['OK', 'Cancel'])
    
    def default_traits_view(self):
        return self.default_view

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


class WireframeEngine(BaseEngine):
    _outlines = True
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        self.set_shader_program(WireFrameShaderProgram)


    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)

        with self.shader_program:
            vertices = layer.get_vertices()
            n_vertices = vertices.shape[0]

            normals = layer.get_normals()
            colors = layer.get_colors()

            glVertexPointerf(vertices)
            glNormalPointerf(normals)
            glColorPointerf(colors)

            glDrawArrays(GL_TRIANGLES, 0, n_vertices)
            
            if self._outlines:
                sc = np.array([0.5, 0.5, 0.5, 1])
                glColorPointerf(colors*sc[None,:])
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glDrawArrays(GL_TRIANGLES, 0, n_vertices)

            if layer.display_normals:
                normal_buffer = np.empty((vertices.shape[0]+normals.shape[0],3), dtype=vertices.dtype)
                if layer.normal_mode == 'Per vertex':
                    normal_buffer[0::2,:] = vertices
                    normal_buffer[1::2,:] = vertices
                else:
                    vtemp = np.repeat((1./3)*(vertices[0::3]+vertices[1::3]+vertices[2::3]), 3, axis=0)
                    normal_buffer[0::2,:] = vtemp
                    normal_buffer[1::2,:] = vtemp
                assert(np.allclose(np.linalg.norm(normals,axis=1),1))
                normal_buffer[1::2,:] += layer.normal_scaling*normals
                
                glVertexPointerf(normal_buffer)
                sc = np.array([1, 1, 1, 1])
                glColorPointerf(np.ones((normal_buffer.shape[0],4),dtype=colors.dtype)*sc[None,:])  # white normals
                glNormalPointerf(np.ones((normal_buffer.shape[0],3),dtype=normals.dtype))
                glLineWidth(3)  # slightly thick
                glDrawArrays(GL_LINES, 0, 2*n_vertices)



class FlatFaceEngine(WireframeEngine):
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)

    def render(self, gl_canvas, layer):
        self.set_shader_program(DefaultShaderProgram)
        WireframeEngine.render(self, gl_canvas, layer)

class ShadedFaceEngine(WireframeEngine):
    _outlines = False
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)

    def render(self, gl_canvas, layer):
        self.set_shader_program(GouraudShaderProgram)
        WireframeEngine.render(self, gl_canvas, layer)
        
class OITShadedFaceEngine(WireframeEngine):
    _outlines = False
    
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        
    def use_oit(self, layer):
        return layer.alpha < 0.99

    def render(self, gl_canvas, layer):
        if self.use_oit(layer):
            self.set_shader_program(OITGouraudShaderProgram)
        else:
            self.set_shader_program(GouraudShaderProgram)
        WireframeEngine.render(self, gl_canvas, layer)
        
    
        
class TesselEngine(WireframeEngine):
    _outlines = False
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)

    def render(self, gl_canvas, layer):
        self.set_shader_program(TesselShaderProgram)
        WireframeEngine.render(self, gl_canvas, layer)


ENGINES = {
    'wireframe' : WireframeEngine,
    'flat' : FlatFaceEngine,
    #'shaded' : ShadedFaceEngine,
    'tessel' : TesselEngine,
    'shaded' : OITShadedFaceEngine,
}


class TriangleRenderLayer(EngineLayer):
    """
    Layer for viewing triangle meshes.
    """
    # properties to show in the GUI. Note that we also inherit 'visible' from BaseLayer
    vertexColour = CStr('constant', desc='Name of variable used to colour our points')
    cmap = Enum(*cm.cmapnames, default='gist_rainbow', desc='Name of colourmap used to colour faces')
    clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')
    alpha = Float(1.0, desc='Face tranparency')
    method = Enum(*ENGINES.keys(), desc='Method used to display faces')
    normal_mode = Enum(['Per vertex', 'Per face'])
    display_normals = Bool(False)
    normal_scaling = Float(10.0)
    dsname = CStr('output', desc='Name of the datasource within the pipeline to use as a source of triangles (should be a TriangularMesh object)')
    _datasource_choices = List()
    _datasource_keys = List()

    def __init__(self, pipeline, method='wireframe', dsname='', context=None, **kwargs):
        EngineLayer.__init__(self, context=context, **kwargs)
        self._pipeline = pipeline
        self.engine = None
        self.cmap = 'gist_rainbow'

        self.x_key = 'x'  # TODO - make these traits?
        self.y_key = 'y'
        self.z_key = 'z'

        self.xn_key = 'xn'
        self.yn_key = 'yn'
        self.zn_key = 'zn'

        self._bbox = None

        # define a signal so that people can be notified when we are updated (currently used to force a redraw when
        # parameters change)
        self.on_update = dispatch.Signal()

        # define responses to changes in various traits
        self.on_trait_change(self._update, 'vertexColour')
        self.on_trait_change(lambda: self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, alpha, dsname, normal_mode, display_normals, normal_scaling')
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
        Return the datasource we are connected to (does not go through the pipeline for triangles_mesh).
        """
        try:
            return self._pipeline.get_layer_data(self.dsname)
        except AttributeError:
            try:
                return self._pipeline[self.dsname]
            except AttributeError:
                return None
        #return self.datasource
    
    @property
    def _ds_class(self):
        # from PYME.experimental import triangle_mesh
        from PYME.experimental import _triangle_mesh as triangle_mesh
        return triangle_mesh.TrianglesBase

    def _set_method(self):
        self.engine = ENGINES[self.method](self._context)
        self.update()

    def _get_cdata(self):
        try:
            cdata = self.datasource[self.vertexColour]
        except (KeyError, TypeError):
            cdata = np.array([0, 1])

        return cdata

    def _update(self, *args, **kwargs):
        #pass
        cdata = self._get_cdata()
        self.clim = [float(np.nanmin(cdata)), float(np.nanmax(cdata))]
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        try:
            self._datasource_choices = [k for k, v in self._pipeline.dataSources.items() if isinstance(v, self._ds_class)]
        except AttributeError:
            pass
        
        if not self.datasource is None:
            dks = ['constant',]
            if hasattr(self.datasource, 'keys'):
                 dks = dks + sorted(self.datasource.keys())
            self._datasource_keys = dks
        
        if not (self.engine is None or self.datasource is None):
            print('lw update')
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)

    @property
    def bbox(self):
        return self._bbox

    def update_from_datasource(self, ds):
        """
        Pulls vertices/normals from a binary STL file. See PYME.IO.FileUtils.stl for more info. Calls update_data on the input.

        Parameters
        ----------
        ds :
            PYME.experimental.triangular_mesh.TriangularMesh object

        Returns
        -------
        None
        """
        #t = ds.vertices[ds.faces]
        #n = ds.vertex_normals[ds.faces]
        
        x, y, z = ds.vertices[ds.faces].reshape(-1, 3).T
        
        if self.normal_mode == 'Per vertex':
            xn, yn, zn = ds.vertex_normals[ds.faces].reshape(-1, 3).T
        else:
            xn, yn, zn = np.repeat(ds.face_normals.T, 3, axis=1)
            
        if self.vertexColour in ['', 'constant']:
            c = np.ones(len(x))
            clim = [0, 1]
        #elif self.vertexColour == 'vertex_index':
        #    c = np.arange(0, len(x))
        else:
            c = ds[self.vertexColour][ds.faces].ravel()
            clim = self.clim

        cmap = cm[self.cmap]
        alpha = float(self.alpha)

        # Do we have coordinates? Concatenate into vertices.
        if x is not None and y is not None and z is not None:
            vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            self._vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)

            if not xn is None:
                self._normals = np.vstack((xn.ravel(), yn.ravel(), zn.ravel())).T.ravel().reshape(len(x.ravel()), 3)
            else:
                self._normals = -0.69 * np.ones(self._vertices.shape)

            self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
        else:
            self._bbox = None

        # TODO: This temporarily sets all triangles to the color red. User should be able to select color.
        if c is None:
            c = np.ones(self._vertices.shape[0]) * 255  # vector of pink
            
        

        if clim is not None and c is not None and cmap is not None:
            cs_ = ((c - clim[0]) / (clim[1] - clim[0]))
            cs = cmap(cs_)

            if self.method in ['flat', 'tessel']:
                alpha = cs_ * alpha
            
            cs[:, 3] = alpha
            
            if self.method == 'tessel':
                cs = np.power(cs, 0.333)

            self._colors = cs.ravel().reshape(len(c), 4)
        else:
            # cs = None
            if not self._vertices is None:
                self._colors = np.ones((self._vertices.shape[0], 4), 'f')
            
        self._alpha = alpha
        self._color_map = cmap
        self._color_limit = clim


    def get_vertices(self):
        return self._vertices

    def get_normals(self):
        return self._normals

    def get_colors(self):
        return self._colors

    def get_color_map(self):
        return self._color_map

    @property
    def colour_map(self):
        return self._color_map

    def get_color_limit(self):
        return self._color_limit

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices'), visible_when='_datasource_choices')]),
                     Item('method'),
                     Item('normal_mode', visible_when='method=="shaded"'),
                     Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour'),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False), ], visible_when='vertexColour != "constant"'),
                     Group([Item('cmap', label='LUT'),
                            Item('alpha', visible_when='method in ["flat", "tessel", "shaded"]')
                            ])
                     ], )
        # buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view
from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram, OpaquePointShaderProgram
from PYME.LMVis.shader_programs.PointSpriteShaderProgram import PointSpriteShaderProgram
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram, GouraudSphereShaderProgram

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List, Bool
# from pylab import cm
from PYME.misc.colormaps import cm
import numpy as np
from PYME.contrib import dispatch

import logging
logger = logging.getLogger(__name__)

from OpenGL.GL import *

class Points3DEngine(BaseEngine):
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        self.set_shader_program(OpaquePointShaderProgram)
        self.point_scale_correction = 1.0

    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)
    
        with self.shader_program:
            vertices = layer.get_vertices()
            if vertices is None:
                return False
            
            n_vertices = vertices.shape[0]
            normals = layer.get_normals()
            colors = layer.get_colors()
        
            glVertexPointerf(vertices)
            glNormalPointerf(normals)
            glColorPointerf(colors)
        
            if gl_canvas:
                if layer.point_size == 0:
                    glPointSize(1 / gl_canvas.pixelsize)
                else:
                    glPointSize(layer.point_size*self.point_scale_correction / gl_canvas.pixelsize)
            else:
                glPointSize(layer.point_size*self.point_scale_correction)
            glDrawArrays(GL_POINTS, 0, n_vertices)

            if layer.display_normals:
                normal_buffer = np.empty((vertices.shape[0]+normals.shape[0],3), dtype=vertices.dtype)
                normal_buffer[0::2,:] = vertices
                normal_buffer[1::2,:] = vertices
                # assert(np.allclose(np.linalg.norm(normals,axis=1),1))
                normal_buffer[1::2,:] += layer.normal_scaling*normals
                
                glVertexPointerf(normal_buffer)
                sc = np.array([1, 1, 1, 1])
                glColorPointerf(np.ones((normal_buffer.shape[0],4),dtype=colors.dtype)*sc[None,:])  # white normals
                glNormalPointerf(np.ones((normal_buffer.shape[0],3),dtype=normals.dtype))
                glLineWidth(3)  # slightly thick
                glDrawArrays(GL_LINES, 0, 2*n_vertices)
            


class PointSpritesEngine(Points3DEngine):
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        self.set_shader_program(PointSpriteShaderProgram)
        self.point_scale_correction = self.shader_program.size_factor
        
class ShadedPointsEngine(Points3DEngine):
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        self.set_shader_program(GouraudShaderProgram)
        self.point_scale_correction = 1.0
        
class TransparentPointsEngine(Points3DEngine):
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        self.set_shader_program(DefaultShaderProgram)
        self.point_scale_correction = 1.0
        
class SpheresEngine(Points3DEngine):
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        self.set_shader_program(GouraudSphereShaderProgram)
        self.point_scale_correction = 1.0
        

ENGINES = {
    'points' : Points3DEngine,
    'transparent_points' : TransparentPointsEngine,
    'pointsprites' : PointSpritesEngine,
    'shaded_points' : ShadedPointsEngine,
    'spheres' : SpheresEngine,
}

class PointCloudRenderLayer(EngineLayer):
    """
    A layer for viewing point-cloud data, using one of 3 engines (indicated above)
    
    """
    
    # properties to show in the GUI. Note that we also inherit 'visible' from BaseLayer
    vertexColour = CStr('', desc='Name of variable used to colour our points')
    point_size = Float(30.0, desc='Rendered size of the points in nm')
    cmap = Enum(*cm.cmapnames, default='gist_rainbow', desc='Name of colourmap used to colour points')
    clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')
    alpha = Float(1.0, desc='Point tranparency')
    method = Enum(*ENGINES.keys(), desc='Method used to display points')
    display_normals = Bool(False)
    normal_scaling = Float(10.0)
    dsname = CStr('output', desc='Name of the datasource within the pipeline to use as a source of points')
    _datasource_keys = List()
    _datasource_choices = List()

    def __init__(self, pipeline, method='points', dsname='', context=None, **kwargs):
        EngineLayer.__init__(self, context=context, **kwargs)
        self._pipeline = pipeline
        self.engine = None
        self.cmap = 'gist_rainbow'

        self.x_key = 'x' #TODO - make these traits?
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
        self.on_trait_change(self.update, 'cmap, clim, alpha, dsname, point_size')
        self.on_trait_change(self._set_method, 'method')

        # update any of our traits which were passed as command line arguments
        self.set(**kwargs)
        
        # update datasource name and method
        #logger.debug('Setting dsname and method')
        self.dsname = dsname
        self.method = method
        
        self._set_method()
        
        # if we were given a pipeline, connect ourselves to the onRebuild signal so that we can automatically update
        # ourselves
        if not self._pipeline is None:
            self._pipeline.onRebuild.connect(self.update)


    @property
    def datasource(self):
        """
        Return the datasource we are connected to (through our dsname property).
        """
        return self._pipeline.get_layer_data(self.dsname)

    def _set_method(self):
        #logger.debug('Setting layer method to %s' % self.method)
        self.engine = ENGINES[self.method](self._context)
        self.update()

    def _get_cdata(self):
        try:
            cdata = self.datasource[self.vertexColour]
        except KeyError:
            cdata = np.array([0, 1])
    
        return cdata

    def _update(self, *args, **kwargs):
        cdata = self._get_cdata()
        self.clim = [float(np.nanmin(cdata)), float(np.nanmax(cdata))+1e-9]
        #self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        print('lw update')
        self._datasource_choices = self._pipeline.layer_data_source_names
        if not self.datasource is None:
            self._datasource_keys = sorted(self.datasource.keys())
            
        if not (self.engine is None or self.datasource is None):
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)
            


    @property
    def bbox(self):
        return self._bbox
    
    
    def update_from_datasource(self, ds):
        x, y = ds[self.x_key], ds[self.y_key]
        
        if not self.z_key is None:
            try:
                z = ds[self.z_key]
            except KeyError:
                z = 0*x
        else:
            z = 0 * x
        
        if not self.vertexColour == '':
            c = ds[self.vertexColour]
        else:
            c = 0*x
            
        if self.xn_key in ds.keys():
            xn, yn, zn = ds[self.xn_key], ds[self.yn_key], ds[self.zn_key]
            self.update_data(x, y, z, c, cmap=cm[self.cmap], clim=self.clim, alpha=self.alpha, xn=xn, yn=yn, zn=zn)
        else:
            self.update_data(x, y, z, c, cmap=cm[self.cmap], clim=self.clim, alpha=self.alpha)
    
    
    def update_data(self, x=None, y=None, z=None, colors=None, cmap=None, clim=None, alpha=1.0, xn=None, yn=None, zn=None):
        self._vertices = None
        self._normals = None
        self._colors = None
        self._color_map = None
        self._color_limit = 0
        self._alpha = 0
        
        if x is not None and y is not None and z is not None and len(x) > 0:
            vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)
            
            if not xn is None:
                normals = np.vstack((xn.ravel(), yn.ravel(), zn.ravel())).T.ravel().reshape(len(x.ravel()), 3)
            else:
                normals = -0.69 * np.ones(vertices.shape)
            
            self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
        else:
            vertices = None
            normals = None
            self._bbox = None
        
        if clim is not None and colors is not None and clim is not None:
            cs_ = ((colors - clim[0]) / (clim[1] - clim[0]))
            cs = cmap(cs_)
            cs[:, 3] = alpha
            
            cs = cs.ravel().reshape(len(colors), 4)
        else:
            if not vertices is None:
                cs = np.ones((vertices.shape[0], 4), 'f')
            else:
                cs = None

        self.set_values(vertices, normals, cs, cmap, clim, alpha)

    def set_values(self, vertices=None, normals=None, colors=None, color_map=None, color_limit=None, alpha=None):
        if vertices is not None:
            self._vertices = vertices
        if normals is not None:
            self._normals = normals
        if color_map is not None:
            self._color_map = color_map
        if colors is not None:
            self._colors = colors
        if color_limit is not None:
            self._color_limit = color_limit
        if alpha is not None:
            self._alpha = alpha
    
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
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor, TextEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        vis_when = 'cmap not in %s' % cm.solid_cmaps
    
        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     Item('method'),
                     Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour', visible_when=vis_when),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata, update_signal=self.on_update), show_label=False), ], visible_when=vis_when),
                     Group(Item('cmap', label='LUT'),
                           Item('alpha', visible_when="method in ['pointsprites', 'transparent_points']", editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                           Item('point_size', label=u'Point\u00A0size', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)))])
        #buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view
    
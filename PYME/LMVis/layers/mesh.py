from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram
from PYME.LMVis.shader_programs.WireFrameShaderProgram import WireFrameShaderProgram
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram, OITGouraudShaderProgram, OITCompositorProgram
from PYME.LMVis.shader_programs.TesselShaderProgram import TesselShaderProgram

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List
from pylab import cm
import numpy as np
import dispatch

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

            glVertexPointerf(vertices)
            glNormalPointerf(layer.get_normals())
            glColorPointerf(layer.get_colors())

            glDrawArrays(GL_TRIANGLES, 0, n_vertices)
            
            if self._outlines:
                sc = np.array([0.5, 0.5, 0.5, 1])
                glColorPointerf(layer.get_colors()*sc[None,:])
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glDrawArrays(GL_TRIANGLES, 0, n_vertices)


from PYME.LMVis.shader_programs.ShaderProgramFactory import ShaderProgramFactory
class OITEngine(BaseEngine):
    _outlines = False
    
    def __init__(self, context=None):
        BaseEngine.__init__(self, context=context)
        self.set_shader_program(OITGouraudShaderProgram)
        self.composite_shader_program = ShaderProgramFactory.get_program(OITCompositorProgram, self._context)
        
        self._w, self._h = None,None
        
        print('init_gl')
        self.init_gl()
        print('init_gl done')
        
    def init_gl(self):
        self._fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fb)

        self._accumT = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._accumT)

        self._revealT = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._revealT)

        self._db = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._db)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
    def cleanup(self):
        glDeleteFramebuffers(1, self._fb)
        glDeleteTextures(1, self._accumT)
        glDeleteTextures(1, self._revealT)
        glDeleteRenderbuffers(1, self._db)
        
    def resize_gl(self, w, h):
        #print('resize_gl')
        self._w = w
        self._h = h
    
        glViewport(0, 0, w, h)
        #self._accumdata = np.zeros([4,w,h], 'f')

        #print('bind fb')
        glBindFramebuffer(GL_FRAMEBUFFER, self._fb)

        #print('bind accum texture')
        glBindTexture(GL_TEXTURE_2D, self._accumT)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
    
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        #print('framebuffer texture')
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._accumT, 0)

        #print('bind reveal texture')
        glBindTexture(GL_TEXTURE_2D, self._revealT)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, None)
    
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self._revealT, 0)
    
        glBindTexture(GL_TEXTURE_2D, 0)

        #print('bind render buffer')
        glBindRenderbuffer(GL_RENDERBUFFER, self._db)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._db)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
    
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        #print('resize_gl done')
        
    
    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)
        w, h = gl_canvas.view_port_size
        if not ((self._w == w) and (self._h == h)):
            self.resize_gl(w,h)
            
        #print('render')
        
        # draw to an offscreen framebuffer
        current_fb = glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING) #keep a reference to the current framebuffer so we can restore after we're done
        glBindFramebuffer(GL_FRAMEBUFFER, self._fb) #bind our offscreen framebuffer
        
        
        #clear the offscreen framebuffer

        #glClearBufferfv(GL_COLOR, 0, (0., 0., 0., 1.))
        #glClearBufferfv(GL_COLOR, 1, (1., 0., 0., 0.))
        glClearColor(0.0,0.0,0.0,1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        #glDrawBuffer(GL_COLOR_ATTACHMENT0)
        #glClear(GL_COLOR_BUFFER_BIT) #| GL_DEPTH_BUFFER_BIT)
        
        
        #glDrawBuffer(GL_COLOR_ATTACHMENT1)
        #glClearColor(1.0, 0.0, 0.0, 1.0)
        #glClear(GL_COLOR_BUFFER_BIT)

        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
        
        with self.shader_program:
            vertices = layer.get_vertices()
            n_vertices = vertices.shape[0]
            
            glVertexPointerf(vertices)
            glNormalPointerf(layer.get_normals())
            glColorPointerf(layer.get_colors())
            
            glDrawArrays(GL_TRIANGLES, 0, n_vertices)
            
            if self._outlines:
                sc = np.array([0.5, 0.5, 0.5, 1])
                glColorPointerf(layer.get_colors() * sc[None, :])
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glDrawArrays(GL_TRIANGLES, 0, n_vertices)

        # set the framebuffer back
        glBindFramebuffer(GL_FRAMEBUFFER, current_fb)
        #glDrawBuffer(GL_COLOR_ATTACHMENT0)

        #print('compositing pass')
        #now do the compositing pass
        with self.composite_shader_program as c:
            # bind our pre-rendered textures
             
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self._accumT)
            self._acc_buf = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
            glUniform1i(c.get_uniform_location('accum_t'),0)

            glActiveTexture(GL_TEXTURE1)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self._revealT)
            self._reveal_buf = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
            glUniform1i(c.get_uniform_location('reveal_t'), 1)
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA)
            
            # Draw triangles to display texture on
            glColor4f(1., 1., 1., 1.)
            glBegin(GL_QUADS)
            glTexCoord2f(0., 0.) # lower left corner of image */
            glVertex3f(-1, -1, 0.0)
            glTexCoord2f(1., 0.) # lower right corner of image */
            glVertex3f(1, -1, 0.0)
            glTexCoord2f(1.0, 1.0) # upper right corner of image */
            glVertex3f(1, 1, 0.0)
            glTexCoord2f(0.0, 1.0) # upper left corner of image */
            glVertex3f(-1, 1, 0.0)
            glEnd()
            
            #unbind our textures
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D,0)
            glDisable(GL_TEXTURE_2D)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
            
        #print('render done')

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
    'shaded' : ShadedFaceEngine,
    'tessel' : TesselEngine,
    'shaded_oit' : OITEngine,
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
        self.on_trait_change(self.update, 'cmap, clim, alpha, dsname, normal_mode')
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
        self.clim = [float(cdata.min()), float(cdata.max())]
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

        cmap = getattr(cm, self.cmap)
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

        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     Item('method'),
                     Item('normal_mode', visible_when='method=="shaded"'),
                     Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour'),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False), ], visible_when='vertexColour != "constant"'),
                     Group([Item('cmap', label='LUT'),
                            Item('alpha', visible_when='method in ["flat", "tessel"]')
                            ])
                     ], )
        # buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view
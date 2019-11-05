from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram, ImageShaderProgram
from PYME.LMVis.shader_programs.WireFrameShaderProgram import WireFrameShaderProgram
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram
from PYME.LMVis.shader_programs.TesselShaderProgram import TesselShaderProgram

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List, Int
from pylab import cm
import numpy as np
import dispatch

from OpenGL.GL import *


class ImageEngine(BaseEngine):
    _outlines = True
    def __init__(self):
        BaseEngine.__init__(self)
        self.set_shader_program(ImageShaderProgram)
        
        self._texture_id = None
        self._img = None
        
    def set_texture(self, image):
        if self._texture_id is None:
            self._texture_id = glGenTextures(1)
    
        if image is None:
            return
        
        if not image is self._img:
            self._img = image
    
            image = image.T.reshape(*image.shape) #get our byte order right
    
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._texture_id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            #glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            #glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, image.shape[0], image.shape[1], 0, GL_RED, GL_FLOAT, image.astype('f4'))


    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)

        with self.shader_program:
            self.set_texture(layer._im)
            glUniform2f(self.shader_program.get_uniform_location("clim"), *layer.get_color_limit())
            
            glEnable(GL_TEXTURE_2D) # enable texture mapping */
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._texture_id) # bind to our texture, has id of 1 */
            glUniform1i(self.shader_program.get_uniform_location("im_sampler"), 0)
            
            x0, y0, x1, y1 = layer._bounds

            glDisable(GL_TEXTURE_GEN_S)
            glDisable(GL_TEXTURE_GEN_T)
            glDisable(GL_TEXTURE_GEN_R)
    
            #glColor3f(1.,0.,0.)
            glColor4f(1., 1., 1., 1.)
            glBegin(GL_QUADS)
            glTexCoord2f(0., 0.) # lower left corner of image */
            glVertex3f(x0, y0, 0.0)
            glTexCoord2f(1., 0.) # lower right corner of image */
            glVertex3f(x1, y0, 0.0)
            glTexCoord2f(1.0, 1.0) # upper right corner of image */
            glVertex3f(x1, y1, 0.0)
            glTexCoord2f(0.0, 1.0) # upper left corner of image */
            glVertex3f(x0, y1, 0.0)
            glEnd()
    
            glDisable(GL_TEXTURE_2D)
                


ENGINES = {
    'image' : ImageEngine,
}


class ImageRenderLayer(EngineLayer):
    """
    Layer for viewing images.
    """
    # properties to show in the GUI. Note that we also inherit 'visible' from BaseLayer
    cmap = Enum(*cm.cmapnames, default='gray', desc='Name of colourmap used to colour faces')
    clim = ListFloat([0, 1], desc='How our data should be scaled prior to colour mapping')
    alpha = Float(1.0, desc='Tranparency')
    method = Enum(*ENGINES.keys(), desc='Method used to display image')
    dsname = CStr('output', desc='Name of the datasource within the pipeline to use as an image')
    channel = Int(0)
    slice = Int(0)
    z_pos = Float(0)
    _datasource_choices = List()
    _datasource_keys = List()

    def __init__(self, pipeline, method='image', dsname='', **kwargs):
        self._pipeline = pipeline
        self.engine = None
        self.cmap = 'gray'

        self._bbox = None
        
        self._im_key = None

        # define a signal so that people can be notified when we are updated (currently used to force a redraw when
        # parameters change)
        self.on_update = dispatch.Signal()

        # define responses to changes in various traits
        #self.on_trait_change(self._update, 'vertexColour')
        self.on_trait_change(lambda: self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, alpha, dsname')
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
            self._pipeline.onRebuild.connect(self.update)

    @property
    def datasource(self):
        """
        Return the datasource we are connected to (does not go through the pipeline for triangles_mesh).
        """
        return self._pipeline.get_layer_data(self.dsname)
        #return self.datasource
    
    @property
    def _ds_class(self):
        # from PYME.experimental import triangle_mesh
        from PYME.IO import image
        return image.ImageStack

    def _set_method(self):
        self.engine = ENGINES[self.method]()
        self.update()


    # def _update(self, *args, **kwargs):
    #     #pass
    #     cdata = self._get_cdata()
    #     self.clim = [float(cdata.min()), float(cdata.max())]
    #     self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        self._datasource_choices = [k for k, v in self._pipeline.dataSources.items() if isinstance(v, self._ds_class)]
        
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

        Parameters
        ----------
        ds :
            PYME.IO.image.ImageStack object

        Returns
        -------
        None
        """

        
        clim = self.clim
        cmap = getattr(cm, self.cmap)
        alpha = float(self.alpha)
        
        c0, c1 = clim
        
        im_key = (self.dsname, self.slice, self.channel)
        
        if not self._im_key == im_key:
            self._im_key = im_key
            self._im = ds.data[:,:,self.slice, self.channel].astype('f4')# - c0)/(c1-c0)
        
            x0, y0, x1, y1, _, _ = ds.imgBounds.bounds

            self._bbox = np.array([x0, y0, 0, x1, y1, 0])
        
            self._bounds = [x0, y0, x1, y1]
            
        self._alpha = alpha
        self._color_map = cmap
        self._color_limit = clim

    def get_color_map(self):
        return self._color_map

    @property
    def colour_map(self):
        return self._color_map

    def get_color_limit(self):
        return self._color_limit
    
    def _get_cdata(self):
        return self._im.ravel()[::20]

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     #Item('method'),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata), show_label=False), ]),
                     Group([Item('cmap', label='LUT'),
                            Item('alpha', visible_when='method in ["flat", "tessel"]')
                            ])
                     ], )
        # buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view
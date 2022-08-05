from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram, ImageShaderProgram
from PYME.LMVis.shader_programs.WireFrameShaderProgram import WireFrameShaderProgram
from PYME.LMVis.shader_programs.GouraudShaderProgram import GouraudShaderProgram
from PYME.LMVis.shader_programs.TesselShaderProgram import TesselShaderProgram

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List, Int, observe
# from pylab import cm
from PYME.misc.colormaps import cm
import numpy as np
from PYME.contrib import dispatch

from OpenGL.GL import *


class ImageEngine(BaseEngine):
    _outlines = True
    def __init__(self):
        BaseEngine.__init__(self)
        self.set_shader_program(ImageShaderProgram)
        
        self._texture_id = None
        self._lut_id = None
        self._img = None
        self._lut = None
        
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

    def set_lut(self, lut):
        if self._lut_id is None:
            self._lut_id = glGenTextures(1)
    
        if lut is None:
            return
    
        if not lut is self._lut:
            self._lut = lut
            
            lut_array = lut(np.linspace(0, 1.0, 255))
            
            print(lut_array.shape, lut_array[-1])
        
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_1D, self._lut_id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            #glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, lut_array.shape[0], 0, GL_RGBA, GL_FLOAT,
                         lut_array.astype('f4'))


    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)

        with self.get_shader_program(gl_canvas) as sp:
            self.set_lut(layer.colour_map)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_1D, self._lut_id) # bind to our texture, has id of 1 */
            glUniform1i(sp.get_uniform_location("lut"), 1)
            
            self.set_texture(layer._im)
            glUniform2f(sp.get_uniform_location("clim"), *layer.get_color_limit())
            
            glEnable(GL_TEXTURE_2D) # enable texture mapping */
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._texture_id) # bind to our texture, has id of 1 */
            glUniform1i(sp.get_uniform_location("im_sampler"), 0)
            
            x0, y0, x1, y1 = layer._bounds

            glDisable(GL_TEXTURE_GEN_S)
            glDisable(GL_TEXTURE_GEN_T)
            glDisable(GL_TEXTURE_GEN_R)
    
            #glColor3f(1.,0.,0.)
            glColor4f(1., 1., 1., 1.)
            glBegin(GL_QUADS)
            glTexCoord2f(0., 0.) # lower left corner of image */
            glVertex3f(x0, y0, layer.z_nm)
            glTexCoord2f(1., 0.) # lower right corner of image */
            glVertex3f(x1, y0, layer.z_nm)
            glTexCoord2f(1.0, 1.0) # upper right corner of image */
            glVertex3f(x1, y1, layer.z_nm)
            glTexCoord2f(0.0, 1.0) # upper left corner of image */
            glVertex3f(x0, y1, layer.z_nm)
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
    #slice = Int(0)
    z_pos = Int(0)
    t_pos = Int(0)
    _datasource_choices = List()
    _datasource_keys = List()

    def __init__(self, pipeline, method='image', dsname='', display_opts=None, **kwargs):
        EngineLayer.__init__(self, **kwargs)
        self._pipeline = pipeline
        self.engine = None
        self.cmap = 'gray'

        self._bbox = None
        self._do = display_opts #a dh5view display_options instance - if provided, this over-rides the the clim, cmap properties
        
        self._im_key = None
        self._zn_nm = 0

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
        if (not self._pipeline is None) and hasattr(pipeline, 'onRebuild'):
            self._pipeline.onRebuild.connect(self.update)

        if self._do:
            self.sync_to_display_opts()
            self._do.WantChangeNotification.append(self.sync_to_display_opts)

    @property
    def datasource(self):
        """
        Return the datasource we are connected to (does not go through the pipeline for triangles_mesh).
        """
        try:
            return self._pipeline.get_layer_data(self.dsname)
        except AttributeError:
            #fallback if pipeline is a dictionary
            return self._pipeline[self.dsname]
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

    @observe('z_pos, t_pos')
    def update(self, *args, **kwargs):
        try:
            self._datasource_choices = [k for k, v in self._pipeline.dataSources.items() if isinstance(v, self._ds_class)]
        except AttributeError:
            self._datasource_choices = [k for k, v in self._pipeline.items() if
                                        isinstance(v, self._ds_class)]
        
        if not (self.engine is None or self.datasource is None):
            print('lw update')
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)

    @property
    def z_nm(self):
        return self._z_nm

    @property
    def bbox(self):
        return self._bbox
    
    def sync_to_display_opts(self, do=None):
        if (do is None):
            if not (self._do is None):
                do = self._do
            else:
                return

        o = do.Offs[self.channel]
        g = do.Gains[self.channel]
        clim = [o, o + 1.0 / g]

        cmap = do.cmaps[self.channel].name
        visible = do.show[self.channel]
        
        self.set(clim=clim, cmap=cmap, visible=visible, z_pos=do.zp, t_pos=do.tp)
        

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

        
        #if self._do is not None:
            # Let display options (if provied) over-ride our settings (TODO - is this the right way to do this?)
        #    o = self._do.Offs[self.channel]
        #    g = self._do.Gains[self.channel]
        #    clim = [o, o + 1.0/g]
            #self.clim = clim
            
        #    cmap = self._do.cmaps[self.channel]
            #self.visible = self._do.show[self.channel]
        #else:
        
        clim = self.clim
        cmap = cm[self.cmap]
            
        alpha = float(self.alpha)
        
        c0, c1 = clim
        
        im_key = (self.dsname, self.z_pos, self.t_pos, self.channel)
        
        if not self._im_key == im_key:
            self._im_key = im_key
            self._im = ds.data_xyztc[:,:,self.z_pos, self.t_pos,self.channel].astype('f4').squeeze()# - c0)/(c1-c0)
        
            x0, y0, x1, y1, z0, z1 = ds.imgBounds.bounds

            self._z_nm = z0 + self.z_pos*ds.voxelsize_nm.z 

            self._bbox = np.array([x0, y0, z0, x1, y1, z1])
        
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
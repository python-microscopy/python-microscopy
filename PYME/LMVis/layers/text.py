from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram, TextShaderProgram

from OpenGL.GL import *
import numpy as np

class Text(BaseEngine):
    """
    A text engine - the idea is for this to be used slightly differently than other rendering engines, in that,
    e.g. an annotation layer might have multiple Text instances for each annotation, rather than a 1:1 mapping between
    a layer and an engine as with other layers.
    
    As currently written, it will only produce suitably sized text when used in a layer which uses screen pixel coords
    (i.e. projection and modelview matrices are the identity). This is currently just the LUTOverlayLayer.
    
    TODO - allow scaled coordinates
    TODO - allow 3D positions
    TODO - rotation
    TODO - text colour
    TODO - Shader. My first instinct here would be to make the text values (as derived from the array) control transparency.
           This should give an attractive anti-aliasing/blending effect, but we'd need to try it out in practice.
    """
    def __init__(self, text='', pos=(0,0)):
        BaseEngine.__init__(self)
        self.set_shader_program(TextShaderProgram)
        
        self._texture_id = None
        self._img = None
        
        self.text = text
        self.pos = pos
        
        
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, val):
        self._text = val
        
        #just take red channel
        self._im = np.ascontiguousarray(self.gen_text_image(self._text)[:, :, 0]/255.)
        self._h, self._w = self._im.shape
    
    
    @classmethod
    def gen_text_image(cls, text='', size=12, font='courier'):
        # TODO - implement size, font, etc ...
        import wx
        
        dc = wx.MemoryDC()
        w, h = dc.GetTextExtent(text)
        w, h = max(w, 1), max(h,1)
        bmp = wx.Bitmap(w, h)
        dc.SelectObject(bmp)
        dc.SetTextForeground(wx.WHITE)
        dc.DrawText(text, 0, 0)
        dc.SelectObject(wx.NullBitmap)
        
        im = np.zeros([h, w, 3], dtype='uint8')
        bmp.CopyToBuffer(im.data)
        
        return im
        
        
    
    def set_texture(self, image):
        if self._texture_id is None:
            self._texture_id = glGenTextures(1)
        
        if image is None:
            return
        
        if not image is self._img:
            self._img = image
            
            #image = image.T.reshape(*image.shape) #get our byte order right
            
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
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, image.shape[1], image.shape[0], 0, GL_RED, GL_FLOAT,
                         image.astype('f4'))
    
    def render(self, gl_canvas):
        with self.get_shader_program(gl_canvas) as sp:
            self.set_texture(self._im)
        
            glEnable(GL_TEXTURE_2D) # enable texture mapping */
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._texture_id) # bind to our texture, has id of 1 */
            glUniform1i(sp.get_uniform_location("im_sampler"), 0)
        
            # FIXME - choose appropriately for current viewport - want to make it so that the text renders real size
            # (i.e. unwind any model-view / and projection stuff).
            x0, y0 = self.pos
            x1 = x0 + self._w #*scale
            y1 = y0 + self._h #*scale
        
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
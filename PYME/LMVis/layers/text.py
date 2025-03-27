from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram, TextShaderProgram

from OpenGL.GL import *
from OpenGL import GLU
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
    def __init__(self, text='', pos=(0,0), color=None, font_size=10, **kwargs):
        BaseEngine.__init__(self)
        self.set_shader_program(TextShaderProgram)
        
        self._texture_id = None
        self._img = None
        self._color = color
        self._font_size = font_size

        self._im_key = None
        self._im = None
        
        self.text = text
        self.pos = pos
        
        
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, val):
        self._text = val
        self._im_key = None
        
        #just take red channel
        #self._im = np.ascontiguousarray(self.gen_text_image(self._text)[:, :, 0]/255.)
        #self._h, self._w = self._im.shape
    
    
    @classmethod
    def gen_text_image(cls, text='', size=10, dip_scale=1.0):
        # TODO - implement size, font, etc ...
        import wx
        
        dc = wx.MemoryDC()
        # TODO - use CreateWithDIPSize to avoid manually scaling font size
        dc.SetFont(wx.Font(wx.FontInfo(size*dip_scale)))
        w, h = dc.GetTextExtent(text)
        w, h = max(w, 1), max(h,1)
        # TODO - use CreateWithDIPSize
        #bmp = wx.Bitmap.CreateWithDIPSize(w, h, dip_scale)
        bmp = wx.Bitmap(w, h)
        dc.SelectObject(bmp)
        dc.SetTextForeground(wx.WHITE)
        dc.DrawText(text, 0, 0)
        dc.SelectObject(wx.NullBitmap)
        
        im = np.zeros([h, w, 3], dtype='uint8')
        bmp.CopyToBuffer(im.data)
        
        return im
        
    def get_img_array(self, gl_canvas):
        im_key = (self.text, self._font_size, gl_canvas.content_scale_factor)

        if im_key != self._im_key:
            self._im_key = im_key
            self._im = np.ascontiguousarray(self.gen_text_image(self.text, size=self._font_size, dip_scale=gl_canvas.content_scale_factor)[:, :, 0]/255.)

        h, w = self._im.shape
        return self._im, h, w 
    
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
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            #glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            #glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            #glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, image.shape[1], image.shape[0], 0, GL_RED, GL_FLOAT,
                         image.astype('f4'))
    
    def render(self, gl_canvas):
        im, h, w = self.get_img_array(gl_canvas)
        with self.get_shader_program(gl_canvas) as sp:
            self.set_texture(im)
            
            if gl_canvas.core_profile:
                from PYME.LMVis import mv_math as mm
                # find the on-screen position of our text
                vp = glGetIntegerv(GL_VIEWPORT)
                #print('vp:', vp)

                pos = np.zeros(4)
                pos[:len(self.pos)] = self.pos
                pos[3] = 1.0
                x0, y0, _, _ = np.dot(gl_canvas.mvp, pos)

                #print('textPos:', x0, y0)
                # convert to screen pixel coordinates

                x0  = (x0 + 1.) * vp[2] / 2.
                y0 = (y0 + 1.) * vp[3] / 2.

                #print('textPos (px):', x0, y0)

                verts = self._gen_rect_triangles(x0, y0, w, -h)
                tex_coords = self._gen_rect_texture_coords()

                if self._color is None:
                    cols = np.ones([6, 4], dtype='f4')
                else:
                    cols = np.tile(self._color, 6).astype('f')
                    #print(f'text cols: {cols}')

                proj = mm.ortho(0, vp[2], 0, vp[3], -1000, 1000)
                #print('proj:', proj)
                sp.set_modelviewprojectionmatrix(np.array(proj))

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self._texture_id) # bind to our texture, has id of 1 */
                glUniform1i(sp.get_uniform_location("im_sampler"), 0)

                vao = glGenVertexArrays(1)
                vbo = glGenBuffers(3)
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
                glBufferData(GL_ARRAY_BUFFER, verts, GL_STATIC_DRAW)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
                glEnableVertexAttribArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
                glBufferData(GL_ARRAY_BUFFER, tex_coords, GL_STATIC_DRAW)
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
                glEnableVertexAttribArray(1)
                glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
                glBufferData(GL_ARRAY_BUFFER, cols, GL_STATIC_DRAW)
                glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, None)
                glEnableVertexAttribArray(2)

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDrawArrays(GL_TRIANGLES, 0, 6)

                glBindVertexArray(0)
                glDeleteVertexArrays(1, [vao,])
                glDeleteBuffers(3, vbo)

            else:
                #legacy mode
                mv = glGetDoublev(GL_MODELVIEW_MATRIX)
                p = glGetDoublev(GL_PROJECTION_MATRIX)
                

                #print(mv, p, vp)

                # calculate on-screen pixel coordinates for our text (perform model-view transformation)
                pos = np.zeros(3)
                pos[:len(self.pos)] = self.pos
                x0, y0, _ = GLU.gluProject(pos[0], pos[1], pos[2], mv, p, vp)

                #print('textPos:', x0, y0)

                try:
                    # set model-view so that we are drawing in screen pixel coordinates
                    
                    glMatrixMode(GL_PROJECTION)
                    glPushMatrix()
                    glLoadIdentity()
                    glOrtho(0, vp[2], 0, vp[3], -1000, 1000)

                    glMatrixMode(GL_MODELVIEW)
                    glPushMatrix()
                    glLoadIdentity()

            
                    glEnable(GL_TEXTURE_2D) # enable texture mapping */
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, self._texture_id) # bind to our texture, has id of 1 */
                    glUniform1i(sp.get_uniform_location("im_sampler"), 0)
                
                    # FIXME - choose appropriately for current viewport - want to make it so that the text renders real size
                    # (i.e. unwind any model-view / and projection stuff).
                    #x0, y0 = self.pos
                    x1 = x0 + w#*gl_canvas.content_scale_factor
                    y1 = y0 - h#*gl_canvas.content_scale_factor
                
                    glDisable(GL_TEXTURE_GEN_S)
                    glDisable(GL_TEXTURE_GEN_T)
                    glDisable(GL_TEXTURE_GEN_R)
                
                    #glColor3f(1.,0.,0.)
                    if self._color is not None:
                        glColor4f(*self._color)
                    else:
                        glColor4f(1., 1., 1., 1.)

                    #vertex_data = np.array([[x0, y0, 0.0], 
                    #                        [x1, y0, 0.0], 
                    #                        [x1, y1, 0.0], 
                    #                        [x0, y1, 0.0]], dtype='f4') 

                    #_vbo = glGenBuffers(1)
                    #glBindBuffer(GL_ARRAY_BUFFER, _vbo)
                    #glBufferData(GL_ARRAY_BUFFER, vertex_data, GL_STATIC_DRAW)   

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
                finally:
                    glMatrixMode(GL_PROJECTION)
                    glPopMatrix()
                    glMatrixMode(GL_MODELVIEW)
                    glPopMatrix()
#!/usr/bin/python

# LUTOverlayLayer.py
#
# Copyright Michael Graff
#   graff@hm.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy
import numpy as np
# import pylab

from PYME.recipes.traits import Bool

from PYME.LMVis.layers.OverlayLayer import OverlayLayer
from OpenGL.GL import *
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram, ImageShaderProgram


class _LUTOverlayLayer(OverlayLayer):
    """
    This OverlayLayer produces a bar that indicates the given color map.
    """

    show_bounds = Bool(False)
    
    def __init__(self, offset=None, **kwargs):
        """

        Parameters
        ----------

        offset      offset of the canvas origin where it should be drawn.
                    Currently only offset[0] is used
        """
        if not offset:
            offset = [10, 10]
            
        OverlayLayer.__init__(self, offset, **kwargs)

        self.set_offset(offset)

        self._lut_width_px = 10.0
        self._border_colour = [.5, .5, 0]
        self.set_shader_program(DefaultShaderProgram)
        
        self._labels = {}
        
    def _get_label(self, layer):
        from . import text
        try:
            return self._labels[layer]
        except KeyError:
            self._labels[layer] = [text.Text(), text.Text()]

            return self._labels[layer]


    def render(self, gl_canvas):
        if not self.visible:
            return
        
        labels = []
        
        with self.get_shader_program(gl_canvas) as sp:
            sp.clear_shader_clipping()
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            glDisable(GL_BLEND)
            #view_size_x = gl_canvas.xmax - gl_canvas.xmin
            #view_size_y = gl_canvas.ymax - gl_canvas.ymin
            
            view_size_x, view_size_y = gl_canvas.Size

            # upper right y
            lb_ur_y = .1 * view_size_y
            # lower right y
            lb_lr_y = 0.9 * view_size_y

            lb_len =lb_lr_y -  lb_ur_y
            
            lb_width = self._lut_width_px #* view_size_x / gl_canvas.Size[0]
            
            visible_layers = [l for l in gl_canvas.layers if (getattr(l, 'visible', True) and getattr(l, 'show_lut', True))]
            
            
            
            for j, l in enumerate(visible_layers):
                cmap = l.colour_map
                
                # upper right x
                lb_ur_x = view_size_x - self.get_offset()[0] - j*1.5*lb_width
                
                # upper left x
                lb_ul_x = lb_ur_x - lb_width
                
                #print(lb_ur_x, lb_ur_y, lb_lr_y, lb_ul_x, lb_width, lb_len, view_size_x, view_size_y)

                glBegin(GL_QUAD_STRIP)
    
                for i in numpy.arange(0, 1.01, .01):
                    glColor3fv(cmap(i)[:3])
                    glVertex2f(lb_ul_x, lb_ur_y + (1.- i) * lb_len)
                    glVertex2f(lb_ur_x, lb_ur_y + (1.- i) * lb_len)
    
                glEnd()
    
                glBegin(GL_LINE_LOOP)
                glColor3fv(self._border_colour)
                glVertex2f(lb_ul_x, lb_lr_y)
                glVertex2f(lb_ur_x, lb_lr_y)
                glVertex2f(lb_ur_x, lb_ur_y)
                glVertex2f(lb_ul_x, lb_ur_y)
                glEnd()
                
                if hasattr(l, 'clim') and self.show_bounds:
                    tl, tu = self._get_label(l)
                    cl, cu = l.clim
                    tu.text = '%.3G' % cu
                    tl.text = '%.3G' % cl
                    
                    xc = lb_ur_x - 0.5*lb_width

                    _, hu, wu = tu.get_img_array(gl_canvas)
                    _, hl, wl = tl.get_img_array(gl_canvas)
                    
                    tu.pos = (xc - wu/2, lb_ur_y - hu)
                    tl.pos = (xc - wl/2, lb_lr_y)
                    
                    labels.extend([tl, tu])
                
        for l in labels:
            l.render(gl_canvas)
                
class LUTOverlayLayer(OverlayLayer):
    """
    This OverlayLayer produces a bar that indicates the given color map.
    """

    show_bounds = Bool(False)
    
    def __init__(self, offset=None, **kwargs):
        """

        Parameters
        ----------

        offset      offset of the canvas origin where it should be drawn.
                    Currently only offset[0] is used
        """
        if not offset:
            offset = [10, 10]
            
        OverlayLayer.__init__(self, offset, **kwargs)

        self.set_offset(offset)

        self._lut_width_px = 10.0
        self._border_colour = [.5, .5, 0]
        self.set_shader_program(ImageShaderProgram)
        
        self._labels = {}

        self._texture_id = None
        self._lut_id = None
        #self._lut_key = None
        
    def _get_label(self, layer):
        from . import text
        try:
            return self._labels[layer]
        except KeyError:
            self._labels[layer] = [text.Text(), text.Text()]

            return self._labels[layer]
        
    def set_texture(self):
        if self._texture_id is None:
            self._texture_id = glGenTextures(1)
    
            image = np.linspace(0, 1, 255).reshape([1, 255]).astype('f')# image.T.reshape(*image.shape) #get our byte order right
    
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
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, image.shape[0], image.shape[1], 0, GL_RED, GL_FLOAT, image.astype('f4'))
        else:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._texture_id)

    def set_lut(self, cmap):
        if self._lut_id is None:
            self._lut_id = glGenTextures(1)
    
        lut_array = cmap(np.linspace(0, 1.0, 255))
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_1D, self._lut_id)


        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        #glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        #glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, lut_array.shape[0], 0, GL_RGBA, GL_FLOAT,
                    lut_array.astype('f4'))


    def render(self, gl_canvas):
        if not self.visible:
            return
        
        labels = []
        
        with self.get_shader_program(gl_canvas) as sp:
            sp.clear_shader_clipping()
            glDisable(GL_DEPTH_TEST)
            #glDisable(GL_LIGHTING)
            glDisable(GL_BLEND)

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
            #view_size_x = gl_canvas.xmax - gl_canvas.xmin
            #view_size_y = gl_canvas.ymax - gl_canvas.ymin
            
            view_size_x, view_size_y = gl_canvas.Size

            # upper right y
            lb_ur_y = .1 * view_size_y
            # lower right y
            lb_lr_y = 0.9 * view_size_y

            lb_len =lb_lr_y -  lb_ur_y
            
            lb_width = self._lut_width_px #* view_size_x / gl_canvas.Size[0]
            
            visible_layers = [l for l in gl_canvas.layers if (getattr(l, 'visible', True) and getattr(l, 'show_lut', True))]
            
            self.set_texture()
            glUniform1i(sp.get_uniform_location('im_sampler'), 0)

            if gl_canvas.core_profile:
                sp.set_modelviewprojectionmatrix(np.array(gl_canvas.mvp, 'f'))
            
            for j, l in enumerate(visible_layers):
                cmap = l.colour_map
                self.set_lut(cmap)
                glUniform1i(sp.get_uniform_location('lut'), 1)
                glUniform2f(sp.get_uniform_location('clim'), 0, 1)
                
                # upper right x
                lb_ur_x = view_size_x - self.get_offset()[0] - j*1.5*lb_width
                
                # upper left x
                lb_ul_x = lb_ur_x - lb_width
                
                #print(lb_ur_x, lb_ur_y, lb_lr_y, lb_ul_x, lb_width, lb_len, view_size_x, view_size_y)

                verts = self._gen_rect_triangles(lb_ul_x, lb_ur_y, lb_width , lb_len)
                tex_coords = self._gen_rect_texture_coords()
                
                # glBegin(GL_QUAD_STRIP)
    
                # for i in numpy.arange(0, 1.01, .01):
                #     glColor3fv(cmap(i)[:3])
                #     glVertex2f(lb_ul_x, lb_ur_y + (1.- i) * lb_len)
                #     glVertex2f(lb_ur_x, lb_ur_y + (1.- i) * lb_len)
    
                # glEnd()

                if gl_canvas.core_profile:
                    vao = glGenVertexArrays(1)
                    vbo = glGenBuffers(2)

                    glBindVertexArray(vao)
                    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
                    glBufferData(GL_ARRAY_BUFFER, verts, GL_STATIC_DRAW)
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
                    glEnableVertexAttribArray(0)
                    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
                    glBufferData(GL_ARRAY_BUFFER, tex_coords, GL_STATIC_DRAW)
                    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
                    glEnableVertexAttribArray(1)
                else:
                    glVertexPointerf(verts)
                    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                    glTexCoordPointerf(tex_coords)
                
                glDrawArrays(GL_TRIANGLES, 0, 6)

                if gl_canvas.core_profile:
                    glBindVertexArray(0)
                    glDeleteVertexArrays(1, [vao,])
                    glDeleteBuffers(1, vbo)
                else:
                    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    
                # glBegin(GL_LINE_LOOP)
                # glColor3fv(self._border_colour)
                # glVertex2f(lb_ul_x, lb_lr_y)
                # glVertex2f(lb_ur_x, lb_lr_y)
                # glVertex2f(lb_ur_x, lb_ur_y)
                # glVertex2f(lb_ul_x, lb_ur_y)
                # glEnd()
                
                if hasattr(l, 'clim') and self.show_bounds:
                    tl, tu = self._get_label(l)
                    cl, cu = l.clim
                    tu.text = '%.3G' % cu
                    tl.text = '%.3G' % cl
                    
                    xc = lb_ur_x - 0.5*lb_width

                    _, hu, wu = tu.get_img_array(gl_canvas)
                    _, hl, wl = tl.get_img_array(gl_canvas)
                    
                    tu.pos = (xc - wu/2, lb_ur_y - hu)
                    tl.pos = (xc - wl/2, lb_lr_y)
                    
                    labels.extend([tl, tu])
                
        for l in labels:
            l.render(gl_canvas)
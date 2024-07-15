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
# import pylab

from PYME.recipes.traits import Bool

from PYME.LMVis.layers.OverlayLayer import OverlayLayer
from OpenGL.GL import *
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram


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
        
        self._clear_shader_clipping(gl_canvas)
        labels = []
        
        with self.get_shader_program(gl_canvas):
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
                

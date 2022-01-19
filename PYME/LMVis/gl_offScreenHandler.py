#!/usr/bin/python

# gl_offScreenHandler.py
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
from OpenGL.GL import *


class OffScreenHandler(object):
    def __init__(self, viewport_size, mode, multisample_buffers=4):
        self._viewport_size = viewport_size
        self._mode = mode
        self._snap = None
        self._multisample_buffers=multisample_buffers
        self._frame_buffer_object, self._render_buffer, self._depth_buffer, self._ss_framebuffer, self._ss_renderbuffer = self.setup_off_screen()

    def __enter__(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self._frame_buffer_object)
        glPushAttrib(GL_VIEWPORT_BIT)
        print('Viewport: ', self._viewport_size)
        glViewport(0, 0, self._viewport_size[0], self._viewport_size[1])
        glDrawBuffer(GL_COLOR_ATTACHMENT0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._multisample_buffers > 1:
            width, height = self._viewport_size
            # copy rendered image from MSAA (multi-sample) to normal (single-sample)
            # NOTE: The multi samples at a pixel in read buffer will be converted
            # to a single sample at the target pixel in draw buffer.
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._frame_buffer_object) # src FBO (multi-sample)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._ss_framebuffer)     # dst FBO (single-sample)
    
            glBlitFramebuffer(0, 0, width, height,             # src rect
                      0, 0, width, height,             # dst rect
                      GL_COLOR_BUFFER_BIT,             # buffer mask
                      GL_LINEAR)                      # scale filter
    
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._ss_framebuffer)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        # glNamedFramebufferReadBuffer(self._frame_buffer_object, GL_COLOR_ATTACHMENT0) # only OpenGL 4.5
        self._snap = glReadPixelsf(0, 0, self._viewport_size[0], self._viewport_size[1], self._mode)
        if self._mode == GL_LUMINANCE:
            self._snap.strides = (4, 4 * self._snap.shape[0])
        elif self._mode == GL_RGB:
            self._snap.strides = (12, 12 * self._snap.shape[0], 4)
        else:
            raise RuntimeError('{} is not a supported mode.'.format(self._mode))
        glPopAttrib()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def __del__(self):
        #print(repr(self._frame_buffer_object), self._render_buffer, self._depth_buffer, self._ss_framebuffer)
        glDeleteFramebuffers(1, [self._frame_buffer_object,])
        glDeleteRenderbuffers(2, [self._render_buffer, self._depth_buffer])
        
        if not self._ss_framebuffer is None:
            glDeleteFramebuffers(1, [self._ss_framebuffer,])
            glDeleteRenderbuffers(1, [self._ss_renderbuffer, ])

    def get_viewport_size(self):
        return self.get_viewport_size()

    def get_snap(self):
        return self._snap

    def setup_off_screen(self):
        if any([self._viewport_size[ind] >= GL_MAX_RENDERBUFFER_SIZE for ind in range(2)]):
            raise RuntimeError('Both width and height (%d, %d) must be smaller than GL_MAX_RENDERBUFFER_SIZE: %d' % (
            self._viewport_size[0], self._viewport_size[1], GL_MAX_RENDERBUFFER_SIZE))
        
        # Create colour renderbuffer
        render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, render_buffer)
        
        fmt = GL_RGB8 if (self._mode == GL_RGB) else GL_LUMINANCE
        
        if self._multisample_buffers > 1:
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, self._multisample_buffers, fmt, self._viewport_size[0], self._viewport_size[1])
        else:
            glRenderbufferStorage(GL_RENDERBUFFER, fmt, self._viewport_size[0], self._viewport_size[1])

        # create depth buffer
        depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
        if self._multisample_buffers > 1:
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, self._multisample_buffers, GL_DEPTH_COMPONENT, self._viewport_size[0], self._viewport_size[1])
        else:
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self._viewport_size[0], self._viewport_size[1])
        
        # create frame buffer
        frame_buffer_object = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object)
        
        # attach colour buffer
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, render_buffer)
        
        # attach depth buffer
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer)
        
        
        #if we are multisampling, also generate a single-sampled buffer for readout
        ss_framebuffer = None
        ss_renderbuffer = None
        if self._multisample_buffers > 1:
            # Create colour renderbuffer  (don't need depthbuffer as we're just doing a 2D blit)
            ss_renderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, ss_renderbuffer)
    
            fmt = GL_RGB8 if (self._mode == GL_RGB) else GL_LUMINANCE
            glRenderbufferStorage(GL_RENDERBUFFER, fmt, self._viewport_size[0], self._viewport_size[1])

            ss_framebuffer = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, ss_framebuffer)

            # attach colour buffer
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ss_renderbuffer)
            
        
        return frame_buffer_object, render_buffer, depth_buffer, ss_framebuffer, ss_renderbuffer

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
    def __init__(self, viewport_size, mode):
        self._viewport_size = viewport_size
        self._mode = mode
        self._snap = None
        (self._frame_buffer_object, self._render_buffer) = self.setup_off_screen()

    def __enter__(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self._frame_buffer_object)
        glPushAttrib(GL_VIEWPORT_BIT)
        print('Viewport: ', self._viewport_size)
        glViewport(0, 0, self._viewport_size[0], self._viewport_size[1])
        glDrawBuffer(GL_COLOR_ATTACHMENT0)

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        glDeleteFramebuffers(1, self._frame_buffer_object)
        glDeleteRenderbuffers(1, self._render_buffer)

    def get_viewport_size(self):
        return self.get_viewport_size()

    def get_snap(self):
        return self._snap

    def setup_off_screen(self):
        frame_buffer_object = glGenFramebuffers(1)
        render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, render_buffer)
        if any([self._viewport_size[ind] >= GL_MAX_RENDERBUFFER_SIZE for ind in range(2)]):
            raise RuntimeError('Both width and height (%d, %d) must be smaller than GL_MAX_RENDERBUFFER_SIZE: %d' % (self._viewport_size[0], self._viewport_size[1], GL_MAX_RENDERBUFFER_SIZE))
        if self._mode == GL_RGB:
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, self._viewport_size[0], self._viewport_size[1])
        else:
            glRenderbufferStorage(GL_RENDERBUFFER, GL_LUMINANCE, self._viewport_size[0], self._viewport_size[1])
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, render_buffer)
        return frame_buffer_object, render_buffer

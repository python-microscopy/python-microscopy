#!/usr/bin/python

##################
# gl_render3D.py
#
# Copyright David Baddeley, Michael Graff
# graff@hm.edu
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
##################
from math import floor

import numpy
import numpy as np
import wx
import wx.glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *

from PYME.LMVis.layers import AxesOverlayLayer, LUTOverlayLayer, ScaleBarOverlayLayer, SelectionOverlayLayer
    #QuadTreeRenderLayer, VertexRenderLayer, , ShadedPointRenderLayer, \
    #TetrahedraRenderLayer

from PYME.LMVis.gl_offScreenHandler import OffScreenHandler
from wx.glcanvas import GLCanvas


from PYME import config
from PYME.LMVis.views import View

try:
    from PYME.Analysis.points.gen3DTriangs import gen3DTriangs, gen3DBlobs
except:
    pass

try:
    # location in Python 2.7 and 3.1
    from weakref import WeakSet
except ImportError:
    # separately installed py 2.6 compatibility
    from weakrefset import WeakSet

# import time

from warnings import warn
import logging
logger = logging.getLogger(__name__)

import sys

# if sys.platform == 'darwin':
#     # osx gives us LOTS of scroll events
#     # ajust the mag in smaller increments
#     ZOOM_FACTOR = 1.1
# else:
#     ZOOM_FACTOR = 2.0
ZOOM_FACTOR = config.get('pymevis-zoom-factor', 1.1)

# import statusLog

name = 'ball_glut'

from . import views
from PYME.contrib import dispatch


class SelectionSettings(object):
    def __init__(self):
        self.start = (0, 0)
        self.finish = (0, 0)
        self.colour = [1, 1, 0]
        self.show = False

class LMGLShaderCanvas(GLCanvas):
    LUTOverlayLayer = None
    AxesOverlayLayer = None
    ScaleBarOverlayLayer = None
    ScaleBoxOverlayLayer = None
    _is_initialized = False

    def __init__(self, parent, show_lut=True, display_mode='2D', view=None):
        print("New Canvas")
        attribute_list = [wx.glcanvas.WX_GL_RGBA, wx.glcanvas.WX_GL_STENCIL_SIZE, 8, wx.glcanvas.WX_GL_DOUBLEBUFFER]
        self._num_antialias_samples = int(config.get('VisGUI-antialias_samples', 4))
        if self._num_antialias_samples > 0:
            attribute_list.extend([wx.glcanvas.WX_GL_SAMPLE_BUFFERS, 1, wx.glcanvas.WX_GL_SAMPLES, self._num_antialias_samples])
        
        GLCanvas.__init__(self, parent, -1, attribList=attribute_list)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.OnMiddleDown)
        self.Bind(wx.EVT_MIDDLE_UP, self.OnMiddleUp)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnMiddleDown)
        self.Bind(wx.EVT_RIGHT_UP, self.OnMiddleUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyPress)
        # wx.EVT_MOVE(self, self.OnMove)
        try:
            self.gl_context = wx.glcanvas.GLContext(self)
        except:
            logger.exception('Error creating OpenGL context, try modifying the number of anti-aliasing samples using the "VisGUI-antialias_samples" config setting (0 to disable antialiasing)')
            raise

        self.displayMode = display_mode  # 3DPersp' #one of 3DPersp, 3DOrtho, 2D

        self.parent = parent

        self._scaleBarLength = 1000
        self.clear_colour = [0,0,0,1.0]

        self.LUTDraw = show_lut

        self.zmin = -10
        self.zmax = 10

        self.angup = 0
        self.angright = 0
        
        if view is None:
            self.view = views.View()
        else:
            self.view = view

        self.sx = 100
        self.sy = 100

        self.zc_o = 0

        self.stereo = False

        self.eye_dist = .01

        self.dragging = False
        self.panning = False

        self.selectionSettings = SelectionSettings()
        self.selectionDragging = False

        self.layers = []
        self.overlays = []
        self.underlays = [] #draw these before data (assuming depth-testing is disabled)

        self.wantViewChangeNotification = WeakSet()
        self.pointSelectionCallbacks = []

        self.on_screen = True
        self.view_port_size = (self.Size[0], self.Size[1])
        
        self._old_bbox = None

        self.layer_added = dispatch.Signal()
    
    @property
    def xc(self):
        warn('Use view.translation[0] instead', DeprecationWarning, stacklevel=2)
        return self.view.translation[0]
    
    @property
    def yc(self):
        warn('Use view.translation[1] instead', DeprecationWarning, stacklevel=2)
        return self.view.translation[1]

    @property
    def zc(self):
        warn('Use view.translation[2] instead', DeprecationWarning, stacklevel=2)
        return self.view.translation[2]

    @property
    def scale(self):
        warn('Use view.scale instead', DeprecationWarning, stacklevel=2)
        return self.view.scale
    
    @property
    def bounds(self):
        return self.view.clipping

    @property
    def scaleBarLength(self):
        return self._scaleBarLength

    @scaleBarLength.setter
    def scaleBarLength(self, value):
        self._scaleBarLength = value
        self.ScaleBarOverlayLayer.set_scale_bar_length(value)

    def OnPaint(self, events):
        if not self.IsShown():
            print('ns')
            return
        
        #wx.PaintDC(self)
        #self.gl_context.SetCurrent(self)
        self.SetCurrent(self.gl_context)

        if not self._is_initialized:
            self.initialize()
        else:
            self.OnDraw()
        return
    
    @property
    def bbox(self):
        """ Bounding box in format [x0,y0,z0, x1, y1, z1]
        
        Calculated as the spanning box of all visible layers
        """
        
        bb = np.zeros(6)
        bb[:3] = 1e12
        bb[3:] = -1e12
        
        nLayers = 0
        
        for l in self.layers:
            # if getattr(l, 'visible', True):
            if True:
                bbl = l.bbox
                
                if not bbl is None:
                    bb[:3] = np.minimum(bb[:3], bbl[:3])
                    bb[3:] = np.maximum(bb[3:], bbl[3:])
                    nLayers += 1
                
        if nLayers > 0:
            #print('bbox: %s' % bb)
            if not self._old_bbox is None and not np.allclose(self._old_bbox, bb):
                #bbox has changed - update our cliping region
                self.bounds['x'] = [bb[0]-1, bb[3]+1]
                self.bounds['y'] = [bb[1]-1, bb[4]+1]
                self.bounds['z'] = [bb[2]-1, bb[5]+1]
            self._old_bbox = bb
            return bb
        else:
            return None
        

    def initialize(self):
        from .layers.ScaleBoxOverlayLayer import ScaleBoxOverlayLayer
        self.InitGL()
        self.ScaleBarOverlayLayer = ScaleBarOverlayLayer(context=self.gl_context)
        self.ScaleBoxOverlayLayer = ScaleBoxOverlayLayer(context=self.gl_context)

        self.LUTOverlayLayer = LUTOverlayLayer(context=self.gl_context)
        self.AxesOverlayLayer = AxesOverlayLayer(context=self.gl_context)
        
        self.overlays.append(SelectionOverlayLayer(self.selectionSettings,context=self.gl_context))
        self.underlays.append(self.ScaleBoxOverlayLayer)

        self._is_initialized = True

    def OnSize(self, event):
        self.view_port_size = (self.Size[0], self.Size[1])
        if self._is_initialized:
            self.OnDraw()
        self.Refresh()

        # self.interlace_stencil()

    def OnMove(self, event):
        self.Refresh()
        
    def add_layer(self, layer, connect=True):
        self.layers.append(layer)
        if connect:
            layer.on_update.connect(self.refresh)
            
        self.layer_added.send(self)

    def setOverlayMessage(self, message=''):
        # self.messageOverlay.set_message(message)
        # if self._is_initialized:
        #     self.Refresh()
        pass

    def interlace_stencil(self):
        window_width = self.view_port_size[0]
        window_height = self.view_port_size[1]
        # setting screen-corresponding geometry
        glViewport(0, 0, window_width, window_height)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0.0, window_width - 1, 0.0, window_height - 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # clearing and configuring stencil drawing
        if self.on_screen:
            glDrawBuffer(GL_BACK)
        glEnable(GL_STENCIL_TEST)
        glClearStencil(0)
        glClear(GL_STENCIL_BUFFER_BIT)
        glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE)  # colorbuffer is copied to stencil
        glDisable(GL_DEPTH_TEST)
        glStencilFunc(GL_ALWAYS, 1, 1)  # to avoid interaction with stencil content

        # drawing stencil pattern
        glColor4f(1, 1, 1, 0)  # alfa is 0 not to interfere with alpha tests

        start = self.ScreenPosition[1] % 2
        # print start

        for y in range(start, window_height, 2):
            glLineWidth(1)
            glBegin(GL_LINES)
            glVertex2f(0, y)
            glVertex2f(window_width, y)
            glEnd()

        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)  # // disabling changes in stencil buffer
        glFlush()

        # print 'is'

    @property
    def _stereo_views(self):
        if self.displayMode == '2D':
            return ['2D']
        elif self.stereo:
            return ['left', 'right']
        else:
            return ['centre']
    
    def OnDraw(self):
        self.interlace_stencil()
        glEnable(GL_DEPTH_TEST)
        
        glClearColor(*self.clear_colour)
        glClear(GL_COLOR_BUFFER_BIT)

        #aspect ratio of window
        ys = float(self.view_port_size[1]) / float(self.view_port_size[0])

        for stereo_view in self._stereo_views:
            if stereo_view == 'left':
                eye = -self.eye_dist
                glStencilFunc(GL_NOTEQUAL, 1, 1)
            elif stereo_view == 'right':
                eye = +self.eye_dist
                glStencilFunc(GL_EQUAL, 1, 1)
            else:
                eye = 0

            glClear(GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()

            if self.displayMode == '3DPersp':
                # our object will be be scaled to fit a 2x2x2 box at z=10 - see translate and scale calls below
                glFrustum(-1 + eye, 1 + eye, ys, -ys, 8.5, 11.5)
            else:
                glOrtho(-1, 1, ys, -ys, -1000, 1000)

            glMatrixMode(GL_MODELVIEW)
            glTranslatef(eye, 0.0, 0.0)

            # move our object to be centred at -10
            glTranslatef(0, 0, -10)

            if not self.displayMode == '2D':
                self.AxesOverlayLayer.render(self)

            # scale object to fit a 2x2x2 box
            glScalef(self.view.scale, self.view.scale, self.view.scale)

            try:
                glPushMatrix()
                # rotate object
                glMultMatrixf(self.object_rotation_matrix)
                glTranslatef(-self.view.translation[0], -self.view.translation[1], -self.view.translation[2])
                
                for l in self.underlays:
                    l.render(self)
    
                
                oit_layers = [] #layers which support order independent transparency - render these all together
                for l in self.layers:
                    if getattr(l.engine, 'use_oit', lambda l: False)(l) or getattr(l.engine, 'oit', False):
                        #defer rendering
                        oit_layers.append(l)
                    else:
                        l.render(self)
                        
                if len(oit_layers) > 0:
                    self.render_oit_layers(oit_layers)
                    
    
                for o in self.overlays:
                    o.render(self)
            finally:
                glPopMatrix()

            self.ScaleBarOverlayLayer.render(self)
            if self.LUTDraw:
                # set us up to draw in pixel coordinates
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                glOrtho(0, self.Size[0], self.Size[1], 0, -1000, 1000)

                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                
                self.LUTOverlayLayer.render(self)

        glFlush()

        self.SwapBuffers()

    def init_oit(self):
        self._fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fb)
    
        self._accumT = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._accumT)
    
        self._revealT = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._revealT)
    
        self._db = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._db)
    
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def cleanup_oit(self):
        glDeleteFramebuffers(1, self._fb)
        glDeleteTextures(1, self._accumT)
        glDeleteTextures(1, self._revealT)
        glDeleteRenderbuffers(1, self._db)
    
    def resize_gl_oit(self, w, h):
        #print('resize_gl')
        self._oit_w = w
        self._oit_h = h
    
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
    
    def render_oit_layers(self, layers):
        # initialise OIT if not already done
        if not hasattr(self, '_fb'):
            self.init_oit()
            
        # reallocate OIT textures if window has changed size
        w, h = self.view_port_size
        if not ((getattr(self, '_oit_w', None) == w) and (getattr(self,'_oit_h', None) == h)):
            self.resize_gl_oit(w, h)
    
    
        # draw to an offscreen framebuffer
        current_fb = glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING) #keep a reference to the current framebuffer so we can restore after we're done
        glBindFramebuffer(GL_FRAMEBUFFER, self._fb) #bind our offscreen framebuffer
    
        #clear the offscreen framebuffer
    
        #glClearBufferfv(GL_COLOR, 0, (0., 0., 0., 1.))
        #glClearBufferfv(GL_COLOR, 1, (1., 0., 0., 0.))
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
        #Copy depth buffer from opaque stuff rendered previously
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._fb)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, self._oit_w, self._oit_h,
                          0, 0, self._oit_w, self._oit_h,
                          GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fb)
    
        #glDrawBuffer(GL_COLOR_ATTACHMENT0)
        #glClear(GL_COLOR_BUFFER_BIT) #| GL_DEPTH_BUFFER_BIT)
    
        #clear the second colour attachment buffer
        glDrawBuffers(1, [GL_COLOR_ATTACHMENT1, ])
        glClearColor(1.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
    
        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
        
        #draw the OIT layers
        for l in layers:
            l.render(self)

        # set the framebuffer back
        glBindFramebuffer(GL_FRAMEBUFFER, current_fb)

        #now do the compositing pass
        with self.composite_shader as c:
            # bind our pre-rendered textures

            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self._accumT)
            self._acc_buf = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
            glUniform1i(c.get_uniform_location('accum_t'), 0)

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
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
                
    @property
    def composite_shader(self):
        from PYME.LMVis.shader_programs.ShaderProgramFactory import ShaderProgramFactory
        from PYME.LMVis.shader_programs.oit_compositor import OITCompositorProgram
        if not hasattr(self, '_composite_shader'):
            self._composite_shader = ShaderProgramFactory.get_program(OITCompositorProgram, self.gl_context)
            
        return self._composite_shader

    @property
    def object_rotation_matrix(self):
        """
        The transformation matrix used to map coordinates in real space to our 3D view space. Currently implements
        rotation, defined by 3 vectors (up, right, and back). Does not include scaling or projection.
        
        Returns
        -------
        a 4x4 matrix describing the rotation of the points within our 3D world
        
        """
        if not self.displayMode == '2D':
            return numpy.array(
                [numpy.hstack((self.view.vec_right, 0)), numpy.hstack((self.view.vec_up, 0)), numpy.hstack((self.view.vec_back, 0)),
                 [0, 0, 0, 1]])
        else:
            return numpy.eye(4)

    def InitGL(self):
        print('OpenGL - Version: {}'.format(glGetString(GL_VERSION)))
        print('Shader - Version: {}'.format(glGetString(GL_SHADING_LANGUAGE_VERSION)))
        
        max_samples = glGetInteger(GL_MAX_SAMPLES)
        n_samples = glGetInteger(GL_SAMPLES)
        print('GL_MAX_SAMPLES: %d, GL_SAMPLES: %d' % (max_samples, n_samples))
        
        if (max_samples >= 4) and (n_samples < 4):
            logger.info('Your machine supports OpenGL antialiasing, but antialiasing disabled - enable by setting the "VisGUI-antialias_samples" PYME config setting to 4 or higher')
            
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glEnable(GL_MULTISAMPLE)

        glLoadIdentity()
        glOrtho(self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

        self.ResetView()

    def ResetView(self):
        self.view.vec_up = numpy.array([0, 1, 0])
        self.view.vec_right = numpy.array([1, 0, 0])
        self.view.vec_back = numpy.array([0, 0, 1])

        self.Refresh()

    @property
    def xmin(self):
        return self.view.translation[0] - 0.5 * self.pixelsize * self.view_port_size[0]

    @property
    def xmax(self):
        return self.view.translation[0] + 0.5 * self.pixelsize * self.view_port_size[0]

    @property
    def ymin(self):
        return self.view.translation[1] - 0.5 * self.pixelsize * self.view_port_size[1]

    @property
    def ymax(self):
        return self.view.translation[1] + 0.5 * self.pixelsize * self.view_port_size[1]

    @property
    def pixelsize(self):
        return 2. / (self.view.scale * self.view_port_size[0])
    
    def _view_changed(self):
        """Notify anyone who is synced to our view"""
        for callback in self.wantViewChangeNotification:
            if callback:
                callback.Refresh()

    def setView(self, xmin, xmax, ymin, ymax):

        self.view.translation[0] = (xmin + xmax) / 2.0
        self.view.translation[1] = (ymin + ymax) / 2.0
        self.view.translation[2] = self.zc_o  # 0#z.mean()

        self.view.scale = 2. / (xmax - xmin)

        self.Refresh()
        if 'OnGLViewChanged' in dir(self.parent):
            self.parent.OnGLViewChanged()

        self._view_changed()

    def moveView(self, dx, dy, dz=0):
        return self.pan(dx, dy, dz)

    def pan(self, dx, dy, dz=0):
        self.view.translation[0] += dx
        self.view.translation[1] += dy
        self.view.translation[2] += dz

        self.Refresh()
        self._view_changed()

    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        xp, yp = self._ScreenCoordinatesToNm(event.GetX(), event.GetY())

        dx, dy = (xp - self.view.translation[0]), (yp - self.view.translation[1])
        dx_, dy_, dz_, c_ = numpy.dot(self.object_rotation_matrix, [dx, dy, 0, 0])
        xp_, yp_, zp_ = (self.view.translation[0] + dx_), (self.view.translation[1] + dy_), (self.view.translation[2] + dz_)

        if event.MiddleIsDown():
            self.WheelFocus(rot, xp_, yp_, zp_)
        else:
            self.WheelZoom(rot, xp_, yp_, zp_)

    def WheelZoom(self, rot, xp, yp, zp=0):
        dx = xp - self.view.translation[0]
        dy = yp - self.view.translation[1]
        dz = zp - self.view.translation[2]

        if rot > 0:
            # zoom out
            self.view.scale *= ZOOM_FACTOR

            self.view.translation[0] += dx * (1. - 1. / ZOOM_FACTOR)
            self.view.translation[1] += dy * (1. - 1. / ZOOM_FACTOR)
            self.view.translation[2] += dz * (1. - 1. / ZOOM_FACTOR)

        if rot < 0:
            # zoom in
            self.view.scale /= ZOOM_FACTOR

            self.view.translation[0] += dx * (1. - ZOOM_FACTOR)
            self.view.translation[1] += dy * (1. - ZOOM_FACTOR)
            self.view.translation[2] += dz * (1. - ZOOM_FACTOR)

        self.Refresh()
        self._view_changed()

    def WheelFocus(self, rot, xp, yp, zp=0):
        if rot > 0:
            self.view.translation[2] -= 1.

        if rot < 0:
            self.view.translation[2] += 1.

        self.Refresh()
        self._view_changed()

    def OnLeftDown(self, event):
        if not self.displayMode == '2D':
            # dragging the mouse rotates the object
            self.xDragStart = event.GetX()
            self.yDragStart = event.GetY()

            self.angyst = self.angup
            self.angxst = self.angright

            self.dragging = True
        else:  # 2D
            # dragging the mouse sets an ROI
            xp, yp = self._ScreenCoordinatesToNm(event.GetX(), event.GetY())

            self.selectionDragging = True
            self.selectionSettings.show = True

            self.selectionSettings.start = (xp, yp)
            self.selectionSettings.finish = (xp, yp)

        event.Skip()

    def OnLeftUp(self, event):
        self.dragging = False

        if self.selectionDragging:
            xp, yp = self._ScreenCoordinatesToNm(event.GetX(), event.GetY())

            self.selectionSettings.finish = (xp, yp)
            self.selectionDragging = False

            self.Refresh()
            self.Update()

        event.Skip()

    def OnMiddleDown(self, event):
        self.xDragStart = event.GetX()
        self.yDragStart = event.GetY()

        self.panning = True
        event.Skip()

    def OnMiddleUp(self, event):
        self.panning = False
        event.Skip()

    def _ScreenCoordinatesToNm(self, x, y):
        # FIXME!!!
        x_ = self.pixelsize * (x - 0.5 * float(self.view_port_size[0])) + self.view.translation[0]
        y_ = self.pixelsize * (y - 0.5 * float(self.view_port_size[1])) + self.view.translation[1]
        # print x_, y_
        return x_, y_

    def OnMouseMove(self, event):
        x = event.GetX()
        y = event.GetY()

        if self.selectionDragging:
            self.selectionSettings.finish = self._ScreenCoordinatesToNm(x, y)
            self.Refresh()
            event.Skip()

        elif self.dragging:
            angx = numpy.pi * (x - self.xDragStart) / 180
            angy = numpy.pi * (y - self.yDragStart) / 180

            r_mat1 = numpy.matrix(
                [[numpy.cos(angx), 0, numpy.sin(angx)], [0, 1, 0], [-numpy.sin(angx), 0, numpy.cos(angx)]])
            r_mat = r_mat1 * numpy.matrix(
                [[1, 0, 0], [0, numpy.cos(angy), numpy.sin(angy)], [0, -numpy.sin(angy), numpy.cos(angy)]])

            vec_right_n = numpy.array(r_mat * numpy.matrix(self.view.vec_right).T).squeeze()
            vec_up_n = numpy.array(r_mat * numpy.matrix(self.view.vec_up).T).squeeze()
            vec_back_n = numpy.array(r_mat * numpy.matrix(self.view.vec_back).T).squeeze()

            self.view.vec_right = vec_right_n

            self.view.vec_up = vec_up_n
            self.view.vec_back = vec_back_n

            self.xDragStart = x
            self.yDragStart = y

            self.Refresh()
            event.Skip()

        elif self.panning:
            dx = self.pixelsize * (x - self.xDragStart)
            dy = self.pixelsize * (y - self.yDragStart)

            dx_, dy_, dz_, c_ = numpy.dot(self.object_rotation_matrix, [dx, dy, 0, 0])

            self.xDragStart = x
            self.yDragStart = y

            self.pan(-dx_, -dy_, -dz_)

            event.Skip()

    def OnKeyPress(self, event):
        # print event.GetKeyCode()
        if event.GetKeyCode() == 83:  # S - toggle stereo
            self.stereo = not self.stereo
            self.Refresh()
        elif event.GetKeyCode() == 67:  # C - centre
            self.recenter_bbox()
            self.Refresh()

        elif event.GetKeyCode() == 91:  # [ decrease eye separation
            self.eye_dist /= 1.5
            self.Refresh()

        elif event.GetKeyCode() == 93:  # ] increase eye separation
            self.eye_dist *= 1.5
            self.Refresh()

        elif event.GetKeyCode() == 82:  # R reset view
            self.ResetView()
            self.Refresh()

        elif event.GetKeyCode() == 314:  # left
            pos = numpy.array([self.view.translation[0], self.view.translation[1], self.view.translation[2]], 'f')
            pos -= 300 * self.view.vec_right
            self.view.translation[0], self.view.translation[1], self.view.translation[2] = pos
            # print 'l'
            self.Refresh()

        elif event.GetKeyCode() == 315:  # up
            pos = numpy.array([self.view.translation[0], self.view.translation[1], self.view.translation[2]])
            pos -= 300 * self.view.vec_back
            self.view.translation[0], self.view.translation[1], self.view.translation[2] = pos
            self.Refresh()

        elif event.GetKeyCode() == 316:  # right
            pos = numpy.array([self.view.translation[0], self.view.translation[1], self.view.translation[2]])
            pos += 300 * self.view.vec_right
            self.view.translation[0], self.view.translation[1], self.view.translation[2] = pos
            self.Refresh()

        elif event.GetKeyCode() == 317:  # down
            pos = numpy.array([self.view.translation[0], self.view.translation[1], self.view.translation[2]])
            pos += 300 * self.view.vec_back
            self.view.translation[0], self.view.translation[1], self.view.translation[2] = pos
            self.Refresh()

        else:
            event.Skip()

    def getSnapshot(self, mode=GL_RGB):
        # glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        # width, height = self.view_port_size[0], self.view_port_size[1]
        # snap = glReadPixelsf(0, 0, width, height, mode)
        # snap = snap.ravel().reshape(width, height, -1, order='F')
        #
        # if mode == GL_LUMINANCE:
        #     # snap.strides = (4, 4 * snap.shape[0])
        #     pass
        # elif mode == GL_RGB:
        #     snap.strides = (12, 12 * snap.shape[0], 4)
        # else:
        #     raise RuntimeError('{} is not a supported format.'.format(mode))
        # img.show()
        self.on_screen = False
        off_screen_handler = OffScreenHandler(self.view_port_size, mode, self._num_antialias_samples)
        with off_screen_handler:
            self.OnDraw()
        snap = off_screen_handler.get_snap()
        self.on_screen = True
        return  (255*np.asarray(snap)).astype('uint8')

    def getIm(self, pixel_size=None, mode=GL_RGB, image_bounds=None):
        if ((pixel_size is None) or (abs(1 - pixel_size) < 0.001) and image_bounds is None):  # use current pixel size
            return self.getSnapshot(mode=mode)
        else:
            # set size before moving the view, since self.setView includes a self.Refresh() call
            if image_bounds is None:
                from PYME.IO.image import ImageBounds
                x0, y0, z0, x1, y1, z1 = self.bbox
                #FIXME - do model-view translation to cope with 3D rotated views.
                image_bounds = ImageBounds(x0, x1, y0, y1, z0, z1)
                
            self.view_port_size = (int((image_bounds.x1 - image_bounds.x0) / pixel_size),
                                   int((image_bounds.y1 - image_bounds.y0) / pixel_size))
            logging.debug('viewport size %s' % (self.view_port_size,))
            self.setView(image_bounds.x0, image_bounds.x0 + self.view_port_size[0]*pixel_size,
                         image_bounds.y0, image_bounds.y0 + self.view_port_size[1]*pixel_size)
            snap = self.getSnapshot(mode=mode)
            
            print(pixel_size, self.pixelsize)
            assert(self.pixelsize == pixel_size)
            self.view_port_size = self.Size
            return snap

    def recenter(self, x, y):
        self.view.translation[0] = x.mean()
        self.view.translation[1] = y.mean()
        self.view.translation[2] = 0  # z.mean()

        self.sx = x.max() - x.min()
        self.sy = y.max() - y.min()
        self.sz = 0  # z.max() - z.min()

        self.view.scale = 2. / (max(self.sx, self.sy))
        
    def recenter_bbox(self):
        bb = self.bbox
        if bb is None:
            return
        
        centre = 0.5*(bb[:3] + bb[3:])
        
        self.view.translation[0], self.view.translation[1], self.view.translation[2] = centre
        
    def fit_bbox(self):
        bb = self.bbox
        if bb is None:
            return
    
        sz = (bb[3:] - bb[:3])
    
        self.view.scale = 1.5/max(np.abs(sz))
        self.recenter_bbox()

    def set_view(self, view):
        self.view=view
        self.Refresh()

    def get_view(self, view_id='id'):
        view = View.copy(self.view)
        view.view_id = view_id
        return view
    
    def refresh(self, *args, **kwargs):
        bb = self.bbox #force an update of our bounding box
        self.Refresh()


class LegacyGLCanvas(LMGLShaderCanvas):
    """ Catch all for stuff which has largely been deprecated by layers, but might still be required for the non-layers mode"""
    
    def __init__(self, *args, **kwargs):
        LMGLShaderCanvas.__init__(self, *args, **kwargs)
        
        import matplotlib.cm

        self.cmap = matplotlib.cm.gist_rainbow
        self.clim = [0, 1]
        self.alim = [0, 1]

        self.pointSize = 30  # default point size = 30nm

        self.c = numpy.array([1, 1, 1])
        self.a = numpy.array([1, 1, 1])
        
    
    def setColour(self, IScale=None, zeroPt=None):
        self.Refresh()

    def setCMap(self, cmap):
        self.cmap = cmap
        self.setColour()

    def setCLim(self, clim, alim=None):
        self.clim = clim
        if alim is None:
            self.alim = clim
        else:
            self.alim = alim
        self.setColour()
        
        
    def setTriang3D(self, x, y, z, c=None, sizeCutoff=1000., zrescale=1, internalCull=True, wireframe=False, alpha=1,
                    recenter=True):

        if recenter:
            self.recenter(x, y)
        self.layers.append(TetrahedraRenderLayer(x, y, z, c, self.cmap, sizeCutoff,
                                                 internalCull, zrescale, alpha, is_wire_frame=wireframe))
        self.Refresh()

    def setTriang(self, T, c=None, sizeCutoff=1000., zrescale=1, internalCull=True, alpha=1,
                  recenter=True):
        # center data
        x = T.x
        y = T.y
        xs = x[T.triangles]
        ys = y[T.triangles]
        zs = np.zeros_like(xs)  # - z.mean()

        if recenter:
            self.recenter(x, y)

        if c is None:
            a = numpy.vstack((xs[:, 0] - xs[:, 1], ys[:, 0] - ys[:, 1])).T
            b = numpy.vstack((xs[:, 0] - xs[:, 2], ys[:, 0] - ys[:, 2])).T
            b2 = numpy.vstack((xs[:, 1] - xs[:, 2], ys[:, 1] - ys[:, 2])).T

            c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)], 0)
            c = 1.0 / (c + 1)

        self.c = numpy.vstack((c, c, c)).T.ravel()

        self.view.vec_up = numpy.array([0, 1, 0])
        self.view.vec_right = numpy.array([1, 0, 0])
        self.view.vec_back = numpy.array([0, 0, 1])

        self.SetCurrent(self.gl_context)

        self.layers.append(
            VertexRenderLayer(T.x[T.triangles], T.y[T.triangles], 0 * (T.x[T.triangles]), self.c,
                              self.cmap, self.clim, alpha))
        self.Refresh()

    def setTriangEdges(self, T):
        self.setTriang(T)

    def setPoints3D(self, x, y, z, c=None, a=None, recenter=False, alpha=1.0, mode='points',
                    normal_x = None, normal_y = None, normal_z = None):  # , clim=None):
        from PYME.LMVis.layers.pointcloud import PointCloudRenderLayer
        # center data
        x = x  # - x.mean()
        y = y  # - y.mean()
        z = z  # - z.mean()

        if recenter:
            self.recenter(x, y)

        self.view.translation[2] = z.mean()
        self.zc_o = 1.0 * self.view.translation[2]

        if c is None:
            self.c = numpy.ones(x.shape).ravel()
        else:
            self.c = c

        if a:
            self.a = a
        else:
            self.a = numpy.ones(x.shape).ravel()

        self.sx = x.max() - x.min()
        self.sy = y.max() - y.min()
        self.sz = z.max() - z.min()

        self.SetCurrent(self.gl_context)

        if mode is 'pointsprites':
            self.layers.append(PointSpritesRenderLayer(x, y, z, self.c, self.cmap, self.clim, alpha, self.pointSize))
        elif mode is 'shadedpoints':
            self.layers.append(ShadedPointRenderLayer(x, y, z, normal_x, normal_y, normal_z, self.c, self.cmap,
                                                      self.clim, alpha=alpha, point_size=self.pointSize))
        else:
            self.layers.append(Point3DRenderLayer(x, y, z, self.c, self.cmap, self.clim,
                                                  alpha=alpha, point_size=self.pointSize))
        self.Refresh()

    def setPoints(self, x, y, c=None, a=None, recenter=True, alpha=1.0):
        """Set 2D points"""
        self.setPoints3D(x, y, 0 * x, c, a, recenter, alpha)

    def setQuads(self, qt, max_depth=100, md_scale=False):
        lvs = qt.getLeaves(max_depth)

        xs = numpy.zeros((len(lvs), 4))
        ys = numpy.zeros((len(lvs), 4))
        c = numpy.zeros(len(lvs))

        i = 0

        real_max_depth = 0
        for l in lvs:
            xs[i, :] = [l.x0, l.x1, l.x1, l.x0]
            ys[i, :] = [l.y0, l.y0, l.y1, l.y1]
            c[i] = float(l.numRecords) * 2 ** (2 * l.depth)
            i += 1
            real_max_depth = max(real_max_depth, l.depth)

        if not md_scale:
            c /= 2 ** (2 * real_max_depth)

        self.c = numpy.vstack((c, c, c, c)).T.ravel()

        self.SetCurrent(self.gl_context)
        self.layers.append(QuadTreeRenderLayer(xs.ravel(), ys.ravel(), 0 * xs.ravel(),
                                               self.c, self.cmap, self.clim, alpha=1))
        self.Refresh()
    


def showGLFrame():
    f = wx.Frame(None, size=(800, 800))
    #sizer = wx.BoxSizer(wx.VERTICAL)
    c = LMGLShaderCanvas(f)
    #sizer.Add(c, 1, wx.EXPAND, 0)
    #f.SetSizer(sizer)
    f.Show()
    return c

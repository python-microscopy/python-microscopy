#!/usr/bin/python

##################
# gl_render3D.py
#
# Copyright David Baddeley, 209
# d.baddeley@auckland.ac.nz
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

import numpy
import numpy as np
# import pylab
#import matplotlib.cm
from PYME.misc.colormaps import cm
import wx
import wx.glcanvas
from OpenGL import GLUT
from OpenGL.GL import *
from OpenGL.GLU import *
from six.moves import xrange
from wx.glcanvas import GLCanvas

#from PYME.LMVis.tests.gl_test_objects import NineCollections

try:
    from PYME.Analysis.points.gen3DTriangs import gen3DTriangs, gen3DBlobs, testObj
except:
    pass

try:
    # location in Python 2.7 and 3.1
    from weakref import WeakSet
except ImportError:
    # separately installed py 2.6 compatibility
    from weakrefset import WeakSet

# import time

import sys
if sys.platform == 'darwin':
    # osx gives us LOTS of scroll events
    # ajust the mag in smaller increments
    ZOOM_FACTOR = 1.1
else:
    ZOOM_FACTOR = 2.0

# import statusLog

name = 'ball_glut'




class cmap_mult:
    def __init__(self, gains, zeros):
        self.gains = gains
        self.zeros = zeros

    def __call__(self, cvals):
        return numpy.minimum(numpy.vstack((self.gains[0]*cvals - self.zeros[0], self.gains[1]*cvals - self.zeros[1],
                                           self.gains[2]*cvals - self.zeros[2], 1 + 0*cvals)), 1).astype('f').T

cm_hot = cmap_mult(8.0*numpy.ones(3)/3, [0, 3.0/8, 6.0/8])
cm_grey = cmap_mult(numpy.ones(3), [0, 0, 0])

    
class RenderLayer(object):
    drawModes = {'triang': GL_TRIANGLES, 'quads': GL_QUADS, 'edges': GL_LINES, 'points': GL_POINTS,
                 'wireframe': GL_TRIANGLES, 'tracks': GL_LINE_STRIP}
    
    def __init__(self, vertices, normals, colours, cmap, clim, mode='triang', pointsize=5, alpha=1):
        self.verts = vertices
        self.normals = normals
        self.cols = colours
        
        self.cmap = cmap
        self.clim = clim
        self.alpha = alpha
        
        cs_ = ((self.cols - self.clim[0])/(self.clim[1] - self.clim[0]))
        cs = self.cmap(cs_)
        cs[:, 3] = self.alpha
        
        self.cs = cs.ravel().reshape(len(self.cols), 4)
        
        self.mode = mode
        self.pointSize = pointsize

    def render(self, glcanvas=None):
        # with default_shader:
        if self.mode in ['points']:
            glDisable(GL_LIGHTING)
            #glDisable(GL_DEPTH_TEST)
            #glPointSize(self.pointSize*self.scale*(self.xmax - self.xmin))

            glEnable(GL_POINT_SMOOTH)
            if glcanvas:
                if self.pointSize == 0:
                    glPointSize(1 / glcanvas.pixelsize)
                else:
                    glPointSize(self.pointSize/glcanvas.pixelsize)
            else:
                glPointSize(self.pointSize)
        else:
            glEnable(GL_LIGHTING)
            pass
        
        if self.mode in ['wireframe']:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT, GL_FILL)
            
        nVertices = self.verts.shape[0]
            
        self.vs_ = glVertexPointerf(self.verts)
        self.n_ = glNormalPointerf(self.normals)
        self.c_ = glColorPointerf(self.cs)
            
        glPushMatrix ()
        glColor4f(0,0.5,0, 1)

        glDrawArrays(self.drawModes[self.mode], 0, nVertices)

        glPopMatrix()


class TrackLayer(RenderLayer):
    def __init__(self, vertices, colours, cmap, clim, clumpSizes, clumpStarts, alpha=1):
        self.verts = vertices
        self.cols = colours
        self.clumpSizes = clumpSizes
        self.clumpStarts = clumpStarts
        
        self.cmap = cmap
        self.clim = clim
        self.alpha = alpha
        
        cs_ = ((self.cols - self.clim[0])/(self.clim[1] - self.clim[0]))
        cs = self.cmap(cs_)
        cs[:,3] = self.alpha
        
        self.cs = cs.ravel().reshape(len(self.cols), 4)

    def render(self, glcanvas=None):
        glDisable(GL_LIGHTING)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        self.vs_ = glVertexPointerf(self.verts)
        #self.n_ = glNormalPointerf(self.normals)
        self.c_ = glColorPointerf(self.cs)

        glPushMatrix ()
        glColor4f(0,0.5,0, 1)
        
        for i, cl in enumerate(self.clumpSizes):
            if cl > 0:
                glDrawArrays(self.drawModes['tracks'], self.clumpStarts[i], cl)

        glPopMatrix()


class SelectionSettings(object):
    def __init__(self):
        self.start = (0,0)
        self.finish = (0,0)
        self.colour = [1,1,0]
        self.show = False


class SelectionOverlay(object):

    def __init__(self, selectionSettings):
        self.selectionSettings = selectionSettings

    def render(self, glcanvas):
        if self.selectionSettings.show:
            glDisable(GL_LIGHTING)
            x0, y0 = self.selectionSettings.start
            x1, y1 = self.selectionSettings.finish

            glColor3fv(self.selectionSettings.colour)
            glBegin(GL_LINE_LOOP)
            glVertex3f(x0, y0, glcanvas.zc)
            glVertex3f(x1, y0, glcanvas.zc)
            glVertex3f(x1, y1, glcanvas.zc)
            glVertex3f(x0, y1, glcanvas.zc)
            glEnd()


class MessageOverlay(object):
    def __init__(self, message = '', x=-.7, y=0):
        self.message = message
        self.x = x
        self.y = y
        
    def set_message(self, message):
        self.message = message

    def render(self, glcanvas):
        if not self.message == '':
            glDisable(GL_LIGHTING)
            glPushMatrix()
            glLoadIdentity()

            glOrtho(-1, 1, -1, 1, -1, 1)
            # def glut_print(x, y, font, text, r, g, b, a):

            # blending = False
            # if glIsEnabled(GL_BLEND):
            #     blending = True

            # glEnable(GL_BLEND)
            glColor3f(1, 1, 1)
            glRasterPos2f(self.x, self.y)
            # on windows we must have a glutInit call
            # on mac this seems not absolutely vital
            GLUT.glutInit()
            for ch in self.message:
                GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_9_BY_15, ctypes.c_int(ord(ch)))

            # GLUT.glutBitmapString(GLUT.GLUT_BITMAP_9_BY_15, ctypes.c_char_p(self.message))

            # if not blending:
            #     glDisable(GL_BLEND)

            glPopMatrix()
            glEnable(GL_LIGHTING)



class LMGLCanvas(GLCanvas):
    defaultProgram = None
    LUTOverlayLayer = None
    AxesOverlayLayer = None
    ScaleBarOverlayLayer = None
    _is_initialized = False

    def __init__(self, parent):
        attriblist = [wx.glcanvas.WX_GL_RGBA,wx.glcanvas.WX_GL_STENCIL_SIZE,8, wx.glcanvas.WX_GL_DOUBLEBUFFER, 16]
        GLCanvas.__init__(self, parent,-1, attribList = attriblist)
        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_LEFT_DOWN(self, self.OnLeftDown)
        wx.EVT_LEFT_UP(self, self.OnLeftUp)
        wx.EVT_MIDDLE_DOWN(self, self.OnMiddleDown)
        wx.EVT_MIDDLE_UP(self, self.OnMiddleUp)
        wx.EVT_RIGHT_DOWN(self, self.OnMiddleDown)
        wx.EVT_RIGHT_UP(self, self.OnMiddleUp)
        wx.EVT_MOTION(self, self.OnMouseMove)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)
        #wx.EVT_MOVE(self, self.OnMove)
        self.gl_context = wx.glcanvas.GLContext(self)

        self.nVertices = 0
        self.IScale = [1.0, 1.0, 1.0]
        self.zeroPt = [0, 1.0/3, 2.0/3]
        self.cmap = cm.hsv
        self.clim = [0,1]
        self.alim = [0,1]

        self.displayMode = '2D'#3DPersp' #one of 3DPersp, 3DOrtho, 2D
        
        self.wireframe = False

        self.parent = parent

        self.pointSize=5 #default point size = 5nm

        #self.pixelsize = 5./800

        #self.xmin =0
        #self.xmax = self.pixelsize*self.Size[0]
        #self.ymin = 0
        #self.ymax = self.pixelsize*self.Size[1]

        self.scaleBarLength = 1000

        self.scaleBarOffset = (10, 10) #pixels from corner
        self.scaleBarDepth = 10.0 #pixels
        self.scaleBarColour = [1,1,0]

        self.centreCross=False

        self.LUTDraw = True

        self.mode = 'triang'

        self.colouring = 'area'

        self.drawModes = {'triang':GL_TRIANGLES, 'quads':GL_QUADS, 'edges':GL_LINES, 'points':GL_POINTS}

        self.c = numpy.array([1,1,1])
        self.a = numpy.array([1,1,1])
        self.zmin = -10
        self.zmax = 10

        self.angup = 0
        self.angright = 0

        self.vecUp = numpy.array([0,1,0])
        self.vecRight = numpy.array([1,0,0])
        self.vecBack = numpy.array([0,0,1])

        self.xc = 0
        self.yc = 0
        self.zc = 0

        self.zc_o = 0

        self.sx = 100
        self.sy = 100

        self.scale = 1
        self.stereo = False
        
        self.eye_dist = .01
        
        self.dragging = False
        self.panning = False

        self.edgeThreshold = 20

        self.selectionSettings = SelectionSettings()
        self.selectionDragging = False
        
        self.layers = []
        self.overlays = []

        self.messageOverlay = MessageOverlay()
        self.overlays.append(self.messageOverlay)
        self.wantViewChangeNotification = WeakSet()
        self.pointSelectionCallbacks = []

        return

    def OnPaint(self, event):
        if not self.IsShown():
            print('ns')
            return
        #wx.PaintDC(self)
        # print self.GetContext()
        self.gl_context.SetCurrent(self)
        self.SetCurrent(self.gl_context)

        if not self._is_initialized:
            self.InitGL()

            self.overlays.append(SelectionOverlay(self.selectionSettings))

            self._is_initialized = True
        else:
            self.OnDraw()
        return

    def OnSize(self, event):
        #self.SetCurrent(self.gl_context)
        #glViewport(0,0, self.Size[0], self.Size[1])

        #self.xmax = self.xmin + self.Size[0]*self.pixelsize
        #self.ymax = self.ymin + self.Size[1]*self.pixelsize
        if self._is_initialized:
            self.OnDraw()
        self.Refresh()
        
        #self.interlace_stencil()
        
    def OnMove(self, event):
        self.Refresh()

    def setOverlayMessage(self, message=''):
        self.messageOverlay.set_message(message)
        if self._is_initialized:
            self.Refresh()
        
    def interlace_stencil(self):
        WindowWidth = self.Size[0]
        WindowHeight = self.Size[1]
        # setting screen-corresponding geometry
        glViewport(0,0,WindowWidth,WindowHeight)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0.0,WindowWidth-1,0.0,WindowHeight-1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        	
        #clearing and configuring stencil drawing
        glDrawBuffer(GL_BACK)
        glEnable(GL_STENCIL_TEST)
        glClearStencil(0)
        glClear(GL_STENCIL_BUFFER_BIT)
        glStencilOp (GL_REPLACE, GL_REPLACE, GL_REPLACE) # colorbuffer is copied to stencil
        glDisable(GL_DEPTH_TEST)
        glStencilFunc(GL_ALWAYS,1,1) # to avoid interaction with stencil content
        	
        # drawing stencil pattern
        glColor4f(1,1,1,0)	# alfa is 0 not to interfere with alpha tests

        start = self.ScreenPosition[1] % 2     
        #print start
        
        for y in range(start, WindowHeight, 2):
            glLineWidth(1)
            glBegin(GL_LINES)
            glVertex2f(0,y)
            glVertex2f(WindowWidth,y)
            glEnd()	
            
        glStencilOp (GL_KEEP, GL_KEEP, GL_KEEP)# // disabling changes in stencil buffer
        glFlush()
        
        #print 'is'


    def OnDraw(self):
        self.interlace_stencil()
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT)
        
        #print 'od'

        if self.displayMode == '2D':
            views = ['2D']
        elif self.stereo:
            views = ['left', 'right']
        else:
            views = ['centre']
        
        for view in views:
            if view == 'left':
                eye = -self.eye_dist
                glStencilFunc(GL_NOTEQUAL,1,1)
            elif view == 'right':
                eye = +self.eye_dist
                glStencilFunc(GL_EQUAL,1,1)
            else:
                eye = 0
            
            glClear(GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)        
            glLoadIdentity()
            glMatrixMode(GL_PROJECTION)        
            glLoadIdentity()

            ys = float(self.Size[1])/float(self.Size[0])

            if self.displayMode == '3DPersp':
                glFrustum(-1 + eye,1 + eye,-ys,ys,8.5,11.5)
            else:
                glOrtho(-1,1,-ys,ys,-1000,1000)
            
            glMatrixMode(GL_MODELVIEW)
            glTranslatef(eye,0.0,0.0)
    
            self.setupLights()

            glTranslatef(0, 0, -10)

            if not self.displayMode == '2D':
                self.drawAxes(self.object_rotation_matrix, ys)

            glScalef(self.scale, self.scale, self.scale)

            glPushMatrix()
            #rotate object
            glMultMatrixf(self.object_rotation_matrix)

            glTranslatef(-self.xc, -self.yc, -self.zc)
            
            for l in self.layers:
                l.render(self)

            for o in self.overlays:
                o.render(self)

            glPopMatrix()

            self.drawScaleBar()
            self.drawLUT()

        glFlush()

        self.SwapBuffers()
        
        return
    
    @property
    def object_rotation_matrix(self):
        """
        The transformation matrix used to map coordinates in real space to our 3D view space. Currently implements
        rotation, defined by 3 vectors (up, right, and back). Does not include scaling or projection.
        
        Returns
        -------
        a 4x4 matrix describing the rotation of the pints within our 3D world
        
        """
        if not self.displayMode == '2D':
            return numpy.array([numpy.hstack((self.vecRight, 0)), numpy.hstack((self.vecUp, 0)), numpy.hstack((self.vecBack, 0)), [0,0,0, 1]])
        else:
            return numpy.eye(4)

    def setupLights(self):
        # set viewing projection
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        # light_diffuse = [1., 1., 1., 1.0]
        light_position = [2.0, 2.00, 2.0, 0.0]

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.5, 0.5, 0.5, 1.0]);
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE)

        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3,0.3,0.3,1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    def InitGL(self):
        # GLUT.glutInitContextVersion(3,2) #; /* or later versions, core was introduced only with 3.2 */
        # GLUT.glutInitContextProfile(GLUT.GLUT_CORE_PROFILE)#;
        
#        # set viewing projection
#        light_diffuse = [0.5, 0.5, 0.5, 1.0]
#        #light_diffuse = [1., 1., 1., 1.0]
#        light_position = [2.0, 2.00, 2.0, 0.0]
#
#        #glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.5, 0.5, 0.5, 1.0]);
#        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
#
#        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
#        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3,0.3,0.3,1.0])
#        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
#
#        glEnable(GL_LIGHTING)
#        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        #glEnable(GL_STENCIL)

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.3, 0.3, 0.3, 1])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 50)


        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)

        glShadeModel(GL_SMOOTH)

        glLoadIdentity()
        glOrtho(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)


        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnable (GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glBlendFunc (GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

        #self.vs = glVertexPointerf(numpy.array([[5, 5],[10000, 10000], [10000, 5]]).astype('f'))
        #self.cs = glColorPointerf(numpy.ones((3,3)).astype('f'))

        #self.nVertices = 3

        #to = test_obj()
        #print('OpenGL version: %s' % glGetString(GL_VERSION))

        #self.setBlob(to[0], to[1], to[2], smScale=[1e3,1e3,1e3])
        #self.setTriang(to[0], to[1], to[2])
        #self.setPoints(to[0], to[1], to[2], to[2])
        self.ResetView()

        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #gluPerspective(40.0, 1.0, 1.0, 30.0)

        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()
        #gluLookAt(0.0, 0.0, 10.0,
        #          0.0, 0.0, 0.0,
        #          0.0, 1.0, 0.0)
        return

    def drawAxes(self,trafMatrix, ys=1.0):
        glDisable(GL_LIGHTING)
        glPushMatrix ()

        glTranslatef(.9, .1 - ys, 0)
        glScalef(.1, .1, .1)
        glLineWidth(3)
        glMultMatrixf(trafMatrix)

        glColor3fv([1,.5,.5])
        glBegin(GL_LINES)
        glVertex3f(0,0,0)
        glVertex3f(1,0,0)
        glEnd()

        glColor3fv([.5,1,.5])
        glBegin(GL_LINES)
        glVertex3f(0,0,0)
        glVertex3f(0,1,0)
        glEnd()

        glColor3fv([.5,.5,1])
        glBegin(GL_LINES)
        glVertex3f(0,0,0)
        glVertex3f(0,0,1)
        glEnd()
        
        glLineWidth(1)

        glPopMatrix ()
        glEnable(GL_LIGHTING)

    def setBlob3D(self, x,y,z, sizeCutoff=1000., zrescale=1, smooth=False, smScale=[10,10,10], recenter=True):
        #center data
        x = x #- x.mean()
        y = y #- y.mean()
        z = z #- z.mean()
        
        if recenter:        
            self.xc = x.mean()
            self.yc = y.mean()
            self.zc = z.mean()

        P, A, N = gen3DBlobs(x,y,z/zrescale, sizeCutoff, smooth, smScale)
        P[:,2] = P[:,2]*zrescale
        
        self.sx = x.max() - x.min()
        self.sy = y.max() - y.min()
        self.sz = z.max() - z.min()

        self.c = A
        self.a = self.c
        vs = P

        self.SetCurrent(self.gl_context)
        
        self.layers.append(RenderLayer(vs, N, self.c, self.cmap, [self.c.min(), self.c.max()]))

    def setBlobs(self, objects, sizeCutoff):
        from PYME.Analysis.points import gen3DTriangs

        vs, self.c = gen3DTriangs.blobify2D(objects, sizeCutoff)

        #vs = vs.ravel()
        #self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        #self.mode = 'triang'

        #self.nVertices = vs.shape[0]
        #self.setColour(self.IScale, self.zeroPt)
        #self.setCLim((0, len(objects)))
        N = vs.shape[0]

        vs = np.vstack([vs, 0*vs[:,0]])

        self.SetCurrent(self.gl_context)
        self.layers.append(RenderLayer(vs, N, self.c, self.cmap, [self.c.min(), self.c.max()]))

    def setTriang3D(self, x,y,z, c = None, sizeCutoff=1000., zrescale=1, internalCull = True, wireframe=False, alpha=1,
                    recenter=True):

        #center data
        x = x #- x.mean()
        y = y #- y.mean()
        z = z #- z.mean()

        if recenter:
            self.xc = x.mean()
            self.yc = y.mean()
            self.zc = z.mean()

        self.sx = x.max() - x.min()
        self.sy = y.max() - y.min()
        self.sz = z.max() - z.min()

        P, A, N = gen3DTriangs(x,y,z/zrescale, sizeCutoff, internalCull=internalCull)
        P[:,2] = P[:,2]*zrescale

        self.scale = 2./(x.max() - x.min())



        self.vecUp = numpy.array([0,1,0])
        self.vecRight = numpy.array([1,0,0])
        self.vecBack = numpy.array([0,0,1])

        if c == 'z':
            self.c = P[:,2]
        else:
            self.c = 1./A

        #self.a = 1./A
        self.a = 0.5*numpy.ones_like(A)
        vs = P

        self.SetCurrent(self.gl_context)

        if wireframe:
            mode = 'wireframe'
        else:
            mode = 'triang'

        self.layers.append(RenderLayer(vs, N, self.c, self.cmap, [self.c.min(), self.c.max()], mode=mode, alpha = alpha))
        self.Refresh()
        
    def setTriang(self, T, c = None, sizeCutoff=1000., zrescale=1, internalCull = True, wireframe=False, alpha=1, recenter=True):
        #center data
        #x = x #- x.mean()
        #y = y #- y.mean()
        x = T.x
        y = T.y
        xs = x[T.triangles]
        ys = y[T.triangles]
        zs = np.zeros_like(xs) #- z.mean()
        
        if recenter:        
            self.xc = x.mean()
            self.yc = y.mean()
            self.zc = 0#z.mean()
        
            self.sx = x.max() - x.min()
            self.sy = y.max() - y.min()
            self.sz = 0#z.max() - z.min()
            
            self.scale = 2./(max(self.sx, self.sy))

        if c is None:
            a = numpy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
            b = numpy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T
            b2 = numpy.vstack((xs[:,1] - xs[:,2], ys[:,1] - ys[:,2])).T
            
            c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)], 0)
            c = 1.0/(c + 1)
            
        self.c = numpy.vstack((c,c,c)).T.ravel()
        
        vs = numpy.vstack((xs.ravel(), ys.ravel(), zs.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 3)
        
        N = -0.69*numpy.ones_like(vs)
            

        self.vecUp = numpy.array([0,1,0])
        self.vecRight = numpy.array([1,0,0])
        self.vecBack = numpy.array([0,0,1])

        #if c == 'z':
        #    self.c = P[:,2]
        #else:
        #    self.c = 1./A
            
        #self.a = 1./A
        #self.a = alpha*numpy.ones_like(c)
        #vs = P

        self.SetCurrent(self.gl_context)
        
        if wireframe:
            mode = 'wireframe'
        else:
            mode = 'triang'
                        
        self.layers.append(RenderLayer(vs, N, self.c, self.cmap, self.clim, mode=mode, alpha = alpha))
        self.Refresh()
        
    def setTriangEdges(self, T):
        self.setTriang(T, wireframe=True)
        
    
    def setQuads(self, qt, maxDepth = 100, mdscale=False):
        lvs = qt.getLeaves(maxDepth)

        xs = numpy.zeros((len(lvs), 4))
        ys = numpy.zeros((len(lvs), 4))
        c = numpy.zeros(len(lvs))

        i = 0

        maxdepth = 0
        for l in lvs:
            xs[i, :] = [l.x0, l.x1, l.x1, l.x0]
            ys[i, :] = [l.y0, l.y0, l.y1, l.y1]
            c[i] = float(l.numRecords)*2**(2*l.depth)
            i +=1
            maxdepth = max(maxdepth, l.depth)

        if not mdscale:
            c = c/(2**(2*maxdepth))

        self.c = numpy.vstack((c,c,c,c)).T.ravel()

        vs = numpy.vstack((xs.ravel(), ys.ravel(), 0*xs.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 3)
        N = -0.69 * numpy.ones_like(vs)
        #vs_ = glVertexPointerf(vs)

        mode = 'quads'
        
        self.SetCurrent(self.gl_context)
        self.layers.append(RenderLayer(vs, N, self.c, self.cmap, self.clim, mode=mode, alpha=1))
        self.Refresh()

        #self.nVertices = vs.shape[0]
        #self.setColour(self.IScale, self.zeroPt)

    def setPoints3D(self, x, y, z, c = None, a = None, recenter=False, alpha=1.0, mode='points'):#, clim=None):
        # center data
        x = x # - x.mean()
        y = y # - y.mean()
        z = z # - z.mean()
        
        if recenter:        
            self.xc = x.mean()
            self.yc = y.mean()

        self.zc = z.mean()
        self.zc_o = 1.0*self.zc

        if c is None:
            self.c = numpy.ones(x.shape).ravel()
        else:
            self.c = c

        if a:
            self.a = a
        else:
            self.a = numpy.ones(x.shape).ravel()

        #if clim == None:
        #    clim = [self.c.min(), self.c.max()]
            
        self.sx = x.max() - x.min()
        self.sy = y.max() - y.min()
        self.sz = z.max() - z.min()
        
        self.SetCurrent(self.gl_context)
        vs = numpy.vstack((x.ravel(), y.ravel(), z.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 3)

        self.layers.append(RenderLayer(vs, -0.69*numpy.ones(vs.shape), self.c, self.cmap, self.clim, mode,
                                           pointsize=self.pointSize, alpha=alpha))
        self.Refresh()
        
    def setPoints(self, x, y, c = None, a = None, recenter=True, alpha=1.0):
        """Set 2D points"""
        self.setPoints3D(x, y, 0*x, c, a, recenter, alpha)
        
    def setTracks3D(self, x, y, z, ci, c = None):
        NClumps = int(ci.max())
        
        clist = [[] for i in xrange(NClumps)]
        for i, cl_i in enumerate(ci):
            clist[int(cl_i-1)].append(i)


        clumpSizes = [len(cl_i) for cl_i in clist]
        clumpStarts = numpy.cumsum([0,] + clumpSizes)

        I = numpy.hstack([numpy.array(cl) for cl in clist])

        if c is None:
            self.c = numpy.ones(x.shape).ravel()
        else:
            self.c = numpy.array(c)[I]

        x = x[I]
        y = y[I]
        z = z[I]
        ci = ci[I]
        
        #there may be a different number of points in each clump; generate a lookup
        #table for clump numbers so we can index into our list of results to get
        #all the points within a certain range of clumps

        vs = numpy.vstack((x.ravel(), y.ravel(), z.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 3)
        
        #self.layers.append(TrackLayer(vs, -0.69*numpy.ones(vs.shape), self.c, self.cmap, [self.c.min(), self.c.max()], mode='tracks'))
        self.layers.append(TrackLayer(vs, self.c, self.cmap, self.clim, clumpSizes, clumpStarts))
        
        self.Refresh()
        
    def setTracks(self, x, y, ci, c = None):
        self.setTracks3D(x, y, np.zeros_like(x), ci, c)
        
       
        
    def ResetView(self):
        #self.xc = self.sx/2
        #self.yc = self.sy/2
        #self.zc = 0

        self.vecUp = numpy.array([0,1,0])
        self.vecRight = numpy.array([1,0,0])
        self.vecBack = numpy.array([0,0,1])


        #self.scale = 2./(self.sx)

        self.Refresh()


    
    def setColour(self, IScale=None, zeroPt=None):
        #self.IScale = IScale
        #self.zeroPt = zeroPt

        #cs = numpy.minimum(numpy.vstack((IScale[0]*self.c - zeroPt[0],IScale[1]*self.c - zeroPt[1],IScale[2]*self.c - zeroPt[2])), 1).astype('f')
        #cs_ = ((self.c - self.clim[0])/(self.clim[1] - self.clim[0]))
        #csa_ = ((self.a - self.alim[0])/(self.alim[1] - self.alim[0]))
        #csa_ = numpy.minimum(csa_, 1.)
        #csa_ = numpy.maximum(csa_, 0.)
        #cs = self.cmap(cs_)
        #cs[:,3] = csa_
        #print cs.shape
        #print cs.shape
        #print cs.strides
        #cs = cs[:, :3] #if we have an alpha component chuck it
        #cs = cs.ravel().reshape(len(self.c), 4)
        #self.cs_ = glColorPointerf(cs)

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

    @property
    def xmin(self):
        return self.xc - 0.5*self.pixelsize*self.Size[0]
    
    @property
    def xmax(self):
        return self.xc + 0.5*self.pixelsize*self.Size[0]
        
    @property
    def ymin(self):
        return self.yc - 0.5*self.pixelsize*self.Size[1]
        
    @property
    def ymax(self):
        return self.yc + 0.5*self.pixelsize*self.Size[1]
        
    @property
    def pixelsize(self):
        return 2./(self.scale*self.Size[0])

    def setView(self, xmin, xmax, ymin, ymax):
        #self.xmin = xmin
        #self.xmax = xmax
        #self.ymin = ymin
        #self.ymax = ymax
        
        self.xc = (xmin + xmax)/2.0
        self.yc = (ymin + ymax)/2.0
        self.zc = self.zc_o# 0#z.mean() 
        
        self.scale = 2./(xmax - xmin)


        self.Refresh()
        if 'OnGLViewChanged' in dir(self.parent):
            self.parent.OnGLViewChanged()

        for callback in self.wantViewChangeNotification:
            callback.Refresh()

    def moveView(self, dx, dy, dz=0):
        return self.pan(dx, dy, dz)
    
    def pan(self, dx, dy, dz=0):
        #self.setView(self.xmin + dx, self.xmax + dx, self.ymin + dy, self.ymax + dy)
        self.xc += dx
        self.yc += dy
        self.zc += dz

        self.Refresh()

        for callback in self.wantViewChangeNotification:
            callback.Refresh()

    def drawScaleBar(self):
        if not self.scaleBarLength is None:
            view_size_x = self.xmax - self.xmin
            view_size_y = self.ymax - self.ymin

            sb_ur_x = -self.xc + self.xmax - self.scaleBarOffset[0]*view_size_x/self.Size[0]
            sb_ur_y = - self.yc + self.ymax - self.scaleBarOffset[1]*view_size_y/self.Size[1]
            sb_depth = self.scaleBarDepth*view_size_y/self.Size[1]

            glDisable(GL_LIGHTING)

            glColor3fv(self.scaleBarColour)
            glBegin(GL_POLYGON)
            glVertex3f(sb_ur_x, sb_ur_y, 0)
            glVertex3f(sb_ur_x, sb_ur_y - sb_depth, 0)
            glVertex3f(sb_ur_x - self.scaleBarLength, sb_ur_y - sb_depth, 0)
            glVertex3f(sb_ur_x - self.scaleBarLength, sb_ur_y, 0)
            glEnd()

    def drawLUT(self):
        if self.LUTDraw:
            mx = self.c.max()

            view_size_x = self.xmax - self.xmin
            view_size_y = self.ymax - self.ymin

            lb_ur_x = -self.xc + self.xmax - self.scaleBarOffset[0]*view_size_x/self.Size[0]
            lb_ur_y = .4*view_size_y  #- #3*self.scaleBarOffset[1]*view_size_y/self.Size[1]

            lb_lr_y = -.4*view_size_y #self.ymin #+ 3*self.scaleBarOffset[1]*view_size_y/self.Size[1]
            lb_width = 2*self.scaleBarDepth*view_size_x/self.Size[0]
            lb_ul_x = lb_ur_x - lb_width

            lb_len = lb_ur_y - lb_lr_y

            #sc = mx*numpy.array(self.IScale)
            #zp = numpy.array(self.zeroPt)
            #sc = mx
            glDisable(GL_LIGHTING)

            glBegin(GL_QUAD_STRIP)

            for i in numpy.arange(0,1, .01):
                #glColor3fv(i*sc - zp)
                #glColor3fv(self.cmap((i*mx - self.clim[0])/(self.clim[1] - self.clim[0]))[:3])
                glColor3fv(self.cmap(i)[:3])
                glVertex2f(lb_ul_x, lb_lr_y + i*lb_len)
                glVertex2f(lb_ur_x, lb_lr_y + i*lb_len)

            glEnd()

            glBegin(GL_LINE_LOOP)
            glColor3f(.5,.5,0)
            glVertex2f(lb_ul_x, lb_lr_y)
            glVertex2f(lb_ur_x, lb_lr_y)
            glVertex2f(lb_ur_x, lb_ur_y)
            glVertex2f(lb_ul_x, lb_ur_y)
            glEnd()


    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        #view_size_x = self.xmax - self.xmin
        #view_size_y = self.ymax - self.ymin

        #get translated coordinates
        #xp = 1*(event.GetX() - self.Size[0]/2)/float(self.Size[0])
        #yp = 1*(self.Size[1]/2 - event.GetY())/float(self.Size[1])

        #x, y = event.GetX(), event.GetY()
        xp, yp = self._ScreenCoordinatesToNm(event.GetX(), event.GetY())

        dx, dy = (xp - self.xc), (yp - self.yc)

        dx_, dy_, dz_, c_ = numpy.dot(self.object_rotation_matrix, [dx, dy, 0, 0])

        xp_, yp_, zp_ = (self.xc + dx_), (self.yc + dy_), (self.zc + dz_)

        #print xp
        #print yp
        if event.MiddleIsDown():
            self.WheelFocus(rot, xp_, yp_, zp_)
        else:
            self.WheelZoom(rot, xp_, yp_, zp_)


    

    def WheelZoom(self, rot, xp, yp, zp=0):
        dx = xp - self.xc
        dy = yp - self.yc
        dz = zp - self.zc

        if rot > 0:
            #zoom out
            self.scale *=ZOOM_FACTOR

            self.xc += dx*(1.- 1./ZOOM_FACTOR)
            self.yc += dy*(1.- 1./ZOOM_FACTOR)
            self.zc += dz*(1.- 1./ZOOM_FACTOR)

        if rot < 0:
            #zoom in
            self.scale /=ZOOM_FACTOR

            self.xc += dx*(1.- ZOOM_FACTOR)
            self.yc += dy*(1.- ZOOM_FACTOR)
            self.zc += dz*(1.- ZOOM_FACTOR)

        self.Refresh()

        for callback in self.wantViewChangeNotification:
            callback.Refresh()
            
    def WheelFocus(self, rot, xp, yp, zp = 0):
        if rot > 0:
            #zoom out
            self.zc -= 1.

        if rot < 0:
            #zoom in
            self.zc +=1.

        self.Refresh()

        for callback in self.wantViewChangeNotification:
            callback.Refresh()
 


    def OnLeftDown(self, event):
        if not self.displayMode == '2D':
            #dragging the mouse rotates the object
            self.xDragStart = event.GetX()
            self.yDragStart = event.GetY()

            self.angyst = self.angup
            self.angxst = self.angright

            self.dragging = True
        else: #2D
            #dragging the mouse sets an ROI
            xp, yp = self._ScreenCoordinatesToNm(event.GetX(), event.GetY())
            if True:#self.vp is None:
                self.selectionDragging = True
                self.selectionSettings.show = True

                self.selectionSettings.start = (xp, yp)
                self.selectionSettings.finish = (xp, yp)
            else:
                self.vp.xp = xp/self.vpVoxSize
                self.vp.yp = yp/self.vpVoxSize

                self.Refresh()
                self.Update()

        event.Skip()

    def OnLeftUp(self, event):
        self.dragging=False

        if self.selectionDragging:
            xp, yp = self._ScreenCoordinatesToNm(event.GetX(), event.GetY())
            
            self.selectionSettings.finish = (xp, yp)
            self.selectionDragging=False
            
            self.Refresh()
            self.Update()

        event.Skip()
        
    def OnMiddleDown(self, event):
        self.xDragStart = event.GetX()
        self.yDragStart = event.GetY()

        self.panning = True
        event.Skip()

    def OnMiddleUp(self, event):
        self.panning=False
        event.Skip()

    def _ScreenCoordinatesToNm(self, x, y):
        #FIXME!!!
        x_ = self.pixelsize*(x - 0.5*float(self.Size[0])) + self.xc
        y_ = -self.pixelsize*(y - 0.5*float(self.Size[1])) + self.yc
        #print x_, y_
        return x_, y_ 

    def OnMouseMove(self, event):
        x = event.GetX()
        y = event.GetY()
        
        if self.selectionDragging:
            self.selectionSettings.finish = self._ScreenCoordinatesToNm(x, y)

            self.Refresh()
            event.Skip()

        elif self.dragging:    
            #self.angup = self.angyst + x - self.xDragStart
            #self.angright = self.angxst + y - self.yDragStart

            angx = numpy.pi*(x - self.xDragStart)/180
            angy = -numpy.pi*(y - self.yDragStart)/180

            #vecRightN = numpy.cos(angx) * self.vecRight + numpy.sin(angx) * self.vecBack
            #vecBackN = numpy.cos(angx) * self.vecBack - numpy.sin(angx) * self.vecRight
            rMat1 = numpy.matrix([[numpy.cos(angx), 0, numpy.sin(angx)], [0,1,0], [-numpy.sin(angx), 0, numpy.cos(angx)]])
            rMat = rMat1*numpy.matrix([[1,0,0],[0,numpy.cos(angy), numpy.sin(angy)], [0,-numpy.sin(angy), numpy.cos(angy)]])

            vecRightN = numpy.array(rMat*numpy.matrix(self.vecRight).T).squeeze()
            vecUpN = numpy.array(rMat*numpy.matrix(self.vecUp).T).squeeze()
            vecBackN = numpy.array(rMat*numpy.matrix(self.vecBack).T).squeeze()

            self.vecRight = vecRightN
            #self.vecBack = vecBackN

            #vecUpN = numpy.cos(angy) * self.vecUp + numpy.sin(angy) * self.vecBack
            #vecBackN = numpy.cos(angy) * self.vecBack - numpy.sin(angy) * self.vecUp

            self.vecUp = vecUpN
            self.vecBack = vecBackN

            #print self.vecUp, self.vecRight, self.vecBack

            self.xDragStart = x
            self.yDragStart = y

            self.Refresh()
            event.Skip()
            
        elif self.panning:
            #x = event.GetX()
            #y = event.GetY()
            
            dx = self.pixelsize*(x - self.xDragStart)
            dy = -self.pixelsize*(y - self.yDragStart)
            
            #print dx

            dx_, dy_, dz_, c_ = numpy.dot(self.object_rotation_matrix, [dx, dy, 0, 0])
            
            #self.xc -= dx_
            #self.yc -= dy_
            #self.zc -= dz_

            self.xDragStart = x
            self.yDragStart = y

            self.pan(-dx_, -dy_, -dz_)

            #self.Refresh()

            #for callback in self.wantViewChangeNotification:
            #    callback.Refresh()

            event.Skip()
            
    def OnKeyPress(self, event):
        # print event.GetKeyCode()
        if event.GetKeyCode() == 83:  # S - toggle stereo
            self.stereo = not self.stereo
            self.Refresh()
        elif event.GetKeyCode() == 67:  # C - centre
            self.xc = self.sx/2
            self.yc = self.sy/2
            self.zc = self.sz/2
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
            pos = numpy.array([self.xc, self.yc, self.zc], 'f')
            pos -= 300*self.vecRight
            self.xc, self.yc, self.zc = pos
            # print 'l'
            self.Refresh()
            
        elif event.GetKeyCode() == 315:  # up
            pos = numpy.array([self.xc, self.yc, self.zc])
            pos -= 300*self.vecBack
            self.xc, self.yc, self.zc = pos
            self.Refresh()
            
        elif event.GetKeyCode() == 316:  # right
            pos = numpy.array([self.xc, self.yc, self.zc])
            pos += 300*self.vecRight
            self.xc, self.yc, self.zc = pos
            self.Refresh()
            
        elif event.GetKeyCode() == 317:  # down
            pos = numpy.array([self.xc, self.yc, self.zc])
            pos += 300*self.vecBack
            self.xc, self.yc, self.zc = pos
            self.Refresh()
            
        else:
            event.Skip()

    def getSnapshot(self, mode = GL_LUMINANCE):
        snap =  glReadPixelsf(0,0,self.Size[0],self.Size[1], mode)
        
        #snap = snap.ravel().reshape(self.Size[0], self.Size[1], -1, order='F')

        if mode == GL_LUMINANCE:
            snap.strides = (4, 4*snap.shape[0])
        else: #GL_RGB
            snap.strides = (12,12*snap.shape[0], 4)

        return snap
    
    def getIm(self, pixelSize=None):
        #FIXME - this is copied from 2D code and is currently broken.
        if pixelSize is None: #use current pixel size
            self.OnDraw()
            return self.getSnapshot(GL_RGB)
        else:
            #status = statusLog.StatusLogger('Tiling image ...')
            #save a copy of the viewport
            minx, maxx, miny, maxy  = (self.xmin, self.xmax, self.ymin, self.ymax)
            #and scalebar and LUT settings
            lutD = self.LUTDraw
            self.LUTDraw = False

            scaleB = self.scaleBarLength
            self.scaleBarLength = None

            sx, sy = self.Size
            dx, dy = (maxx - minx, maxy-miny)

            #print dx
            #print dy

            nx = numpy.ceil(dx/pixelSize) #number of x pixels
            ny = numpy.ceil(dy/pixelSize) #  "    "  y   "

            #print nx
            #print ny

            sxn = pixelSize*sx
            syn = pixelSize*sy

            #print sxn
            #print syn

            #initialise array to hold tiled image
            h = numpy.zeros((nx,ny))

            #do the tiling
            for x0 in numpy.arange(minx, maxx, sxn):
                self.xmin = x0
                self.xmax = x0 + sxn

                #print x0

                xp = numpy.floor((x0 - minx)/pixelSize)
                xd = min(xp+sx, nx) - xp

                #print 'xp = %3.2f, xd = %3.2f' %(xp, xd)

                for y0 in numpy.arange(miny, maxy, syn):
                    status.setStatus('Tiling Image at %3.2f, %3.2f' %(x0, y0))
                    self.ymin = y0
                    self.ymax = y0 + syn

                    yp = numpy.floor((y0 - miny)/pixelSize)
                    yd = min(yp+sy, ny) - yp

                    self.OnDraw()
                    tile = self.getSnapshot(GL_RGB)[:,:,0].squeeze()

                    #print tile.shape
                    #print h[xp:(xp + xd), yp:(yp + yd)].shape
                    #print tile[:xd, :yd].shape
                    #print syn

                    h[xp:(xp + xd), yp:(yp + yd)] = tile[:xd, :yd]
                    #h[xp:(xp + xd), yp:(yp + yd)] = y0 #tile[:xd, :yd]



            #restore viewport
            self.xmin, self.xmax, self.ymin, self.ymax = (minx, maxx, miny, maxy)
            self.LUTDraw = lutD
            self.scaleBarLength = scaleB

            self.Refresh()
            self.Update()
            return h


def showGLFrame():
    f = wx.Frame(None, size=(800, 800))
    c = LMGLCanvas(f)
    f.Show()
    return c


class TestApp(wx.App):
    def __init__(self, *args):
        wx.App.__init__(self, *args)

    def OnInit(self):
        # wx.InitAllImageHandlers()
        frame = wx.Frame(None, -1, 'ball_wx', wx.DefaultPosition, wx.Size(800, 800))
        canvas = LMGLCanvas(frame)
        canvas.gl_context.SetCurrent(canvas)
        # glcontext = wx.glcanvas.GLContext(canvas)
        # glcontext.SetCurrent(canvas)
        to = NineCollections()
        canvas.displayMode = '3D'
        canvas.setPoints3D(to.x, to.y, to.z)
        # canvas.setTriang3D(to.x, to.y, to.z, sizeCutoff=6e3, alpha=0.5)
        canvas.setTriang3D(to.x, to.y, to.z, sizeCutoff=6e3, wireframe=False)
        canvas.Refresh()
        frame.Show()
        self.SetTopWindow(frame)
        return True

def main():
    app = TestApp()
    app.MainLoop()

if __name__ == '__main__':
    main()

#!/usr/bin/python

##################
# gl_render3D.py
#
# Copyright David Baddeley, 2009
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

from wx.glcanvas import GLCanvas
import wx.glcanvas
import wx
#from OpenGL.GLUT import *
#from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys,math
#import sys
import numpy
#import Image
#from matplotlib import delaunay
#from PYME.Analysis.QuadTree import pointQT
#import scipy
import pylab

try:
    from gen3DTriangs import gen3DTriangs, gen3DBlobs, testObj
except:
    pass

#import statusLog

name = 'ball_glut'

def testObj():
    x = 5e3*((numpy.arange(270)%27)/9 + 0.1*numpy.random.randn(270))
    y = 5e3*((numpy.arange(270)%9)/3 + 0.1*numpy.random.randn(270))
    z = 5e3*(numpy.arange(270)%3 + 0.1*numpy.random.randn(270))

    return x, y, z

class cmap_mult:
    def __init__(self, gains, zeros):
        self.gains = gains
        self.zeros = zeros

    def __call__(self, cvals):
        return numpy.minimum(numpy.vstack((self.gains[0]*cvals - self.zeros[0],self.gains[1]*cvals - self.zeros[1],self.gains[2]*cvals - self.zeros[2], 1+ 0*cvals)), 1).astype('f').T

cm_hot = cmap_mult(8.0*numpy.ones(3)/3, [0, 3.0/8, 6.0/8])
cm_grey = cmap_mult(numpy.ones(3), [0, 0, 0])

    
class RenderLayer(object):
    drawModes = {'triang':GL_TRIANGLES, 'quads':GL_QUADS, 'edges':GL_LINES, 'points':GL_POINTS, 'wireframe':GL_TRIANGLES}
    
    def __init__(self, vertices, normals, colours, cmap, clim, mode='triang', pointsize=5, alpha=1):
        self.verts = vertices
        self.normals = normals
        self.cols = colours
        
        self.cmap = cmap
        self.clim = clim
        self.alpha = alpha
        
        cs_ = ((self.cols - self.clim[0])/(self.clim[1] - self.clim[0]))
        cs = self.cmap(cs_)
        cs[:,3] = self.alpha
        
        self.cs = cs.ravel().reshape(len(self.cols), 4)
        
        self.mode = mode
        self.pointSize=pointsize
        
    def render(self):
        if self.mode in ['points']:
            glDisable(GL_LIGHTING)
            #glPointSize(self.pointSize*self.scale*(self.xmax - self.xmin))
            glPointSize(self.pointSize)
        else:
            glEnable(GL_LIGHTING)
            pass        
        
        if self.mode in ['wireframe']:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
        nVertices = self.verts.shape[0]
            
        self.vs_ = glVertexPointerf(self.verts)
        self.n_ = glNormalPointerf(self.normals)
        self.c_ = glColorPointerf(self.cs)
            
        glPushMatrix ()
        glColor4f(0,0.5,0, 1)

        glDrawArrays(self.drawModes[self.mode], 0, nVertices)

        glPopMatrix ()
        

class LMGLCanvas(GLCanvas):
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
        wx.EVT_MOTION(self, self.OnMouseMove)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)
        #wx.EVT_MOVE(self, self.OnMove)
        
        self.gl_context = wx.glcanvas.GLContext(self)

        self.init = 0
        self.nVertices = 0
        self.IScale = [1.0, 1.0, 1.0]
        self.zeroPt = [0, 1.0/3, 2.0/3]
        self.cmap = pylab.cm.hsv
        self.clim = [0,1]
        self.alim = [0,1]
        
        self.wireframe = False

        self.parent = parent

        self.pointSize=5 #default point size = 5nm

        self.pixelsize = 5./800

        self.xmin =0
        self.xmax = self.pixelsize*self.Size[0]
        self.ymin = 0
        self.ymax = self.pixelsize*self.Size[1]

        self.scaleBarLength = 200

        self.scaleBarOffset = (20.0, 20.0) #pixels from corner
        self.scaleBarDepth = 10.0 #pixels
        self.scaleBarColour = [1,1,0]

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

        self.scale = 1
        self.stereo = True
        
        self.eye_dist = .1
        
        self.dragging = False
        self.panning = False

        self.edgeThreshold = 200
        
        self.layers = []

        return

    def OnPaint(self,event):
        if not self.IsShown():
            print('ns')
            return
        print('foo')
        #raise Exception('foo')
        dc = wx.PaintDC(self)
        #print self.GetContext()
        self.gl_context.SetCurrent(self)
        self.SetCurrent()
        if not self.init:
            self.InitGL()
            self.init = 1
        else:
            self.OnDraw()
        return

    def OnSize(self, event):
        #self.SetCurrent(self.gl_context)
        #glViewport(0,0, self.Size[0], self.Size[1])

        self.xmax = self.xmin + self.Size[0]*self.pixelsize
        self.ymax = self.ymin + self.Size[1]*self.pixelsize
        
        #self.interlace_stencil()
        
    def OnMove(self, event):
        self.Refresh()
        
    def interlace_stencil(self):
        WindowWidth = self.Size[0]
        WindowHeight = self.Size[1]
        # setting screen-corresponding geometry
        glViewport(0,0,WindowWidth,WindowHeight)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity
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
        
        if self.stereo:
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
            #glOrtho(-10,10,-10,10,-1000,1000)
            glFrustum(-1 + eye,1 + eye,-1,1,1.5,20)
            #glFrustum(
            #       (-(1+eye) *(ps.znear/ps.zscreen)*xfactor,
            #       (ps.w/(2.0*PIXELS_PER_INCH)+Eye)    *(ps.znear/ps.zscreen)*xfactor,
            #       -(ps.h/(2.0*PIXELS_PER_INCH))*(ps.znear/ps.zscreen)*yfactor,
            #       (ps.h/(2.0*PIXELS_PER_INCH))*(ps.znear/ps.zscreen)*yfactor,
            #       ps.znear, ps.zfar);
            #gluPerspective(60, 1, 1.5, 20)
            
            glMatrixMode(GL_MODELVIEW)
            glTranslatef(eye,0.0,0.0)
    
            self.setupLights()

            trafMatrix = numpy.array([numpy.hstack((self.vecRight, 0)), numpy.hstack((self.vecUp, 0)), numpy.hstack((self.vecBack, 0)), [0,0,0, 1]])
            self.drawAxes(trafMatrix)            
            
            glTranslatef(-self.xc, -self.yc, -self.zc)             
            glTranslatef(0, 0, -10)

    
            glMultMatrixf(trafMatrix)
    
            glScalef(self.scale, self.scale, self.scale)

              
            #glPushMatrix()
            #color = [1.0,0.,0.,1.]
            #glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
            #glutSolidSphere(2,20,20)
            #glColor3f(1,0,0)
    
            #glBegin(GL_POLYGON)
            #glVertex2f(0, 0)
            #glVertex2f(1,0)
            #glVertex2f(1,1)
            #glVertex2f(0,1)
            #glEnd()
    
#            #print self.scale
#            if self.wireframe:
#                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
#            else:
#                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
#    
#            if self.mode in  ['points']:
#                glDisable(GL_LIGHTING)
#                #glPointSize(self.pointSize*self.scale*(self.xmax - self.xmin))
#                glPointSize(self.pointSize)
#            else:
#                pass
#                #glEnable(GL_LIGHTING)
#    
#            #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#            glPushMatrix ()
#    
#            glColor4f(0,0.5,0, 1)
#    
#            glDrawArrays(self.drawModes[self.mode], 0, self.nVertices)
#    
#            glPopMatrix ()
            
            for l in self.layers:
                l.render()
        

        glFlush()
        #glPopMatrix()
        #print 'odf'
        self.SwapBuffers()
        
        #print 'odd'
        return

    def setupLights(self):
        # set viewing projection
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        #light_diffuse = [1., 1., 1., 1.0]
        light_position = [20.0, 20.00, 20.0, 0.0]

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.5, 0.5, 0.5, 1.0]);
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3,0.3,0.3,1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    def InitGL(self):
        
#        # set viewing projection
#        light_diffuse = [0.5, 0.5, 0.5, 1.0]
#        #light_diffuse = [1., 1., 1., 1.0]
#        light_position = [20.0, 20.00, 20.0, 0.0]
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
        glEnable(GL_POINT_SMOOTH)

        #self.vs = glVertexPointerf(numpy.array([[5, 5],[10000, 10000], [10000, 5]]).astype('f'))
        #self.cs = glColorPointerf(numpy.ones((3,3)).astype('f'))

        #self.nVertices = 3

        #to = testObj()

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

    def drawAxes(self,trafMatrix):
        glDisable(GL_LIGHTING)
        glPushMatrix ()

        glTranslatef(.8, -.85, -2)
        glScalef(.1, .1, .1)
        glLineWidth(2)
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

    def setBlob(self, x,y,z, sizeCutoff=1000., zrescale=1, smooth=False, smScale=[10,10,10]):
        #center data
        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()

        P, A, N = gen3DBlobs(x,y,z/zrescale, sizeCutoff, smooth, smScale)
        P[:,2] = P[:,2]*zrescale
        
        self.sx = x.max() - x.min()
        self.sy = y.max() - y.min()
        self.sz = z.max() - z.min()

        self.c = A
        self.a = self.c
        vs = P

        self.SetCurrent()
        
        self.layers.append(RenderLayer(vs, N, self.c, self.cmap, [self.c.min(), self.c.max()]))
        

    def setTriang(self, x,y,z, c = None, sizeCutoff=1000., zrescale=1, internalCull = True, wireframe=False, alpha=1):
        #center data
        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()
        
        self.sx = x.max() - x.min()
        self.sy = y.max() - y.min()
        self.sz = z.max() - z.min()

        P, A, N = gen3DTriangs(x,y,z/zrescale, sizeCutoff, internalCull=internalCull)
        P[:,2] = P[:,2]*zrescale

        self.scale = 10./(x.max() - x.min())

        self.xc = 0
        self.yc = 0
        self.zc = 0

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

        self.SetCurrent()
        
        if wireframe:
            mode = 'wireframe'
        else:
            mode = 'triang'
                        
        self.layers.append(RenderLayer(vs, N, self.c, self.cmap, [self.c.min(), self.c.max()], mode=mode, alpha = alpha))
        self.Refresh()


    def setPoints(self, x, y, z, c = None, a = None):
        #center data
        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()

        if c == None:
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
        
        self.SetCurrent()
        vs = numpy.vstack((x.ravel(), y.ravel(), z.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 3)
        
        self.layers.append(RenderLayer(vs, -0.69*numpy.ones(vs.shape), self.c, self.cmap, [self.c.min(), self.c.max()], mode='points'))
        self.Refresh()

        
    def ResetView(self):
        self.xc = 0
        self.yc = 0
        self.zc = 0

        self.vecUp = numpy.array([0,1,0])
        self.vecRight = numpy.array([1,0,0])
        self.vecBack = numpy.array([0,0,1])


        self.scale = 5./(self.sx)


    
    def setColour(self, IScale=None, zeroPt=None):
        #self.IScale = IScale
        #self.zeroPt = zeroPt

        #cs = numpy.minimum(numpy.vstack((IScale[0]*self.c - zeroPt[0],IScale[1]*self.c - zeroPt[1],IScale[2]*self.c - zeroPt[2])), 1).astype('f')
        cs_ = ((self.c - self.clim[0])/(self.clim[1] - self.clim[0]))
        csa_ = ((self.a - self.alim[0])/(self.alim[1] - self.alim[0]))
        csa_ = numpy.minimum(csa_, 1.)
        csa_ = numpy.maximum(csa_, 0.)
        cs = self.cmap(cs_)
        cs[:,3] = csa_
        #print cs.shape
        #print cs.shape
        #print cs.strides
        #cs = cs[:, :3] #if we have an alpha component chuck it
        cs = cs.ravel().reshape(len(self.c), 4)
        self.cs_ = glColorPointerf(cs)

        self.Refresh()

    def setCMap(self, cmap):
        self.cmap = cmap
        self.setColour()

    def setCLim(self, clim, alim=None):
        self.clim = clim
        if alim == None:
            self.alim = clim
        else:
            self.alim = alim
        self.setColour()


    def setView(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.pixelsize = (xmax - xmin)*1./self.Size[0]

        self.Refresh()
        if 'OnGLViewChanged' in dir(self.parent):
            self.parent.OnGLViewChanged()

    def pan(self, dx, dy):
        self.setView(self.xmin + dx, self.xmax + dx, self.ymin + dy, self.ymax + dy)

    def drawScaleBar(self):
        if not self.scaleBarLength == None:
            view_size_x = self.xmax - self.xmin
            view_size_y = self.ymax - self.ymin

            sb_ur_x = self.xmax - self.scaleBarOffset[0]*view_size_x/self.Size[0]
            sb_ur_y = self.ymax - self.scaleBarOffset[1]*view_size_y/self.Size[1]
            sb_depth = self.scaleBarDepth*view_size_y/self.Size[1]

            glColor3fv(self.scaleBarColour)
            glBegin(GL_POLYGON)
            glVertex2f(sb_ur_x, sb_ur_y)
            glVertex2f(sb_ur_x, sb_ur_y - sb_depth)
            glVertex2f(sb_ur_x - self.scaleBarLength, sb_ur_y - sb_depth)
            glVertex2f(sb_ur_x - self.scaleBarLength, sb_ur_y)
            glEnd()




    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        #view_size_x = self.xmax - self.xmin
        #view_size_y = self.ymax - self.ymin

        #get translated coordinates
        xp = 1*(event.GetX() - self.Size[0]/2)/float(self.Size[0])
        yp = 1*(self.Size[1]/2 - event.GetY())/float(self.Size[1])

        #print xp
        #print yp
        if event.MiddleIsDown():
            self.WheelFocus(rot, xp, yp)
        else:
            self.WheelZoom(rot, xp, yp)
        self.Refresh()

    

    def WheelZoom(self, rot, xp, yp):
        if rot > 0:
            #zoom out
            self.scale *=2.

        if rot < 0:
            #zoom in
            self.scale /=2.
            
    def WheelFocus(self, rot, xp, yp):
        if rot > 0:
            #zoom out
            self.zc -= 1.

        if rot < 0:
            #zoom in
            self.zc +=1.
 


    def OnLeftDown(self, event):
        self.xDragStart = event.GetX()
        self.yDragStart = event.GetY()

        self.angyst = self.angup
        self.angxst = self.angright

        self.dragging = True

        event.Skip()

    def OnLeftUp(self, event):
        self.dragging=False
        event.Skip()
        
    def OnMiddleDown(self, event):
        self.xDragStart = event.GetX()
        self.yDragStart = event.GetY()

        self.panning = True
        event.Skip()

    def OnMiddleUp(self, event):
        self.panning=False
        event.Skip()

    def OnMouseMove(self, event):
        if self.dragging:
            x = event.GetX()
            y = event.GetY()

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
            x = event.GetX()
            y = event.GetY()
            
            dx = 10*(x - self.xDragStart)/float(self.Size[0])
            dy = 10*(y - self.yDragStart)/float(self.Size[1])
            
            #print dx
            
            self.xc -= dx
            self.yc += dy

            self.xDragStart = x
            self.yDragStart = y

            self.Refresh()

            event.Skip()
            
    def OnKeyPress(self, event):
        print event.GetKeyCode()
        if event.GetKeyCode() == 83: #S - toggle stereo
            self.stereo = not self.stereo
            self.Refresh()
        elif event.GetKeyCode() == 67: #C - centre
            self.xc = 0
            self.yc = 0
            self.zc = 0
            self.Refresh()
            
        elif event.GetKeyCode() == 91: #[ decrease eye separation
            self.eye_dist /=1.5
            self.Refresh()
        
        elif event.GetKeyCode() == 93: #] increase eye separation
            self.eye_dist *=1.5
            self.Refresh()
            
        elif event.GetKeyCode() == 82: #R reset view
            self.ResetView()
            self.Refresh()
            
        elif event.GetKeyCode() == 314: #left
            pos = numpy.array([self.xc, self.yc, self.zc], 'f')
            pos -= .1*self.vecRight
            self.xc, self.yc, self.zc = pos
            print 'l'
            self.Refresh()
            
        elif event.GetKeyCode() == 315: #up
            pos = numpy.array([self.xc, self.yc, self.zc])
            pos -= .1*self.vecBack
            self.xc, self.yc, self.zc = pos
            self.Refresh()
            
        elif event.GetKeyCode() == 316: #right
            pos = numpy.array([self.xc, self.yc, self.zc])
            pos += .1*self.vecRight
            self.xc, self.yc, self.zc = pos
            self.Refresh()
            
        elif event.GetKeyCode() == 317: #down
            pos = numpy.array([self.xc, self.yc, self.zc])
            pos += .1*self.vecBack
            self.xc, self.yc, self.zc = pos
            self.Refresh()
            
        else:
            event.Skip()

    def getSnapshot(self, mode = GL_LUMINANCE):
        snap =  glReadPixelsf(0,0,self.Size[0],self.Size[1], mode)

        snap.strides = (12,12*snap.shape[0], 4)

        return snap







def showGLFrame():
    f = wx.Frame(None, size=(800,800))
    c = LMGLCanvas(f)
    f.Show()
    return c





def main():
    app = wx.PySimpleApp()
    frame = wx.Frame(None,-1,'ball_wx',wx.DefaultPosition,wx.Size(800,800))
    canvas = LMGLCanvas(frame)
    #glcontext = wx.glcanvas.GLContext(canvas)
    #glcontext.SetCurrent(canvas)
    to = testObj()
    canvas.setPoints(to[0], to[1], to[2])
    canvas.setTriang(to[0], to[1], to[2], sizeCutoff = 6e3, alpha=0.5)
    canvas.setTriang(to[0], to[1], to[2], sizeCutoff = 6e3, wireframe=True)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__': main()

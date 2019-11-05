#!/usr/bin/python

##################
# gl_render.py
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

from OpenGL.GL import *

import numpy

import statusLog
import weakref
from six.moves import xrange

try:
    # location in Python 2.7 and 3.1
    from weakref import WeakSet
except ImportError:
    # separately installed
    from weakrefset import WeakSet

name = 'ball_glut'

class cmap_mult:
    def __init__(self, gains, zeros):
        self.gains = gains
        self.zeros = zeros

    def __call__(self, cvals):
        return numpy.minimum(numpy.vstack((self.gains[0]*cvals - self.zeros[0],self.gains[1]*cvals - self.zeros[1],self.gains[2]*cvals - self.zeros[2])), 1).astype('f').T

cm_hot = cmap_mult(8.0*numpy.ones(3)/3, [0, 3.0/8, 6.0/8])
cm_grey = cmap_mult(numpy.ones(3), [0, 0, 0])

class LMGLCanvas(GLCanvas):
    def __init__(self, parent, trackSelection=True, vp = None, vpVoxSize = None):
        attriblist = [wx.glcanvas.WX_GL_RGBA,wx.glcanvas.WX_GL_STENCIL_SIZE,8, wx.glcanvas.WX_GL_DOUBLEBUFFER, 16]
        GLCanvas.__init__(self, parent,-1, attribList = attriblist)
        #GLCanvas.__init__(self, parent,-1)
        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_LEFT_DOWN(self, self.OnLeftDown)
        wx.EVT_LEFT_UP(self, self.OnLeftUp)
        wx.EVT_LEFT_DCLICK(self, self.OnLeftDClick)
        if trackSelection:
            wx.EVT_MOTION(self, self.OnMouseMove)
        #wx.EVT_ERASE_BACKGROUND(self, self.OnEraseBackground)
        #wx.EVT_IDLE(self, self.OnIdle)
        
        self.gl_context = wx.glcanvas.GLContext(self)
        #self.gl_context.SetCurrent()

        self.init = False
        self.nVertices = 0
        self.IScale = [1.0, 1.0, 1.0]
        self.zeroPt = [0, 1.0/3, 2.0/3]
        self.cmap = cm_hot
        self.clim = [0,1]

        self.parent = parent

        self.vp = vp
        self.vpVoxSize = vpVoxSize

        self.pointSize=5 #default point size = 5nm

        self.pixelsize = 10

        self.xmin =0
        self.xmax = self.pixelsize*self.Size[0]
        self.ymin = 0
        self.ymax = self.pixelsize*self.Size[1]

        self.scaleBarLength = 200

        self.scaleBarOffset = (20.0, 20.0) #pixels from corner
        self.scaleBarDepth = 10.0 #pixels
        self.scaleBarColour = [1,1,0]

        self.crosshairColour = [0,1,1]
        self.centreCross=False

        self.numBlurSamples = 1
        self.blurSigma = 0.0

        self.LUTDraw = True
        self.mode = 'triang'

        self.backgroundImage = False

        self.colouring = 'area'

        self.drawModes = {'triang':GL_TRIANGLES, 'quads':GL_QUADS, 'edges':GL_LINES, 'points':GL_POINTS, 'tracks': GL_LINE_STRIP}

        self.c = numpy.array([1,1,1])
        self.zmin = -1
        self.zmax = 1
        self.ang = 0

        self.selectionDragging = False
        self.selectionStart = (0,0)
        self.selectionFinish = (0,0)
        self.selection = False

        self.wantViewChangeNotification = WeakSet()
        self.pointSelectionCallbacks = []

        #self.InitGL()

    def OnPaint(self,event):
        dc = wx.PaintDC(self)
        #print self.GetContext()
        self.SetCurrent()
        self.gl_context.SetCurrent(self)
        #print self.GetContext()
        #print 'OnPaint', event.GetId()
        if not self.init:
            self.InitGL()
            self.init = True
            #if 'RefreshView' in dir(self.parent):
            #    wx.CallAfter(self.parent.RefreshView())
            
        self.OnDraw()
        #return

    #def OnEraseBackground(self, event):
    #    pass

    #def OnIdle(self, event):
    #    self.GetParent().GetParent().ProcessPendingEvents()
    #    pass
    #    #self.Refresh()

    def OnSize(self, event):
        if self.init:
            glViewport(0,0, self.Size[0], self.Size[1])

        self.xmax = self.xmin + self.Size[0]*self.pixelsize
        self.ymax = self.ymin + self.Size[1]*self.pixelsize


    def OnDraw(self):
        #self.GetParent().ProcessPendingEvents()
        #print 'OnDraw'
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glOrtho(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        
        glDisable(GL_LIGHTING)

        #glRotatef(self.ang, 0, 1, 0)
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

        #print self.drawModes[self.mode]

        self.drawBackground()

        #glClear(GL_ACCUM_BUFFER_BIT)
        if self.mode =='tracks':
            for cl in self.clumps:
                ns, nf = self.clumpIndices[cl]
                #nf = self.clumpIndices[i+1]

                if nf > ns:
                    glDrawArrays(self.drawModes[self.mode], ns, nf-ns)
        elif self.mode == 'tracks2':
            for i, cl in enumerate(self.clumpSizes):
                if cl > 0:
                    glDrawArrays(self.drawModes['tracks'], self.clumpStarts[i], cl)
        else:
            if self.mode == 'points':
                glPointSize(self.pointSize*(float(self.Size[0])/(self.xmax - self.xmin)))
            for i in range(self.numBlurSamples):
                #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glPushMatrix ()

                glTranslatef (self.blurSigma*numpy.random.normal(), self.blurSigma*numpy.random.normal(),0.0)

                glDrawArrays(self.drawModes[self.mode], 0, self.nVertices)

                glPopMatrix ()
            #glAccum(GL_ACCUM, 1.0)#/self.numBlurSamples)

        #glAccum (GL_RETURN, 1.0);

        #glDrawArrays(self.drawModes[self.mode], 0, self.nVertices)
        #glDrawArrays(GL_LINES, 0, self.nVertices)

        #glBegin(GL_TRIANGLES)
        #for i in range(self.nVertices):
        #    glColor3fv(self.cs[i,:])
        #    glVertex2fv(self.vs[i,:])
        #glEnd()

        self.drawScaleBar()
        self.drawLUT()
        self.drawSelection()
        self.drawCrosshairs()

        glFlush()
        #glPopMatrix()
        self.SwapBuffers()
        return

    def InitGL(self):
        #glutInit(sys.argv)
        # set viewing projection
        #light_diffuse = [1.0, 1.0, 1.0, 1.0]
        #light_position = [1.0, 1.0, 1.0, 0.0]

        #glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        #glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        #glEnable(GL_LIGHTING)
        #glEnable(GL_LIGHT0)
        #glEnable(GL_DEPTH_TEST)

        glLoadIdentity()
        glOrtho(self.xmin,self.xmax,self.ymin,self.ymax,-1,1)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        
        glEnable(GL_POINT_SMOOTH)


        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        self.vs = glVertexPointerf(numpy.array([[5, 5],[10000, 10000], [10000, 5]]).astype('f'))
        self.cs = glColorPointerf(numpy.ones((3,3)).astype('f'))

        self.nVertices = 3

        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #gluPerspective(40.0, 1.0, 1.0, 30.0)

        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()
        #gluLookAt(0.0, 0.0, 10.0,
        #          0.0, 0.0, 0.0,
        #          0.0, 1.0, 0.0)
        return

    def setBackgroundImage(self, image, origin = (0,0), pixelSize=70):
        if self.backgroundImage:
            #if we already have a background image, free it
            glDeleteTextures([1])

        if image is None:
            self.backgroundImage = False
            return

        image = image.T.reshape(*image.shape) #get our byte order right
        
        glBindTexture (GL_TEXTURE_2D, 1)
        glPixelStorei (GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        #glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        #glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glTexImage2D (GL_TEXTURE_2D, 0, 1, image.shape[0], image.shape[1], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, image)

        self.backgroundImage = True
        self.backgroundOrigin = origin
        self.backgroundExtent = (origin[0] + image.shape[0]*pixelSize, origin[1] + image.shape[1]*pixelSize)

    def drawBackground(self):
        if self.backgroundImage:
            glEnable (GL_TEXTURE_2D) # enable texture mapping */
            glBindTexture (GL_TEXTURE_2D, 1) # bind to our texture, has id of 1 */

            #glColor3f(1.,0.,0.)
            glBegin (GL_QUADS)
            glTexCoord2f(0., 0.) # lower left corner of image */
            glVertex3f(self.backgroundOrigin[0],self.backgroundOrigin[1], 0.0)
            glTexCoord2f (1., 0.) # lower right corner of image */
            glVertex3f (self.backgroundExtent[0],self.backgroundOrigin[1], 0.0)
            glTexCoord2f (1.0, 1.0) # upper right corner of image */
            glVertex3f (self.backgroundExtent[0],self.backgroundExtent[1], 0.0)
            glTexCoord2f (0.0, 1.0) # upper left corner of image */
            glVertex3f (self.backgroundOrigin[0],self.backgroundExtent[1], 0.0)
            glEnd ()

            glDisable (GL_TEXTURE_2D) # disable texture mapping */


    def setTriang(self, T, c = None):
        xs = T.x[T.triangles]
        ys = T.y[T.triangles]

        a = numpy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
        b = numpy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T
        b2 = numpy.vstack((xs[:,1] - xs[:,2], ys[:,1] - ys[:,2])).T

        #area of triangle
        #c = 0.5*numpy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*numpy.sqrt((a*a).sum(1))

        #c = 0.5*numpy.sqrt((b*b).sum(1)*(a*a).sum(1) - ((a*b).sum(1)**2))

        #c = numpy.maximum(((b*b).sum(1)),((a*a).sum(1)))

        if c is None:
            if True:#numpy.version.version > '1.2':
                c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)], 0)
            else:
                c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)])

        a_ = ((a*a).sum(1))
        b_ = ((b*b).sum(1))
        b2_ = ((b2*b2).sum(1))
        #c_neighbours = c[T.triangle_neighbors].sum(1)
        #c = 1.0/(c + c_neighbours + 1)
        #c = numpy.maximum(c, self.pixelsize**2)
        c = 1.0/(c + 1)

        self.c = numpy.vstack((c,c,c)).T.ravel()
        #self.c = numpy.vstack((1.0/(a_*b_ + 1),1.0/(a_*b2_ + 1),1.0/(b_*b2_ + 1))).T.ravel()

        #self.c = numpy.sqrt(self.c)
        vs = numpy.vstack((xs.ravel(), ys.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'triang'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setPoints(self, x, y, c = None):
        #if not self.init:
        #    self.InitGL()
        #    self.init = 1

        if c is None:
            self.c = numpy.ones(x.shape).ravel()
        else:
            self.c = c

        vs = numpy.vstack((x.ravel(), y.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'points'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setTracks_(self, x, y, ci, c = None):
        #if not self.init:
        #    self.InitGL()
        #    self.init = 1

        #I = ci.argsort()

        indices = numpy.arange(len(x))
        I = []

        self.clumps = set(ci)
        self.clumpIndices = {}

        ns = 0
        for cl in self.clumps:
            inds = indices[ci == cl]
            I.append(inds)
            nf = ns + len(inds)
            self.clumpIndices[cl] = (ns, nf)
            ns = nf

        I = numpy.hstack(I)

        if c is None:
            self.c = numpy.ones(x.shape).ravel()
        else:
            self.c = c[I]

        x = x[I]
        y = y[I]
        ci = ci[I]
        
        nPts = len(x)

        #there may be a different number of points in each clump; generate a lookup
        #table for clump numbers so we can index into our list of results to get
        #all the points within a certain range of clumps
        #self.clumpIndices = (nPts + 2)*numpy.ones(ci.max() + 10, 'int32')
        #self.cimax = ci.max()

        #for c_i, i in zip(ci, range(nPts)):
        #    self.clumpIndices[:(c_i+1)] = numpy.minimum(self.clumpIndices[:(c_i+1)], i)

        vs = numpy.vstack((x.ravel(), y.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'tracks'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)
        
    def setTracks(self, x, y, ci, c = None):
        #if not self.init:
        #    self.InitGL()
        #    self.init = 1

        #I = ci.argsort()
        NClumps = int(ci.max())
        
        clist = [[] for i in xrange(NClumps)]
        for i, cl_i in enumerate(ci):
            clist[int(cl_i-1)].append(i)

        #indices = numpy.arange(len(x))
        #I = []

        self.clumpSizes = [len(cl_i) for cl_i in clist]
        self.clumpStarts = numpy.cumsum([0,] + self.clumpSizes)

        #self.clumps = set(ci)
        #self.clumpIndices = {}

        #ns = 0
        #for cl in self.clumps:
        #    inds = indices[ci == cl]
        #    I.append(inds)
        #    nf = ns + len(inds)
        #    self.clumpIndices[cl] = (ns, nf)
        #    ns = nf

        I = numpy.hstack([numpy.array(cl) for cl in clist])
        #print I
        #print c.shape

        if c is None:
            self.c = numpy.ones(x.shape).ravel()
        else:
            self.c = numpy.array(c)[I]

        x = x[I]
        y = y[I]
        ci = ci[I]
        
        nPts = len(x)

        #there may be a different number of points in each clump; generate a lookup
        #table for clump numbers so we can index into our list of results to get
        #all the points within a certain range of clumps
        #self.clumpIndices = (nPts + 2)*numpy.ones(ci.max() + 10, 'int32')
        #self.cimax = ci.max()

        #for c_i, i in zip(ci, range(nPts)):
        #    self.clumpIndices[:(c_i+1)] = numpy.minimum(self.clumpIndices[:(c_i+1)], i)

        vs = numpy.vstack((x.ravel(), y.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'tracks2'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setPoints3d(self, x, y, z, c = None):
        if c is None:
            self.c = numpy.ones(x.shape).ravel()
        else:
            self.c = c

        vs = numpy.vstack((x.ravel(), y.ravel(), z.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 3)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'points'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setVoronoi(self, T, cp=None):
        from matplotlib import tri
        tdb = []
        for i in range(len(T.x)):
            tdb.append([])

        for i in range(len(T.triangles)):
            nds = T.triangles[i]
            for n in nds:
                tdb[n].append(i)



        xs_ = None
        ys_ = None
        c_ = None

        area_colouring= True

        if not cp is None:
            area_colouring=False

        for i in range(len(T.x)):
            #get triangles around point
            impingentTriangs = tdb[i] #numpy.where(T.triangle_nodes == i)[0]
            if len(impingentTriangs) >= 3:

                circumcenters = T.circumcenters[impingentTriangs] #get their circumcenters

                #add current point - to deal with edge cases
                newPts = numpy.array(list(circumcenters) + [[T.x[i], T.y[i]]])

                #re-triangulate (we could try and sort the triangles somehow, but this is easier)
                T2 = tri.Triangulation(newPts[:,0],newPts[:,1] )


                #now do the same as for the standard triangulation
                xs = T2.x[T2.triangle_nodes]
                ys = T2.y[T2.triangle_nodes]

                a = numpy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
                b = numpy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T

                #area of triangle
                c = 0.5*numpy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*numpy.sqrt((a*a).sum(1))

                #c = numpy.maximum(((b*b).sum(1)),((a*a).sum(1)))

                #c_neighbours = c[T.triangle_neighbors].sum(1)
                #c = 1.0/(c + c_neighbours + 1)
                c = c.sum()*numpy.ones(c.shape)
                c = 1.0/(c + 1)

                if not area_colouring:
                    c = cp[i]*numpy.ones(c.shape)

                #print xs.shape
                #print c.shape

                if xs_ is None:
                    xs_ = xs
                    ys_ = ys
                    c_ = c
                else:
                    xs_ = numpy.vstack((xs_, xs))
                    ys_ = numpy.vstack((ys_, ys))
                    c_ = numpy.hstack((c_, c))


        self.c = numpy.vstack((c_,c_,c_)).T.ravel()

        vs = numpy.vstack((xs_.ravel(), ys_.ravel()))
        vs = vs.T.ravel().reshape(len(xs_.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'triang'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)


    def setTriangEdges(self, T):
        xs = T.x[T.edges]
        ys = T.y[T.edges]

        a = numpy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
        #b = numpy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T

        #area of triangle
        #c = 0.5*numpy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*numpy.sqrt((a*a).sum(1))

        c = ((a*a).sum(1))

        #c_neighbours = c[T.triangle_neighbors].sum(1)
        c = 1.0/(c + 1)

        self.c = numpy.vstack((c,c)).T.ravel()

        vs = numpy.vstack((xs.ravel(), ys.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'edges'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)


    def setBlobs(self, objects, sizeCutoff):
        from PYME.Analysis.points import gen3DTriangs

        vs, self.c = gen3DTriangs.blobify2D(objects, sizeCutoff)

        #vs = vs.ravel()
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'triang'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)
        self.setCLim((0, len(objects)))

    def setIntTriang(self, T, cs= None):

        if cs is None:
            #make ourselves a quicker way of getting at edge info.
            edb = []
            for i in range(len(T.x)):
                edb.append(([],[]))

            for i in range(len(T.edges)):
                e = T.edges[i]
                edb[e[0]][0].append(i)
                edb[e[0]][1].append(e[1])
                edb[e[1]][0].append(i)
                edb[e[1]][1].append(e[0])



            #gen colour array
            cs = numpy.zeros(T.x.shape)

            for i in range(len(T.x)):
                incidentEdges = T.edges[edb[i][0]]
                #neighbourPoints = edb[i][1]

                #incidentEdges = T.edge_db[edb[neighbourPoints[0]][0]]
                #for j in range(1, len(neighbourPoints)):
                #    incidentEdges = numpy.vstack((incidentEdges, T.edge_db[edb[neighbourPoints[j]][0]]))
                dx = numpy.diff(T.x[incidentEdges])
                dy = numpy.diff(T.y[incidentEdges])

                dist = (dx**2 + dy**2)

                di = numpy.mean(numpy.sqrt(dist))


                neighbourPoints = edb[i][1]

                incidentEdges = T.edges[edb[neighbourPoints[0]][0]]
                for j in range(1, len(neighbourPoints)):
                    incidentEdges = numpy.vstack((incidentEdges, T.edges[edb[neighbourPoints[j]][0]]))
                dx = numpy.diff(T.x[incidentEdges])
                dy = numpy.diff(T.y[incidentEdges])

                dist = (dx**2 + dy**2)

                din = numpy.mean(numpy.sqrt(dist))

                #cs[i] = numpy.absolute(5 + di - 4*di/din)
                cs[i] = di

            cs = 1.0/cs**2


        xs = T.x[T.triangles]
        ys = T.y[T.triangles]

        #a = numpy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
        #b = numpy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T

        #b2 = numpy.vstack((xs[:,1] - xs[:,2], ys[:,1] - ys[:,2])).T

        #area of triangle
        #c = 0.5*numpy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*numpy.sqrt((a*a).sum(1))

        #c = 0.5*numpy.sqrt((b*b).sum(1)*(a*a).sum(1) - ((a*b).sum(1)**2))

        #c = numpy.maximum(((b*b).sum(1)),((a*a).sum(1)))
        #c = numpy.median([(b*b).sum(1), (a*a).sum(1), (b2*b2).sum(1)])

        #a_ = ((a*a).sum(1))
        #b_ = ((b*b).sum(1))
        #b2_ = ((b2*b2).sum(1))
        #c_neighbours = c[T.triangle_neighbors].sum(1)
        #c = 1.0/(c + c_neighbours + 1)
        #c = 1.0/(c + 1)

        #self.c = numpy.vstack((c,c,c)).T.ravel()
        #self.c = numpy.vstack((1.0/(a_*b_ + 1),1.0/(a_*b2_ + 1),1.0/(b_*b2_ + 1))).T.ravel()

        c = cs[T.triangles]

        c = c.mean(1)

        #area of triangle
        #c = 0.5*numpy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*numpy.sqrt((a*a).sum(1))

        #c = numpy.maximum(((b*b).sum(1)),((a*a).sum(1)))

        #c_neighbours = c[T.triangle_neighbors].sum(1)
        #c = 1.0/(c + c_neighbours + 1)
        #c = 1.0/(c + 1)

        #self.c = c.ravel()
        self.c = numpy.vstack((c,c,c)).T.ravel()

        vs = numpy.vstack((xs.ravel(), ys.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = numpy.minimum(numpy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'triang'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

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

        vs = numpy.vstack((xs.ravel(), ys.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 2)
        vs_ = glVertexPointerf(vs)

        self.mode = 'quads'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setColour(self, IScale=None, zeroPt=None):
        #self.IScale = IScale
        #self.zeroPt = zeroPt

        #cs = numpy.minimum(numpy.vstack((IScale[0]*self.c - zeroPt[0],IScale[1]*self.c - zeroPt[1],IScale[2]*self.c - zeroPt[2])), 1).astype('f')

        cs = self.cmap((self.c - self.clim[0])/(self.clim[1] - self.clim[0]))
        #print cs.shape
        #print cs.shape
        #print cs.strides
        cs = cs[:, :3]/self.numBlurSamples #if we have an alpha component chuck it
        cs = cs.ravel().reshape(len(self.c), 3)
        self.cs_ = glColorPointerf(cs)

        self.Refresh()
        self.Update()

    def setCMap(self, cmap):
        self.cmap = cmap
        self.setColour()

    def setCLim(self, clim):
        self.clim = clim
        self.setColour()

    def setPercentileCLim(self, pctile):
        """set clim based on a certain percentile"""
        clim_upper = float(self.c[numpy.argsort(self.c)[len(self.c)*pctile]])
        self.setCLim([0.0, clim_upper])


    def moveView(self, dx, dy):
        self.setView(self.xmin + dx, self.xmax + dx, self.ymin + dy, self.ymax + dy)
    
    
    def setView(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.pixelsize = (xmax - xmin)*1./self.Size[0]

        self.Refresh()
        self.Update()

        for callback in self.wantViewChangeNotification:
            callback.Refresh()
        #if 'OnGLViewChanged' in dir(self.parent):
        #    self.parent.OnGLViewChanged()
        #elif 'OnGLViewChanged' in dir(self.parent.GetParent()):
        #    self.parent.GetParent().OnGLViewChanged()

    def pan(self, dx, dy):
        self.setView(self.xmin + dx, self.xmax + dx, self.ymin + dy, self.ymax + dy)

    def drawScaleBar(self):
        if not self.scaleBarLength is None:
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

    def drawSelection(self):
        if self.selection:
            #view_size_x = self.xmax - self.xmin
            #view_size_y = self.ymax - self.ymin

            #sb_ur_x = self.xmax - self.scaleBarOffset[0]*view_size_x/self.Size[0]
            #sb_ur_y = self.ymax - self.scaleBarOffset[1]*view_size_y/self.Size[1]
            #sb_depth = self.scaleBarDepth*view_size_y/self.Size[1]

            x0,y0 = self.selectionStart
            x1, y1 = self.selectionFinish

            glColor3fv(self.scaleBarColour)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()

    def drawCrosshairs(self):
        if not self.vp is None:
            x = self.vp.xp*self.vpVoxSize
            y = self.vp.yp*self.vpVoxSize

            glColor3fv(self.crosshairColour)
            glBegin(GL_LINES)
            glVertex2f(x, self.ymin)
            glVertex2f(x, self.ymax)
            glEnd()

            glBegin(GL_LINES)
            glVertex2f(self.xmin, y)
            glVertex2f(self.xmax, y)
            glEnd()
        elif self.centreCross:
            x = .5*(self.xmin+self.xmax)
            y = .5*(self.ymin+self.ymax)

            glColor3fv(self.crosshairColour)
            glBegin(GL_LINES)
            glVertex2f(x, self.ymin)
            glVertex2f(x, self.ymax)
            glEnd()

            glBegin(GL_LINES)
            glVertex2f(self.xmin, y)
            glVertex2f(self.xmax, y)
            glEnd()

    def drawLUT(self):
        if self.LUTDraw:
            mx = self.c.max()

            view_size_x = self.xmax - self.xmin
            view_size_y = self.ymax - self.ymin

            lb_ur_x = self.xmax - self.scaleBarOffset[0]*view_size_x/self.Size[0]
            lb_ur_y = self.ymax - 3*self.scaleBarOffset[1]*view_size_y/self.Size[1]

            lb_lr_y = self.ymin + 3*self.scaleBarOffset[1]*view_size_y/self.Size[1]
            lb_width = 2*self.scaleBarDepth*view_size_x/self.Size[0]
            lb_ul_x = lb_ur_x - lb_width

            lb_len = lb_ur_y - lb_lr_y

            #sc = mx*numpy.array(self.IScale)
            #zp = numpy.array(self.zeroPt)
            #sc = mx

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


    def OnLeftDown(self, event):
        view_size_x = self.xmax - self.xmin
        view_size_y = self.ymax - self.ymin

        #get translated coordinates
        xp = event.GetX()*view_size_x/self.Size[0] + self.xmin
        yp = (self.Size[1] - event.GetY())*view_size_y/self.Size[1] + self.ymin

        if self.vp is None:
            self.selectionDragging = True
            self.selection = True

            self.selectionStart = (xp, yp)
            self.selectionFinish = (xp, yp)
        else:
            self.vp.xp = xp/self.vpVoxSize
            self.vp.yp = yp/self.vpVoxSize

            self.Refresh()
            self.Update()

        event.Skip()
        
    def OnLeftDClick(self, event):
        view_size_x = self.xmax - self.xmin
        view_size_y = self.ymax - self.ymin

        #get translated coordinates
        xp = event.GetX()*view_size_x/self.Size[0] + self.xmin
        yp = (self.Size[1] - event.GetY())*view_size_y/self.Size[1] + self.ymin

        for cb in self.pointSelectionCallbacks:
            cb(xp, yp)

        event.Skip()

    def OnLeftUp(self, event):
        if self.selectionDragging:
            view_size_x = self.xmax - self.xmin
            view_size_y = self.ymax - self.ymin

            #get translated coordinates
            xp = event.GetX()*view_size_x/self.Size[0] + self.xmin
            yp = (self.Size[1] - event.GetY())*view_size_y/self.Size[1] + self.ymin

            

            #self.selectionStart = (xp, yp)
            self.selectionFinish = (xp, yp)
            #x = event.GetX()
            #y = event.GetY()
            

            self.selectionDragging=False
            
            self.Refresh()
            self.Update()
            
        event.Skip()

    def OnMouseMove(self, event):
        view_size_x = self.xmax - self.xmin
        view_size_y = self.ymax - self.ymin

        #get translated coordinates
        xp = event.GetX()*view_size_x/self.Size[0] + self.xmin
        yp = (self.Size[1] - event.GetY())*view_size_y/self.Size[1] + self.ymin
        
        #print 'mm'
        if self.selectionDragging:
#            view_size_x = self.xmax - self.xmin
#            view_size_y = self.ymax - self.ymin
#
#            #get translated coordinates
#            xp = event.GetX()*view_size_x/self.Size[0] + self.xmin
#            yp = (self.Size[1] - event.GetY())*view_size_y/self.Size[1] + self.ymin



            #self.selectionStart = (xp, yp)
            self.selectionFinish = (xp, yp)
            #x = event.GetX()
            #y = event.GetY()


            #self.selectionDragging=False

            self.Refresh()
            self.Update()
            
        if event.MiddleIsDown():
            dx = xp - self.last_xp
            dy = yp - self.last_yp
            #self.last_xp = xp
            #self.last_yp = yp
            
            #print dx, dy
            
            self.moveView(-dx, -dy)
            self.last_xp = xp - dx
            self.last_yp = yp - dy
        else:
            self.last_xp = xp
            self.last_yp = yp

        event.Skip()

    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        view_size_x = self.xmax - self.xmin
        view_size_y = self.ymax - self.ymin

        #get translated coordinates
        xp = event.GetX()*view_size_x/self.Size[0] + self.xmin
        yp = (self.Size[1] - event.GetY())*view_size_y/self.Size[1] + self.ymin

        #print xp
        #print yp
        
        self.WheelZoom(rot, xp, yp)
        
        event.Skip()

    def WheelZoom(self, rot, xp, yp):
        view_size_x = self.xmax - self.xmin
        view_size_y = self.ymax - self.ymin

        if rot < 0:
            #zoom out
            self.pixelsize *=2.
            self.setView(xp - view_size_x, xp + view_size_x,yp - view_size_y, yp + view_size_y )



        if rot > 0:
            #zoom in
            self.pixelsize /=2.
            self.setView(xp - view_size_x/4, xp + view_size_x/4,yp - view_size_y/4, yp + view_size_y/4 )

    def getSnapshot(self, mode = GL_LUMINANCE):
        snap =  glReadPixelsf(0,0,self.Size[0],self.Size[1], mode)

        snap.strides = (12,12*snap.shape[0], 4)

        return snap

    def jitMCT(self,x,y,jsig, mcp):
        from matplotlib import tri
        Imc = numpy.random.normal(size=len(x)) < mcp
        if type(jsig) == numpy.ndarray:
            #print jsig.shape, Imc.shape
            jsig = jsig[Imc]
        T = tri.Triangulation(x[Imc] +  jsig*numpy.random.normal(size=Imc.sum()), y[Imc] +  jsig*numpy.random.normal(size=Imc.sum()))
        self.setTriang(T)


    def jitMCQ(self,x,y,jsig, mcp):
        from PYME.Analysis.points.QuadTree import pointQT
        Imc = numpy.rand(len(x)) < mcp
        qt = pointQT.qtRoot(-250,250, 0, 500)
        if type(jsig) == numpy.ndarray:
            jsig = jsig[Imc]
        for xi, yi in zip(x[Imc] +  jsig*numpy.random.normal(size=Imc.sum()), y[Imc] +  jsig*numpy.random.normal(size=Imc.sum())):
            qt.insert(pointQT.qtRec(xi, yi, None))
        self.setQuads(qt, 100, True)

    def genJitQim(self,n, x,y,jsig, mcp, pixelSize=None):
        self.jitMCQ(x,y,jsig, mcp)
        self.setPercentileCLim(.99)
        self.GetParent().Raise()
        self.OnDraw()

        h_ = self.getSnapshot(GL_RGB)

        for i in range(n - 1):
            self.jitMCQ(x,y,jsig, mcp)
            self.OnDraw()
            h_ += self.getSnapshot(GL_RGB)

        return h_/n

    def genJitTim(self,n, x,y,jsig, mcp, pixelSize=None):
        status = statusLog.StatusLogger('Jittering image ...')
        #turn LUT and scalebar off
        sbl = self.scaleBarLength
        self.scaleBarLength = None
        ld = self.LUTDraw
        ld = False

        self.jitMCT(x,y,jsig, mcp)
        #self.setPercentileCLim(.995)
        self.setCLim((0, 1./pixelSize**2))
        self.GetParent().Raise()
        #self.OnDraw()
        #h_ = self.getSnapshot(GL_RGB)
        h_ = self.getIm(pixelSize)
        for i in range(n - 1):
            status.setStatus('Jittering image - permutation %d of %d' %(i+2, n))
            self.jitMCT(x,y,jsig, mcp)
            #self.OnDraw()
            #h_ += self.getSnapshot(GL_RGB)
            h_ += self.getIm(pixelSize)

        self.scaleBarLength = sbl

        return h_/n

    def getIm(self, pixelSize=None):
        if pixelSize is None: #use current pixel size
            self.OnDraw()
            return self.getSnapshot(GL_RGB)
        else:
            status = statusLog.StatusLogger('Tiling image ...')
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
    f = wx.Frame(None, size=(800,800))
    c = LMGLCanvas(f)
    f.Show()
    return c


def genMapColouring(T):
    """Assigns a colour to each of the underlying points of a triangulation (T)
    such that no neighbours have the same colour. To keep complexity down, does
    not do any juggling to reduce the number of colours used to the theoretical
    4. For use with the voronoi diagram visualisation to illustrate the voronoi
    domains (use a colour map with plenty of colours & not too much intensity
    variation - e.g. hsv)."""

    cols = numpy.zeros(T.x.shape)

    for i in range(len(cols)):
        cand = 1 #candidate colour

        #find neighbouring points
        ix, iy = numpy.where(T.edges == i)
        neighb = T.edges[ix, (1-iy)]

        #if one of our neighbours already has the candidate colour, increment
        while cand in cols[neighb]:
            cand +=1

        #else assign candidate as new point colour
        cols[i] = cand

    return cols



def main():
    app = wx.PySimpleApp()
    frame = wx.Frame(None,-1,'ball_wx',wx.DefaultPosition,wx.Size(400,400))
    canvas = LMGLCanvas(frame)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__': main()

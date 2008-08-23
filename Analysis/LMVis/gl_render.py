from wx.glcanvas import GLCanvas
import wx
#from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys,math
import sys
import scipy
import Image
from scikits import delaunay

name = 'ball_glut'

class cmap_mult:
    def __init__(self, gains, zeros):
        self.gains = gains
        self.zeros = zeros

    def __call__(self, cvals):
        return scipy.minimum(scipy.vstack((self.gains[0]*cvals - self.zeros[0],self.gains[1]*cvals - self.zeros[1],self.gains[2]*cvals - self.zeros[2])), 1).astype('f').T

cm_hot = cmap_mult(8.0*scipy.ones(3)/3, [0, 3.0/8, 6.0/8])
cm_grey = cmap_mult(scipy.ones(3), [0, 0, 0])

class LMGLCanvas(GLCanvas):
    def __init__(self, parent):
	GLCanvas.__init__(self, parent,-1)
	wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
	self.init = 0
        self.nVertices = 0
        self.IScale = [1.0, 1.0, 1.0]
        self.zeroPt = [0, 1.0/3, 2.0/3]
        self.cmap = cm_hot
        self.clim = [0,1]

        self.pointSize=5 #default point size = 5nm

        self.xmin =0
        self.xmax = 20000
        self.ymin = 0
        self.ymax = 20000

        self.scaleBarLength = 200

        self.scaleBarOffset = (20.0, 20.0) #pixels from corner
        self.scaleBarDepth = 10.0 #pixels
        self.scaleBarColour = [1,1,0]

        self.numBlurSamples = 1
        self.blurSigma = 0.0

        self.LUTDraw = True
        self.mode = 'triang'

        self.drawModes = {'triang':GL_TRIANGLES, 'quads':GL_QUADS, 'edges':GL_LINES, 'points':GL_POINTS}

        self.c = scipy.array([1,1,1])
	return

    def OnPaint(self,event):
	dc = wx.PaintDC(self)
	self.SetCurrent()
	if not self.init:
	    self.InitGL()
	    self.init = 1
	self.OnDraw()
	return

    def OnSize(self, event):
        glViewport(0,0, self.Size[0], self.Size[1])

    def OnDraw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glOrtho(self.xmin,self.xmax,self.ymin,self.ymax,-1,1)

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

        #glClear(GL_ACCUM_BUFFER_BIT)

        if self.mode == 'points':
            glPointSize(self.pointSize*(float(self.Size[0])/(self.xmax - self.xmin)))
        for i in range(self.numBlurSamples):
            #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glPushMatrix ()

            glTranslatef (self.blurSigma*scipy.randn(), self.blurSigma*scipy.randn(),0.0)
            
            glDrawArrays(self.drawModes[self.mode], 0, self.nVertices)
      
            glPopMatrix ()
            #glAccum(GL_ACCUM, 1.0/self.numBlurSamples)

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


        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        self.vs = glVertexPointerf(scipy.array([[5, 5],[10000, 10000], [10000, 5]]).astype('f'))
        self.cs = glColorPointerf(scipy.ones((3,3)).astype('f'))

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

    def setTriang(self, T):
        xs = T.x[T.triangle_nodes]
        ys = T.y[T.triangle_nodes]

        a = scipy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
        b = scipy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T

        #area of triangle
        #c = 0.5*scipy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*scipy.sqrt((a*a).sum(1))

        c = scipy.maximum(((b*b).sum(1)),((a*a).sum(1)))

        #c_neighbours = c[T.triangle_neighbors].sum(1)
        #c = 1.0/(c + c_neighbours + 1)
        c = 1.0/(c + 1)

        self.c = scipy.vstack((c,c,c)).T.ravel()
        
        vs = scipy.vstack((xs.ravel(), ys.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = scipy.minimum(scipy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'triang'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setPoints(self, x, y, c = None):
        if c == None:
            self.c = scipy.ones(x.shape).ravel()
        else: 
            self.c = c
        
        vs = scipy.vstack((x.ravel(), y.ravel()))
        vs = vs.T.ravel().reshape(len(x.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = scipy.minimum(scipy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'points'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setVoronoi(self, T):
        xs_ = None
        ys_ = None
        c_ = None
        
        for i in range(len(T.x)):
            #get triangles around point
            impingentTriangs = scipy.where(T.triangle_nodes == i)[0]
            if len(impingentTriangs >= 3):

                circumcenters = T.circumcenters[impingentTriangs] #get their circumcenters

                #add current point - to deal with edge cases
                newPts = scipy.array(list(circumcenters) + [[T.x[i], T.y[i]]])

                #re-triangulate (we could try and sort the triangles somehow, but this is easier)
                T2 = delaunay.Triangulation(newPts[:,0],newPts[:,1] )


                #now do the same as for the standard triangulation
                xs = T2.x[T2.triangle_nodes]
                ys = T2.y[T2.triangle_nodes]

                a = scipy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
                b = scipy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T

                #area of triangle
                c = 0.5*scipy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*scipy.sqrt((a*a).sum(1))

                #c = scipy.maximum(((b*b).sum(1)),((a*a).sum(1)))

                #c_neighbours = c[T.triangle_neighbors].sum(1)
                #c = 1.0/(c + c_neighbours + 1)
                c = c.sum()*scipy.ones(c.shape)
                c = 1.0/(c + 1)

                #print xs.shape
                #print c.shape

                if xs_ == None:
                    xs_ = xs
                    ys_ = ys
                    c_ = c
                else:
                    xs_ = scipy.vstack((xs_, xs))
                    ys_ = scipy.vstack((ys_, ys))
                    c_ = scipy.hstack((c_, c))

        
        self.c = scipy.vstack((c_,c_,c_)).T.ravel()
        
        vs = scipy.vstack((xs_.ravel(), ys_.ravel()))
        vs = vs.T.ravel().reshape(len(xs_.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = scipy.minimum(scipy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'triang'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)


    def setTriangEdges(self, T):
        xs = T.x[T.edge_db]
        ys = T.y[T.edge_db]

        a = scipy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
        #b = scipy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T

        #area of triangle
        #c = 0.5*scipy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*scipy.sqrt((a*a).sum(1))

        c = ((a*a).sum(1))

        #c_neighbours = c[T.triangle_neighbors].sum(1)
        c = 1.0/(c + 1)

        self.c = scipy.vstack((c,c,c)).T.ravel()
        
        vs = scipy.vstack((xs.ravel(), ys.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 2)
        self.vs_ = glVertexPointerf(vs)

        #cs = scipy.minimum(scipy.vstack((self.IScale[0]*c,self.IScale[1]*c,self.IScale[2]*c)), 1).astype('f')
        #cs = cs.T.ravel().reshape(len(c), 3)
        #cs_ = glColorPointerf(cs)

        self.mode = 'edges'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setQuads(self, qt, maxDepth = 100):
        lvs = qt.getLeaves(maxDepth)
        
        xs = scipy.zeros((len(lvs), 4))
        ys = scipy.zeros((len(lvs), 4))
        c = scipy.zeros(len(lvs))

        i = 0

        maxdepth = 0
        for l in lvs:
            xs[i, :] = [l.x0, l.x1, l.x1, l.x0]
            ys[i, :] = [l.y0, l.y0, l.y1, l.y1]
            c[i] = float(l.numRecords)*2**(2*l.depth)
            i +=1
            maxdepth = max(maxdepth, l.depth)

        
        c = c/(2**(2*maxdepth))
        self.c = scipy.vstack((c,c,c,c)).T.ravel()
        
        vs = scipy.vstack((xs.ravel(), ys.ravel()))
        vs = vs.T.ravel().reshape(len(xs.ravel()), 2)
        vs_ = glVertexPointerf(vs)

        self.mode = 'quads'

        self.nVertices = vs.shape[0]
        self.setColour(self.IScale, self.zeroPt)

    def setColour(self, IScale=None, zeroPt=None):
        #self.IScale = IScale
        #self.zeroPt = zeroPt

        #cs = scipy.minimum(scipy.vstack((IScale[0]*self.c - zeroPt[0],IScale[1]*self.c - zeroPt[1],IScale[2]*self.c - zeroPt[2])), 1).astype('f')

        cs = self.cmap((self.c - self.clim[0])/(self.clim[1] - self.clim[0]))
        print cs.shape
        cs = cs[:, :3] #if we have an alpha component chuck it
        cs = cs.ravel().reshape(len(self.c), 3)
        self.cs_ = glColorPointerf(cs)

        self.Refresh()

    def setCMap(self, cmap):
        self.cmap = cmap
        self.setColour()

    def setCLim(self, clim):
        self.clim = clim
        self.setColour()

    def setPercentileCLim(self, pctile):
        '''set clim based on a certain percentile'''
        clim_upper = self.c[scipy.argsort(self.c)[len(self.c)*pctile]]
        self.setCLim([0, clim_upper])

        
    def setView(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.Refresh()

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

    def drawLUT(self):
        if self.LUTDraw == True:
            mx = self.c.max()

            view_size_x = self.xmax - self.xmin
            view_size_y = self.ymax - self.ymin

            lb_ur_x = self.xmax - self.scaleBarOffset[0]*view_size_x/self.Size[0]
            lb_ur_y = self.ymax - 3*self.scaleBarOffset[1]*view_size_y/self.Size[1]

            lb_lr_y = self.ymin + 3*self.scaleBarOffset[1]*view_size_y/self.Size[1]
            lb_width = 2*self.scaleBarDepth*view_size_x/self.Size[0]
            lb_ul_x = lb_ur_x - lb_width

            lb_len = lb_ur_y - lb_lr_y

            #sc = mx*scipy.array(self.IScale)
            #zp = scipy.array(self.zeroPt)
            #sc = mx

            glBegin(GL_QUAD_STRIP)

            for i in scipy.arange(0,1, .001):
                #glColor3fv(i*sc - zp)
                glColor3fv(self.cmap((i*mx - self.clim[0])/(self.clim[1] - self.clim[0]))[:3])
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
        view_size_x = self.xmax - self.xmin
        view_size_y = self.ymax - self.ymin

        #get translated coordinates
        xp = event.GetX()*view_size_x/self.Size[0] + self.xmin
        yp = (self.Size[1] - event.GetY())*view_size_y/self.Size[1] + self.ymin

        #print xp
        #print yp

        if rot < 0:
            #zoom out
            self.setView(xp - view_size_x, xp + view_size_x,yp - view_size_y, yp + view_size_y )
            

        if rot > 0:
            #zoom in
            self.setView(xp - view_size_x/4, xp + view_size_x/4,yp - view_size_y/4, yp + view_size_y/4 )

    def getSnapshot(self, mode = GL_LUMINANCE):
        return glReadPixelsf(0,0,self.Size[0],self.Size[1], mode)
        
        
def showGLFrame():
    f = wx.Frame(None, size=(800,800))
    c = LMGLCanvas(f)
    f.Show()
    return c
        

def main():
    app = wx.PySimpleApp()
    frame = wx.Frame(None,-1,'ball_wx',wx.DefaultPosition,wx.Size(400,400))
    canvas = myGLCanvas(frame)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__': main()

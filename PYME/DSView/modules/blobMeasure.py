#!/usr/bin/python

##################
# blobMeasure.py
#
# Copyright David Baddeley, 2012
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
import numpy as np
from numpy import linalg
from scipy import ndimage
import wx
import wx.lib.agw.aui as aui

from ._base import Plugin

#TODO - move non-GUI logic out of here!!

def iGen():
    i = 1
    while True:
        i += 1
        yield i
        
ig = iGen()

def getNewID():
    return next(ig)


class DataBlock(object):
    def __init__(self, data, X, Y, Z, voxelsize=(1,1,1)):
        self.data = data
        self.X = X
        self.Y = Y
        self.Z = Z
        
        self.voxelsize = voxelsize
        
    def __getstate__(self):
        #support for pickling
        return {'data':self.data, 'X':self.X, 'Y':self.Y, 'Z':self.Z, 'voxelsize':self.voxelsize}
        
    @property
    def bbox(self):
        if not '_bbox' in dir(self):
            x0 = self.X.min()*self.voxelsize[0]
            y0 = self.Y.min()*self.voxelsize[1]
            z0 = self.Z.min()*self.voxelsize[2]
            
            x1 = self.X.max()*self.voxelsize[0]
            y1 = self.Y.max()*self.voxelsize[1]
            z1 = self.Z.max()*self.voxelsize[2]
            
            self._bbox = (x0,y0,z0, x1, y1, z1)
        return self._bbox
    
    @property
    def sum(self):
        if not '_sum' in dir(self):
            self._sum = self.data.sum()
        return self._sum
    
    @property    
    def centroid(self):
        if not '_centroid' in dir(self):
            #crop off the bottom of the data to avoid biasing the centroid
            m = np.maximum(self.data - 1.0*self.data.mean(), 0)
            m = m/m.sum()
            xc = (m*self.X).sum()
            yc = (m*self.Y).sum()
            zc = (m*self.Z).sum()
    
            self._centroid =  (xc, yc, zc)
        
        return self._centroid
        
    @property
    def centroidNM(self):
        return self.voxelsize[0]*self.centroid[0], self.voxelsize[1]*self.centroid[1], self.voxelsize[2]*self.centroid[2]
        
    @property
    def XC(self):
        if not '_XC' in dir(self):
            self._XC = self.voxelsize[0]*(self.X - self.centroid[0])
        return self._XC
        
    @property
    def YC(self):
        if not '_YC' in dir(self):
            self._YC = self.voxelsize[1]*(self.Y - self.centroid[1])
        return self._YC
        
    @property
    def ZC(self):
        if not '_ZC' in dir(self):
            self._ZC = self.voxelsize[2]*(self.Z - self.centroid[2])
        return self._ZC
        
    @property
    def principalAxis(self):
        if not '_principalAxis' in dir(self):
            #b = (self.XC*self.data).ravel()
            b = np.vstack([(self.XC*self.data).ravel(), (self.YC*self.data).ravel(), (self.ZC*self.data).ravel()]).T
            
            #use thresholded data for determining principle axis
            m = self.data
            m = m > 0.2*m.max()
            #print((m.sum()))
            l, nl = ndimage.label(m)
            
            #take only the largest contiguous region
            m = 0*m
            for i in range(1,nl+1):
                r = l == i
                if r.sum() > m.sum():
                    m = r
            b = b[m.ravel()>0.5, :]
            #print((m.sum(), b.shape))
            if b.shape[0] < 2:
                self._principalAxis = np.NaN*np.ones(3)
            #print b.shape
            #A = (self.R*self.data*np.sign(self.XC)).ravel()[:,None]
            #print A.shape
            #pz = (np.abs(self.ZC)*self.data).sum()
            #px = 1.0
            else:
                pa = linalg.svd(b, full_matrices=False)[2][0]
            #print pa
            #pa = linalg.lstsq(A, b)[0].squeeze()
            #print pa
            
            #print pa.shape
            
            #pa = np.array([1.0, (1 - /py, 1./pz])
                self._principalAxis = pa/linalg.norm(pa)
            
        return self._principalAxis
        
    @property
    def secondaryAxis(self):
        if not '_secondaryAxis' in dir(self):
            trial = np.roll(self.principalAxis, 1) #gauranteed not to be co-linear with pa
            
            sa = np.cross(self.principalAxis, trial)
            #set the secondary and 3ary axes for now - will be updated later
            sa /= linalg.norm(sa)
            self._secondaryAxis = sa
            
            
            b = np.vstack([(self.A1*self.data).ravel(), (self.A2*self.data).ravel()]).T
            b = b[self.data.ravel()>0]
            if b.shape[0] < 2:
                self._secondaryAxis = np.NaN*np.ones(3)
            else:
                p1, p2 = linalg.svd(b, full_matrices=False)[2][0]
                #p1 = (np.abs(self.A1)*self.data).sum()
                #p2 = (np.abs(self.A2)*self.data).sum()
            
                sa = p1*sa + p2*self.tertiaryAxis
                if self.data.shape[2] == 1:
                    #2D, force sa to be in plane
                    sa[2] = 0
                sa /= linalg.norm(sa)
                self._secondaryAxis = sa
            
            del self._tertiaryAxis, self._A1, self._A2 #force these to be re-calculated
            
        return self._secondaryAxis
        
    def flipSecondaryAxis(self):
        self._secondaryAxis = -self._secondaryAxis
        try:
            del self._tertiaryAxis
        except AttributeError:
            pass
        try:
            del self._A1
        except AttributeError:
            pass
        try:
            del self._A2
        except AttributeError:
            pass
        
    @property
    def tertiaryAxis(self):
        if not '_tertiaryAxis' in dir(self):
            ta = np.cross(self.principalAxis, self.secondaryAxis)
            ta /= linalg.norm(ta)
            self._tertiaryAxis = ta
            
        return self._tertiaryAxis
    
    
    @property
    def A0(self):
        if not '_A0' in dir(self):
            self._A0 = self.principalAxis[0]*self.XC + self.principalAxis[1]*self.YC + self.principalAxis[2]*self.ZC 
        return self._A0
        
    @property
    def A1(self):
        if not '_A1' in dir(self):
            self._A1 = self.secondaryAxis[0]*self.XC + self.secondaryAxis[1]*self.YC + self.secondaryAxis[2]*self.ZC 
        return self._A1
        
    @property
    def A2(self):
        if not '_A2' in dir(self):
            self._A2 = self.tertiaryAxis[0]*self.XC + self.tertiaryAxis[1]*self.YC + self.tertiaryAxis[2]*self.ZC 
        return self._A2
        
    @property
    def SA(self):
        if self.data.ndim > 2 and self.data.shape[2] > 1:
            return self.A2
        else:
            return self.A1
            
    @property
    def R(self):
        return np.sqrt(self.XC**2 + self.YC**2 + self.ZC**2)
        
    @property
    def r(self):
        return np.sqrt(self.A0**2 + self.A1**2)
        
    @property
    def Theta(self):
        return np.angle(self.A0 +1j*self.A1)
        
    @property
    def Phi(self):
        return np.angle(self.A1 +1j*self.A2)
    
    @property
    def geom_length(self):
        vals = self.A0[self.data>0]
        if self.sum == 0:
            return 0
        else:
            return vals.max() - vals.min()
        
    @property
    def geom_width(self):
        if self.sum == 0:
            return 0
        else:
            vals = self.A1[self.data>0]
            return vals.max() - vals.min()
        
    @property
    def geom_depth(self):
        if self.sum == 0:
            return 0
        else:
            vals = self.A2[self.data>0]
            return vals.max() - vals.min()
        
    @property
    def mad_0(self):
        return (abs(self.A0)*self.data).sum()/self.sum
    
    @property
    def mad_1(self):
        return (abs(self.A1)*self.data).sum()/self.sum
    
    @property
    def mad_2(self):
        return (abs(self.A2)*self.data).sum()/self.sum

class BlobObject(object):
    def __init__(self, channels, masterChan=0, orientChan=1, orient_dir=-1):
        self.chans = channels
        self.masterChan = masterChan
        self.orientChan = orientChan
        self.orient_dir = orient_dir
        
        self.shown = True
        
        self.nsteps = 51
        if self.chans[0].voxelsize[0] > 20:
            self.nsteps = 10
            
        mc = self.chans[masterChan]
        oc = self.chans[orientChan]
        
        if (mc.SA*oc.data).sum()*orient_dir < 0:
            #flip the direction of short axis on mc
            mc.flipSecondaryAxis()
            
    def __getstate__(self):
        return {'chans': self.chans, 'masterChan':self.masterChan, 
                'orientChan':self.orientChan, 'orient_dir':self.orient_dir, 
                'shown':self.shown, 'nsteps':self.nsteps}
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        
        mc = self.chans[self.masterChan]
        oc = self.chans[self.orientChan]
        
        if (mc.SA*oc.data).sum()*self.orient_dir < 0:
            #flip the direction of short axis on mc
            mc.flipSecondaryAxis()
        
        
    def longAxisDistN(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(-1,1, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        
        for c in self.chans:
            bn, bm, bs = binAvg(mc.A0/mc.geom_length, c.data, xvs)
            bb = bm*bn
            bms.append(bb/bb.sum())
            
        return xvs, bms
    longAxisDistN.xlabel = 'Longitudinal position [a.u.]'
    
    def longAxisOrthDist(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(-1,1, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        
        for c in self.chans:
            bn, bm, bs = binAvg(mc.A0/mc.geom_length, c.data, xvs)
            bss = bm*bn
            bn, bm, bs = binAvg(mc.A0/mc.geom_length, c.data*mc.SA, xvs)
            bb = bm*bn
            bms.append(bb/bss)
            
        return xvs, bms
    longAxisOrthDist.xlabel = 'Longitudinal position [a.u.]'
        
    def shortAxisDist(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(-500,500, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        
        for c in self.chans:
            bn, bm, bs = binAvg(mc.SA, c.data, xvs)
            bb = bm*bn
            bms.append(bb/bb.sum())
            
        return xvs, bms
    shortAxisDist.xlabel = 'Position on shortest axis [nm]'
        
    def radialDistN(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(0, 2, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        r = mc.r/(mc.geom_length/2)
        
        for c in self.chans:
            bn, bm, bs = binAvg(r, c.data, xvs)
            bb = bm*bn
            bms.append(bb/bb.sum())
            
        return xvs, bms
    radialDistN.xlabel = 'Radial position [a.u.]'
    
    def RadialDistN(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(0, 1, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        R = mc.R/(mc.geom_length/2)
        
        for c in self.chans:
            bn, bm, bs = binAvg(R, c.data, xvs)
            bb = bm*bn
            bms.append(bb/bb.sum())
            
        return xvs, bms
    RadialDistN.xlabel = 'Radial position [a.u.]'
        
    def angularDist(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(-np.pi, np.pi, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        
        for c in self.chans:
            bn, bm, bs = binAvg(mc.Theta, c.data, xvs)
            bb = bm*bn
            bms.append(bb/bb.sum())
            
        return xvs, bms
    angularDist.xlabel=None
        
    def drawOverlay(self, view, dc):
        #import wx
        if self.shown:
            try:
                mc = self.chans[self.masterChan]
                
                #x, y, z = mc.centroid
                l = mc.geom_length/(2*mc.voxelsize[0])
                x0, y0, z0 = mc.centroid - l*mc.principalAxis
                x1, y1, z1 = mc.centroid + l*mc.principalAxis
                
                x0_, y0_ = view.pixel_to_screen_coordinates(x0, y0)
                x1_, y1_ = view.pixel_to_screen_coordinates(x1, y1)
                
                dc.DrawLine(x0_, y0_, x1_, y1_)
                
                l = mc.geom_width/(2*mc.voxelsize[0])
                x0, y0, z0 = mc.centroid
                x1, y1, z1 = mc.centroid + l*mc.secondaryAxis
                
                x0_, y0_ = view.pixel_to_screen_coordinates(x0, y0)
                x1_, y1_ = view.pixel_to_screen_coordinates(x1, y1)
                
                dc.DrawLine(x0_, y0_, x1_, y1_)
            except ValueError:
                pass
            
    def getImage(self, gains=None):
        try:
            import Image
        except ImportError:
            from PIL import Image
            
        from io import BytesIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"
        
        dispChans = []
        ovsc = 2
        if self.chans[0].voxelsize[0] > 50:
            ovsc = 1
            
        for i, c in enumerate(self.chans):
            d = c.data
            if d.ndim > 2 and d.shape[2] > 1:
                d = d.max(2)
            d = np.atleast_3d(d.squeeze().T)
            d -= d.min()
            if gains is None:
                d = np.minimum(ovsc*255*d/d.max(), 255).astype('uint8')
            else:
                d = np.minimum(gains[i]*255*d, 255).astype('uint8')
            
            dispChans.append(d)
            
        while len(dispChans) < 3:
            dispChans.append(0*d)
            
        im = np.concatenate(dispChans, 2)
        
        #scalebar (200 nm)
        nPixels = 200/(c.voxelsize[0])
        w = max(int(15/c.voxelsize[0]), 1)
        if c.voxelsize[0] < 50:
            im[-(2*w):-w, -(nPixels+w):-w, :] = 255
    

        xsize = im.shape[0]
        ysize = im.shape[1]

        zoom = 200./max(xsize, ysize)
        
        out = BytesIO()
        Image.fromarray(im).resize((int(zoom*ysize), int(zoom*xsize))).save(out, 'PNG')
        s = out.getvalue()
        out.close()
        return s
        
    def getGraph(self, graphName):
        #import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from io import BytesIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"
        
        isPolar =  'ang' in graphName
        if isPolar:
            fig = Figure(figsize=(3,3), facecolor='none')
        else:
            fig = Figure(figsize=(4,3), facecolor='none')
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([.1, .15, .85, .8], polar=isPolar)
        
        xv, yv = getattr(self, graphName)()
        
        cols = ['r','g', 'b']
        for i in range(len(yv)):
            ax.plot(xv[:-1], yv[i], c = cols[i], lw=2)
        ax.set_xlabel(getattr(self, graphName).xlabel)
        
        out = BytesIO()
        canvas.print_png(out, dpi=100, facecolor='w')
        s = out.getvalue()
        out.close()
        return s
        
    def getSchematic(self):
        #import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        from io import BytesIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"

        fig = Figure(figsize=(2,2), facecolor='none')
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0.01, 0.01, .98, .98])
        
        mc = self.chans[self.masterChan]
        
        bx0, by0, bz0, bx1, by1, bz1 = mc.bbox
        rect = Rectangle((bx0, by0), (bx1 - bx0), (by1 - by0), facecolor='w')
        ax.add_patch(rect)
            
        #x, y, z = mc.centroid
        l = mc.geom_length/2
        x0, y0, z0 = mc.centroidNM - l*mc.principalAxis
        x1, y1, z1 = mc.centroidNM + l*mc.principalAxis
        
        ax.plot([x0, x1], [y0, y1], 'k', lw=2)
        
        l = mc.geom_width/2
        x0, y0, z0 = mc.centroidNM
        x1, y1, z1 = mc.centroidNM + l*mc.secondaryAxis
        ax.plot([x0, x1], [y0, y1], 'm', lw=2)
        
        l = mc.geom_depth/2
        x0, y0, z0 = mc.centroidNM
        x1, y1, z1 = mc.centroidNM + l*mc.tertiaryAxis
        ax.plot([x0, x1], [y0, y1], 'c--', lw=2)
        
        cols = ['r','g', 'b']
        for i, c in enumerate(self.chans):
            l = c.geom_length/2
            x0, y0, z0 = c.centroidNM - l*c.principalAxis
            x1, y1, z1 = c.centroidNM + l*c.principalAxis
            
            ax.plot([x0, x1], [y0, y1],c = cols[i] , lw=2)
            
            x0, y0, z0 = c.centroidNM
            ax.plot(x0, y0, 'x', c = cols[i], lw=2)
            
        
        ax.axis('scaled')
        ax.set_xlim(bx0, bx1)
        ax.set_ylim(by1, by0)
        
        ax.set_axis_off()
        
        out = BytesIO()
        canvas.print_png(out, dpi=100, facecolor='none')
        s = out.getvalue()
        out.close()
        return s
        
    def getSchematic3D(self):
        #import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.figure import Figure
        from io import BytesIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"

        fig = Figure(figsize=(2,2), facecolor='none')
        canvas = FigureCanvas(fig)
        #ax = fig.add_axes([.1, .15, .85, .8])
        ax = Axes3D(fig)
        
        mc = self.chans[self.masterChan]
            
        #x, y, z = mc.centroid
        l = mc.geom_length/(2*mc.voxelsize[0])
        x0, y0, z0 = mc.centroid - l*mc.principalAxis
        x1, y1, z1 = mc.centroid + l*mc.principalAxis
        
        ax.plot([x0, x1], [y0, y1], [z0, z1], 'k', lw=2)
        
        l = mc.geom_width/(2*mc.voxelsize[0])
        x0, y0, z0 = mc.centroid
        x1, y1, z1 = mc.centroid + l*mc.secondaryAxis
        ax.plot([x0, x1], [y0, y1], [z0, z1], 'k', lw=2)
        
        l = mc.geom_depth/(2*mc.voxelsize[0])
        x0, y0, z0 = mc.centroid
        x1, y1, z1 = mc.centroid + l*mc.tertiaryAxis
        ax.plot([x0, x1], [y0, y1], [z0, z1], 'k:', lw=2)
        
        cols = ['r','g', 'b']
        for i, c in enumerate(self.chans):
            l = c.geom_length/(2*mc.voxelsize[0])
            x0, y0, z0 = c.centroid - l*c.principalAxis
            x1, y1, z1 = c.centroid + l*c.principalAxis
            
            ax.plot([x0, x1], [y0, y1], [z0, z1],c = cols[i] , lw=2)
            
            x0, y0, z0 = c.centroid
            ax.plot([x0], [y0], [z0], 'x', c = cols[i], lw=2)
            
        #ax.axis('equal')
        #ax.set_axis_off()
        
        out = BytesIO()
        canvas.print_png(out, dpi=100, facecolor='none')
        s = out.getvalue()
        out.close()
        return s
        
    def _shade(self, faces, baseColor= [1,0,0, 0], lightVector=[1,1,1]):
        baseColor = np.array(baseColor)
        lightVector = np.array(lightVector)
        v1 = faces[:,1, :] - faces[:,0, :]
        v2 = faces[:,2, :] - faces[:,1, :]
        
        vnorm = np.cross(v1, v2)
        
        vnorm /= np.linalg.norm(vnorm, axis=-1)[:,None]
        
        return baseColor[None, :]*np.clip(np.abs(np.inner(vnorm, lightVector)), .05, 1)[:,None]
        
        
        
    def get3DIsosurf(self, isovalues=None):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.figure import Figure
        from skimage import measure
        from io import BytesIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"
        
        if '_3diso' in dir(self):
            return self._3diso

        fig = Figure(figsize=(8,4), facecolor='none')
        canvas = FigureCanvas(fig)
        #ax = fig.add_axes([.1, .15, .85, .8])
        ax = Axes3D(fig, [0, 0, .5,1], azim = -60, elev=30, axisbg='none')
        #ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        mc = self.chans[self.masterChan]

        meshes = []        
        cols = [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]    
        
        for i, c in enumerate(self.chans):
            d = c.data#.swapaxes(0,1)
            verts, faces = measure.marching_cubes(d, isovalues[i])
            vertsc = verts + np.array(c.bbox[:3])/np.array(mc.voxelsize)
            tris =  vertsc[faces]
            tricols = self._shade(tris, cols[i])
            #mesh = Poly3DCollection(vertsc[faces], facecolor=['r', 'g', 'b'][i], edgecolor=['r', 'g', 'b'][i], alpha=.5)
            #mesh = Poly3DCollection(tris, facecolor=self._shade(tris, cols[i]), edgecolor=['r', 'g', 'b'][i], alpha=.5)
            meshes.append((tris, tricols))
        
        
            
        #x, y, z = mc.centroid
            
        def _drawaxes(ax):
            l = mc.geom_length/(2*mc.voxelsize[0])
            x0, y0, z0 = mc.centroid - l*mc.principalAxis
            x1, y1, z1 = mc.centroid + l*mc.principalAxis
            
            ax.plot([x0, x1], [y0, y1], [z0, z1], 'k', lw=3)
            
            l = mc.geom_width/(2*mc.voxelsize[0])
            x0, y0, z0 = mc.centroid
            x1, y1, z1 = mc.centroid + l*mc.secondaryAxis
            ax.plot([x0, x1], [y0, y1], [z0, z1], 'm', lw=3)
            
            l = mc.geom_depth/(2*mc.voxelsize[0])
            x0, y0, z0 = mc.centroid
            x1, y1, z1 = mc.centroid + l*mc.tertiaryAxis
            ax.plot([x0, x1], [y0, y1], [z0, z1], 'c--', lw=3)
            
            cols = ['r','g', 'b']
            for i, c in enumerate(self.chans):
                l = c.geom_length/(2*mc.voxelsize[0])
                x0, y0, z0 = c.centroid - l*c.principalAxis
                x1, y1, z1 = c.centroid + l*c.principalAxis
                
                ax.plot([x0, x1], [y0, y1], [z0, z1],c = cols[i] , lw=3)
                
                x0, y0, z0 = c.centroid
                ax.plot([x0], [y0], [z0], 'x', c = cols[i], lw=3)
                
        _drawaxes(ax)
            

            
#        cols = [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]
#        for i, c in enumerate(self.chans):
#            d = c.data
#            verts, faces = measure.marching_cubes(d, isovalues[i])
#            vertsc = verts + np.array(c.bbox[:3])/np.array(mc.voxelsize)
#            tris =  vertsc[faces]
#            #mesh = Poly3DCollection(vertsc[faces], facecolor=['r', 'g', 'b'][i], edgecolor=['r', 'g', 'b'][i], alpha=.5)
#            #mesh = Poly3DCollection(tris, facecolor=self._shade(tris, cols[i]), edgecolor=['r', 'g', 'b'][i], alpha=.5)
#            mesh = Poly3DCollection(tris, facecolor=self._shade(tris, cols[i]), edgecolor='none', alpha=.5)
#            ax.add_collection3d(mesh)
        
        for tris, fcols in meshes:
            mesh = Poly3DCollection(tris, facecolor=fcols, edgecolor='none', alpha=.5)
            ax.add_collection3d(mesh)
            

        x0, y0, z0 = np.array(mc.bbox[:3])/np.array(mc.voxelsize)
        x1, y1, z1 = np.array(mc.bbox[3:])/np.array(mc.voxelsize)

        xc = (x0 + x1)/2
        al = x1 - x0
        yc = (y0 + y1)/2
        al = max(al, y1 - y0)        
        zc = (z0 + z1)/2
        al = max(al, z1 - z0)
        
        al2 = al/2.0
        
        ax.set_xlim(xc - al2, xc + al2)
        ax.set_ylim(yc - al2, yc + al2)
        ax.set_zlim(zc - al2, zc + al2)        
        #print c.centroid, vertsc
        #ax.axis('equal')
        #ax.set_axis_off()
        
        #ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax = Axes3D(fig, [.5, 0, .5,1], azim = 60, elev=30, axisbg='none')
#        for m in meshes:
#            ax.add_collection3d(m)
        _drawaxes(ax)
        
        for tris, fcols in meshes:
            mesh = Poly3DCollection(tris, facecolor=fcols, edgecolor='none', alpha=.5)
            ax.add_collection3d(mesh)
            
        ax.set_xlim(xc - al2, xc + al2)
        ax.set_ylim(yc - al2, yc + al2)
        ax.set_zlim(zc - al2, zc + al2)
        
        #ax.view_init(-45, 45)
        
        out = BytesIO()
        canvas.print_png(out, dpi=100, facecolor='none')
        s = out.getvalue()
        out.close()
        
        self._3diso = s
        return s

class Measurements(wx.Panel, Plugin):
    def __init__(self, dsviewer):
        wx.Panel.__init__(self, dsviewer)
        Plugin.__init__(self, dsviewer)
        
        dsviewer.view.add_overlay(self.DrawOverlays, 'Blob measurements')
        
        vsizer=wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Reference Channel:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.chMaster=wx.Choice(self, -1, choices=self.image.names)
        self.chMaster.SetSelection(0)
        hsizer.Add(self.chMaster, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Orient by:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.chOrient=wx.Choice(self, -1, choices=self.image.names)
        self.chOrient.SetSelection(1)
        hsizer.Add(self.chOrient, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.chDirection=wx.Choice(self, -1, choices=['-ve from reference', '+ve from reference'], size=(30, -1))
        self.chDirection.SetSelection(0)
        hsizer.Add(self.chDirection, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        
        sbsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Minimum Amplitudes:'), wx.VERTICAL)
        nChans = len(self.image.names)        
        
        gsizer = wx.FlexGridSizer(nChans, 2, 2, 5)
        
        self.aTexts = []
        for chan in range(nChans):
            gsizer.Add(wx.StaticText(self, -1, self.image.names[chan]), 1, wx.ALIGN_CENTER_VERTICAL)
            at = wx.TextCtrl(self, -1, '0')
            self.aTexts.append(at)
            gsizer.Add(at, 2, wx.ALIGN_CENTER_VERTICAL)
            
        sbsizer.Add(gsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        vsizer.Add(sbsizer, 0, wx.EXPAND, 2)
        
        bCalculate = wx.Button(self, -1, 'Measure')
        bCalculate.Bind(wx.EVT_BUTTON, self.OnCalculate)
        vsizer.Add(bCalculate, 0, wx.EXPAND|wx.ALL, 2)
        
        bHide = wx.Button(self, -1, 'Hide Object')
        bHide.Bind(wx.EVT_BUTTON, self.OnHideObject)
        vsizer.Add(bHide, 0, wx.EXPAND|wx.ALL, 2)
        
        bView = wx.Button(self, -1, 'View Results')
        bView.Bind(wx.EVT_BUTTON, self.OnView)
        vsizer.Add(bView, 0, wx.EXPAND|wx.ALL, 2)
        
        bXls = wx.Button(self, -1, 'Export to xls')
        bXls.Bind(wx.EVT_BUTTON, self.OnExportXls)
        vsizer.Add(bXls, 0, wx.EXPAND|wx.ALL, 2)
        
        self.SetSizerAndFit(vsizer)
        
        self.objects = []
        self.ID = getNewID()
        
        self.StartServing()

    def StartServing(self):
        from PYME.DSView import htmlServe #ensure that our local cherrypy server is running
        import cherrypy
        
        htmlServe.StartServing()
        cherrypy.tree.mount(self, '/measure/%d' % self.ID)
                    
        
#    def StartServing(self):
#        try: 
#            import threading
#            self.serveThread = threading.Thread(target=self._serve)
#            self.serveThread.start()
#        except ImportError:
#            pass
#            
#    def _serve(self):
#        import cherrypy
#        cherrypy.quickstart(self, '/measure/%d' % self.ID)
        
    def OnCalculate(self, event):
        self.RetrieveObjects(self.chMaster.GetSelection(), self.chOrient.GetSelection(), 2*self.chDirection.GetSelection() - 1)
        
        cts = [float(at.GetValue()) for at in self.aTexts]
        
        for obj in self.objects:
            for chan, ct in zip(obj.chans, cts):
                if chan.sum < ct:
                    obj.shown=False
                    
        self.dsviewer.do.OnChange()
        
    def OnHideObject(self, event):
        do = self.dsviewer.do
        ind = self.image.labels[do.xp, do.yp, do.zp]
        
        if ind:
            self.objects[ind - 1].shown = False
            do.OnChange()
    
    def OnView(self, event):
        import webbrowser
        from PYME.DSView import htmlServe
        
        webbrowser.open('%smeasure/%d' % (htmlServe.getURL(), self.ID))
        
    def OnSaveObjects(self, event):
        #import cPickle
        import zipfile
        
        zf = zipfile()
        
    def OnExportXls(self, event):
        import os
        filename = wx.FileSelector('Save to xls', default_filename=os.path.splitext(self.image.filename)[0] + '_blobMeasure.xls', wildcard='*.xls')
        self.toXls(filename)
            
    
    def GetRegion(self, index, objects = None):
        if not objects:
            objects = ndimage.find_objects(self.image.labels)
            
        o = objects[index]
        
        mask = self.image.labels[o] == (index+1)
        
        slx, sly, slz = o
        
        X, Y, Z = np.ogrid[slx, sly, slz]
        vs = self.image.voxelsize
        
        #return [DataBlock(np.maximum(self.image.data[slx, sly, slz, j] - self.image.data[slx, sly, slz, j].min(), 0)*mask , X, Y, Z, vs) for j in range(self.image.data.shape[3])]
        return [DataBlock(np.maximum(self.image.data[slx, sly, slz, j], 0)*mask , X, Y, Z, vs) for j in range(self.image.data.shape[3])]
        
    def RetrieveObjects(self, masterChan=0, orientChan=1, orient_dir=-1):
        objs = ndimage.find_objects(self.image.labels)
        
        self.objects = [BlobObject(self.GetRegion(i, objs), masterChan, orientChan, orient_dir) for i in range(self.image.labels.max())]
        
    def DrawOverlays(self, view, dc):
        import wx
        col = wx.TheColourDatabase.FindColour('CYAN')
        dc.SetPen(wx.Pen(col,2))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        
        for obj in self.objects:
            obj.drawOverlay(view, dc)
            
        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
        
    def index(self, templateName='measureView3.html'):
        from jinja2 import Environment, PackageLoader
        env = Environment(loader=PackageLoader('PYME.DSView.modules', 'templates'))
        
        #if self.dsviewer.image.data.shape[2] > 1:
            #3D image
       #     templateName=
        
        template = env.get_template(templateName)
        
        return template.render(objects=self.objects)
    index.exposed = True
    
    def _xlData(self, book, graphName):
        gs = book.add_sheet(graphName) 
        
        #r = 3
        
        objects = [obj for obj in self.objects if obj.shown]
        
        #write the data first
        c0 = 2
        
        for obj in objects:
            xv, yv = getattr(obj, graphName)()
            
            
            #for x_v in xv:
            #    gs.write(r, c, x_v)
            #    c += 1
            
            #add a gap
            #c += 1
            
            c = c0
            
            
            for yvc in yv:
                gs.write(2, c, 'Obj %d' % self.objects.index(obj), self._boldStyle)
                r = 3
                for y_v in yvc:
                    gs.write(r, c, y_v)
                    r += 1
                c += (len(objects) + 1)
            
            c0 += 1
            #r+=1
            
        c = 0
        r = 3
        for x_v in xv:
            gs.write(r, c, x_v)
            r += 1
            #    c += 1
            
                    
        #now go back and write the header
        #gs.merge(0, 0, 0, len(xv))
        gs.write(0, 0, 'Bin edges', self._boldStyle)
        for i, n in enumerate(self.image.names):
            gs.merge( 0, 0,  2+ i*(len(objects)+1),  1 + (i+1)*(len(objects)+1))
            gs.write(0, 2+ i*(len(objects)+1) , n, self._boldStyle)    
        
    def toXls(self, filename):
        import xlwt
        
        book = xlwt.Workbook()
        
        gs = book.add_sheet('General')
        
        #set up bold style
        self._boldStyle = xlwt.easyxf('font: bold 1')
        
        paramsToWrite=['Centroid x','Centroid y', 'Length', 'Width', 'Sum']
        nParams = len(paramsToWrite)
        
        #Header
        gs.write(1,0,'Image #', self._boldStyle)
        for i, n in enumerate(self.image.names):
            gs.merge( 0, 0, 1+ i*nParams, (i+1)*nParams,)
            gs.write(0, 1+ i*nParams, n, self._boldStyle)
            for j, p in enumerate(paramsToWrite):
                gs.write(1, 1 + i*nParams + j, p, self._boldStyle)
        
        r = 2
        for obj in self.objects:
            if obj.shown:
                gs.write(r, 0, self.objects.index(obj))
                for i, ch in enumerate(obj.chans):
                    gs.write(r, 1 + i*nParams + 0, ch.centroid[0])
                    gs.write(r, 1 + i*nParams + 1, ch.centroid[1])
                    gs.write(r, 1 + i*nParams + 2, ch.mad_0*2.35/0.8)
                    gs.write(r, 1 + i*nParams + 3, ch.mad_1*2.35/0.8)
                    gs.write(r, 1 + i*nParams + 4, float(ch.sum))
                
                r += 1
         
        r+=2
        gs.write(r, 0, 'Reference Channel:', self._boldStyle)
        gs.write(r, 1, self.image.names[self.objects[0].masterChan])
                
        #now do the graph data
        graphs = ['shortAxisDist', 'longAxisDistN', 'angularDist', 'radialDistN', 'longAxisOrthDist']
        
        for gn in graphs:
            self._xlData(book, gn)
                
        book.save(filename)
        
    
    def images(self, num):
        return self.objects[int(num)].getImage(self.dsviewer.do.Gains)
    images.exposed = True
    
    def graphs(self, num, graphName):
        return self.objects[int(num)].getGraph(graphName)
    graphs.exposed = True
    
    def schemes(self, num):
        return self.objects[int(num)].getSchematic()
    schemes.exposed = True
    
    def schemes3D(self, num):
        return self.objects[int(num)].getSchematic3D()
    schemes3D.exposed = True
    
    def isosurface3D(self, num):
        return self.objects[int(num)].get3DIsosurf(0.5/np.array(self.dsviewer.do.Gains))
    isosurface3D.exposed = True
    
    def hide(self, num):
        self.objects[int(num)].shown = False
        self.dsviewer.do.OnChange()
        return ''
    hide.exposed = True
    
        
        
def Plug(dsviewer):
    measure = Measurements(dsviewer)
    
    measure.SetSize(measure.GetBestSize())
    pinfo2 = aui.AuiPaneInfo().Name("measurePanel").Left().Caption('Blob Characterisation').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
    dsviewer._mgr.AddPane(measure, pinfo2)
    dsviewer._mgr.Update()
    
    return measure

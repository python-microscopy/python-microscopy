#!/usr/bin/python

##################
# blobMeasure.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy as np
from numpy import linalg
from scipy import ndimage
import wx
import wx.lib.agw.aui as aui


def iGen():
    i = 1
    while True:
        i += 1
        yield i
        
ig = iGen()

def getNewID():
    return ig.next()


class DataBlock(object):
    def __init__(self, data, X, Y, Z, voxelsize=(1,1,1)):
        self.data = data
        self.X = X
        self.Y = Y
        self.Z = Z
        
        self.voxelsize = voxelsize
    
    @property
    def sum(self):
        if not '_sum' in dir(self):
            self._sum = self.data.sum()
        return self._sum
    
    @property    
    def centroid(self):
        if not '_centroid' in dir(self):
            xc = (self.data*self.X).sum()/self.sum
            yc = (self.data*self.Y).sum()/self.sum
            zc = (self.data*self.Z).sum()/self.sum
    
            self._centroid =  (xc, yc, zc)
        
        return self._centroid
        
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
            px = (np.abs(self.XC)*self.data).sum()
            py = (np.abs(self.YC)*self.data).sum()
            pz = (np.abs(self.ZC)*self.data).sum()
            
            pa = np.array([px, py, pz])
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
            
            p1 = (np.abs(self.A1)*self.data).sum()
            p2 = (np.abs(self.A2)*self.data).sum()
            
            sa = p1*sa + p2*self.tertiaryAxis
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
    def R(self):
        return np.sqrt(self.A0**2 + self.A1**2 + self.A2**2)
        
    @property
    def Theta(self):
        return np.angle(self.A0 +1j*self.A1)
        
    @property
    def Phi(self):
        return np.angle(self.A1 +1j*self.A2)
    
    @property
    def geom_length(self):
        vals = self.A0[self.data>0]
        return vals.max() - vals.min()
        
    @property
    def geom_width(self):
        vals = self.A1[self.data>0]
        return vals.max() - vals.min()
        
    @property
    def geom_depth(self):
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
        
        self.shown = True
        
        self.nsteps = 100
        if self.chans[0].voxelsize[0] > 20:
            self.nsteps = 10
            
        mc = self.chans[masterChan]
        oc = self.chans[orientChan]
        
        if (mc.A1*oc.data).sum()*orient_dir < 0:
            #flip the direction of short axis on mc
            mc.flipSecondaryAxis()
        
        
    def longAxisDistN(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(-1,1, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        
        for c in self.chans:
            bn, bm, bs = binAvg(mc.A0/mc.geom_length, c.data, xvs)
            bms.append(bm/bm.sum())
            
        return xvs, bms
    longAxisDistN.xlabel = 'Longitudinal position [a.u.]'
        
    def shortAxisDist(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(-500,500, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        
        for c in self.chans:
            bn, bm, bs = binAvg(mc.A1, c.data, xvs)
            bms.append(bm/bm.sum())
            
        return xvs, bms
    shortAxisDist.xlabel = 'Transverse position [nm]'
        
    def radialDistN(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(0, 1, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        R = mc.R/(mc.geom_length/2)
        
        for c in self.chans:
            bn, bm, bs = binAvg(R, c.data, xvs)
            bms.append(bm/bm.sum())
            
        return xvs, bms
    radialDistN.xlabel = 'Radial position [a.u.]'
        
    def angularDist(self):
        from PYME.Analysis.binAvg import binAvg
        
        xvs = np.linspace(-np.pi, np.pi, self.nsteps)
        bms = []
        
        mc = self.chans[self.masterChan]
        
        for c in self.chans:
            bn, bm, bs = binAvg(mc.Theta, c.data, xvs)
            bms.append(bm/bm.sum())
            
        return xvs, bms
    angularDist.xlabel=None
        
    def drawOverlay(self, view, dc):
        #import wx
        if self.shown:
            mc = self.chans[self.masterChan]
            
            #x, y, z = mc.centroid
            l = mc.geom_length/(2*mc.voxelsize[0])
            x0, y0, z0 = mc.centroid - l*mc.principalAxis
            x1, y1, z1 = mc.centroid + l*mc.principalAxis
            
            x0_, y0_ = view._PixelToScreenCoordinates(x0, y0)
            x1_, y1_ = view._PixelToScreenCoordinates(x1, y1)
            
            dc.DrawLine(x0_, y0_, x1_, y1_)
            
            l = mc.geom_width/(2*mc.voxelsize[0])
            x0, y0, z0 = mc.centroid
            x1, y1, z1 = mc.centroid + l*mc.secondaryAxis
            
            x0_, y0_ = view._PixelToScreenCoordinates(x0, y0)
            x1_, y1_ = view._PixelToScreenCoordinates(x1, y1)
            
            dc.DrawLine(x0_, y0_, x1_, y1_)
            
    def getImage(self):
        import Image
        import StringIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"
        
        dispChans = []
        ovsc = 2
        if self.chans[0].voxelsize[0] > 50:
            ovsc = 1
            
        for c in self.chans:
            d = np.atleast_3d(c.data.squeeze().T)
            d -= d.min()
            d = np.minimum(ovsc*255*d/d.max(), 255).astype('uint8')
            dispChans.append(d)
            
        while len(dispChans) < 3:
            dispChans.append(0*d)
            
        im = np.concatenate(dispChans, 2)
        
        #scalebar (200 nm)
        nPixels = 200/(c.voxelsize[0])
        if c.voxelsize[0] < 50:
            im[-8:-5, -(nPixels+5):-5, :] = 255
    

        xsize = im.shape[0]
        ysize = im.shape[1]

        zoom = 200./max(xsize, ysize)
        
        out = StringIO.StringIO()
        Image.fromarray(im).resize((int(zoom*ysize), int(zoom*xsize))).save(out, 'PNG')
        s = out.getvalue()
        out.close()
        return s
        
    def getGraph(self, graphName):
        #import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        import StringIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"
        
        isPolar =  'ang' in graphName
        if isPolar:
            fig = Figure(figsize=(3,3))
        else:
            fig = Figure(figsize=(4,3))
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([.1, .15, .85, .8], polar=isPolar)
        
        xv, yv = getattr(self, graphName)()
        
        cols = ['r','g', 'b']
        for i in range(len(yv)):
            ax.plot(xv[:-1], yv[i], c = cols[i], lw=2)
        ax.set_xlabel(getattr(self, graphName).xlabel)
        
        out = StringIO.StringIO()
        canvas.print_png(out, dpi=100)
        s = out.getvalue()
        out.close()
        return s
        
    def getSchematic(self):
        #import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        import StringIO
        import cherrypy
        cherrypy.response.headers["Content-Type"]="image/png"

        fig = Figure(figsize=(2,2))
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([.1, .15, .85, .8])
        
        mc = self.chans[self.masterChan]
            
        #x, y, z = mc.centroid
        l = mc.geom_length/(2*mc.voxelsize[0])
        x0, y0, z0 = mc.centroid - l*mc.principalAxis
        x1, y1, z1 = mc.centroid + l*mc.principalAxis
        
        ax.plot([x0, x1], [-y0, -y1], 'k', lw=2)
        
        l = mc.geom_width/(2*mc.voxelsize[0])
        x0, y0, z0 = mc.centroid
        x1, y1, z1 = mc.centroid + l*mc.secondaryAxis
        ax.plot([x0, x1], [-y0, -y1], 'k', lw=2)
        
        cols = ['r','g', 'b']
        for i, c in enumerate(self.chans):
            l = c.geom_length/(2*mc.voxelsize[0])
            x0, y0, z0 = c.centroid - l*c.principalAxis
            x1, y1, z1 = c.centroid + l*c.principalAxis
            
            ax.plot([x0, x1], [-y0, -y1],c = cols[i] , lw=2)
            
            x0, y0, z0 = c.centroid
            ax.plot(x0, -y0, 'x', c = cols[i], lw=2)
            
        ax.axis('scaled')
        ax.set_axis_off()
        
        out = StringIO.StringIO()
        canvas.print_png(out, dpi=100)
        s = out.getvalue()
        out.close()
        return s

class Measurements(wx.Panel):
    def __init__(self, dsviewer):
        wx.Panel.__init__(self, dsviewer)
        self.dsviewer = dsviewer
        self.image = dsviewer.image
        
        dsviewer.view.overlays.append(self.DrawOverlays)
        
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
        
        bCalculate = wx.Button(self, -1, 'Measure')
        bCalculate.Bind(wx.EVT_BUTTON, self.OnCalculate)
        vsizer.Add(bCalculate, 0, wx.EXPAND|wx.ALL, 2)
        
        bHide = wx.Button(self, -1, 'Hide Object')
        bHide.Bind(wx.EVT_BUTTON, self.OnHideObject)
        vsizer.Add(bHide, 0, wx.EXPAND|wx.ALL, 2)
        
        bView = wx.Button(self, -1, 'View Results')
        bView.Bind(wx.EVT_BUTTON, self.OnView)
        vsizer.Add(bView, 0, wx.EXPAND|wx.ALL, 2)
        
        self.SetSizerAndFit(vsizer)
        
        self.objects = []
        self.ID = getNewID()
        
        self.StartServing()
        
        
    def StartServing(self):
        try: 
            import threading
            self.serveThread = threading.Thread(target=self._serve)
            self.serveThread.start()
        except ImportError:
            pass
            
    def _serve(self):
        import cherrypy
        cherrypy.quickstart(self, '/measure/%d' % self.ID)
        
    def OnCalculate(self, event):
        self.RetrieveObjects(self.chMaster.GetSelection(), self.chOrient.GetSelection(), 2*self.chDirection.GetSelection() - 1)
        self.dsviewer.do.OnChange()
        
    def OnHideObject(self, event):
        do = self.dsviewer.do
        ind = self.image.labels[do.xp, do.yp, do.zp]
        
        if ind:
            self.objects[ind - 1].shown = False
            do.OnChange()
    
    def OnView(self, event):
        import webbrowser
        webbrowser.open('http://localhost:8080/measure/%d' % self.ID)
            
    
    def GetRegion(self, index, objects = None):
        if not objects:
            objects = ndimage.find_objects(self.image.labels)
            
        o = objects[index]
        
        mask = self.image.labels[o] == (index+1)
        
        slx, sly, slz = o
        
        X, Y, Z = np.ogrid[slx, sly, slz]
        vs = (1e3*self.image.mdh['voxelsize.x'], 1e3*self.image.mdh['voxelsize.y'],1e3*self.image.mdh['voxelsize.z'])
        
        return [DataBlock(np.maximum(self.image.data[slx, sly, slz, j] - self.image.labelThresholds[j], 0)*mask , X, Y, Z, vs) for j in range(self.image.data.shape[3])]
        
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
        
    def index(self, templateName='measureView.html'):
        from jinja2 import Environment, PackageLoader
        env = Environment(loader=PackageLoader('PYME.DSView.modules', 'templates'))
        
        template = env.get_template(templateName)
        
        return template.render(objects=self.objects)
    index.exposed = True
    
    def images(self, num):
        return self.objects[int(num)].getImage()
    images.exposed = True
    
    def graphs(self, num, graphName):
        return self.objects[int(num)].getGraph(graphName)
    graphs.exposed = True
    
    def schemes(self, num):
        return self.objects[int(num)].getSchematic()
    schemes.exposed = True
    
    def hide(self, num):
        self.objects[int(num)].shown = False
        self.dsviewer.do.OnChange()
        return ''
    hide.exposed = True
        
        
def Plug(dsviewer):
    dsviewer.measure = Measurements(dsviewer)
    
    dsviewer.measure.SetSize(dsviewer.measure.GetBestSize())
    pinfo2 = aui.AuiPaneInfo().Name("measurePanel").Left().Caption('Blob Characterisation').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
    dsviewer._mgr.AddPane(dsviewer.measure, pinfo2)
    dsviewer._mgr.Update()
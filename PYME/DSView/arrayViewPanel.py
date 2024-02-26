#!/usr/bin/python

##################
# myviewpanel_numarray.py
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

import warnings
import wx

from PYME.DSView import scrolledImagePanel
from PYME.DSView.displayOptions import DisplayOpts, labeled
from PYME.DSView import overlays
from PYME.DSView.LUT import applyLUT

import numpy
import scipy
# import pylab
import matplotlib.cm

from PYME.ui import wx_compat
from PYME.ui import selection

LUTCache = {}

SLICE_AXIS_LUT = {DisplayOpts.SLICE_XY:2, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:0}
TOL_AXIS_LUT = {DisplayOpts.SLICE_XY:0, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:2}

def getLUT(cmap):
    if not cmap.name in LUTCache.keys():
        #calculate and cache LUT
        LUTCache[cmap.name] = (255*(cmap(numpy.linspace(0,1,256))[:,:3].T)).copy().astype('uint8')

    return LUTCache[cmap.name]


default_overlays = [(overlays.ScaleBarOverlay, 'Scale Bar'), 
                    (overlays.CrosshairsOverlay, 'Crosshairs')]          
class ArrayViewPanel(scrolledImagePanel.ScrolledImagePanel):
    def __init__(self, parent, dstack = None, aspect=1, do = None, voxelsize=None, initial_overlays=default_overlays):
        """
        Parameters
        ----------

        parent : wx.window
            The windows parent
        dstack : np.ndarray like object (usually and XYZTCDataSource or subclass), optional
            The data to display, ignored if do is specified
        do : displayOptions.DisplayOpts instance, optional
            The display settings (gain, scale, colour LUTs etc ...). If provided, the dstack parameter is ignored and the
            data associated with the display settings is used. If not provided a new DisplayOpts instance is created for
            the passed dstack.
        voxelsize :  PYME.IO.MetaDataHandler.VoxelSize instance, or callable
            voxel size in nm.  (x, y, z). Specififying a callable here which retuns the image voxelsize rather than the
            current value at initialisation allows changes to the voxelsize to propagate here if metadata voxelsize is changed.
        initial_overlays : list
            A list of tuples, [(OverlayClass, display_name)] for overlays to add at initialisation. Overlays can also be added
            later using the `add_overlay()` method.

        """
        
        if (dstack is None and do is None):
            dstack = scipy.zeros((10,10))

        if do is None:
            self.do = DisplayOpts(dstack, aspect=aspect)
            self.do.Optimise()
        else:
            self.do = do
        
        if voxelsize is None:
            voxelsize=[1,1,1] #compatibility fallback
            
        self._voxelsize = voxelsize

        scrolledImagePanel.ScrolledImagePanel.__init__(self, parent, self._do_paint, style=wx.SUNKEN_BORDER|wx.TAB_TRAVERSAL)

        self.do.WantChangeNotification.append(self._display_options_updated)
        #self.do.WantChangeNotification.append(self.Refresh)

        self.SetVirtualSize(wx.Size(self.do.ds.shape[0],self.do.ds.shape[1]))
        #self.imagepanel.SetSize(wx.Size(self.do.ds.shape[0],self.do.ds.shape[1]))

        self.showContours = True
        self.layerMode = 'Add'

        self.psfROIs = []
        self.psfROISize=[30,30,30]

        self.lastUpdateTime = 0
        self.lastFrameTime = 2e-3

        #self.do.scale = 0
        #self.showSelection = True
        self.selecting = False

        self.aspect = 1.

        self._slice = None
        self._sc = None
        
        # TODO - do these belong here, or with the display opts?
        self.overlays = [kls(display_name=name) for kls, name in initial_overlays]
        
        self._oldIm = None
        self._oldImSig = None
        
        self.CenteringHandlers = []
        self.selectHandlers = []
        
        self.labelPens = [wx.Pen(wx.Colour(*[int(c) for c in matplotlib.cm.hsv(v, alpha=.5, bytes=True)]), 2) for v in numpy.linspace(0, 1, 16)]
        
        #self.SetOpts()
        #self.optionspanel.RefreshHists()
        self.updating = 0

        self.imagepanel.Bind(wx.EVT_MOUSEWHEEL, self._on_wheel)
        self.imagepanel.Bind(wx.EVT_KEY_DOWN, self._on_key_press)
        #wx.EVT_KEY_DOWN(self.Parent(), self.OnKeyPress)
        self.imagepanel.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.imagepanel.Bind(wx.EVT_LEFT_UP, self._on_left_up)

        self.imagepanel.Bind(wx.EVT_MIDDLE_DOWN, self._on_middle_down)
        self.imagepanel.Bind(wx.EVT_MIDDLE_UP, self._on_middle_up)

        self.imagepanel.Bind(wx.EVT_RIGHT_DOWN, self._on_right_down)
        self.imagepanel.Bind(wx.EVT_RIGHT_UP, self._on_right_up)

        self.imagepanel.Bind(wx.EVT_MIDDLE_DCLICK, self._on_middle_double_click)

        self.imagepanel.Bind(wx.EVT_MOTION, self._on_motion)

        #
        self.imagepanel.Bind(wx.EVT_ERASE_BACKGROUND, self._do_nothing)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self._do_nothing)

    @property
    def voxelsize(self):
        if callable(self._voxelsize):
            return self._voxelsize()
        else:
            return self._voxelsize
        
    def SetDataStack(self, ds):
        """
        Make this viewer point at a new data source, resetting all the view options
        (the viewer will behave as though the new data set was cleanly loaded)
        """
        self.do.SetDataStack(ds)
        self.SetVirtualSize(wx.Size(self.do.ds.shape[0],self.do.ds.shape[1]))
                
        self.do.xp=0
        self.do.yp=0
        self.do.zp=0
        self.do.Optimise()
            
        self.do.ResetSelection()
        
        self.Layout()
#        self.Refresh()

    def ResetDataStack(self, ds):
        """
        Make this viewer point at a new data source, whilst keeping all the display options constant / unchanged
        """
        self.do.SetDataStack(ds)

    def _screen_to_abs_coordinates(self, x, y):
        xp,yp = self.CalcUnscrolledPosition(x,y)
        if self.do.orientation == self.do.UPRIGHT:
            return xp, yp
        else:
            return yp, xp

    def _screen_to_pixel_coordinates(self, x, y):
        xp, yp = self._screen_to_abs_coordinates(x, y)
        
        return xp/self.scale, yp/(self.scale*self.aspect)

    def _evt_pixel_coords(self, event, three_d=False):
        dc = wx.ClientDC(self.imagepanel)
        pos = event.GetLogicalPosition(dc)
        if three_d:
            return self._screen_to_pixel_coordinates_3D(*pos)
        else:
            return self._screen_to_pixel_coordinates(*pos)

    def _screen_to_pixel_coordinates_3D(self, x, y):
        xp, yp = self._screen_to_abs_coordinates(x, y)

        if (self.do.slice == self.do.SLICE_XY):
            return xp/self.scale, yp/(self.scale*self.aspect), self.do.zp
        elif (self.do.slice == self.do.SLICE_XZ):
            return xp/self.scale, self.do.yp, yp/(self.scale*self.aspect)
        elif (self.do.slice == self.do.SLICE_YZ):
            return self.do.xp, xp/self.scale, yp/(self.scale*self.aspect)
        

    def _abs_to_screen_coordinates(self, x, y):
        x0,y0 = self.CalcUnscrolledPosition(0,0)

        if self.do.orientation == self.do.UPRIGHT:        
            return x - x0, y - y0
        else:
            return y - x0, x - y0

    def pixel_to_screen_coordinates(self, x, y):
        """
        Return the screen coordinates for a given pixel coordinate, taking view scaling and translation into account

        Useful in overlays, etc ... to position them correctly on the screen

        NOTE: this assumes that slicing has already been accounted for - code calling this should be slice aware, or use 
        pixel_to_screen_coordinates3D() instead

        Parameters
        ==========

        x : float, or np.ndarray
            x position(s) in units of image pixels
        y : float, or np.ndarray
            y position(s) in units of image pixels 

        
        Returns
        =======
         x : float or np.ndarray
            x position(s) in device context (drawing) coordinates
        y : float or np.ndarray
            y position(s) in device context (drawing) coordinates

        """
        return self._abs_to_screen_coordinates(x*self.scale, y*self.scale*self.aspect)
        
    def pixel_to_screen_coordinates3D(self, x, y, z):
        """
        Return the screen coordinates for a given pixel coordinate, taking view scaling, translation and slicing into account

        Useful in overlays, etc ... to position them correctly on the screen

        TODO: change this to a 5D data model with t as well (when we add, e.g. x-t slicing)

        Parameters
        ==========

        x : float, or np.ndarray
            x position(s) in units of image pixels
        y : float, or np.ndarray
            y position(s) in units of image pixels 
        z : float, or np.ndarray
            z position(s) in units of image pixels
        
        Returns
        =======
         x : float or np.ndarray
            x position(s) in device context (drawing) coordinates
        y : float or np.ndarray
            y position(s) in device context (drawing) coordinates

        """
        if (self.do.slice == self.do.SLICE_XY):
            xs, ys = self.pixel_to_screen_coordinates(x,y)
        elif (self.do.slice == self.do.SLICE_XZ):
            xs, ys = self.pixel_to_screen_coordinates(x,z)
        elif (self.do.slice == self.do.SLICE_YZ):
            xs, ys = self.pixel_to_screen_coordinates(y,z)
            
        return xs, ys
        
    def draw_box_pixel_coords(self, dc, x, y, z, w, h, d):
        """Draws a box on a given device contect (dc) given 3D co-ordinates
        in image pixel space.

        Usually called from overlays. NOTE: the dc should be the same one that is passed TO the overlay, and which comes from 
        our OnPaint handler, not any arbitrary device context.

        """
        if (self.do.slice == self.do.SLICE_XY):
            xs, ys = self.pixel_to_screen_coordinates(x,y)
            ws, hs = (w*self.scale, h*self.scale*self.aspect)
        elif (self.do.slice == self.do.SLICE_XZ):
            xs, ys = self.pixel_to_screen_coordinates(x,z)
            ws, hs = (w*self.scale, d*self.scale*self.aspect)
        elif (self.do.slice == self.do.SLICE_YZ):
            xs, ys = self.pixel_to_screen_coordinates(y,z)
            ws, hs = (h*self.scale, d*self.scale*self.aspect)
            
        dc.DrawRectangle(int(xs - 0.5*ws),int( ys - 0.5*hs),int( ws),int(hs))
        
    def draw_cross_pixel_coords(self, dc, x, y, z, w, h, d):
        """Draws a cross on a given device contect (dc) given 3D co-ordinates
        in image pixel space.

        Usually called from overlays. NOTE: the dc should be the same one that is passed TO the overlay, and which comes from 
        our OnPaint handler, not any arbitrary device context.

        """
        if (self.do.slice == self.do.SLICE_XY):
            xs, ys = self.pixel_to_screen_coordinates(x,y)
            ws, hs = (w*self.scale, h*self.scale*self.aspect)
        elif (self.do.slice == self.do.SLICE_XZ):
            xs, ys = self.pixel_to_screen_coordinates(x,z)
            ws, hs = (w*self.scale, d*self.scale*self.aspect)
        elif (self.do.slice == self.do.SLICE_YZ):
            xs, ys = self.pixel_to_screen_coordinates(y,z)
            ws, hs = (h*self.scale, d*self.scale*self.aspect)
            
        #dc.DrawRectangle(xs - 0.5*ws, ys - 0.5*hs, ws,hs)
        dc.DrawLine(int(xs - 0.5*ws),int( ys-0.5*hs),int( xs + 0.5*ws),int( ys+0.5*hs))
        dc.DrawLine(int(xs - 0.5*ws),int( ys+0.5*hs),int( xs + 0.5*ws),int( ys-0.5*hs))
        

    @property
    def scale(self):
        """
        The scaling between image pixels and display pixels

        NOTE: this is linear, DisplayOptions.scale is log2(this)
        """
        return pow(2.0,(self.do.scale))
            
    def _draw_selection(self, view, dc):
        if self.do.showSelection:
            col = wx.TheColourDatabase.FindColour('YELLOW')
            #col.Set(col.red, col.green, col.blue, 125)
            dc.SetPen(wx.Pen(col,1))
            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            lx, ly, hx, hy = self.do.GetSliceSelection()
            lx, ly = self.pixel_to_screen_coordinates(lx, ly)
            hx, hy = self.pixel_to_screen_coordinates(hx, hy)
            
            if self.do.selection.mode == selection.SELECTION_RECTANGLE:
                dc.DrawRectangle(int(lx),int(ly),int( (hx-lx)),int((hy-ly)))
                
            elif self.do.selection.mode == selection.SELECTION_SQUIGGLE:
                if len(self.do.selection.trace) > 2:
                    x, y = numpy.array(self.do.selection.trace).T
                    pts = numpy.vstack(self.pixel_to_screen_coordinates(x, y)).T
                    dc.DrawSpline(pts.astype('i'))
            elif self.do.selection.width == 1:
                dc.DrawLine(int(lx),int(ly),int( hx),int(hy))
            else:
                lx, ly, hx, hy = self.do.GetSliceSelection()
                dx = hx - lx
                dy = hy - ly

                if dx == 0 and dy == 0: #special case - profile is orthogonal to current plane
                    d_x = 0.5*self.do.selection.width
                    d_y = 0.5*self.do.selection.width
                else:
                    d_x = 0.5*self.do.selection.width*dy/numpy.sqrt((dx**2 + dy**2))
                    d_y = 0.5*self.do.selection.width*dx/numpy.sqrt((dx**2 + dy**2))
                    
                x_0, y_0 = self.pixel_to_screen_coordinates(lx + d_x, ly - d_y)
                x_1, y_1 = self.pixel_to_screen_coordinates(lx - d_x, ly + d_y)
                x_2, y_2 = self.pixel_to_screen_coordinates(hx - d_x, hy + d_y)
                x_3, y_3 = self.pixel_to_screen_coordinates(hx + d_x, hy - d_y)
                
                lx, ly = self.pixel_to_screen_coordinates(lx, ly)
                hx, hy = self.pixel_to_screen_coordinates(hx, hy)
               

                dc.DrawLine(int(lx),int( ly),int( hx),int( hy))
                dc.DrawPolygon([(int(x_0), int(y_0)), (int(x_1), int(y_1)), (int(x_2), int(y_2)), (int(x_3), int(y_3))])
                
            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)
                
    def _draw_contours(self, view, dc):
        # TODO - shift to overlay [currently triggered as a result of specific recipe output - work out how to detect this in the recipe handling]
        if self.showContours and 'filter' in dir(self) and 'contour' in self.filter.keys() and self.do.slice ==self.do.SLICE_XY:
            t = self.filter['t'] # prob safe as int
            x = self.filter['x']/self.voxelsize[0]
            y = self.filter['y']/self.voxelsize[1]
            
            xb, yb, zb, tb = self.visible_bounds
            
            IFoc = (x >= xb[0])*(y >= yb[0])*(t >= zb[0])*(x < xb[1])*(y < yb[1])*(t < zb[1])

            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            #pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            dc.SetPen(pGreen)
            
            contours = self.filter['contour'][IFoc]
            if 'clumpIndex' in self.filter.keys():
                colInds = self.filter['clumpIndex'][IFoc] %len(self.labelPens)
            else:
                colInds = numpy.zeros(len(contours), 'i') #%len(self.labelPens)
            for c, colI in zip(contours, colInds):
                xc, yc = c.T
                dc.SetPen(self.labelPens[int(colI)])
                dc.DrawSpline(numpy.vstack(self.pixel_to_screen_coordinates(xc, yc)).T.astype('i'))
                

    @property
    def visible_bounds(self):
        """
        The currently visible bounds of the image, in image pixel coordinates [x, y, z, t]

        Used to avoid drawing overlays in regions of the image which are not shown.

        TODO - make 4D (when we add xt etc .... slicing). Overlays (and especially those in plugins) should be written
        so they will also work if this method retuns a 4-tuple, [x,y,z,t]
        """
        sc = self.scale
        x0,y0 = self.CalcUnscrolledPosition(0,0)
        sX, sY = self.imagepanel.Size
        
        if self.do.slice == self.do.SLICE_XY:
            bnds = [(x0/sc, (x0+sX)/sc), (y0/sc, (y0+sY)/sc), (self.do.zp-.5, self.do.zp+.5), (self.do.tp-.5, self.do.tp+.5)]
        elif self.do.slice == self.do.SLICE_XZ:
            bnds = [(x0/sc, (x0+sX)/sc), (self.do.yp-.5, self.do.yp+.5), (y0/sc, (y0+sY)/sc), (self.do.tp-.5, self.do.tp+.5)]
        elif self.do.slice == self.do.SLICE_YZ:
            bnds = [(self.do.xp-.5, self.do.xp+.5),(x0/sc, (x0+sX)/sc), (y0/sc, (y0+sY)/sc), (self.do.tp-.5, self.do.tp+.5)]

        return bnds
        
    
    
    def _do_paint(self, dc, fullImage=False):
        #print 'p'
        
        dc.Clear()
                                     
        im = self._render(fullImage)

        sc = self.scale
        sc2 = sc
        
        if sc >= 1:
            step = 1
        else:
            step = 2**(-numpy.ceil(numpy.log2(sc)))
            sc2 = sc*step

        
        im2 = wx_compat.BitmapFromImage(im)
        dc.DrawBitmap(im2,int(-sc2/2),int(-sc2/2))
        
        self._draw_selection(self, dc) 
        self._draw_contours(self, dc)

        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
            
        for ovl in self.overlays:
            ovl(self, dc)

    def GrabImage(self, fullImage=True):
        #TODO - get suitable image dependent viewport

        xs, ys = self._unscrolled_view_size()
        if fullImage:
            from PYME import warnings
            if (xs > 2e3 or ys > 2e3) and not warnings.warn('Captured image will be very large, continue?',allow_cancel=True):
                return
        else:
            s = self.GetClientSize()
            xs = min(s.GetWidth(), xs)
            ys = min(s.GetHeight(), ys)

        MemBitmap = wx_compat.EmptyBitmap(xs, ys)
        MemDC = wx.MemoryDC()
        OldBitmap = MemDC.SelectObject(MemBitmap)

        self._do_paint(MemDC, fullImage)

        return MemBitmap

    def GrabPNG(self, filename, fullImage=True):
        MemBitmap = self.GrabImage(fullImage)
        img = MemBitmap.ConvertToImage()
        img.SaveFile(filename, wx.BITMAP_TYPE_PNG)
    
    def ExportStackToPNG(self, filename, fullImage=True):
        """Save current view to a series of PNG files with z (or t) index as suffix, suitable for use in making a movie
        via ffmpeg or similar tools

        Parameters
        ----------
        filename : str
            fully qualified path, with extension. Note that _%d will be appended to the filename to generate the
            individual files
        fullImage : bool, optional
            whether to export the full image even if it is clipped in the GUI, by default True
        FIXME - make this work with time series / 5D image data model.
        """
        import os
        filestub, ext = os.path.splitext(filename)
        for ind in range(self.do.ds.shape[2]):
            self.do.zp = ind
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
            self.GrabPNG(filestub + '_%d' % ind + ext, fullImage)
        
    def GrabPNGToBuffer(self, fullImage=True):
        '''Get PNG data in a buffer (rather than writing directly to file)'''
        from io import BytesIO

        img = self.GrabImage(fullImage)
        out = BytesIO()
        # NB - using wx functionality rather than pillow here as wxImage.GetData() returns a BytesArray object rather
        # than a buffer on py3. This underlying problem may need to be revisited.
        img.ConvertToImage().SaveFile(out, wx.BITMAP_TYPE_PNG)
        return out.getvalue()

    def CopyImage(self, fullImage=True):
        """ Copies the currently displayed image to the clipboard"""
        bmp = self.GrabImage(fullImage)
        try:
            wx.TheClipboard.Open()
            bmpDataObject = wx.BitmapDataObject(bmp)
            wx.TheClipboard.SetData(bmpDataObject)
        finally:
            wx.TheClipboard.Close()
            

    def _on_wheel(self, event):
        rot = event.GetWheelRotation()
        if rot < 0:
            if event.RightIsDown():
                self.do.yp = max(self.do.yp - 1, 0)
            elif event.MiddleIsDown(): 
                self.do.xp = max(self.do.xp - 1, 0)
            elif event.ShiftDown():
                self.do.SetScale(self.do.scale - 1)
            else:
                self.do.zp = max(self.do.zp - 1, 0)
        if rot > 0:
            if event.RightIsDown():
                self.do.yp = min(self.do.yp + 1, self.do.ds.shape[1] -1)
            elif event.MiddleIsDown(): 
                self.do.xp = min(self.do.xp + 1, self.do.ds.shape[0] -1)
            elif event.ShiftDown():
                self.do.SetScale(self.do.scale + 1)
            else:
                self.do.zp = min(self.do.zp + 1, self.do.ds.shape[2] -1)
                
        if ('update' in dir(self.GetParent())):
             self.GetParent().update()
        else:
            self.imagepanel.Refresh()
        #self.update()
    
    def _on_key_press(self, event):
        if event.GetKeyCode() == wx.WXK_PAGEUP:
            self.do.zp = max(0, self.do.zp - 1)
            #self.optionspanel.RefreshHists()
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
                

        elif event.GetKeyCode() == wx.WXK_PAGEDOWN:
            self.do.zp = min(self.do.zp + 1, self.do.ds.shape[2] - 1)
            #self.optionspanel.RefreshHists()
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
                #print 'upd'
            else:
                self.imagepanel.Refresh()
                
        elif event.GetKeyCode() == 74: #J
            self.do.xp = (self.do.xp - 1)
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 76: #L
            self.do.xp +=1
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 73: #I
            self.do.yp -= 1
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 75: #K
            self.do.yp += 1
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 77: #M
            #print 'o'
            self.do.Optimise(method='min-max')
        elif event.GetKeyCode() == ord('P'): #M
            #print 'p'
            self.do.Optimise(method='percentile')
        elif event.GetKeyCode() == ord('C'):
            if event.GetModifiers() == wx.MOD_CMD:
                self.CopyImage()
            elif event.GetModifiers() == wx.MOD_CMD|wx.MOD_SHIFT:
                self.CopyImage(False)
            else:
                event.Skip()
        else:
            event.Skip()
        

        
    def _display_options_updated(self,event=None):
        if (self.updating == 0):

            sc = pow(2.0,(self.do.scale))
            s = self._calc_im_size()
            self.SetVirtualSize(wx.Size(int(s[0]*sc),int(s[1]*sc)))

            if (self._slice != self.do.slice) or (self._sc != sc):
                #print('recentering')
                #if the slice has changed, change our aspect and do some
                self._slice = self.do.slice
                self._sc = sc
                #if not event is None and event.GetId() in [self.cbSlice.GetId(), self.cbScale.GetId()]:
                #recenter the view
                if(self.do.slice == self.do.SLICE_XY):
                    lx = self.do.xp
                    ly = self.do.yp
                    self.aspect = self.do.aspect[1]/self.do.aspect[0]
                elif(self.do.slice == self.do.SLICE_XZ):
                    lx = self.do.xp
                    ly = self.do.zp
                    self.aspect = self.do.aspect[2]/self.do.aspect[0]
                elif(self.do.slice == self.do.SLICE_YZ):
                    lx = self.do.yp
                    ly = self.do.zp
                    self.aspect = self.do.aspect[2]/self.do.aspect[1]

                sx,sy =self.imagepanel.GetClientSize()

                ppux, ppuy = self.GetScrollPixelsPerUnit()
                self.Scroll(max(0, lx*sc - sx/2)/ppux, max(0, ly*sc*self.aspect - sy/2)/ppuy)

            #self.imagepanel.Refresh()
            self.Refresh()
            self.Update()
            
        
    def _calc_im_size(self):
        # calculate the full size of an image when grabbing a full-size colour mapped image as PNG or similar
        if (self.do.slice == self.do.SLICE_XY):
            if (self.do.orientation == self.do.UPRIGHT):
                return (self.do.ds.shape[0],self.do.ds.shape[1])
            else:
                return (self.do.ds.shape[1],self.do.ds.shape[0])
        elif (self.do.slice == self.do.SLICE_XZ):
            return (self.do.ds.shape[0],self.do.ds.shape[2])
        else:
            return(self.do.ds.shape[1],self.do.ds.shape[2] )
        
    def _do_nothing(self, event):
        # used to catch ERASE_BACKGROUND events, to reduce flicker
        pass

    def _on_left_down(self,event):
        if self.do.leftButtonAction == self.do.ACTION_SELECTION:
            self._start_selection(event)
            
        event.Skip()
    
    def _on_left_up(self,event):
        if self.do.leftButtonAction == self.do.ACTION_SELECTION:
            self._progress_selection(event)
            self._end_selection()
        elif self.do.leftButtonAction == self.do.ACTION_SELECT_OBJECT:
            self._on_select_object(event)
        else:
            self._on_set_position(event)
            
        event.Skip()
        
    def _on_middle_down(self,event):
        dc = wx.ClientDC(self.imagepanel)
#        self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        self.middleDownPos = self.CalcUnscrolledPosition(*pos)
        event.Skip()
    
    def _on_middle_up(self,event):
        dc = wx.ClientDC(self.imagepanel)
#        self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)

        dx = pos[0] - self.middleDownPos[0]
        dy = pos[1] - self.middleDownPos[1]
        
        sc = pow(2.0,(self.do.scale))

        if (abs(dx) > 5) or (abs(dy) > 5):
            for h in self.CenteringHandlers:
                h(-dx/sc,-dy/sc)
        
        event.Skip()
        
    def _on_middle_double_click(self,event):
        dc = wx.ClientDC(self.imagepanel)
#        self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)
        #print pos
        sc = pow(2.0,(self.do.scale))
        if (self.do.slice == self.do.SLICE_XY):
            x = (pos[0]/sc) - 0.5*self.do.ds.shape[0]
            y = (pos[1]/(sc*self.aspect)) - 0.5*self.do.ds.shape[1]
            
            for h in self.CenteringHandlers:
                h(x,y)
            
        event.Skip()
    
            
    def _unscrolled_view_size(self):
        sc = pow(2.0, (self.do.scale))
        shp = self.do.ds.shape

        if (self.do.slice == self.do.SLICE_XY):
            xs = int(shp[0] * sc)
            ys = int(shp[1] * sc*self.aspect)
        elif (self.do.slice == self.do.SLICE_XZ):
            xs = int(shp[0] * sc)
            ys = int(shp[2] * sc * self.aspect)
        elif (self.do.slice == self.do.SLICE_YZ):
            xs = int(shp[1] * sc)
            ys = int(shp[2] * sc * self.aspect)

        return xs, ys

    def _on_set_position(self,event):
        pos_3d = self._evt_pixel_coords(event, three_d=True)

        self.do.inOnChange = True
        try:
            self.do.xp, self.do.yp, self.do.zp = [int(p) for p in pos_3d]
        finally:
            self.do.inOnChange = False
            
        self.do.OnChange()
        
        for cb in self.selectHandlers:
            if cb(pos_3d):
                break

    def _on_select_object(self, event):
        pos_3d = self._evt_pixel_coords(event, three_d=True)
        
        for cb in self.selectHandlers:
            if cb(pos_3d):
                #only continue until we hit something
                break


    def _on_right_down(self, event):
        self._start_selection(event)
            
    def _start_selection(self,event):
        self.selecting = True
        
        pos = self._evt_pixel_coords(event)
        
        if (self.do.slice == self.do.SLICE_XY):
            self.do.selection.start.x, self.do.selection.start.y = [int(p) for p in pos]
        elif (self.do.slice == self.do.SLICE_XZ):
            self.do.selection.start.x, self.do.selection.start.z = [int(p) for p in pos]
        elif (self.do.slice == self.do.SLICE_YZ):
            self.do.selection.start.y, self.do.selection.start.z = [int(p) for p in pos]
            
        self.do.selection.trace = []
        self.do.selection.trace.append(tuple(pos))

    def _on_right_up(self,event):
        self._progress_selection(event)
        self._end_selection()

    def _on_motion(self, event):
        if event.Dragging() and self.selecting:
            self._progress_selection(event)
            
    def _progress_selection(self,event):
        pos = self._evt_pixel_coords(event)
        
        if (self.do.slice == self.do.SLICE_XY):
            self.do.selection.finish.x, self.do.selection.finish.y = [int(p) for p in pos]
        elif (self.do.slice == self.do.SLICE_XZ):
            self.do.selection.finish.x, self.do.selection.finish.z = [int(p) for p in pos]
        elif (self.do.slice == self.do.SLICE_YZ):
            self.do.selection.finish.y, self.do.selection.finish.z = [int(p) for p in pos]


        if event.ShiftDown(): #lock
            if (self.do.slice == self.do.SLICE_XY):

                dx = abs(self.do.selection.finish.x - self.do.selection.start.x)
                dy = abs(self.do.selection.finish.y - self.do.selection.start.y)

                if dx > 1.5*dy: #horizontal
                    self.do.selection.finish.y = self.do.selection.start.y
                elif dy > 1.5*dx: #vertical
                    self.do.selection.finish.x = self.do.selection.start.x
                else: #diagonal
                    self.do.selection.finish.y = self.do.selection.start.y + dx*numpy.sign(self.do.selection.finish.y - self.do.selection.start.y)
                
        self.do.selection.trace.append(pos)

        self.Refresh()
        self.Update()

    def _end_selection(self):
        self.selecting = False
        self.do.EndSelection()
            

        
    def _image_signature(self, x0, y0, sX,sY, do):
        # generate a signature for the current image settings to work out if we need to re-render (colour-map) the image
        # or if we can use a cached copy
        sig = [x0, y0, sX, sY, do.scale, do.slice, do.GetActiveChans(), do.ds.shape]
        if do.slice == DisplayOpts.SLICE_XY:
            sig += [do.zp, do.maximumProjection]
        if do.slice == DisplayOpts.SLICE_XZ:
            sig += [do.yp]
        if do.slice == DisplayOpts.SLICE_YZ:
            sig += [do.xp]
            
        return sig
    
    def Redraw(self, sender=None, **kwargs):
        self._oldImSig = None
        self.Refresh()
        self.Update()
        
    def _map_colour(self, seg, gain, offset, cmap, ima):
        lut = getLUT(cmap)
        
        if cmap == labeled:
            # special case for labelled colourmap - use slow matplotlib lookup and rely on matplotlib roll-around to
            # cycle colour map TODO - check if recent matplotlibs actually roll around or not.
            ima[:] = numpy.minimum(ima[:] + (255 * cmap(gain * (seg - offset))[:, :, :3])[:], 255)
    
        elif numpy.iscomplexobj(seg):
            if self.do.colourMax or (self.do.complexMode == 'imag coloured'):
                applyLUT(numpy.imag(seg), self.do.cmax_scale / self.do.ds.shape[2], self.do.cmax_offset, lut, ima)
                ima[:] = (ima * numpy.clip((numpy.real(seg) - offset) * gain, 0, 1)[:, :, None]).astype('uint8')
            elif self.do.complexMode == 'real':
                applyLUT(seg.real, gain, offset, lut, ima)
            elif self.do.complexMode == 'imag':
                applyLUT(seg.imag, gain, offset, lut, ima)
            elif self.do.complexMode == 'abs':
                applyLUT(numpy.abs(seg), gain, offset, lut, ima)
            elif self.do.complexMode == 'angle':
                applyLUT(numpy.angle(seg), gain, offset, lut, ima)
            else:
                applyLUT(numpy.angle(seg), self.do.cmax_scale / self.do.ds.shape[2], self.do.cmax_offset, lut, ima)
                ima[:] = (ima * numpy.clip((numpy.abs(seg) - offset) * gain, 0, 1)[:, :, None]).astype('uint8')
        else:
            #print seg.shape
            applyLUT(seg, gain, offset, lut, ima)

    def _render(self, fullImage=False):
        #print 'rend'
        if fullImage:
            x0, y0 = 0,0
            sX, sY = self._unscrolled_view_size()
        else:
            x0,y0 = self.CalcUnscrolledPosition(0,0)
            sX, sY = self.imagepanel.Size
        
        sig = self._image_signature(x0, y0, sX, sY, self.do)
        if sig == self._oldImSig:# and not self._oldIm is None:
            #if nothing has changed, don't re-render
            return self._oldIm

        sc = pow(2.0,self.do.scale)
        sc2 = sc

        if sc >= 1:
            step = 1
        else:
            step = 2**(-numpy.ceil(numpy.log2(sc)))
            sc2 = sc*step
        
        sX_ = int(sX/(sc))
        sY_ = int(sY/(sc*self.aspect))
        x0_ = int(x0/sc)
        y0_ = int(y0/(sc*self.aspect))
            
        fstep = float(step)
        step = int(step)
        
        if (step > 1) and hasattr(self.do.ds, 'levels'):
            # we have a pyramidal data source
            
            level = -self.do.scale
            
            #if (level > len(self.do.ds.levels)):
            level = int(min(level, len(self.do.ds.levels)-1))
            step = int(2**(-numpy.ceil(numpy.log2(sc))-level))
            
            _s = 1.0/(2**level)
            
            x0_, y0_, sX_, sY_ = [int(numpy.ceil(v*_s)) for v in [x0_, y0_, sX_, sY_]]
            
            # x0_ = int(numpy.ceil(x0_*_s))
            # y0_ = int(y0_*_s)
            # sX_ = int(sX_*_s)
            # sY_ = int(sY_*_s)

            print('level:', level)
            
            ds = self.do.ds.levels[level]
        else:
            ds = self.do.ds
            _s = 1
        
        #XY
        if self.do.slice == DisplayOpts.SLICE_XY:
            dmy, dmx  = self.do.ds.shape[1], self.do.ds.shape[0]
            slice_key = (slice(x0_,(x0_+sX_),step),
                         slice(y0_,(y0_+sY_),step),
                         int(self.do.zp*_s),
                         int(self.do.tp*_s))
            
            proj_axis = 2
                    
        #XZ
        elif self.do.slice == DisplayOpts.SLICE_XZ:
            dmy, dmx = self.do.ds.shape[2], self.do.ds.shape[0]
            slice_key = (slice(x0_, (x0_ + sX_), step),
                         int(self.do.yp*_s),
                         slice(y0_, (y0_ + sY_), step),
                         int(self.do.tp*_s))
            
            proj_axis = 1
        #YZ
        elif self.do.slice == DisplayOpts.SLICE_YZ:
            dmy, dmx = self.do.ds.shape[2], self.do.ds.shape[1]
            slice_key = (int(self.do.xp*_s),
                         slice(x0_, (x0_ + sX_), step),
                         slice(y0_, (y0_ + sY_), step),
                         int(self.do.tp*_s))
            
            proj_axis = 0
            
        if ds.ndim < 5:
            # for old-style data, drop the time dimension
            slice_key = slice_key[:3]
            
        #ima = numpy.zeros((int(numpy.ceil(min(sY_/_s, dmy)/fstep)), int(numpy.ceil(min(sX_/_s, dmx)/fstep)), 3), 'uint8')

        segs = []
        for chan, offset, gain, cmap in self.do.GetActiveChans():
            if self.do.maximumProjection and (self.do.slice == DisplayOpts.SLICE_XY):
                # special case for max projection - fixme - remove after we get colour coded projections in the projection module
                seg = self.do.ds[slice_key[:2] + (slice(None), chan)].max(2).squeeze().T
                if self.do.colourMax:
                    seg = seg + 1j*self.do.ds[slice_key[:2] + (slice(None), chan)].argmax(2).squeeze().T
            else:
                seg = ds[slice_key +  (chan,)].squeeze().T
                
            segs.append((seg, chan, offset, gain, cmap))
            
        
        if len(segs) > 0:
            ima = numpy.zeros(segs[0][0].shape[:2] + (3,), 'uint8')
        else:
            ima = numpy.zeros((int(numpy.ceil(min(sY_ / _s, dmy) / fstep)), int(numpy.ceil(min(sX_ / _s, dmx) / fstep)), 3), 'uint8')
        
        for seg, chan, offset, gain, cmap in segs:
            #'slice_key:', slice_key)
            #print('seg.shape, ima.shape:', seg.shape, ima.shape)
            self._map_colour(seg, gain, offset, cmap, ima)
#        
        img = wx_compat.ImageFromData(ima.shape[1], ima.shape[0], ima.ravel())
        img.Rescale(int(img.GetWidth()*sc2),int(img.GetHeight()*sc2*self.aspect))
        self._oldIm = img
        self._oldImSig = sig
        return img

    def add_overlay(self, ovl, display_name=None):
        """
        Add an overlay. 
        
        The overlay should ideally be an object derived from overlays.Overlay and follow the instance outlined in that class, 
        function overlays are also supported in the interim for backwards compatibility.

        If using a functional overlay, a display_name must be provided

        Parameters
        ==========

        ovl : overlays.Overlay instance, optionaly a function or method
            The overlay code

        display_name : string
            a name to use when controlling overlay visibility. Must be provided for function overlays, ignored for instances of the Overlay class (which have a display_name property)

        """
        if not isinstance(ovl, overlays.Overlay):
            warnings.warn(numpy.VisibleDeprecationWarning('using old-style overlay function, please re-write as a class'))
            ovl = overlays.FunctionOverlay(ovl, display_name)
        
        self.overlays.append(ovl)
        self.do.OnChange()


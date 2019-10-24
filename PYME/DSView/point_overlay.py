import wx
import numpy as np
import six
from PYME.DSView.arrayViewPanel import SLICE_AXIS_LUT, TOL_AXIS_LUT
from PYME.localization import splitting

class PointOverlay(object):
    def __init__(self, datasource, point_mode=None, point_size=5):
        self._datasource = datasource
        
        self._point_mode = point_mode
        self.point_size = point_size
        self.show = True
        self.show_adjacent_points = False
        
    @property
    def point_mode(self):
        if not self._point_mode is None:
            return self._point_mode
        else:
            return 'splitter' if ('gFrac' in self._datasource.keys()) else 'standard'
        
    def _draw_points(self, view, dc, points, point_colours=None):
        ps = self.point_size

        if isinstance(point_colours, six.string_types):
            pGreen = wx.Pen(wx.TheColourDatabase.FindColour(point_colours), 1)
            point_colours = np.ones(len(points), np.bool)
        elif point_colours is None:
            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('GREEN'), 1)
            point_colours = np.ones(len(points), np.bool)
        else:
            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('GREEN'), 1)
            pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'), 1)
            
        for p, c, dxi, dyi in zip(points, point_colours, dx, dy):
            if c:
                dc.SetPen(pGreen)
            else:
                dc.SetPen(pRed)
        
            view._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)
    
        dc.SetPen(wx.NullPen)
    
    def DrawOverlays(self, view, dc):
        dx = 0
        dy = 0
    
        aN = SLICE_AXIS_LUT[view.do.slice]
        tolN = TOL_AXIS_LUT[view.do.slice]
        pos = [view.do.xp, view.do.yp, view.do.zp]
    
        if self.show and not self._datasource is None:
            t = self._datasource['t'] #prob safe as int
            x = self._datasource['x'] / view.voxelsize[0]
            y = self._datasource['y'] / view.voxelsize[1]
        
            xb, yb, zb = view._calcVisibleBounds()
        
            IFoc = (x >= xb[0]) * (y >= yb[0]) * (t >= zb[0]) * (x < xb[1]) * (y < yb[1]) * (t < zb[1])
            INFoc = 0*IFoc # FIXME
        
            pFoc = np.vstack((x[IFoc], y[IFoc], t[IFoc])).T
            
            if self.point_mode == 'splitter':
                x1, y1 = splitting.remap_splitter_coords(self._datasource.mdh, view.do.ds.shape[:2], x, y)
                pCol = self._datasource['gFrac'][IFoc] > .5
            
            pNFoc = []
        
            # #intrinsic points
            # elif len(self.points) > 0:
            #     pointTol = self.pointTolNFoc[self.pointMode]
            #
            #     IFoc = abs(self.points[:, aN] - pos[aN]) < 1
            #     INFoc = abs(self.points[:, aN] - pos[aN]) < pointTol[tolN]
            #
            #     pFoc = self.points[IFoc]
            #     pNFoc = self.points[INFoc]
            #
            #     if self.pointMode == 'splitter':
            #         pCol = self.pointColours[IFoc]
        
        
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
        
            #Draw points in planes above and below the current one
            if self.show_adjacent_points and (len(pNFoc) > 0):
                dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('BLUE'), 1))
                
                self._draw_points(view, dc, pNFoc, 'BLUE')
            
                if self.point_mode == 'splitter':
                    self._draw_points(view, dc, np.vstack((x1[INFoc], y1[INFoc], t[INFoc])).T, 'BLUE')
            
            # draw points in current plane
            if self.point_mode == 'splitter':
                self._draw_points(view, dc, pFoc, pCol)
                self._draw_points(view, dc, np.vstack((x1[IFoc], y1[IFoc], t[IFoc])).T, pCol)
            else:
                self._draw_points(view, dc, pFoc)
        
            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)
        
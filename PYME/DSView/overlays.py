# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:28:03 2015

@author: david
"""
from asyncio.log import logger
import wx
import numpy as np
from PYME.DSView.displayOptions import DisplayOpts
from PYME.LMVis.layers.base import SimpleLayer
from PYME.recipes.traits import CStr, Int
import abc
import logging
logger = logging.getLogger(__name__)

class Overlay(SimpleLayer):
    # make our overlays inherit from PYMEVis layers, even if we don't implement opengl display for now
    # NOTE this gives us a 'visible' property

    display_name = CStr('')

        
    @abc.abstractmethod
    def __call__(self, vp, dc):
        """
        Draw this overly using wx, as an overlay on an arrayviewpanel


        Parameters
        ==========

        vp : arrayViewPanle.ArrayViewPanel instance
            Generally only to be used for getting display options, pixel coordinate transformations, etc ... please do not store or acess data through vp
            (vp was a mess, containing lots of data, rather than just view, and we are slowly trying to move all fot that out)
        dc : wx.DC instance
            The device context to draw onto

        TODO - refactor this to e.g. draw_wx
        """
        pass

SLICE_AXIS_LUT = {DisplayOpts.SLICE_XY:2, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:0}
TOL_AXIS_LUT = {DisplayOpts.SLICE_XY:0, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:2}
class PointDisplayOverlay(Overlay):
    def __init__(self, points = [], filter=None, md=None, **kwargs):
        self.points = points
        self.md = md
        self.pointColours = []
        self.pointSize = 11
        self.pointMode = 'confoc'
        self.pointTolNFoc = {'confoc' : (5,5,5), 'lm' : (2, 5, 5), 'splitter' : (2,5,5)}
        self.showAdjacentPoints = False
        self.display_as = 'box' # 'box' or 'cross'
        self.z_mode='t'

        # keep a single, global copy of pens
        self._pGreen = wx.Pen(wx.TheColourDatabase.FindColour('GREEN'))
        self._pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'))
        self._pBlue = wx.Pen(wx.TheColourDatabase.FindColour('BLUE'))

        self._pens = [self._pGreen, self._pRed, self._pBlue]

        if filter:
            self.filter = filter

        Overlay.__init__(self, **kwargs)

    
    def _map_splitter_coords(self, x, y, ds_shape):
        from PYME.localization import splitting

        if self.md is None:
            md = getattr(self.filter, 'mdh', {})
        else:
            md = self.md

        xgs, xrs, ygs, yrs = splitting.get_splitter_rois(md, ds_shape)
        return splitting.remap_splitter_coords_(x, y, [xgs, xrs], [ygs, yrs], quadrant=1, flip=(yrs.step < 0))
    
    
    def __call__(self, vp, dc):
        dx = 0
        dy = 0
        
        aN = SLICE_AXIS_LUT[vp.do.slice]
        tolN = TOL_AXIS_LUT[vp.do.slice]
        if vp.do.ds.shape[2] > 1:
            # stack has z
            pos = [vp.do.xp, vp.do.yp, vp.do.zp]
        else:
            # stack is a time series
            pos = [vp.do.xp, vp.do.yp, vp.do.tp]

        try:
            vx, vy, vz = vp.voxelsize
        except KeyError:
            logger.exception('No voxelsize set, cannot display scale bar.')
            print('No voxelsize set, cannot display scale bar.')
            return

        if self.visible and ('filter' in dir(self) or len(self.points) > 0):
            #print('plotting points')
            
            if 'filter' in dir(self):
                f_keys = self.filter.keys()

                t = self.filter['t'] #prob safe as int
                x = self.filter['x']/vx
                y = self.filter['y']/vy
                if self.z_mode == 'z':
                    if 'z' in f_keys:
                        t = self.filter['z']/vz
                    else:
                        t = self.filter['fitResults_z0']/vz
                
                xb, yb, zb, tb = vp.visible_bounds
                
                if vp.do.ds.shape[3] <= 1:
                    # stack is a pure z-series, pretend it's a time series for backwards compatibility
                    tb = zb
                
                IFoc = (x >= xb[0])*(y >= yb[0])*(t >= tb[0])*(x < xb[1])*(y < yb[1])*(t < tb[1])
                    
                pFoc = np.vstack((x[IFoc], y[IFoc], t[IFoc])).T

                
                #assume splitter, then test for keys
                pm = 'splitter'    
                if 'gFrac' in f_keys:
                    pCol = self.filter['gFrac'][IFoc] > .5 
                elif 'ratio' in f_keys:
                    pCol = self.filter['ratio'][IFoc] > 0.5
                elif 'fitResults_Ag' in f_keys:
                    pCol = self.filter['fitResults_Ag'][IFoc] > self.filter['fitResults_Ar'][IFoc]
                else:
                    pm = self.pointMode
                    if 'channel' in f_keys:
                        pCol = self.filter['channel'][IFoc]
                    elif 'probe' in f_keys:
                        pCol = self.filter['probe'][IFoc]
                    else:
                        pCol = np.zeros(len(pFoc), dtype='i4')               
                
                pNFoc = []

            #intrinsic points            
            elif len(self.points) > 0:
                pm = self.pointMode
                pointTol = self.pointTolNFoc[self.pointMode]
                
                IFoc = abs(self.points[:,aN] - pos[aN]) < 1
                INFoc = abs(self.points[:,aN] - pos[aN]) < pointTol[tolN]
                    
                pFoc = self.points[IFoc]
                pNFoc = self.points[INFoc]

                if len(self.pointColours) == len(self.points):
                    pCol = self.pointColours[IFoc]
                else:
                    pCol = np.zeros(len(pFoc), dtype='i4')
                    

            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            ps = self.pointSize

            if self.showAdjacentPoints:
                dc.SetPen(self._pBlue)
                
                for xi, yi, zi in pNFoc:
                    vp.draw_box_pixel_coords(dc, xi, yi, zi, ps, ps, ps)
                
                if pm == 'splitter':
                    x, y, z = pNFoc.T
                    x_, y_ = self._map_splitter_coords(x, y, vp.do.ds.shape)
                    for xi, yi, zi in zip(x_, y_, z):#, dxi, dyi in zip(pNFoc, dxn, dyn):
                        vp.draw_box_pixel_coords(dc, xi, yi, zi, ps, ps, ps)
            
            if pm == 'splitter':
                x, y, z = pFoc.T
                x_, y_ = self._map_splitter_coords(x, y, vp.do.ds.shape)

                self._draw_points(dc, vp, x, y, z, pCol)
                self._draw_points(dc, vp, x_, y_, z, pCol)
            else:
                x, y, z = pFoc.T
                self._draw_points(dc, vp, x, y, z, pCol)
            
            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)

    def _draw_points(self, dc, vp, x, y, z, c):
        ps = self.pointSize

        for xi, yi, zi, ci in zip(x, y, z, c):
            dc.SetPen(self._pens[int(ci)])

            if self.display_as == 'box':
                vp.draw_box_pixel_coords(dc, xi, yi, zi, ps, ps, ps)
            elif self.display_as == 'cross':
                vp.draw_cross_pixel_coords(dc, xi, yi, zi, ps, ps, ps)
            else:
                raise ValueError('Unknown display_as type: %s - expected \'box\' or \'cross\'' % self.display_as)


    def points_hit_test(self, xp, yp, zp, voxelsize=None):
        if len(self.points) > 0:
            x, y, z = self.points
        elif hasattr(self, 'filter'):
            vx, vy = voxelsize[:2]
            z = self.filter['t'] #prob safe as int
            x = self.filter['x']/vx
            y = self.filter['y']/vy
        else:
            return None

        iCand = np.where((abs(z - zp) < 1)*(abs(x - xp) < 3)*(abs(y - yp) < 3))[0]

        if len(iCand) == 0:
            return None
        elif len(iCand) == 1:
            return iCand[0]
        else:
            iNearest = np.argmin((x[iCand] - xp)**2 + (y[iCand] - yp)**2)
            return iCand[iNearest]

        
class ScaleBarOverlay(Overlay):
    # TODO - combine with LMVis scale bar
    length_nm = Int(2000)

    def __call__(self, vp, dc):
        if self.visible:
            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('WHITE'),10)
            pGreen.SetCap(wx.CAP_BUTT)
            dc.SetPen(pGreen)
            sX, sY = vp.imagepanel.Size
            
            try:
                vx, vy = vp.voxelsize[:2]
            except KeyError:
                logger.exception('No voxelsize set, cannot display scale bar.')
                print('No voxelsize set, cannot display scale bar.')
                return
            
            sbLen = int(self.length_nm*vp.scale/vx)
            
            y1 = 20
            x1 = 20 + sbLen
            x0 = x1 - sbLen
            dc.DrawLine(x0, y1, x1, y1)
            
            dc.SetTextForeground(wx.TheColourDatabase.FindColour('WHITE'))
            if self.length_nm > 1000:
                s = u'%1.1f \u00B5m' % (self.length_nm / 1000.)
            else:
                s = u'%d nm' % int(self.length_nm)
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, x0 + (sbLen - w)/2, y1 + 7)

class CrosshairsOverlay(Overlay):
    def __call__(self, vp, dc):
        if self.visible:
            sX, sY = vp.imagepanel.Size
            
            dc.SetPen(wx.Pen(wx.CYAN,1))
            if(vp.do.slice == vp.do.SLICE_XY):
                lx = vp.do.xp
                ly = vp.do.yp
            elif(vp.do.slice == vp.do.SLICE_XZ):
                lx = vp.do.xp
                ly = vp.do.zp
            elif(vp.do.slice == vp.do.SLICE_YZ):
                lx = vp.do.yp
                ly = vp.do.zp
        
            
            xc, yc = vp.pixel_to_screen_coordinates(lx, ly)            
            dc.DrawLine(0, yc, sX, yc)
            dc.DrawLine(xc, 0, xc, sY)
            
            dc.SetPen(wx.NullPen)

class FunctionOverlay(Overlay):
    """
    Class to permit backwards compatible use of overlay functions
    """

    def __init__(self, fcn, display_name):
        self._fcn = fcn
        Overlay.__init__(self, display_name=display_name)

    def __call__(self, vp, dc):
        if self.visible:
            self._fcn(vp, dc)
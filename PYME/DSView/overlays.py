# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:28:03 2015

@author: david
"""
import wx
import numpy as np
from PYME.DSView.displayOptions import DisplayOpts

SLICE_AXIS_LUT = {DisplayOpts.SLICE_XY:2, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:0}
TOL_AXIS_LUT = {DisplayOpts.SLICE_XY:0, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:2}
class PointDisplayOverlay(object):
    def __init__(self, points = [], filter=None):
        self.show = True

        self.points = points
        self.pointColours = []
        self.pointSize = 11
        self.pointMode = 'confoc'
        self.pointTolNFoc = {'confoc' : (5,5,5), 'lm' : (2, 5, 5), 'splitter' : (2,5,5)}
        self.showAdjacentPoints = False

        if filter:
            self.filter = filter

    
    def __call__(self, vp, dc):
        dx = 0
        dy = 0
        
        aN = SLICE_AXIS_LUT[vp.do.slice]
        tolN = TOL_AXIS_LUT[vp.do.slice]
        pos = [vp.do.xp, vp.do.yp, vp.do.zp]

        vx, vy = vp.voxelsize[:2]

        if self.show and ('filter' in dir(self) or len(self.points) > 0):
            print('plotting points')
            if 'filter' in dir(self):
                t = self.filter['t'] #prob safe as int
                x = self.filter['x']/vx
                y = self.filter['y']/vy
                
                xb, yb, zb = vp._calcVisibleBounds()
                
                IFoc = (x >= xb[0])*(y >= yb[0])*(t >= zb[0])*(x < xb[1])*(y < yb[1])*(t < zb[1])
                    
                pFoc = np.vstack((x[IFoc], y[IFoc], t[IFoc])).T
                if self.pointMode == 'splitter':
                    f_keys = self.filter.keys()
                    if 'gFrac' in f_keys:
                        pCol = self.filter['gFrac'][IFoc] > .5 
                    elif 'ratio' in f_keys:
                        pCol = self.fitResults['ratio'][IFoc] > 0.5
                    else:
                        pCol = self.fitResults['fitResults']['Ag'][IFoc] > self.fitResults['fitResults']['Ar'][IFoc]               
                
                pNFoc = []

            #intrinsic points            
            elif len(self.points) > 0:
                pointTol = self.pointTolNFoc[self.pointMode]
                
                IFoc = abs(self.points[:,aN] - pos[aN]) < 1
                INFoc = abs(self.points[:,aN] - pos[aN]) < pointTol[tolN]
                    
                pFoc = self.points[IFoc]

                print(pFoc)

                pNFoc = self.points[INFoc]
                
                if self.pointMode == 'splitter':
                    pCol = self.pointColours[IFoc]
                    
            if self.pointMode == 'splitter':
                if 'chroma' in dir(self):
                    vx_nm, vy_nm = 1e3*vx, 1e3*vy

                    dx = self.chroma.dx.ev(pFoc[:,0]*vx_nm, pFoc[:,1]*vy_nm)/(vx_nm)
                    dy = self.chroma.dy.ev(pFoc[:,0]*vx_nm, pFoc[:,1]*vy_nm)/(vx_nm)

                    dxn = self.chroma.dx.ev(pNFoc[:,0]*vx_nm, pNFoc[:,1]*vy_nm)/(vx_nm)
                    dyn = self.chroma.dy.ev(pNFoc[:,0]*vx_nm, pNFoc[:,1]*vy_nm)/(vx_nm)
                else:
                    dx = 0*pFoc[:,0]
                    dy = 0*pFoc[:,0]

                    dxn = 0*pNFoc[:,0]
                    dyn = 0*pNFoc[:,0]
                    

            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            ps = self.pointSize

            if self.showAdjacentPoints:
                dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('BLUE'),1))
                
                if self.pointMode == 'splitter':
                    for p, dxi, dyi in zip(pNFoc, dxn, dyn):
                        vp._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)
                        vp._drawBoxPixelCoords(dc, p[0]-dxi, 0.5*vp.do.ds.shape[1] + p[1]-dyi, p[2], ps, ps, ps)

                else:
                    for p in pNFoc:
                        vp._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)


            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('GREEN'),1)
            pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            dc.SetPen(pGreen)
            
            if self.pointMode == 'splitter':
                for p, c, dxi, dyi in zip(pFoc, pCol, dx, dy):
                    if c:
                        dc.SetPen(pGreen)
                    else:
                        dc.SetPen(pRed)
                        
                    vp._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)
                    vp._drawBoxPixelCoords(dc, p[0]-dxi, 0.5*vp.do.ds.shape[1] + p[1]-dyi, p[2], ps, ps, ps)
                    
            else:
                for p in pFoc:
                    vp._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)
            
            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)
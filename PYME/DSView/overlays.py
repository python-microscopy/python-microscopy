# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:28:03 2015

@author: david
"""
import wx
import numpy as np

class PSFROIs(object):
    def __call__(self, vp, dc):
        if (len(self.psfROIs) > 0):
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('GREEN'),1))
            if(self.do.slice == self.do.SLICE_XY):
                for p in self.psfROIs:
                    dc.DrawRectangle(sc*p[0]-self.psfROISize[0]*sc - x0,sc*p[1] - self.psfROISize[1]*sc - y0, 2*self.psfROISize[0]*sc,2*self.psfROISize[1]*sc)
            elif(self.do.slice == self.do.SLICE_XZ):
                for p in self.psfROIs:
                    dc.DrawRectangle(sc*p[0]-self.psfROISize[0]*sc - x0,sc*p[2]*self.aspect - self.psfROISize[2]*sc*self.aspect - y0, 2*self.psfROISize[0]*sc,2*self.psfROISize[2]*sc*self.aspect)
            elif(self.do.slice == self.do.SLICE_YZ):
                for p in self.psfROIs:
                    dc.DrawRectangle(sc*p[1]-self.psfROISize[1]*sc - x0,sc*p[2]*self.aspect - self.psfROISize[2]*sc*self.aspect - y0, 2*self.psfROISize[1]*sc,2*self.psfROISize[2]*sc*self.aspect)

class TrackDisplay(object):
    def __call__(self, vp, dc):
        if self.showTracks and 'filter' in dir(self) and 'clumpIndex' in self.filter.keys():
            if(self.do.slice == self.do.SLICE_XY):
                IFoc = (abs(self.filter['t'] - vp.do.zp) < 1)
                               
            elif(self.do.slice == self.do.SLICE_XZ):
                IFoc = (abs(self.filter['y'] - vp.do.yp*self.vox_y) < 3*vp.vox_y)*(self.filter['t'] > y0/sc)*(self.filter['t'] < (y0 +sY)/sc)      

            else:#(self.do.slice == self.do.SLICE_YZ):
                IFoc = (abs(self.filter['x'] - vp.do.xp*self.vox_x) < 3*vp.vox_x)*(self.filter['t'] > y0/sc)*(self.filter['t'] < (y0 +sY)/sc)

            tFoc = list(set(self.filter['clumpIndex'][IFoc]))

            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            #pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            dc.SetPen(pGreen)

            for tN in tFoc:
                IFoc = (self.filter['clumpIndex'] == tN)
                if(self.do.slice == self.do.SLICE_XY):
                    pFoc = np.vstack((sc*self.filter['x'][IFoc]/self.vox_x - x0, sc*self.filter['y'][IFoc]/self.vox_y - y0)).T

                elif(self.do.slice == self.do.SLICE_XZ):
                    pFoc = np.vstack((sc*self.filter['x'][IFoc]/self.vox_x - x0, sc*self.filter['t'][IFoc] - y0)).T

                else:#(self.do.slice == self.do.SLICE_YZ):
                    pFoc = np.vstack((sc*self.filter['y'][IFoc]/self.vox_y - y0, sc*self.filter['t'][IFoc] - y0)).T

                dc.DrawLines(pFoc)


        dx = 0
        dy = 0

class PointDisplay(object):
    def __call__(self, vp, dc):
        if self.showPoints and ('filter' in dir(self) or len(self.points) > 0):
            if 'filter' in dir(self):
                #pointTol = self.pointTolNFoc[self.pointMode]

                if(self.do.slice == self.do.SLICE_XY):
                    IFoc = (abs(self.filter['t'] - self.do.zp) < 1)
                    pFoc = np.vstack((self.filter['x'][IFoc]/vp.vox_x, self.filter['y'][IFoc]/vp.vox_y)).T
                    if self.pointMode == 'splitter':
                        pCol = self.filter['gFrac'] > .5

                        if 'chroma' in dir(self):
                            dx = self.chroma.dx.ev(self.filter['x'][IFoc], self.filter['y'][IFoc])/vp.vox_x
                            dy = self.chroma.dy.ev(self.filter['x'][IFoc], self.filter['y'][IFoc])/vp.vox_y
                        else:
                            dx = 0*pFoc[:,0]
                            dy = 0*pFoc[:,0]
                            

                elif(self.do.slice == self.do.SLICE_XZ):
                    IFoc = (abs(self.filter['y'] - self.do.yp*vp.vox_y) < 3*vp.vox_y)*(self.filter['t'] > y0/sc)*(self.filter['t'] < (y0 +sY)/sc)
                    pFoc = np.vstack((self.filter['x'][IFoc]/vp.vox_x, self.filter['t'][IFoc])).T

                else:#(self.do.slice == self.do.SLICE_YZ):
                    IFoc = (abs(self.filter['x'] - vp.do.xp*vp.vox_x) < 3*vp.vox_x)*(self.filter['t'] > y0/sc)*(self.filter['t'] < (y0 +sY)/sc)
                    pFoc = np.vstack((self.filter['y'][IFoc]/vp.vox_y, self.filter['t'][IFoc])).T

                #pFoc = numpy.vstack((self.filter['x'][IFoc]/self.vox_x, self.filter['y'][IFoc]/self.vox_y, self.filter['t'][IFoc])).T
                pNFoc = []

            elif len(self.points) > 0 and self.showPoints:
                #if self.pointsMode == 'confoc':
                pointTol = self.pointTolNFoc[self.pointMode]
                if(self.do.slice == self.do.SLICE_XY):
                    pFoc = self.points[abs(self.points[:,2] - self.do.zp) < 1][:,:2]
                    if self.pointMode == 'splitter':
                        pCol = self.pointColours[abs(self.points[:,2] - self.do.zp) < 1]
                        
                        if 'chroma' in dir(self):
                            dx = self.chroma.dx.ev(pFoc[:,0]*1e3*self.vox_x, pFoc[:,1]*1e3*self.vox_y)/(1e3*self.vox_x)
                            dy = self.chroma.dy.ev(pFoc[:,0]*1e3*self.vox_x, pFoc[:,1]*1e3*self.vox_y)/(1e3*self.vox_y)
                        else:
                            dx = 0*pFoc[:,0]
                            dy = 0*pFoc[:,0]

                    pNFoc = self.points[abs(self.points[:,2] - self.do.zp) < pointTol[0]][:,:2]
                    if self.pointMode == 'splitter':
                        if 'chroma' in dir(self):
                            dxn = self.chroma.dx.ev(pFoc[:,0]*1e3*self.vox_x, pFoc[:,1]*1e3*self.vox_y)/(1e3*self.vox_x)
                            dyn = self.chroma.dy.ev(pFoc[:,0]*1e3*self.vox_x, pFoc[:,1]*1e3*self.vox_y)/(1e3*self.vox_y)
                        else:
                            dxn = 0*pFoc[:,0]
                            dyn = 0*pFoc[:,0]

                elif(self.do.slice == self.do.SLICE_XZ):
                    pFoc = self.points[abs(self.points[:,1] - self.do.yp) < 1][:, ::2]
                    pNFoc = self.points[abs(self.points[:,1] - self.do.yp) < pointTol[1]][:,::2]

                else:#(self.do.slice == self.do.SLICE_YZ):
                    pFoc = self.points[abs(self.points[:,0] - self.do.xp) < 1][:, 1:]
                    pNFoc = self.points[abs(self.points[:,0] - self.do.xp) < pointTol[2]][:,1:]




            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            ps = self.pointSize
            ps2 = ps/2

            if self.showAdjacentPoints:
                dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('BLUE'),1))
                
                if self.pointMode == 'splitter' and self.do.slice == self.do.SLICE_XY:
                    for p, dxi, dyi in zip(pNFoc, dxn, dyn):
                        px, py = self._PixelToScreenCoordinates(p[0] - ps2, p[1] - ps2)
                        dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)
                        px, py = self._PixelToScreenCoordinates(p[0] -dxi - ps2, self.do.ds.shape[1] - p[1] + dyi - ps2)
                        dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)

                else:
                    for p in pNFoc:
                        px, py = self._PixelToScreenCoordinates(p[0] - ps2, p[1] - ps2)
                        dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)


            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('GREEN'),1)
            pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            dc.SetPen(pGreen)
            
            if self.pointMode == 'splitter' and self.do.slice == self.do.SLICE_XY:
                for p, c, dxi, dyi in zip(pFoc, pCol, dx, dy):
                    if c:
                        dc.SetPen(pGreen)
                    else:
                        dc.SetPen(pRed)
                    px, py = self._PixelToScreenCoordinates(p[0] - ps2, p[1] - ps2)
                    dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)
                    px, py = self._PixelToScreenCoordinates(p[0] -dxi - ps2, self.do.ds.shape[1] - p[1] + dyi - ps2)
                    dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)
                    
            else:
                for p in pFoc:
                    px, py = self._PixelToScreenCoordinates(p[0] - ps2, p[1] - ps2)
                    dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)

            
            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)
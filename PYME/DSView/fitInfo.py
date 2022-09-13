#!/usr/bin/python

##################
# fitInfo.py
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

import wx
import math
# import pylab
from matplotlib import cm
import numpy as np

from PYME.contrib import wxPlotPanel
#from PYME.IO.MetaDataHandler import NestedClassMDHandler

import logging
logger = logging.getLogger(__name__)

class FitInfoPanel(wx.Panel):
    def __init__(self, parent, fitResults, mdh, ds=None, id=-1):
        wx.Panel.__init__(self, id=id, parent=parent)

        self.fitResults = fitResults
        self.mdh = mdh

        vsizer = wx.BoxSizer(wx.VERTICAL)
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.stSliceNum = wx.StaticText(self, -1, 'No event selected')

        vsizer.Add(self.stSliceNum, 0, wx.LEFT|wx.TOP|wx.BOTTOM, 5)

        sFitRes = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Fit Results'), wx.VERTICAL)

        self.stFitRes = wx.StaticText(self, -1, self.genResultsText(None))
        self.stFitRes.SetFont(wx.Font(10, wx.MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sFitRes.Add(self.stFitRes, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        vsizer.Add(sFitRes, 0, wx.EXPAND|wx.LEFT|wx.TOP|wx.BOTTOM|wx.RIGHT, 5)

        if self.mdh.getEntry('Analysis.FitModule') == 'LatGaussFitFR':
            #we know what the fit parameters are, and how to convert to photons
            tPhotons = self.genGaussPhotonStats(None)
        else:
            tPhotons = ''

        sPhotons = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Photon Stats'), wx.VERTICAL)

        self.stPhotons = wx.StaticText(self, -1, tPhotons)
        self.stPhotons.SetFont(wx.Font(10, wx.MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sPhotons.Add(self.stPhotons, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        vsizer.Add(sPhotons, 0, wx.EXPAND|wx.LEFT|wx.TOP|wx.BOTTOM|wx.RIGHT, 5)

        self.fitViewPan = fitDispPanel(self, fitResults, mdh, ds, size=(300, 700))
        vsizer.Add(self.fitViewPan, 1, wx.EXPAND|wx.ALL, 5)


        self.SetSizerAndFit(vsizer)
        
    def SetResults(self, results, mdh):
        self.fitResults = results
        self.mdh = mdh
        self.fitViewPan.SetFitResults(results, mdh)
        logger.debug('running SetResults')
        

    def genResultsText(self, index):
        s =  u''
        ns = self.fitResults['fitResults'].dtype.names

        nl = max([len(n) for n in ns])

        #print nl

        if not index is None:
            index = int(index)
            r = self.fitResults[index]
            #print r

            for n in ns:
                #\u00B1 is the plus-minus sign
                if 'fitError' in r.dtype.names:
                    s += u'%s %8.2f \u00B1 %3.2f\n' % ((n + ':').ljust(nl+1), r['fitResults'][n], r['fitError'][n])
                else:
                    s += u'%s %8.2f\n' % ((n + ':').ljust(nl+1), r['fitResults'][n])

            #s = s[:-1]
            if 'resultCode' in r.dtype.names:
                s += '\nresultCode: %d' % r['resultCode']
                
            if 'Ag' in r['fitResults'].dtype.names:
                rf = r['fitResults']
                s += '\n\ngFrac: %3.2f' % (rf['Ag']/(rf['Ag'] + rf['Ar']))
            
            if 'startParams' in r.dtype.names:
                s += '\n\nStart Params:\n%s' % str(r['startParams'])
            if 'nchi2' in r.dtype.names:
                s += u'\n\u03A7\u00B2/\u03BD: %3.2f' % r['nchi2']
        else:    
            for n in ns:
                s += u'%s:\n' % (n)
                
        return s

    def genGaussPhotonStats(self, index):
        s =  u''

        if not index is None:
            r = self.fitResults[index]['fitResults']

            nPh = (r['A']*2*math.pi*(r['sigma']/(self.mdh.voxelsize_nm.x))**2)
            nPh = nPh*self.mdh.getEntry('Camera.ElectronsPerCount')/self.mdh.getEntry('Camera.TrueEMGain')

            bPh = r['background']
            bPh = bPh*self.mdh.getEntry('Camera.ElectronsPerCount')/self.mdh.getEntry('Camera.TrueEMGain')

            ron = self.mdh.getEntry('Camera.ReadNoise')/self.mdh.getEntry('Camera.TrueEMGain')

            s += 'Number of photons: %3.2f' %nPh

            deltaX = (r['sigma']**2 + ((self.mdh.voxelsize_nm.x)**2)/12)/nPh + 8*math.pi*(r['sigma']**4)*(bPh + ron**2)/(nPh*self.mdh.voxelsize_nm.x)**2

            s += '\nPredicted accuracy: %3.2f' % math.sqrt(deltaX)
        else:
            s += 'Number of photons:\nPredicted accuracy'

        return s



    def UpdateDisp(self, index):
        slN = 'No event selected'

        if not index is None:
            slN = 'Point #: %d    Slice: %d' % (index, self.fitResults['tIndex'][index])

        self.stSliceNum.SetLabel(slN)

        self.stFitRes.SetLabel(self.genResultsText(index))
        if self.mdh.getEntry('Analysis.FitModule') in ['LatGaussFitFR','LatGaussFitFRforZyla']:
            self.stPhotons.SetLabel(self.genGaussPhotonStats(index))

        self.fitViewPan.draw(index)
        logger.debug('running UpdateDisp')
        
    def DrawOverlays(self, vp, dc):
        do = vp.do

        if do.ds.shape[2] > 1:
            # stack has z
            zp = do.zp
        else:
            # stack is a time series
            zp = do.tp
        
        frameResults = self.fitResults[self.fitResults['tIndex'] == zp]
        
        vx, vy, _ = self.mdh.voxelsize_nm
        
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        
        pGreen = wx.Pen(wx.TheColourDatabase.FindColour('ORANGE'),1)
        pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
        dc.SetPen(pGreen)
        
        if False:#self.pointMode == 'splitter' and self.do.slice == self.do.SLICE_XY:
            for p, c, dxi, dyi in zip(pFoc, pCol, dx, dy):
                if c:
                    dc.SetPen(pGreen)
                else:
                    dc.SetPen(pRed)
                px, py = vp.pixel_to_screen_coordinates(p[0] - ps2, p[1] - ps2)
                dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)
                px, py = vp.pixel_to_screen_coordinates(p[0] -dxi - ps2, self.do.ds.shape[1] - p[1] + dyi - ps2)
                dc.DrawRectangle(px,py, ps*sc,ps*sc*self.aspect)
                
        else:
            for res in frameResults:
                #print res
                #print res['slicesUsed']                
                
                #draw ROI
                x0 = res['slicesUsed']['x']['start']
                x1 = res['slicesUsed']['x']['stop']
                
                y0 = res['slicesUsed']['y']['start']
                y1 = res['slicesUsed']['y']['stop']
                
                px0, py0 = vp.pixel_to_screen_coordinates(x0, y0)
                px1, py1 = vp.pixel_to_screen_coordinates(x1, y1)
                
                dc.DrawRectangle(px0,py0, px1 - px0,py1 - py0)
                
                #draw start pos
                xs = res['startParams']['x0']/vx
                ys = res['startParams']['y0']/vy
                
                pxs, pys = vp.pixel_to_screen_coordinates(xs, ys)
                
                #print xs, ys, pxs, pys
                
                dc.DrawLine(pxs-3, pys, pxs+3, pys)
                dc.DrawLine(pxs, pys-3, pxs, pys+3)
                
                #draw fitted position
                xs = res['fitResults']['x0']/vx
                ys = res['fitResults']['y0']/vy
                
                pxs, pys = vp.pixel_to_screen_coordinates(xs, ys)
                
                dc.DrawLine(pxs-3, pys-3, pxs+3, pys+3)
                dc.DrawLine(pxs-3, pys+3, pxs+3, pys-3)
        logger.debug('running DrawOverlays')


class fitDispPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, fitResults, mdh, ds, **kwargs ):
        self.fitResults = fitResults
        self.mdh = mdh
        self.ds = ds      

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )
        
    def SetFitResults(self, fitResults, mdh):
        self.fitResults = fitResults
        self.mdh = mdh
        
    def _extractROI(self, fri):
        from PYME.IO.MetaDataHandler import get_camera_roi_origin
        roi_x0, roi_y0 = get_camera_roi_origin(self.mdh)

        if (self.ds.shape[2] > 1):
            # TODO - revisit for proper 3D fits
            logger.warning('z and t dimensions potentially incorrectly assigned, working around and assuming z dimension really t') 
            zi = int(fri['tIndex'])
            ti = 0
            ci = 0 # we only support monochrome at present, but pull it out here so we can alter in one place if needed
        else:
            # t=t
            zi = 0
            ti = int(fri['tIndex'])
            ci = 0 # we only support monochrome at present, but pull it out here so we can alter in one place if needed
        
        if 'Splitter' in self.mdh['Analysis.FitModule']:
             # is a splitter fit
            if 'Splitter.Channel0ROI' in self.mdh.getEntryNames():
                x0, y0, w, h = self.mdh['Splitter.Channel0ROI']
                
                x0 -= roi_x0
                y0 -= roi_y0
                #g = self.data[x0:(x0+w), y0:(y0+h)]
                x1, y1, w1, h1 = self.mdh['Splitter.Channel1ROI']
                x1 -= roi_x0
                y1 -= roi_y0
                #r = self.data[x0:(x0+w), y0:(y0+h)]
                
                h_ = min(y0 + h - max(y0, 0), min(y1 + h1, self.ds.shape[1]) - y1)
                
            else:
                x0, y0 = 0,0
                x1, y1 = 0, (self.mdh['Camera.ROIHeight'] + 1)/2
                h = y1
                h_ = h
                
            slux = fri['slicesUsed']['x']
            sluy = fri['slicesUsed']['y']
            
            slx = slice(slux[0], slux[1])
            sly = slice(sluy[0], sluy[1])
             
            if False: #self.mdh.get('Splitter.Flip', True):
                sly = slice(sluy[0] + h - h_, sluy[1] + h - h_)
            
            #sx0 = slice(x0+ slux[0], x0+slux[1])
            #sy0 = slice(y0+ sluy[0], y0+sluy[1])

            
            if 'NR' in self.mdh['Analysis.FitModule']:
                #for fits which take chromatic shift into account when selecting ROIs
                #pixel size in nm
                vx, vy, _ = self.mdh.voxelsize_nm
                
                #position in nm from camera origin
                x_ = ((slux[0] + slux[1])/2. + roi_x0)*vx
                y_ = ((sluy[0] + sluy[1])/2. + roi_y0)*vy
                
                #look up shifts
                if not self.mdh.getOrDefault('Analysis.FitShifts', False):
                    DeltaX = self.mdh['chroma.dx'].ev(x_, y_)
                    DeltaY = self.mdh['chroma.dy'].ev(x_, y_)
                else:
                    DeltaX = 0
                    DeltaY = 0
                
                #find shift in whole pixels
                dxp = int(DeltaX/vx)
                dyp = int(DeltaY/vy)
                
                print((DeltaX, DeltaY, dxp, dyp))
                
                x1 -= dxp
                y1 -= dyp
            
            sx1 = slice(x1 - x0 + slux[0], x1 - x0 + slux[1])
            
            if ('Splitter.Flip' in self.mdh.getEntryNames() and not self.mdh.getEntry('Splitter.Flip')):
                sy1 = slice(y1 - y0 + sluy[0], y1 - y0 +sluy[1])
            else:
                sy1 = slice(y1 + h + y0 - sluy[0], y1 + h + y0 -sluy[1], -1) #FIXME
                
            print((slx, sx1, sly, sy1))
            print(h, y0, y1, sluy)
                
            g = self.ds[slx, sly, zi, ti, ci].squeeze()
            r = self.ds[sx1, sy1, zi, ti, ci].squeeze()
                
            return np.hstack([g,r])  - self.mdh.get('Camera.ADOffset', 0)
        else:
            return self.ds[slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y']), zi, ti, ci].squeeze()  - self.mdh.get('Camera.ADOffset', 0)

    def _extractROI_1(self, fri):
        from PYME.IO.MetaDataHandler import get_camera_roi_origin
        roi_x0, roi_y0 = get_camera_roi_origin(self.mdh)

        if 'Splitter' in self.mdh['Analysis.FitModule']:
            # is a splitter fit
            if 'Splitter.Channel0ROI' in self.mdh.getEntryNames():
                x0, y0, w, h = self.mdh['Splitter.Channel0ROI']

                x0 -= roi_x0
                y0 -= roi_y0
                # g = self.data[x0:(x0+w), y0:(y0+h)]
                x1, y1, w, h = self.mdh['Splitter.Channel1ROI']
                x1 -= roi_x0
                y1 -= roi_y0
                # r = self.data[x0:(x0+w), y0:(y0+h)]
            else:
                x0, y0 = 0, 0
                x1, y1 = 0, (self.mdh['Camera.ROIHeight'] + 1) / 2
                h = y1

            slux = fri['slicesUsed']['x']
            sluy = fri['slicesUsed']['y']

            slx = slice(slux[0], slux[1])
            sly = slice(sluy[0], sluy[1])

            # sx0 = slice(x0+ slux[0], x0+slux[1])
            # sy0 = slice(y0+ sluy[0], y0+sluy[1])

            if 'NR' in self.mdh['Analysis.FitModule']:
                # for fits which take chromatic shift into account when selecting ROIs
                # pixel size in nm
                vx, vy, _ = self.mdh.voxelsize_nm
                
                # position in nm from camera origin
                x_ = ((slux[0] + slux[1]) / 2. + roi_x0) * vx
                y_ = ((sluy[0] + sluy[1]) / 2. + roi_y0) * vy

                # look up shifts
                if not self.mdh.getOrDefault('Analysis.FitShifts', False):
                    DeltaX = self.mdh.chroma.dx.ev(x_, y_)
                    DeltaY = self.mdh.chroma.dy.ev(x_, y_)
                else:
                    DeltaX = 0
                    DeltaY = 0

                # find shift in whole pixels
                dxp = int(DeltaX / vx)
                dyp = int(DeltaY / vy)

                print((DeltaX, DeltaY, dxp, dyp))

                x1 -= dxp
                y1 -= dyp

            sx1 = slice(x1 - x0 + slux[0], x1 - x0 + slux[1])

            if ('Splitter.Flip' in self.mdh.getEntryNames() and not self.mdh.getEntry('Splitter.Flip')):
                sy1 = slice(y1 - y0 + sluy[0], y1 - y0 + sluy[1])
            else:
                sy1 = slice(y1 + h + y0 - sluy[0], y1 + h + y0 - sluy[1], -1)  # FIXME

            print((slx, sx1, sly, sy1))
            print(h, y0, y1, sluy)

            g = self.ds[slx, sly, int(fri['tIndex'])].squeeze()
            r = self.ds[sx1, sy1, int(fri['tIndex'])].squeeze()

            return np.hstack([g, r])
        else:
            return self.ds[slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y']), int(fri['tIndex'])].squeeze()

    def draw( self, i = None):
            """Draw data."""
            if len(self.fitResults) == 0:
                return

            if not hasattr( self, 'subplot1' ):
                self.subplot1 = self.figure.add_subplot( 511 )
                self.subplot2 = self.figure.add_subplot( 512 )
                self.subplot3 = self.figure.add_subplot( 513 )
                self.subplot4 = self.figure.add_subplot( 514 )
                self.subplot5 = self.figure.add_subplot( 515 )

#            a, ed = numpy.histogram(self.fitResults['tIndex'], self.Size[0]/2)
#            print float(numpy.diff(ed[:2]))

            
#            self.subplot1.plot(ed[:-1], a/float(numpy.diff(ed[:2])), color='b' )
#            self.subplot1.set_xticks([0, ed.max()])
#            self.subplot1.set_yticks([0, numpy.floor(a.max()/float(numpy.diff(ed[:2])))])
            if not i is None:
                # only clear the panels if we have been called with a valid fitResults index
                self.subplot1.cla()
                self.subplot2.cla()
                self.subplot3.cla()
                self.subplot4.cla()
                self.subplot5.cla()
                logger.debug('in draw: clearing panels')
            
                fri = self.fitResults[i]
                #print fri
                #print fri['tIndex'], slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y'])
                #print self.ds[slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y']), int(fri['tIndex'])].shape
                #imd = self.ds[slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y']), int(fri['tIndex'])].squeeze()
                imd = self._extractROI(fri)

                self.subplot1.imshow(imd, interpolation='nearest', cmap=cm.hot)
                self.subplot1.set_title('Data')
                logger.debug('in draw: showing ROI image')
                
                logger.debug('in draw: importing fitMod')
                fitMod = __import__('PYME.localization.FitFactories.' + self.mdh.getEntry('Analysis.FitModule'), fromlist=['PYME', 'localization', 'FitFactories']) #import our fitting module
                logger.debug('in draw: imported fitMod')

                if 'genFitImage' in dir(fitMod):
                    imf = fitMod.genFitImage(fri, self.mdh).squeeze()

                    self.subplot2.imshow(imf, interpolation='nearest', cmap=cm.hot)
                    self.subplot2.set_title('Fit')
                    self.subplot3.imshow(imd - imf, interpolation='nearest', cmap=cm.hot)
                    self.subplot3.set_title('Residuals')
                    self.subplot4.plot(imd.sum(0))
                    self.subplot4.plot(imf.sum(0))
                    # required for indexing: changed to explicit integer division below
                    self.subplot5.plot(np.hstack([imd[:,:(imd.shape[1]//2)].sum(1), imd[:,(imd.shape[1]//2):].sum(1)]))
                    self.subplot5.plot(np.hstack([imf[:,:(imd.shape[1]//2)].sum(1), imf[:,(imd.shape[1]//2):].sum(1)]))
                    #self.subplot5.plot(imf.sum(1))
                    logger.debug('in draw: plotting curves')
                else:
                    logger.debug('in draw: fitModel has no genFitImage function')
            else:
                logger.debug('running draw with i as none')

            
#            self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g' )
#            self.subplot2.set_xticks([0, ed.max()])
#            self.subplot2.set_yticks([0, a.sum()])

            self.canvas.draw()
            if not i is None:
                logger.debug('running draw, i = %d' % i)
            else:
                logger.debug('running draw in NONE mode!!!')

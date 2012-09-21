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
import pylab
#import numpy

from PYME.misc import wxPlotPanel
#from PYME.Acquire.MetaDataHandler import NestedClassMDHandler

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

            sPhotons = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Photon Stats'), wx.VERTICAL)

            self.stPhotons = wx.StaticText(self, -1, self.genGaussPhotonStats(None))
            self.stPhotons.SetFont(wx.Font(10, wx.MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            sPhotons.Add(self.stPhotons, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

            vsizer.Add(sPhotons, 0, wx.EXPAND|wx.LEFT|wx.TOP|wx.BOTTOM|wx.RIGHT, 5)

        self.fitViewPan = fitDispPanel(self, fitResults, mdh, ds, size=(300, 700))
        vsizer.Add(self.fitViewPan, 1, wx.EXPAND|wx.ALL, 5)


        self.SetSizerAndFit(vsizer)

    def genResultsText(self, index):
        s =  u''
        ns = self.fitResults['fitResults'].dtype.names

        nl = max([len(n) for n in ns])

        #print nl

        if index:
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

        if not index == None:
            r = self.fitResults[index]['fitResults']

            nPh = (r['A']*2*math.pi*(r['sigma']/(1e3*self.mdh.getEntry('voxelsize.x')))**2)
            nPh = nPh*self.mdh.getEntry('Camera.ElectronsPerCount')/self.mdh.getEntry('Camera.TrueEMGain')

            bPh = r['background']
            bPh = bPh*self.mdh.getEntry('Camera.ElectronsPerCount')/self.mdh.getEntry('Camera.TrueEMGain')

            ron = self.mdh.getEntry('Camera.ReadNoise')/self.mdh.getEntry('Camera.TrueEMGain')

            s += 'Number of photons: %3.2f' %nPh

            deltaX = (r['sigma']**2 + ((1e3*self.mdh.getEntry('voxelsize.x'))**2)/12)/nPh + 8*math.pi*(r['sigma']**4)*(bPh + ron**2)/(nPh*1e3*self.mdh.getEntry('voxelsize.x'))**2

            s += '\nPredicted accuracy: %3.2f' % math.sqrt(deltaX)
        else:
            s += 'Number of photons:\nPredicted accuracy'

        return s



    def UpdateDisp(self, index):
        slN = 'No event selected'

        if not index == None:
            slN = 'Point #: %d    Slice: %d' % (index, self.fitResults['tIndex'][index])

        self.stSliceNum.SetLabel(slN)

        self.stFitRes.SetLabel(self.genResultsText(index))
        if self.mdh.getEntry('Analysis.FitModule') == 'LatGaussFitFR':
            self.stPhotons.SetLabel(self.genGaussPhotonStats(index))

        self.fitViewPan.draw(index)


class fitDispPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, fitResults, mdh, ds, **kwargs ):
        self.fitResults = fitResults
        self.mdh = mdh
        self.ds = ds

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )

    def draw( self, i = None):
            """Draw data."""
            if len(self.fitResults) == 0:
                return

            if not hasattr( self, 'subplot1' ):
                self.subplot1 = self.figure.add_subplot( 311 )
                self.subplot2 = self.figure.add_subplot( 312 )
                self.subplot3 = self.figure.add_subplot( 313 )

#            a, ed = numpy.histogram(self.fitResults['tIndex'], self.Size[0]/2)
#            print float(numpy.diff(ed[:2]))

            self.subplot1.cla()
            self.subplot2.cla()
            self.subplot3.cla()
#            self.subplot1.plot(ed[:-1], a/float(numpy.diff(ed[:2])), color='b' )
#            self.subplot1.set_xticks([0, ed.max()])
#            self.subplot1.set_yticks([0, numpy.floor(a.max()/float(numpy.diff(ed[:2])))])
            if i:
                fri = self.fitResults[i]
                #print fri
                #print fri['tIndex'], slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y'])
                #print self.ds[slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y']), int(fri['tIndex'])].shape
                imd = self.ds[slice(*fri['slicesUsed']['x']), slice(*fri['slicesUsed']['y']), int(fri['tIndex'])].squeeze()

                self.subplot1.imshow(imd, interpolation='nearest', cmap=pylab.cm.hot)
                self.subplot1.set_title('Data')

                fitMod = __import__('PYME.Analysis.FitFactories.' + self.mdh.getEntry('Analysis.FitModule'), fromlist=['PYME', 'Analysis','FitFactories']) #import our fitting module
                #print dir()

                if 'genFitImage' in dir(fitMod):
                    imf = fitMod.genFitImage(fri, self.mdh).squeeze()

                    self.subplot2.imshow(imf, interpolation='nearest', cmap=pylab.cm.hot)
                    self.subplot2.set_title('Fit')
                    self.subplot3.imshow(imd - imf, interpolation='nearest', cmap=pylab.cm.hot)
                    self.subplot3.set_title('Residuals')



            
#            self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g' )
#            self.subplot2.set_xticks([0, ed.max()])
#            self.subplot2.set_yticks([0, a.sum()])

            self.canvas.draw()
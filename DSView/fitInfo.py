#!/usr/bin/python

##################
# fitInfo.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import math
#import numpy

class FitInfoPanel(wx.Panel):
    def __init__(self, parent, fitResults, mdh, id=-1):
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


        self.SetSizerAndFit(vsizer)

    def genResultsText(self, index):
        s =  u''
        ns = self.fitResults['fitResults'].dtype.names

        nl = max([len(n) for n in ns])

        #print nl

        if not index == None:
            r = self.fitResults[index]



            for n in ns:
                #\u00B1 is the plus-minus sign
                s += u'%s %8.2f \u00B1 %3.2f\n' % ((n + ':').ljust(nl+1), r['fitResults'][n], r['fitError'][n])
            s = s[:-1]
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
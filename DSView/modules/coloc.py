#!/usr/bin/python
##################
# coloc.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy
import wx
import pylab

class colocaliser:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        PROC_COLOCALISE = wx.NewId()
        
        
        dsviewer.mProcessing.Append(PROC_COLOCALISE, "&Colocalisation", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, PROC_COLOCALISE, self.OnColoc)



    
    def OnColoc(self, event):
        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc

        #assume we have exactly 2 channels #FIXME - add a selector
        #grab image data
        imA = self.image.data[:,:,:,0].squeeze()
        imB = self.image.data[:,:,:,1].squeeze()

        #assume threshold is half the colour bounds - good if using threshold mode
        tA = self.do.Offs[0] + .5/self.do.Gains[0] #pylab.mean(self.ivps[0].clim)
        tB = self.do.Offs[1] + .5/self.do.Gains[1] #pylab.mean(self.ivps[0].clim)

        try:
            nameA, nameB = self.image.mdh.getEntry('ChannelNames')[:2]
        except:
            nameA = 'Channel 1'
            nameB = 'Channel 2'

        voxelsize = [1e3*self.image.mdh.getEntry('voxelsize.x') ,1e3*self.image.mdh.getEntry('voxelsize.y'), 1e3*self.image.mdh.getEntry('voxelsize.z')]
        voxelsize = voxelsize[:imA.ndim] #trunctate to number of dimensions

        pearson = correlationCoeffs.pearson(imA, imB)
        MA, MB = correlationCoeffs.thresholdedManders(imA, imB, tA, tB)

        bnA, bmA, binsA = edtColoc.imageDensityAtDistance(imB, imA > tA, voxelsize)
        bnB, bmB, binsB = edtColoc.imageDensityAtDistance(imA, imB > tB, voxelsize)

        pylab.figure()
        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
        pylab.subplot(211)
        pylab.bar(binsA[:-1], bmA, binsA[1] - binsA[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameA)
        pylab.ylabel('Density of %s' % nameB)

        pylab.subplot(212)
        pylab.bar(binsB[:-1], bmB, binsB[1] - binsB[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameB)
        pylab.ylabel('Density of %s' % nameA)

        pylab.figure()
        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
        pylab.subplot(211)
        pylab.bar(binsA[:-1], bmA*bnA, binsA[1] - binsA[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameA)
        pylab.ylabel('Fraction of %s' % nameB)

        pylab.subplot(212)
        pylab.bar(binsB[:-1], bmB*bnB, binsB[1] - binsB[0])
        pylab.xlabel('Distance from edge of %s [nm]' % nameB)
        pylab.ylabel('Fraction of %s' % nameA)



def Plug(dsviewer):
    dsviewer.coloc = colocaliser(dsviewer)




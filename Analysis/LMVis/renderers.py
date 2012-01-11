#!/usr/bin/python
##################
# renderers.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from PYME.Analysis.LMVis.visHelpers import ImageBounds#, GeneratedImage
from PYME.DSView.image import GeneratedImage
from PYME.Analysis.LMVis import genImageDialog
from PYME.Analysis.LMVis import visHelpers
#from PYME.Analysis.LMVis import imageView
from PYME.Analysis.LMVis import statusLog

from PYME.Acquire import MetaDataHandler

from PYME.DSView import ViewIm3D

from PYME.Analysis.QuadTree import QTrend

import wx
import pylab
import numpy as np

renderMetadataProviders = []

class CurrentRenderer:
    '''Renders current view (in black and white). Only renderer not to take care
    of colour channels. Simplest renderer and as such also the base class for all 
    others'''

    name = 'Current'
    mode = 'current'
    
    def __init__(self, visFr):
        self.visFr = visFr

        self._addMenuItems()

    def _addMenuItems(self):
        ID = wx.NewId()
        self.visFr.gen_menu.Append(ID, self.name)

        self.visFr.Bind(wx.EVT_MENU, self.Generate, id=ID)

    def _getImBounds(self):
        x0 = max(self.visFr.glCanvas.xmin, self.visFr.imageBounds.x0)
        y0 = max(self.visFr.glCanvas.ymin, self.visFr.imageBounds.y0)
        x1 = min(self.visFr.glCanvas.xmax, self.visFr.imageBounds.x1)
        y1 = min(self.visFr.glCanvas.ymax, self.visFr.imageBounds.y1)

        if 'x' in self.visFr.filterKeys.keys():
            x0 = max(x0, self.visFr.filterKeys['x'][0])
            x1 = min(x1, self.visFr.filterKeys['x'][1])

        if 'y' in self.visFr.filterKeys.keys():
            y0 = max(y0, self.visFr.filterKeys['y'][0])
            y1 = min(y1, self.visFr.filterKeys['y'][1])

        #imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)
        return ImageBounds(x0, y0, x1, y1)

    def _getDefaultJitVar(self, jitVars):
        return jitVars.index('neighbourDistances')

    def _getDefaultZJitVar(self, jitVars):
        if 'fitError_z0' in jitVars:
            return jitVars.index('fitError_z0')
        else:
            return 0

    def _genJitVals(self, jitParamName, jitScale):
        #print jitParamName
        if jitParamName == '1.0':
            jitVals = np.ones(self.visFr.colourFilter['x'].shape)
        elif jitParamName in self.visFr.colourFilter.keys():
            jitVals = self.visFr.colourFilter[jitParamName]
        elif jitParamName in self.genMeas:
            #print 'f'
            if jitParamName == 'neighbourDistances':
                self.visFr.genNeighbourDists(True)
            jitVals = self.visFr.GeneratedMeasures[jitParamName]

        return jitVals*jitScale


    def Generate(self, event=None):
        dlg = genImageDialog.GenImageDialog(self.visFr, mode=self.mode)
        ret = dlg.ShowModal()

        bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            mdh = MetaDataHandler.NestedClassMDHandler()
            mdh['Rendering.Method'] = self.name
            if 'imageID' in self.visFr.mdh.getEntryNames():
                mdh['Rendering.SourceImageID'] = self.visFr.mdh['imageID']
            mdh['Rendering.SourceFilename'] = self.visFr.filename
            
            for cb in renderMetadataProviders:
                cb(mdh)            
            
            pixelSize = dlg.getPixelSize()

            imb = self._getImBounds()

            im = self.genIm(dlg, imb, mdh)
            img = GeneratedImage(im,imb, pixelSize, 0, ['Image'] , mdh = mdh)
            imf = ViewIm3D(img, mode='visGUI', title='Generated %s - %3.1fnm bins' % (self.name, pixelSize), glCanvas=self.visFr.glCanvas, parent=self.visFr)
            #imf = imageView.ImageViewFrame(self.visFr,img, self.visFr.glCanvas)
            #imageView.MultiChannelImageViewFrame(self.visFr, self.visFr.glCanvas, img, title='Generated %s - %3.1fnm bins' % (self.name, pixelSize))
            #self.visFr.generatedImages.append(imf)
            #imf.Show()

            self.visFr.RefreshView()

        dlg.Destroy()
        return imf

    def genIm(self, dlg, imb, mdh):
        oldcmap = self.visFr.glCanvas.cmap
        self.visFr.glCanvas.setCMap(pylab.cm.gray)
        im = self.visFr.glCanvas.getIm(dlg.getPixelSize())

        self.visFr.glCanvas.setCMap(oldcmap)

        return im

class ColourRenderer(CurrentRenderer):
    '''Base class for all other renderers which know about the colour filter'''
    
    def Generate(self, event=None):
        jitVars = ['1.0']
        jitVars += self.visFr.colourFilter.keys()

        self.genMeas = self.visFr.GeneratedMeasures.keys()
        if not 'neighbourDistances' in self.genMeas:
            self.genMeas.append('neighbourDistances')
        jitVars += self.genMeas
        
        if 'z' in self.visFr.mapping.keys():
            zvals = self.visFr.mapping['z']
        else:
            zvals = None

        dlg = genImageDialog.GenImageDialog(self.visFr, mode=self.mode, colours=self.visFr.fluorSpecies.keys(), zvals = zvals, jitterVariables = jitVars, jitterVarDefault=self._getDefaultJitVar(jitVars), jitterVarDefaultZ=self._getDefaultZJitVar(jitVars))
        ret = dlg.ShowModal()

        bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            mdh = MetaDataHandler.NestedClassMDHandler()
            mdh['Rendering.Method'] = self.name
            if 'imageID' in self.visFr.mdh.getEntryNames():
                mdh['Rendering.SourceImageID'] = self.visFr.mdh['imageID']
            mdh['Rendering.SourceFilename'] = self.visFr.filename
            
            for cb in renderMetadataProviders:
                cb(mdh)           
            
            pixelSize = dlg.getPixelSize()

            status = statusLog.StatusLogger('Generating %s Image ...' % self.name)

            imb = self._getImBounds()
            
            colours =  dlg.getColour()
            oldC = self.visFr.colourFilter.currentColour

            ims = []

            for c in  colours:
                self.visFr.colourFilter.setColour(c)
                ims.append(np.atleast_3d(self.genIm(dlg, imb, mdh)))

            im = GeneratedImage(ims,imb, pixelSize,  dlg.getZSliceThickness(), colours, mdh = mdh)

            imfc = ViewIm3D(im, mode='visGUI', title='Generated %s - %3.1fnm bins' % (self.name, pixelSize), glCanvas=self.visFr.glCanvas, parent=self.visFr)

            #imfc = imageView.MultiChannelImageViewFrame(self.visFr, self.visFr.glCanvas, im, title='Generated %s - %3.1fnm bins' % (self.name, pixelSize))

            #self.visFr.generatedImages.append(imfc)
            #imfc.Show()

            self.visFr.colourFilter.setColour(oldC)
        else:
            imfc = None

        dlg.Destroy()
        return imfc


class HistogramRenderer(ColourRenderer):
    '''2D histogram rendering'''

    name = 'Histogram'
    mode = 'histogram'

    def genIm(self, dlg, imb, mdh):
        return visHelpers.rendHist(self.visFr.colourFilter['x'],self.visFr.colourFilter['y'], imb, dlg.getPixelSize())

class Histogram3DRenderer(HistogramRenderer):
    '''3D histogram rendering'''

    name = '3D Histogram'
    mode = '3Dhistogram'

    def genIm(self, dlg, imb, mdh):
        return visHelpers.rendHist3D(self.visFr.colourFilter['x'],self.visFr.colourFilter['y'], self.visFr.colourFilter['z'], imb, dlg.getPixelSize(), dlg.getZBounds(), dlg.getZSliceThickness())
    

class GaussianRenderer(ColourRenderer):
    '''2D Gaussian rendering'''

    name = 'Gaussian'
    mode = 'gaussian'

    def _getDefaultJitVar(self, jitVars):
        if 'error_x' in jitVars:
            return jitVars.index('error_x')
        else:
            return 0

    def genIm(self, dlg, imb, mdh):
        pixelSize = dlg.getPixelSize()
        jitParamName = dlg.getJitterVariable()
        jitScale = dlg.getJitterScale()
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)

        return visHelpers.rendGauss(self.visFr.colourFilter['x'],self.visFr.colourFilter['y'], jitVals, imb, pixelSize)

class Gaussian3DRenderer(GaussianRenderer):
    '''3D Gaussian rendering'''

    name = '3D Gaussian'
    mode = '3Dgaussian'

    def genIm(self, dlg, imb, mdh):
        pixelSize = dlg.getPixelSize()
        jitParamName = dlg.getJitterVariable()
        jitScale = dlg.getJitterScale()
        jitParamNameZ = dlg.getJitterVariableZ()
        jitScaleZ = dlg.getJitterScaleZ()
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale
        mdh['Rendering.JitterVariableZ'] = jitParamNameZ
        mdh['Rendering.JitterScaleZ'] = jitScaleZ

        jitVals = self._genJitVals(jitParamName, jitScale)
        jitValsZ = self._genJitVals(jitParamNameZ, jitScaleZ)

        return visHelpers.rendGauss3D(self.visFr.colourFilter['x'],self.visFr.colourFilter['y'],self.visFr.colourFilter['z'], jitVals, jitValsZ, imb, pixelSize, dlg.getZBounds(), dlg.getZSliceThickness())


class TriangleRenderer(ColourRenderer):
    '''2D triangulation rendering'''

    name = 'Jittered Triangulation'
    mode = 'triangles'

    def genIm(self, dlg, imb, mdh):
        pixelSize = dlg.getPixelSize()
        jitParamName = dlg.getJitterVariable()
        jitScale = dlg.getJitterScale()
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)

        if dlg.getSoftRender():
            status = statusLog.StatusLogger("Rendering triangles ...")
            return visHelpers.rendJitTriang(self.visFr.colourFilter['x'],self.visFr.colourFilter['y'], dlg.getNumSamples(), jitVals, dlg.getMCProbability(),imb, pixelSize)
        else:
            return self.visFr.glCanvas.genJitTim(dlg.getNumSamples(),self.visFr.colourFilter['x'],self.visFr.colourFilter['y'], jitVals, dlg.getMCProbability(),pixelSize)

class Triangle3DRenderer(TriangleRenderer):
    '''3D Triangularisation rendering'''

    name = '3D Triangularisation'
    mode = '3Dtriangles'

    def genIm(self, dlg, imb, mdh):
        pixelSize = dlg.getPixelSize()
        jitParamName = dlg.getJitterVariable()
        jitScale = dlg.getJitterScale()
        jitParamNameZ = dlg.getJitterVariableZ()
        jitScaleZ = dlg.getJitterScaleZ()
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale
        mdh['Rendering.JitterVariableZ'] = jitParamNameZ
        mdh['Rendering.JitterScaleZ'] = jitScaleZ

        jitVals = self._genJitVals(jitParamName, jitScale)
        jitValsZ = self._genJitVals(jitParamNameZ, jitScaleZ)

        return visHelpers.rendJitTet(self.visFr.colourFilter['x'],self.visFr.colourFilter['y'],self.visFr.colourFilter['z'], dlg.getNumSamples(), jitVals, jitValsZ, dlg.getMCProbability(), imb, pixelSize, dlg.getZBounds(), dlg.getZSliceThickness())

class QuadTreeRenderer(ColourRenderer):
    '''2D quadtree rendering'''

    name = 'QuadTree'
    mode = 'quadtree'

    def genIm(self, dlg, imb, mdh):
        pixelSize = dlg.getPixelSize()

        if not pylab.mod(pylab.log2(pixelSize/self.visFr.QTGoalPixelSize), 1) == 0:#recalculate QuadTree to get right pixel size
                self.visFr.QTGoalPixelSize = pixelSize
                self.visFr.Quads = None

        self.visFr.GenQuads()

        qtWidth = self.visFr.Quads.x1 - self.visFr.Quads.x0
        qtWidthPixels = pylab.ceil(qtWidth/pixelSize)

        im = pylab.zeros((qtWidthPixels, qtWidthPixels))
        QTrend.rendQTa(im, self.visFr.Quads)

        return im[(imb.x0/pixelSize):(imb.x1/pixelSize),(imb.y0/pixelSize):(imb.y1/pixelSize)]


RENDERER_GROUPS = ((CurrentRenderer,),(HistogramRenderer, GaussianRenderer, TriangleRenderer, QuadTreeRenderer), (Histogram3DRenderer, Gaussian3DRenderer, Triangle3DRenderer))

def init_renderers(visFr):
    for g in RENDERER_GROUPS:
        for r in g:
            r(visFr)
        visFr.gen_menu.AppendSeparator()
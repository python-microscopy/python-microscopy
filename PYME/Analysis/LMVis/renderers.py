#!/usr/bin/python
##################
# renderers.py
#
# Copyright David Baddeley, 2010
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
    _defaultPixelSize = 5.0
    
    def __init__(self, visFr, pipeline, mainWind = None):
        self.visFr = visFr

        if mainWind == None:
            #menu handlers must be bound to the top level window
            mainWind = self.visFr
        self.mainWind = mainWind
        
        self.pipeline = pipeline

        self._addMenuItems()

    def _addMenuItems(self):
        ID = wx.NewId()
        self.visFr.gen_menu.Append(ID, self.name)

        self.mainWind.Bind(wx.EVT_MENU, self.Generate, id=ID)

    def _getImBounds(self):
        x0 = max(self.visFr.glCanvas.xmin, self.pipeline.imageBounds.x0)
        y0 = max(self.visFr.glCanvas.ymin, self.pipeline.imageBounds.y0)
        x1 = min(self.visFr.glCanvas.xmax, self.pipeline.imageBounds.x1)
        y1 = min(self.visFr.glCanvas.ymax, self.pipeline.imageBounds.y1)

        if 'x' in self.pipeline.filterKeys.keys():
            x0 = max(x0, self.pipeline.filterKeys['x'][0])
            x1 = min(x1, self.pipeline.filterKeys['x'][1])

        if 'y' in self.pipeline.filterKeys.keys():
            y0 = max(y0, self.pipeline.filterKeys['y'][0])
            y1 = min(y1, self.pipeline.filterKeys['y'][1])

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
            jitVals = np.ones(self.pipeline.colourFilter['x'].shape)
        elif jitParamName in self.pipeline.colourFilter.keys():
            jitVals = self.pipeline.colourFilter[jitParamName]
        elif jitParamName in self.genMeas:
            #print 'f'
            if jitParamName == 'neighbourDistances':
                jitVals = self.pipeline.getNeighbourDists(True)
            elif jitParamName == 'neighbourErrorMin':
                jitVals = np.minimum(self.pipeline.colourFilter['error_x'], self.pipeline.getNeighbourDists(True))
            else:
                jitVals = self.pipeline.GeneratedMeasures[jitParamName]

        return jitVals*jitScale


    def Generate(self, event=None):
        dlg = genImageDialog.GenImageDialog(self.mainWind, mode=self.mode)
        ret = dlg.ShowModal()

        bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            mdh = MetaDataHandler.NestedClassMDHandler()
            mdh['Rendering.Method'] = self.name
            if 'imageID' in self.pipeline.mdh.getEntryNames():
                mdh['Rendering.SourceImageID'] = self.pipeline.mdh['imageID']
            mdh['Rendering.SourceFilename'] = self.pipeline.filename
            
            for cb in renderMetadataProviders:
                cb(mdh)            
            
            pixelSize = dlg.getPixelSize()

            imb = self._getImBounds()

            im = self.genIm(dlg, imb, mdh)
            img = GeneratedImage(im,imb, pixelSize, 0, ['Image'] , mdh = mdh)
            imf = ViewIm3D(img, mode='visGUI', title='Generated %s - %3.1fnm bins' % (self.name, pixelSize), glCanvas=self.visFr.glCanvas, parent=self.mainWind)
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
        jitVars += self.pipeline.colourFilter.keys()

        self.genMeas = self.pipeline.GeneratedMeasures.keys()
        if not 'neighbourDistances' in self.genMeas:
            self.genMeas.append('neighbourDistances')
            
        if not 'neighbourErrorMin' in self.genMeas:
            self.genMeas.append('neighbourErrorMin')
            
        jitVars += self.genMeas
        
        
        if 'z' in self.pipeline.mapping.keys():
            zvals = self.pipeline.mapping['z']
        else:
            zvals = None

        dlg = genImageDialog.GenImageDialog(self.mainWind, mode=self.mode, defaultPixelSize=self._defaultPixelSize, colours=self.pipeline.colourFilter.getColourChans(), zvals = zvals, jitterVariables = jitVars, jitterVarDefault=self._getDefaultJitVar(jitVars), jitterVarDefaultZ=self._getDefaultZJitVar(jitVars))
        ret = dlg.ShowModal()

        bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            mdh = MetaDataHandler.NestedClassMDHandler()
            mdh['Rendering.Method'] = self.name
            if 'imageID' in self.pipeline.mdh.getEntryNames():
                mdh['Rendering.SourceImageID'] = self.pipeline.mdh['imageID']
            mdh['Rendering.SourceFilename'] = self.pipeline.filename
            
            mdh.Source = MetaDataHandler.NestedClassMDHandler(self.pipeline.mdh)
            
            for cb in renderMetadataProviders:
                cb(mdh)           
            
            pixelSize = dlg.getPixelSize()

            status = statusLog.StatusLogger('Generating %s Image ...' % self.name)

            imb = self._getImBounds()
            
            #record the pixel origin in nm from the corner of the camera for futrue overlays            
            if 'Source.Camera.ROIPosX' in mdh.getEntryNames():
                #a rendered image with information about the source ROI
                voxx, voxy = 1e3*mdh['Source.voxelsize.x'], 1e3*mdh['Source.voxelsize.y']
                
                ox = (mdh['Source.Camera.ROIPosX'] - 1)*voxx + imb.x0
                oy = (mdh['Source.Camera.ROIPosY'] - 1)*voxy + imb.y0
                if 'Source.Positioning.PIFoc' in mdh.getEntryNames():
                    oz = mdh['Source.Positioning.PIFoc']*1e3
                else:
                    oz = 0
            else:
                ox = imb.x0
                oy = imb.y0 
                oz = 0
            
            mdh['Origin.x'] = ox
            mdh['Origin.y'] = oy
            mdh['Origin.z'] = oz
            
            colours =  dlg.getColour()
            oldC = self.pipeline.colourFilter.currentColour

            ims = []

            for c in  colours:
                self.pipeline.colourFilter.setColour(c)
                ims.append(np.atleast_3d(self.genIm(dlg, imb, mdh)))

            im = GeneratedImage(ims,imb, pixelSize,  dlg.getZSliceThickness(), colours, mdh = mdh)

            imfc = ViewIm3D(im, mode='visGUI', title='Generated %s - %3.1fnm bins' % (self.name, pixelSize), glCanvas=self.visFr.glCanvas, parent=self.mainWind)

            #imfc = imageView.MultiChannelImageViewFrame(self.visFr, self.visFr.glCanvas, im, title='Generated %s - %3.1fnm bins' % (self.name, pixelSize))

            #self.visFr.generatedImages.append(imfc)
            #imfc.Show()

            self.pipeline.colourFilter.setColour(oldC)
        else:
            imfc = None

        dlg.Destroy()
        return imfc


class HistogramRenderer(ColourRenderer):
    '''2D histogram rendering'''

    name = 'Histogram'
    mode = 'histogram'

    def genIm(self, dlg, imb, mdh):
        return visHelpers.rendHist(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], imb, dlg.getPixelSize())

class Histogram3DRenderer(HistogramRenderer):
    '''3D histogram rendering'''

    name = '3D Histogram'
    mode = '3Dhistogram'

    def genIm(self, dlg, imb, mdh):
        mdh['Origin.z'] = dlg.getZBounds()[0]
        return visHelpers.rendHist3D(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], self.pipeline.colourFilter['z'], imb, dlg.getPixelSize(), dlg.getZBounds(), dlg.getZSliceThickness())
    

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

        return visHelpers.rendGauss(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], jitVals, imb, pixelSize)
        
class LHoodRenderer(ColourRenderer):
    '''Log-likelihood of object'''

    name = 'Log Likelihood'
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
        
        print('starting render')

        im =  visHelpers.rendGaussProd(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], jitVals, imb, pixelSize)
        
        print('done rendering')
        print((im.max()))
        
        return im - im.min()


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
        mdh['Origin.z'] = dlg.getZBounds()[0]

        jitVals = self._genJitVals(jitParamName, jitScale)
        jitValsZ = self._genJitVals(jitParamNameZ, jitScaleZ)

        return visHelpers.rendGauss3D(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'],self.pipeline.colourFilter['z'], jitVals, jitValsZ, imb, pixelSize, dlg.getZBounds(), dlg.getZSliceThickness())


class TriangleRenderer(ColourRenderer):
    '''2D triangulation rendering'''

    name = 'Jittered Triangulation'
    mode = 'triangles'
    _defaultPixelSize = 5.0

    def genIm(self, dlg, imb, mdh):
        pixelSize = dlg.getPixelSize()
        jitParamName = dlg.getJitterVariable()
        jitScale = dlg.getJitterScale()
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)

        if dlg.getSoftRender():
            status = statusLog.StatusLogger("Rendering triangles ...")
            return visHelpers.rendJitTriang(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], dlg.getNumSamples(), jitVals, dlg.getMCProbability(),imb, pixelSize)
        else:
            return self.visFr.glCanvas.genJitTim(dlg.getNumSamples(),self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], jitVals, dlg.getMCProbability(),pixelSize)
            
class TriangleRendererW(ColourRenderer):
    '''2D triangulation rendering - weighted'''

    name = 'Jittered Triangulation - weighted'
    mode = 'trianglesw'
    _defaultPixelSize = 5.0

    def genIm(self, dlg, imb, mdh):
        pixelSize = dlg.getPixelSize()
        jitParamName = dlg.getJitterVariable()
        jitScale = dlg.getJitterScale()
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)

        if dlg.getSoftRender():
            status = statusLog.StatusLogger("Rendering triangles ...")
            return visHelpers.rendJitTriang2(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], dlg.getNumSamples(), jitVals, dlg.getMCProbability(),imb, pixelSize)
        else:
            return self.visFr.glCanvas.genJitTim(dlg.getNumSamples(),self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'], jitVals, dlg.getMCProbability(),pixelSize)


class Triangle3DRenderer(TriangleRenderer):
    '''3D Triangularisation rendering'''

    name = '3D Triangularisation'
    mode = '3Dtriangles'
    _defaultPixelSize = 20.0

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
        mdh['Origin.z'] = dlg.getZBounds()[0]

        jitVals = self._genJitVals(jitParamName, jitScale)
        jitValsZ = self._genJitVals(jitParamNameZ, jitScaleZ)

        return visHelpers.rendJitTet(self.pipeline.colourFilter['x'],self.pipeline.colourFilter['y'],self.pipeline.colourFilter['z'], dlg.getNumSamples(), jitVals, jitValsZ, dlg.getMCProbability(), imb, pixelSize, dlg.getZBounds(), dlg.getZSliceThickness())

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


RENDERER_GROUPS = ((CurrentRenderer,),(HistogramRenderer, GaussianRenderer, TriangleRenderer, TriangleRendererW,LHoodRenderer, QuadTreeRenderer), (Histogram3DRenderer, Gaussian3DRenderer, Triangle3DRenderer))

def init_renderers(visFr, mainWind = None):
    for g in RENDERER_GROUPS:
        for r in g:
            r(visFr, visFr.pipeline, mainWind)
        visFr.gen_menu.AppendSeparator()
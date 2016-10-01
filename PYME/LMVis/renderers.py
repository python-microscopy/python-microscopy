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
#from PYME.LMVis.visHelpers import ImageBounds#, GeneratedImage
from PYME.IO.image import GeneratedImage, ImageBounds
from PYME.LMVis import genImageDialog
from PYME.LMVis import visHelpers
#from PYME.LMVis import imageView
from PYME.LMVis import statusLog
from PYME.LMVis import inpFilt

from PYME.IO import MetaDataHandler

try:
    import wx
    from PYME.DSView import ViewIm3D
except SystemExit:
    print('GUI load failed (probably OSX)')

from PYME.Analysis.points.QuadTree import QTrend


import pylab
import numpy as np

renderMetadataProviders = []

class CurrentRenderer:
    """Renders current view (in black and white). Only renderer not to take care
    of colour channels. Simplest renderer and as such also the base class for all 
    others"""

    name = 'Current'
    mode = 'current'
    _defaultPixelSize = 5.0
    
    def __init__(self, visFr, pipeline, mainWind = None):
        self.visFr = visFr

        if mainWind is None:
            #menu handlers must be bound to the top level window
            mainWind = self.visFr
        self.mainWind = mainWind
        
        self.pipeline = pipeline

        if isinstance(pipeline, inpFilt.colourFilter):
            self.colourFilter = pipeline
        else:
            self.colourFilter = pipeline.colourFilter

        self._addMenuItems()

    def _addMenuItems(self):
        #ID = wx.NewId()
        #self.visFr.gen_menu.Append(ID, self.name)

        #self.mainWind.Bind(wx.EVT_MENU, self.GenerateGUI, id=ID)
        if not self.visFr is None:
            self.visFr.AddMenuItem('Generate', self.name, self.GenerateGUI)

    def _getImBounds(self):
        if self.visFr is None:
            return self.pipeline.imageBounds

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
            jitVals = np.ones(self.colourFilter['x'].shape)
        elif jitParamName in self.colourFilter.keys():
            jitVals = self.colourFilter[jitParamName]
        elif jitParamName in self.genMeas:
            #print 'f'
            if jitParamName == 'neighbourDistances':
                jitVals = self.pipeline.getNeighbourDists(True)
            elif jitParamName == 'neighbourErrorMin':
                jitVals = np.minimum(self.colourFilter['error_x'], self.pipeline.getNeighbourDists(True))
            else:
                jitVals = self.pipeline.GeneratedMeasures[jitParamName]

        return jitVals*jitScale

    def Generate(self, settings):
        mdh = MetaDataHandler.NestedClassMDHandler()
        mdh['Rendering.Method'] = self.name
        if 'imageID' in self.pipeline.mdh.getEntryNames():
            mdh['Rendering.SourceImageID'] = self.pipeline.mdh['imageID']
        mdh['Rendering.SourceFilename'] = getattr(self.pipeline, 'filename', '')

        for cb in renderMetadataProviders:
            cb(mdh)

        pixelSize = settings['pixelSize']

        imb = self._getImBounds()

        im = self.genIm(settings, imb, mdh)
        return GeneratedImage(im, imb, pixelSize, 0, ['Image'], mdh=mdh)


    def GenerateGUI(self, event=None):
        dlg = genImageDialog.GenImageDialog(self.mainWind, mode=self.mode)
        ret = dlg.ShowModal()

        #bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            img = self.Generate(dlg.get_settings())
            imf = ViewIm3D(img, mode='visGUI', title='Generated %s - %3.1fnm bins' % (self.name, img.pixelSize), glCanvas=self.visFr.glCanvas, parent=self.mainWind)

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
    """Base class for all other renderers which know about the colour filter"""
    
    def Generate(self, settings):
        mdh = MetaDataHandler.NestedClassMDHandler()
        mdh['Rendering.Method'] = self.name
        if 'imageID' in self.pipeline.mdh.getEntryNames():
            mdh['Rendering.SourceImageID'] = self.pipeline.mdh['imageID']
        mdh['Rendering.SourceFilename'] = getattr(self.pipeline, 'filename', '')

        mdh.Source = MetaDataHandler.NestedClassMDHandler(self.pipeline.mdh)

        for cb in renderMetadataProviders:
            cb(mdh)

        pixelSize = settings['pixelSize']

        status = statusLog.StatusLogger('Generating %s Image ...' % self.name)

        imb = self._getImBounds()

        #record the pixel origin in nm from the corner of the camera for futrue overlays
        if 'Source.Camera.ROIPosX' in mdh.getEntryNames():
            #a rendered image with information about the source ROI
            voxx, voxy = 1e3 * mdh['Source.voxelsize.x'], 1e3 * mdh['Source.voxelsize.y']

            ox = (mdh['Source.Camera.ROIPosX'] - 1) * voxx + imb.x0
            oy = (mdh['Source.Camera.ROIPosY'] - 1) * voxy + imb.y0
            if 'Source.Positioning.PIFoc' in mdh.getEntryNames():
                oz = mdh['Source.Positioning.PIFoc'] * 1e3
            else:
                oz = 0
        else:
            ox = imb.x0
            oy = imb.y0
            oz = 0

        mdh['Origin.x'] = ox
        mdh['Origin.y'] = oy
        mdh['Origin.z'] = oz

        colours = settings['colours']
        oldC = self.colourFilter.currentColour

        ims = []

        for c in colours:
            self.colourFilter.setColour(c)
            ims.append(np.atleast_3d(self.genIm(settings, imb, mdh)))

        self.colourFilter.setColour(oldC)

        return GeneratedImage(ims, imb, pixelSize, settings['zSliceThickness'], colours, mdh=mdh)

    def GenerateGUI(self, event=None):
        jitVars = ['1.0']
        jitVars += self.colourFilter.keys()

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

        dlg = genImageDialog.GenImageDialog(self.mainWind, mode=self.mode, defaultPixelSize=self._defaultPixelSize, colours=self.colourFilter.getColourChans(), zvals = zvals, jitterVariables = jitVars, jitterVarDefault=self._getDefaultJitVar(jitVars), jitterVarDefaultZ=self._getDefaultZJitVar(jitVars))
        ret = dlg.ShowModal()

        #bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            im = self.Generate(dlg.get_settings())
            imfc = ViewIm3D(im, mode='visGUI', title='Generated %s - %3.1fnm bins' % (self.name, im.pixelSize),
                            glCanvas=self.visFr.glCanvas, parent=self.mainWind)
        else:
            imfc = None

        dlg.Destroy()
        return imfc


class HistogramRenderer(ColourRenderer):
    """2D histogram rendering"""

    name = 'Histogram'
    mode = 'histogram'

    def genIm(self, settings, imb, mdh):
        return visHelpers.rendHist(self.colourFilter['x'],self.colourFilter['y'], imb, settings['pixelSize'])

class Histogram3DRenderer(HistogramRenderer):
    """3D histogram rendering"""

    name = '3D Histogram'
    mode = '3Dhistogram'

    def genIm(self, settings, imb, mdh):
        mdh['Origin.z'] = settings['zBounds'][0]
        return visHelpers.rendHist3D(self.colourFilter['x'],self.colourFilter['y'], self.colourFilter['z'], imb, settings['pixelSize'], settings['zBounds'], settings['zSliceThickness'])
    

class GaussianRenderer(ColourRenderer):
    """2D Gaussian rendering"""

    name = 'Gaussian'
    mode = 'gaussian'

    def _getDefaultJitVar(self, jitVars):
        if 'error_x' in jitVars:
            return jitVars.index('error_x')
        else:
            return 0

    def genIm(self, settings, imb, mdh):
        pixelSize = settings['pixelSize']
        jitParamName = settings['jitterVariable']
        jitScale = settings['jitterScale']
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)

        return visHelpers.rendGauss(self.colourFilter['x'],self.colourFilter['y'], jitVals, imb, pixelSize)
        
class LHoodRenderer(ColourRenderer):
    """Log-likelihood of object"""

    name = 'Log Likelihood'
    mode = 'gaussian'

    def _getDefaultJitVar(self, jitVars):
        if 'error_x' in jitVars:
            return jitVars.index('error_x')
        else:
            return 0

    def genIm(self, settings, imb, mdh):
        pixelSize = settings['pixelSize']
        jitParamName = settings['jitterVariable']
        jitScale = settings['jitterScale']
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)
        
        print('starting render')

        im =  visHelpers.rendGaussProd(self.colourFilter['x'],self.colourFilter['y'], jitVals, imb, pixelSize)
        
        print('done rendering')
        print((im.max()))
        
        return im - im.min()


class Gaussian3DRenderer(GaussianRenderer):
    """3D Gaussian rendering"""

    name = '3D Gaussian'
    mode = '3Dgaussian'

    def genIm(self, settings, imb, mdh):
        pixelSize = settings['pixelSize']
        jitParamName = settings['jitterVariable']
        jitScale = settings['jitterScale']
        jitParamNameZ = settings['jitterVariableZ']
        jitScaleZ = settings['jitterScaleZ']
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale
        mdh['Rendering.JitterVariableZ'] = jitParamNameZ
        mdh['Rendering.JitterScaleZ'] = jitScaleZ
        mdh['Origin.z'] = settings['zBounds'][0]

        jitVals = self._genJitVals(jitParamName, jitScale)
        jitValsZ = self._genJitVals(jitParamNameZ, jitScaleZ)

        return visHelpers.rendGauss3D(self.colourFilter['x'],self.colourFilter['y'],
                                      self.colourFilter['z'], jitVals, jitValsZ, imb, pixelSize,
                                      settings['zBounds'], settings['zSliceThickness'])


class TriangleRenderer(ColourRenderer):
    """2D triangulation rendering"""

    name = 'Jittered Triangulation'
    mode = 'triangles'
    _defaultPixelSize = 5.0

    def genIm(self, settings, imb, mdh):
        pixelSize = settings['pixelSize']
        jitParamName = settings['jitterVariable']
        jitScale = settings['jitterScale']
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)

        if settings['softRender']:
            status = statusLog.StatusLogger("Rendering triangles ...")
            return visHelpers.rendJitTriang(self.colourFilter['x'],self.colourFilter['y'],
                                            settings['numSamples'], jitVals, settings['MCProbability'],imb, pixelSize)
        else:
            return self.visFr.glCanvas.genJitTim(settings['numSamples'],self.colourFilter['x'],
                                                 self.colourFilter['y'], jitVals,
                                                 settings['MCProbability'],pixelSize)
            
class TriangleRendererW(ColourRenderer):
    """2D triangulation rendering - weighted"""

    name = 'Jittered Triangulation - weighted'
    mode = 'trianglesw'
    _defaultPixelSize = 5.0

    def genIm(self, settings, imb, mdh):
        pixelSize = settings['pixelSize']
        jitParamName = settings['jitterVariable']
        jitScale = settings['jitterScale']
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale

        jitVals = self._genJitVals(jitParamName, jitScale)

        if settings['softRender']:
            status = statusLog.StatusLogger("Rendering triangles ...")
            return visHelpers.rendJitTriang2(self.colourFilter['x'],self.colourFilter['y'],
                                             settings['numSamples'], jitVals, settings['MCProbability'],imb, pixelSize)
        else:
            return self.visFr.glCanvas.genJitTim(settings['numSamples'],self.colourFilter['x'],
                                                 self.colourFilter['y'], jitVals,
                                                 settings['MCProbability'],pixelSize)


class Triangle3DRenderer(TriangleRenderer):
    """3D Triangularisation rendering"""

    name = '3D Triangularisation'
    mode = '3Dtriangles'
    _defaultPixelSize = 20.0

    def genIm(self, settings, imb, mdh):
        pixelSize = settings['pixelSize']
        jitParamName = settings['jitterVariable']
        jitScale = settings['jitterScale']
        jitParamNameZ = settings['jitterVariableZ']
        jitScaleZ = settings['jitterScaleZ']
        
        mdh['Rendering.JitterVariable'] = jitParamName
        mdh['Rendering.JitterScale'] = jitScale
        mdh['Rendering.JitterVariableZ'] = jitParamNameZ
        mdh['Rendering.JitterScaleZ'] = jitScaleZ
        mdh['Origin.z'] = settings['zBounds'][0]

        jitVals = self._genJitVals(jitParamName, jitScale)
        jitValsZ = self._genJitVals(jitParamNameZ, jitScaleZ)

        return visHelpers.rendJitTet(self.colourFilter['x'],self.colourFilter['y'],
                                     self.colourFilter['z'], settings['numSamples'], jitVals, jitValsZ,
                                     settings['MCProbability'], imb, pixelSize, settings['zBounds'], settings['zSliceThickness'])

class QuadTreeRenderer(ColourRenderer):
    """2D quadtree rendering"""

    name = 'QuadTree'
    mode = 'quadtree'

    def genIm(self, settings, imb, mdh):
        pixelSize = settings['pixelSize']

        if not pylab.mod(pylab.log2(pixelSize/self.visFr.QTGoalPixelSize), 1) == 0:#recalculate QuadTree to get right pixel size
                self.visFr.QTGoalPixelSize = pixelSize
                self.visFr.Quads = None

        self.visFr.GenQuads()

        qtWidth = self.visFr.Quads.x1 - self.visFr.Quads.x0
        qtWidthPixels = pylab.ceil(qtWidth/pixelSize)

        im = pylab.zeros((qtWidthPixels, qtWidthPixels))
        QTrend.rendQTa(im, self.visFr.Quads)

        return im[(imb.x0/pixelSize):(imb.x1/pixelSize),(imb.y0/pixelSize):(imb.y1/pixelSize)]


RENDERER_GROUPS = ((CurrentRenderer,),
                   (HistogramRenderer, GaussianRenderer, TriangleRenderer, TriangleRendererW,LHoodRenderer, QuadTreeRenderer),
                   (Histogram3DRenderer, Gaussian3DRenderer, Triangle3DRenderer))

RENDERERS = {i.name : i for s in RENDERER_GROUPS for i in s}

def init_renderers(visFr, mainWind = None):
    for g in RENDERER_GROUPS:
        for r in g:
            r(visFr, visFr.pipeline, mainWind)
        visFr.AddMenuItem('Generate', itemType='separator')

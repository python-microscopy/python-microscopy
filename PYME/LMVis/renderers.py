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
from PYME.LMVis import visHelpers
from PYME.LMVis import statusLog
from PYME.IO import tabular

from PYME.IO import MetaDataHandler

#import pylab
import numpy as np

renderMetadataProviders = []

SAMPLE_MD_KEYS = [
    'Sample.SlideRef',
    'Sample.Creator',
    'Sample.Notes',
    'Sample.Labelling',
    'AcquiringUser'
]

def copy_sample_metadata(old_mdh, new_mdh):
    pass_through = [k for k in SAMPLE_MD_KEYS if k in old_mdh.keys()]
    for k in pass_through:
        new_mdh[k] = old_mdh[k]

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



        self._addMenuItems()

    @property
    def colourFilter(self):
        if isinstance(self.pipeline, tabular.ColourFilter):
            return self.pipeline
        else:
            return self.pipeline.colourFilter

    def _addMenuItems(self):
        #ID = wx.NewId()
        #self.visFr.gen_menu.Append(ID, self.name)

        #self.mainWind.Bind(wx.EVT_MENU, self.GenerateGUI, id=ID)
        if not self.visFr is None:
            self.visFr.AddMenuItem('Generate', self.name, self.GenerateGUI)

    def _get_image_bounds(self, pixel_size, slice_size=None, zmin=None, zmax=None):
        """
        Returns the image bounds for the image to be generated, rounded to the nearest pixel.
        
        Generates the image bounds as follows:
        
        - Takes image bounds from pipeline (these will either come from the ROI metadata (if present), or have been
         estimated from the maximum extent of localizations in the pipeline.
        
        - Clips these to the currently displayed region
        
        - Clips again to any ROI the user has set
        
        
        Parameters
        ----------
        pixel_size : float - pixel size in nm
        slice_size : float - z slice thickness in nm
        zmin : float [optional] - lower z bound
        zmax : float [optional] - upper z bound

        Returns
        -------

        """
        if self.visFr is None:
            x0, y0, x1, y1, z0, z1 = self.pipeline.imageBounds.bounds
            
            if not zmin is None:
                z0 = zmin
            
            if not zmax is None:
                z1 = zmax
                
            return ImageBounds(x0, y0, x1, y1, z0, z1)

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

        if not zmin is None:
            z0 = zmin
        else:
            z0 = 0

        if not zmax is None:
            z1 = zmax
        else:
            z1 = 0

        # extent x1, y1, and z1 if necessary to make sure that bound ranges are integer multiples of pixel or slice size
        x1 = x0 + (np.ceil((x1 - x0) / pixel_size) * pixel_size)
        y1 = y0 + (np.ceil((y1 - y0) / pixel_size) * pixel_size)
        if slice_size is not None and (z0 != z1):
            z1 = z0 + (np.ceil((z1 - z0) / slice_size) * slice_size)

        return ImageBounds(x0, y0, x1, y1, z0, z1)

    def _getDefaultJitVar(self, jitVars):
        return jitVars.index('neighbourDistances')

    def _getDefaultZJitVar(self, jitVars):
        if 'fitError_z0' in jitVars:
            return jitVars.index('fitError_z0')
        else:
            return 0

    def _get_neighbour_dists(self):
        from matplotlib import tri
        triangles = tri.Triangulation(
            self.colourFilter['x'] + .1 * np.random.normal(size=len(self.colourFilter['x'])),
            self.colourFilter['y'] + .1 * np.random.normal(size=len(self.colourFilter['x'])))

        return np.array(visHelpers.calcNeighbourDists(triangles))

    def _genJitVals(self, jitParamName, jitScale):
        #print jitParamName
        if jitParamName == '1.0':
            jitVals = np.ones(self.colourFilter['x'].shape)
        elif jitParamName in self.colourFilter.keys():
            jitVals = self.colourFilter[jitParamName]
        elif jitParamName == 'neighbourDistances':
            jitVals = self._get_neighbour_dists()
        elif jitParamName == 'neighbourErrorMin':
            jitVals = np.minimum(self.colourFilter['error_x'], self._get_neighbour_dists())
        elif jitParamName in self.genMeas:
            jitVals = self.pipeline.GeneratedMeasures[jitParamName]

        return jitVals*jitScale

    def Generate(self, settings):
        mdh = MetaDataHandler.NestedClassMDHandler()
        copy_sample_metadata(self.pipeline.mdh, mdh)
        mdh['Rendering.Method'] = self.name
        if 'imageID' in self.pipeline.mdh.getEntryNames():
            mdh['Rendering.SourceImageID'] = self.pipeline.mdh['imageID']
        mdh['Rendering.SourceFilename'] = getattr(self.pipeline, 'filename', '')
        mdh['Rendering.NEventsRendered'] = len(self.pipeline[self.pipeline.keys()[0]]) # in future good to use colourfilter for per channel info?

        for cb in renderMetadataProviders:
            cb(mdh)

        pixelSize = settings['pixelSize']

        imb = self._get_image_bounds(pixelSize)  # get image bounds at integer multiple of pixel size
        im = self.genIm(settings, imb, mdh)
        return GeneratedImage(im, imb, pixelSize, 0, ['Image'], mdh=mdh)


    def GenerateGUI(self, event=None):
        import wx
        from PYME.LMVis import genImageDialog
        from PYME.DSView import ViewIm3D
        
        dlg = genImageDialog.GenImageDialog(self.mainWind, mode=self.mode)
        ret = dlg.ShowModal()

        #bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            img = self.Generate(dlg.get_settings())
            imf = ViewIm3D(img, mode='visGUI', title='Generated %s - %3.1fnm bins' % (self.name, img.pixelSize), glCanvas=self.visFr.glCanvas, parent=self.mainWind)

            self.visFr.RefreshView()

        dlg.Destroy()
        return imf

    def genIm(self, settings, imb, mdh):
        # import matplotlib.pyplot as plt
        # oldcmap = self.visFr.glCanvas.cmap
        # self.visFr.glCanvas.setCMap(plt.cm.gray)
        im = self.visFr.glCanvas.getIm(settings['pixelSize'], image_bounds=imb)

        # self.visFr.glCanvas.setCMap(oldcmap)

        return np.atleast_3d(im)

class ColourRenderer(CurrentRenderer):
    """Base class for all other renderers which know about the colour filter"""
    
    def Generate(self, settings):
        mdh = MetaDataHandler.NestedClassMDHandler()
        copy_sample_metadata(self.pipeline.mdh, mdh)
        mdh['Rendering.Method'] = self.name
        if 'imageID' in self.pipeline.mdh.getEntryNames():
            mdh['Rendering.SourceImageID'] = self.pipeline.mdh['imageID']
        mdh['Rendering.SourceFilename'] = getattr(self.pipeline, 'filename', '')
        mdh['Rendering.NEventsRendered'] = len(self.pipeline[self.pipeline.keys()[0]]) # in future good to use colourfilter for per channel info?
        mdh.Source = MetaDataHandler.NestedClassMDHandler(self.pipeline.mdh)

        for cb in renderMetadataProviders:
            cb(mdh)

        pixelSize = settings['pixelSize']
        sliceThickness = settings['zSliceThickness']

        status = statusLog.StatusLogger('Generating %s Image ...' % self.name)

        # get image bounds at integer multiple of pixel size
        imb = self._get_image_bounds(pixelSize, sliceThickness, *settings.get('zBounds', [None, None]))

        #record the pixel origin in nm from the corner of the camera for futrue overlays
        ox, oy, oz = MetaDataHandler.origin_nm(mdh)
        if not imb.z0 == 0:
            # single plane in z stack
            # FIXME - what is z for 3D fitting at a single focal plane? Check for pipeline['focus']==0 instead?
            oz = 0

        mdh['Origin.x'] = ox + imb.x0
        mdh['Origin.y'] = oy + imb.y0
        mdh['Origin.z'] = oz + imb.z0

        colours = settings['colours']
        oldC = self.colourFilter.currentColour

        ims = []

        for c in colours:
            self.colourFilter.setColour(c)
            ims.append(np.atleast_3d(self.genIm(settings, imb, mdh)))

        self.colourFilter.setColour(oldC)

        return GeneratedImage(ims, imb, pixelSize, sliceThickness, colours, mdh=mdh)

    def GenerateGUI(self, event=None):
        import wx
        from PYME.LMVis import genImageDialog
        from PYME.DSView import ViewIm3D
        
        jitVars = ['1.0']
        jitVars += self.colourFilter.keys()

        self.genMeas = list(self.pipeline.GeneratedMeasures.keys())
        if not 'neighbourDistances' in self.genMeas:
            self.genMeas.append('neighbourDistances')
            
        if not 'neighbourErrorMin' in self.genMeas:
            self.genMeas.append('neighbourErrorMin')
            
        jitVars += self.genMeas
        
        
        if 'z' in self.pipeline.keys():
            zvals = self.pipeline['z']
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
        return visHelpers.rendHist3D(self.colourFilter['x'],self.colourFilter['y'], self.colourFilter['z'], imb, settings['pixelSize'], settings['zSliceThickness'])
        
class DensityFitRenderer(HistogramRenderer):
    """3D histogram rendering"""

    name = 'Density Fit'
    mode = 'densityfit'

    def genIm(self, settings, imb, mdh):
        return visHelpers.rend_density_estimate(self.colourFilter['x'],self.colourFilter['y'], imb, settings['pixelSize'], settings['numSamples'])
    

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

        jitVals = np.maximum(self._genJitVals(jitParamName, jitScale), pixelSize)

        if settings['softRender']:
            status = statusLog.StatusLogger("Rendering triangles ...")
            return visHelpers.rendJitTriang(self.colourFilter['x'],self.colourFilter['y'],
                                            settings['numSamples'], jitVals, settings['MCProbability'],imb, pixelSize,
                                            geometric_mean=settings.get('geometricMean', False), mdh=mdh)
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

        jitVals = np.maximum(self._genJitVals(jitParamName, jitScale), pixelSize)

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

        jitVals = np.maximum(self._genJitVals(jitParamName, jitScale), pixelSize)
        jitValsZ = np.maximum(self._genJitVals(jitParamNameZ, jitScaleZ), settings['zSliceThickness'])

        return visHelpers.rendJitTet(self.colourFilter['x'],self.colourFilter['y'],
                                     self.colourFilter['z'], settings['numSamples'], jitVals, jitValsZ,
                                     settings['MCProbability'], imb, pixelSize, settings['zSliceThickness'])

class QuadTreeRenderer(ColourRenderer):
    """2D quadtree rendering"""

    name = 'QuadTree'
    mode = 'quadtree'

    def genIm(self, settings, imb, mdh):
        from PYME.Analysis.points.QuadTree import QTrend
        pixelSize = settings['pixelSize']
        leaf_size = settings.get('qtLeafSize', 10) #default to 10 record leaf size

        if not np.mod(np.log2(pixelSize/self.pipeline.QTGoalPixelSize), 1) == 0:#recalculate QuadTree to get right pixel size
                self.pipeline.QTGoalPixelSize = pixelSize
                self.pipeline.Quads = None

        self.pipeline.GenQuads(max_leaf_size=leaf_size)
        quads = self.pipeline.Quads #TODO remove GenQuads from pipeline

        qtWidth = max(quads.x1 - quads.x0, quads.y1 - quads.y0)
        qtWidthPixels = int(np.ceil(qtWidth/pixelSize))

        im = np.zeros((qtWidthPixels, qtWidthPixels))
        QTrend.rendQTa(im, quads)
        
        #FIXME - make this work for imb > quadtree size
        return im[int(max(imb.x0 - quads.x0, 0)/pixelSize):int((imb.x1 - quads.x0)/pixelSize),int(max(imb.y0 - quads.y0, 0)/pixelSize):int((imb.y1 - quads.y0)/pixelSize)]


class VoronoiRenderer(ColourRenderer):
    """2D histogram rendering"""

    name = 'Voronoi'
    mode = 'voronoi'

    def genIm(self, settings, imb, mdh):
        return visHelpers.rendVoronoi(self.colourFilter['x'],self.colourFilter['y'], imb, settings['pixelSize'])



RENDERER_GROUPS = ((HistogramRenderer, GaussianRenderer, TriangleRenderer, TriangleRendererW,LHoodRenderer, QuadTreeRenderer, DensityFitRenderer, VoronoiRenderer),
                   (Histogram3DRenderer, Gaussian3DRenderer, Triangle3DRenderer))

RENDERERS = {i.name : i for s in RENDERER_GROUPS for i in s}

def init_renderers(visFr, mainWind = None):
    for g in RENDERER_GROUPS:
        for r in g:
            r(visFr, visFr.pipeline, mainWind)
        visFr.AddMenuItem('Generate', itemType='separator')

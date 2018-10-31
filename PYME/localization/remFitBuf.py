#!/usr/bin/python

##################
# remFitBuf.py
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

from PYME.ParallelTasks import taskDef
from PYME.localization import ofind
#import ofind_nofilt #use for drift estimation - faster
#import ofind_xcorr
#try:
    #try to use the FFTW version if we have fftw installed
#    import ofind_xcorr_fw as ofind_xcorr
#except:
    #fall back on fftpack in scipy
    #this was only marginally slower at last benchmark implying much of the 
    #cost is not in the ffts
from PYME.localization import ofind_xcorr
from PYME.localization import ofind_pri
from PYME.localization import splitting

from PYME.IO import buffers
from PYME.IO.image import ImageStack

import logging
logger = logging.getLogger(__name__)
    
import numpy
import numpy as np

from PYME.IO.FileUtils.nameUtils import getFullExistingFilename

bufferMisses = 0
nTasksProcessed = 0

splitterFitModules = ['SplitterFitFR', 'SplitterFitFNR','SplitterFitQR', 'SplitterFitCOIR', 'SplitterFitFNSR',
                      'BiplaneFitR', 'SplitterShiftEstFR', 'SplitterFitFusionR',
                      'SplitterObjFindR', 'SplitterFitInterpR', 'SplitterFitInterpQR', 'SplitterFitInterpNR', 'SplitterFitInterpBNR', 'SplitterROIExtractNR']

#from pylab import *

import copy
from PYME.IO.MetaDataHandler import get_camera_roi_origin

def tqPopFcn(workerN, NWorkers, NTasks):
    #let each task work on its own chunk of data ->
    return workerN * NTasks/NWorkers 
    
class fitResult(taskDef.TaskResult):
    def __init__(self, task, results, driftResults):
        taskDef.TaskResult.__init__(self, task)
        self.index = task.index #the task number so that the taskQueue know which task this was (used to manage the list of in-progress tasks)
        self.results = results # a numpy ndarray with the positions of the fitted molecules
        self.driftResults = driftResults # a numpy ndarray with the positions of fiducials used to track drift

class BufferManager(object):
    """Keeps track of data sources and buffering over individual fitTask instances"""
    def __init__(self):
        self.dBuffer = None
        self.bBuffer = None
        self.dataSourceID = None
        
    def updateBuffers(self, md, dataSourceModule, bufferLen):
        """Update the various buffers. """
        if dataSourceModule is None:
            #if the data source module is not specified, guess based on data source ID
            import PYME.IO.DataSources
            DataSource = PYME.IO.DataSources.getDataSourceForFilename(md.dataSourceID)
        else:
            DataSource = __import__('PYME.IO.DataSources.' + dataSourceModule, fromlist=['DataSource']).DataSource #import our data source

        #read the data
        if not self.dataSourceID == md.dataSourceID: #avoid unnecessary opening and closing 
            self.dBuffer = buffers.dataBuffer(DataSource(md.dataSourceID, md.taskQueue), bufferLen)
            self.bBuffer = None
        
        #fix our background buffers
        if md.getOrDefault('Analysis.PCTBackground', 0) > 0:
            if not isinstance(self.bBuffer, buffers.backgroundBufferM):
                try:
                    from warpDrive.buffers import Buffer as GPUPercentileBuffer
                    HAVE_GPU_PCT_BUFFER = True
                except ImportError:
                    HAVE_GPU_PCT_BUFFER = False
                
                if (HAVE_GPU_PCT_BUFFER and md.getOrDefault('Analysis.GPUPCTBackground', False)):
                    # calculate percentile buffer on the GPU. Only applies if warpDrive module is available AND we explcitly ask for the GPU version
                    # NB: The GPU version should result in a uniform background but will NOT completely remove the background. As such it will only work 
                    # for fits which have a constant background as a fit parameter, and not those which assume that background subtraction reduces the 
                    # background to zero (as is the case for our CPU based background estimation). Use with caution.
                    self.bBuffer = GPUPercentileBuffer(self.dBuffer, md['Analysis.PCTBackground'],
                                                       buffer_length=bufferLen, dark_map=cameraMaps.getDarkMap(md))
                else: 
                    # use our default CPU implementation
                    self.bBuffer = buffers.backgroundBufferM(self.dBuffer, md['Analysis.PCTBackground'])
            else:
                # we already have a percentile buffer - just change the settings. TODO - Does this need to change to reflect introduction of GPU based buffering?
                self.bBuffer.pctile = md['Analysis.PCTBackground']
        else:
            if not isinstance(self.bBuffer, buffers.backgroundBuffer):
                self.bBuffer = buffers.backgroundBuffer(self.dBuffer)
                
        self.dataSourceID = md.dataSourceID

#instance of our buffer manager
bufferManager = BufferManager()

class CameraInfoManager(object):
    """Manages camera information such as dark frames, variance maps, and flatfielding"""
    def __init__(self):
        self._cache = {}

    def _parseROI(self, md):
        """Extract ROI coordinates from metadata"""
        
        x0, y0 = get_camera_roi_origin(md)

        x1 = x0 + md['Camera.ROIWidth']
        y1 = y0 + md['Camera.ROIHeight']

        return x0, y0, x1, y1

    def _fetchMap(self, md, mapName):
        """retrive a map, with a given name. First try and get it from the Queue,
        then try finding it locally"""
        try:
            varmap = md.taskQueue.getQueueData(md.dataSourceID, 'MAP',  mapName)
        except:
            fn = getFullExistingFilename(mapName)
            varmap = ImageStack(filename=fn).data[:,:,0].squeeze() #this should handle .tif, .h5, and a few others

        return varmap

    def _getMap(self, md, mapName):
        """Returns the map specified, from cache if possible"""
        if mapName is None or mapName == '' or mapName == '<none>':
            return None

        ROI = self._parseROI(md)
        mapKey = (mapName, ROI)

        try:
            mp = self._cache[mapKey]
        except KeyError: 
            #cache miss
            x0, y0, x1, y1 = ROI
            mp = self._fetchMap(md, mapName)[x0:x1, y0:y1].astype('f') #everything should be float (rather than double)
            
            self._cache[mapKey] = mp

        return mp

    def getVarianceMap(self, md):
        """Returns the pixel variance map specified in the supplied metadata, from cache if possible.
        The variance map should be in units of *photoelectrons*."""
        varMapName = md.getOrDefault('Camera.VarianceMapID', None)

        mp = self._getMap(md, varMapName)

        if (mp is None):
            #default to uniform read noise
            rn = float(md['Camera.ReadNoise'])
            return rn*rn
        else:
            return mp


    def getDarkMap(self, md):
        """Returns the dark map specified in the supplied metadata, from cache if possible.
        The dark map is in units of camera counts"""
        darkMapName = md.getOrDefault('Camera.DarkMapID', None)

        mp = self._getMap(md, darkMapName)
        if (mp is None):
            return float(md.getOrDefault('Camera.ADOffset', 0))
        else:
            return mp

    def getFlatfieldMap(self, md):
        """Returns the flatfield map specified in the supplied metadata, from cache if possible
        The flatfield is a (floating point) value which is multiplied with the image to correct
        variations in response. It should (usually) have a mean value of 1."""
        flatfieldMapName = md.getOrDefault('Camera.FlatfieldMapID', None)

        mp = self._getMap(md, flatfieldMapName)

        if (mp is None):
            return 1.0
        else:
            return mp

    def correctImage(self, md, img):
        dk = self.getDarkMap(md)
        flat = self.getFlatfieldMap(md)

        return (img.astype('f')-dk)*flat

cameraMaps = CameraInfoManager()

def createFitTaskFromTaskDef(task):
    """
    Creates a fit task from a new-style json task definition
    Parameters
    ----------
    task : dict
        The parsed task definition. As the task definition will need to be parsed by the worker before we get here,
        we expect this to take the form of a python dictionary.

    Returns
    -------

    a fitTask instance

    """
    from PYME.IO import MetaDataHandler

    dataSourceID = task['inputs']['frames']
    frameIndex = int(task['taskdef']['frameIndex'])

    #logger.debug('Creating a task for %s - frame %d' % (dataSourceID, frameIndex))

    md = task['taskdef']['metadata']

    #sort out our metadata
    #TODO - Move this somewhere saner - e.g. a helper function in the MetaDataHandler module
    mdh = MetaDataHandler.NestedClassMDHandler()
    if isinstance(md, dict):
        #metadata was parsed with the enclosing json
        mdh.update(md)
    elif isinstance(md, str) or isinstance(md, unicode):
        if md.startswith('{'):
            #metadata is a quoted json dump
            import json
            mdh.update(json.loads(md))
        else:
            #metadata entry is a filename/URI
            from PYME.IO import unifiedIO
            if md.endswith('.json'):
                import json
                mdh.update(json.loads(unifiedIO.read(md)))
            else:
                raise NotImplementedError('Loading metadata from a URI in task description is not yet supported')

    return fitTask(dataSourceID=dataSourceID, frameIndex=frameIndex, metadata=mdh)

class fitTask(taskDef.Task):
    def __init__(self, dataSourceID, frameIndex, metadata, dataSourceModule=None, resultsURI=None):
        """
        Create a new fit task for performing single molecule localization tasks.

        Parameters
        ----------
        dataSourceID : str
            A filename or URI which identifies the data source
        frameIndex : int
            The frame to fit
        metadata : PYME.IO.MetaDataHandler object
            The image metadata. This defines most of the analysis parameters.
        dataSourceModule : str
            The name of the data source module to use. If None, the dataSourceModule is inferred from the dataSourceID.
        resultsURI : str
            A URI dictating where to store the analysis results.
        """
        
        taskDef.Task.__init__(self, resultsURI=resultsURI)

        self.dataSourceID = dataSourceID
        self.dataSourceModule = dataSourceModule
        self.index = frameIndex

        self.md = metadata
        self._fitMod = None

        #extract remaining parameters from metadata
        self.fitModule = self.md['Analysis.FitModule']

        self.threshold = self.md['Analysis.DetectionThreshold']
        self.SNThreshold = self.md.getOrDefault('Analysis.SNRThreshold', True)
        
        self.driftEst = self.md.getOrDefault('Analysis.TrackFiducials', False)

        self._get_bgindices()  # NB - this injects Analysis.BGRange into metadata if not already present
        #  make sure that our buffer is large enough for drift correction or background subtraction
        if self.driftEst:
            drift_ind = self.index + self.md.getOrDefault('Analysis.DriftIndices', np.array([-10, 0, 10]))
        else:
            drift_ind = [0,0]
        # a fix for Analysis.BGRange[0] == Analysis.BGRange[1], i.e. '0:0' as we tend to do in DNA-PAINT
        # makes bufferLen at least 1
        self.bufferLen = max(1,max(self.md['Analysis.BGRange'][1], drift_ind[-1]) - min(self.md['Analysis.BGRange'][0], drift_ind[0]))
        
    @property
    def fitMod(self):
        if self._fitMod is None:
            self._fitMod = __import__('PYME.localization.FitFactories.' + self.fitModule, fromlist=['PYME', 'localization', 'FitFactories']) #import our fitting module
         
        return self._fitMod
    
    @property
    def is_splitter_fit(self):
        return self.fitModule in splitterFitModules or getattr(self.fitMod, 'SPLITTER_FIT', False)
    

    def _get_bgindices(self):
        if not 'Analysis.BGRange' in self.md.getEntryNames():
            if 'Analysis.NumBGFrames' in self.md.getEntryNames():
                nBGFrames = self.md.Analysis.NumBGFrames
            else:
                nBGFrames = 10

            self.md.setEntry('Analysis.BGRange', (-nBGFrames, 0))

        self.bgindices = range(max(self.index + self.md['Analysis.BGRange'][0], self.md.getOrDefault('EstimatedLaserOnFrameNo', 0)),
                               max(self.index + self.md['Analysis.BGRange'][1], self.md.getOrDefault('EstimatedLaserOnFrameNo', 0)))
            
    def __mapSplitterCoords(self, x,y):
        return splitting.map_splitter_coords(self.md, self.data.shape, x, y)
        
    def __remapSplitterCoords(self, x,y):
        return splitting.remap_splitter_coords(self.md, self.data.shape, x, y)
        
    def _getSplitterROIs(self):
        if not '_splitterROICache' in dir(self):
            self._splitterROICache = splitting.get_splitter_rois(self.md, self.data.shape)
            
        return self._splitterROICache
        
    def _splitImage(self, img):
        xgs, xrs, ygs, yrs = self._getSplitterROIs()
        g = img[xgs, ygs]
        r = img[xrs, yrs]
        
        #print xgs, xrs, ygs, yrs, g.shape, r.shape
        
        return numpy.concatenate((g.reshape(g.shape[0], -1, 1), r.reshape(g.shape[0], -1, 1)),2)


    def __call__(self, gui=False, taskQueue=None):
        global dBuffer, bBuffer, dataSourceID, nTasksProcessed
        
        #create a local copy of the metadata        
        md = copy.copy(self.md)
        md.tIndex = self.index
        md.taskQueue = taskQueue
        md.dataSourceID = self.dataSourceID

        #logging.debug('dataSourceID: %s, cachedDSID: %s', md.dataSourceID, bufferManager.dataSourceID)

        #make sure we're buffering the right data stream
        bufferManager.updateBuffers(md, self.dataSourceModule, self.bufferLen)
        
        self.data = bufferManager.dBuffer.getSlice(self.index)
        #if logger.isEnabledFor(logging.DEBUG):
        #    logger.debug('data: min - %3.2f, max - %3.2f, mean - %3.2f' % (self.data.min(), self.data.max(), self.data.mean()))
        nTasksProcessed += 1
        #print self.index

        #when camera buffer overflows, empty pictures are produced - deal with these here
        if self.data.max() == 0:
            return fitResult(self, [], [])
        
        #squash 4th dimension
        #NB - this now subtracts the ADOffset
        self.data = cameraMaps.correctImage(md, self.data.squeeze()).reshape((self.data.shape[0], self.data.shape[1],1))

        #calculate background
        self.bg = 0
        if not len(self.bgindices) == 0:
            if hasattr(bufferManager.bBuffer, 'calc_background') and 'GPU_BUFFER_READY' in dir(self.fitMod):
                # special case to support GPU based percentile background calculation. In this case we simply update the frame range the buffer needs
                # to look at and leave the buffer on the GPU. self.bg is then a proxy for the GPU background buffer rather than being the background data itself.
                # NB: this has the consequence that the background estimate is not subjected to dark and flat-field corrections, potentially making it unsuitable
                # for fits that are not specifically constructed to deal with uncorrected data.
                bufferManager.bBuffer.calc_background(self.bgindices)
                self.bg = bufferManager.bBuffer  # NB - GPUFIT flag means the fit can handle an asynchronous background calculation
                
                # TODO - due to the way the GPU fits in warpDrive works (undoing our flatfielding corrections), it would make sense to have a special case 
                # before the data correction 
            else:
                # the "normal" way - calculate the background for this frame and correct this for camera characteristics
                self.bg = cameraMaps.correctImage(md, bufferManager.bBuffer.getBackground(self.bgindices)).reshape(self.data.shape)

        #calculate noise
        self.sigma = self.calcSigma(md, self.data)

        #if logger.isEnabledFor(logging.DEBUG):
        #    logger.debug('data_mean: %3.2f, bg: %3.2f, sigma: %3.2f' % (self.data.mean(), self.sigma.mean(), self.bg.mean()))
        
        #############################################
        # Special cases - defer object finding to fit module
        
        if self.fitModule == 'ConfocCOIR': #special case - no object finding
            self.res = self.fitMod.ConfocCOI(self.data, md, background = self.bg)
            return fitResult(self, self.res, [])
            
        if 'MULTIFIT' in dir(self.fitMod):
            #fit module does it's own object finding
            ff = self.fitMod.FitFactory(self.data, md, background = self.bg, noiseSigma=self.sigma)
            self.res = ff.FindAndFit(self.threshold, gui=gui, cameraMaps=cameraMaps)
            return fitResult(self, self.res, [])
            

        ##############################################        
        # Find candidate molecule positions
        bgd = (self.data.astype('f') - self.bg)
        
        #print bgd.shape, self.calcThreshold().shape

        if 'Splitter.TransmittedChannel' in self.md.getEntryNames():
            #don't find points in transmitted light channel
            transChan = md.getEntry('Splitter.TransmitedChannel')
            if transChan == 'Top':
                bgd[:, :(self.data.shape[1]/2)] = 0 #set upper half of image to zero
    
        #define splitter mapping function (if appropriate) for use in object finding
        sfunc = None        
        if self.is_splitter_fit:
            sfunc = self.__mapSplitterCoords

        #Choose which version of ofind to use
        if 'PRI.Axis' in self.md.getEntryNames() and not self.md['PRI.Axis'] == 'none':
            self.ofd = ofind_pri.ObjectIdentifier(bgd * (bgd > 0), md, axis = self.md['PRI.Axis'])
        else:# not 'PSFFile' in self.md.getEntryNames():
            filtRadiusLowpass = md.getOrDefault('Analysis.DetectionRadiusLowpass', 1.0)
            filtRadiusHighpass = md.getOrDefault('Analysis.DetectionRadiusHighpass', 3.0)
            self.ofd = ofind.ObjectIdentifier(bgd * (bgd > 0), filterRadiusLowpass=filtRadiusLowpass, filterRadiusHighpass=filtRadiusHighpass)
        #else: #if we've got a PSF then use cross-correlation object identificatio      
        #    self.ofd = ofind_xcorr.ObjectIdentifier(bgd * (bgd > 0), md, 7, 5e-2)
        
        debounce = self.md.getOrDefault('Analysis.DebounceRadius', 5)    
        discardClumpRadius = self.md.getOrDefault('Analysis.ClumpRejectRadius', 0)
            
        self.ofd.FindObjects(self.calcThreshold(),0, splitter=sfunc, debounceRadius=debounce, discardClumpRadius = discardClumpRadius)
        
        
        ####################################################
        # Find Fiducials
        if self.driftEst:
            self.driftEstInd = self.index + self.md.getOrDefault('Analysis.DriftIndices', np.array([-10, 0, 10]))        
            self.calObjThresh = self.md.getOrDefault('Analysis.FiducialThreshold', 6)
            fiducialROISize = self.md.getOrDefault('Analysis.FiducialROISize', 11)
            fiducialSize = self.md.getOrDefault('Analysis.FiducalSize', 1000.)
            self._findFiducials(sfunc, debounce, fiducialSize)
                
        
        #####################################################################
        #If we are using a splitter, chop the largest common ROI out of the two channels
        
        if self.is_splitter_fit:
            self.data = self._splitImage(self.data)
            self.sigma = self._splitImage(self.sigma)
            
            if not len(self.bgindices) == 0:
                self.bg = self._splitImage(self.bg)
                 
        
        #If we're running under a gui - display found objects
        if gui:
            self._displayFoundObjects()

        #########################################################
        #Create a fit 'factory'
        fitFac = self.fitMod.FitFactory(self.data, md, background = self.bg, noiseSigma = self.sigma)
        
        #perform fit for each point that we detected
        if 'FromPoints' in dir(self.fitMod):
            self.res = self.fitMod.FromPoints(self.ofd)
        elif 'FitResultsDType' in dir(self.fitMod): #legacy fit modules
            self.res = numpy.empty(len(self.ofd), self.fitMod.FitResultsDType)
            if 'Analysis.ROISize' in md.getEntryNames():
                rs = md.getEntry('Analysis.ROISize')
                for i in range(len(self.ofd)):
                    p = self.ofd[i]
                    self.res[i] = fitFac.FromPoint(p.x, p.y, roiHalfSize=rs)
            else:
                for i in range(len(self.ofd)):
                    p = self.ofd[i]
                    self.res[i] = fitFac.FromPoint(p.x, p.y)
        else:
            self.res  = [fitFac.FromPoint(p.x, p.y) for p in self.ofd]

        #Fit Fiducials NOTE: This is potentially broken        
        self.drRes = []
        if self.driftEst:
            fitFac = self.fitMod.FitFactory(self.data, md, noiseSigma = self.sigma)
            nToFit = min(10,len(self.ofdDr)) #don't bother fitting lots of calibration objects 
            if 'FitResultsDType' in dir(self.fitMod):
                self.drRes = numpy.empty(nToFit, self.fitMod.FitResultsDType)
                for i in range(nToFit):
                    p = self.ofdDr[i]
                    self.drRes[i] = fitFac.FromPoint(p.x, p.y, roiHalfSize=fiducialROISize)
            else:
                self.drRes  = [fitFac.FromPoint(p.x, p.y) for p in self.ofd[:nToFit]] 
                
            #print 'Fitted %d fiducials' % nToFit, len(self.drRes)


        return fitResult(self, self.res, self.drRes)

    @classmethod
    def calcSigma(cls, md, data):
        var = np.atleast_3d(cameraMaps.getVarianceMap(md)) # this must be float type!! Should we enforce with an 'astype' call?
        return np.sqrt(var + (float(md.Camera.NoiseFactor)**2)*(float(md.Camera.ElectronsPerCount)*float(md.Camera.TrueEMGain)*np.maximum(data, 1.0) + float(md.Camera.TrueEMGain)*float(md.Camera.TrueEMGain)))/float(md.Camera.ElectronsPerCount)
    
    def calcThreshold(self):
        #from scipy import ndimage
        if self.SNThreshold:
            fudgeFactor = 1 #to account for the fact that the blurring etc... in ofind doesn't preserve intensities - at the moment completely arbitrary so a threshold setting of 1 results in reasonable detection.
            #return self.calcSigma(self.md, self.data - self.md.Camera.ADOffset)*fudgeFactor*self.threshold 
            return (self.sigma*fudgeFactor*self.threshold).squeeze()
            #return (numpy.sqrt(self.md.Camera.ReadNoise**2 + numpy.maximum(self.md.Camera.ElectronsPerCount*(self.md.Camera.NoiseFactor**2)*(ndimage.gaussian_filter((self.data.astype('f') - self.md.Camera.ADOffset).sum(2), 2))*self.md.Camera.TrueEMGain, 1))/self.md.Camera.ElectronsPerCount)*fudgeFactor*self.threshold
        else:
            return self.threshold


    
    def _findFiducials(self, sfunc, debounce, fiducalSize):
        ####################################################
        # Find Fiducials

        self.mIm = numpy.ones(self.data.shape, 'f')
        
        mdnm  = 1./np.median((self.data/self.sigma).ravel())
        
        for dri in self.driftEstInd:
            bs = cameraMaps.correctImage(self.md, bufferManager.dBuffer.getSlice(dri).squeeze())
            bs = bs.reshape(self.data.shape)/self.sigma
            bs = bs*mdnm
            #multiply images together, thus favouring images which are on over multiple frames
            self.mIm = self.mIm*bs
        
        #self.mIm = numpy.absolute(self.mIm)
        #if not 'PSFFile' in self.md.getEntryNames():
        fid_scale = max((fiducalSize/100.)*0.5, 0)
        self.ofdDr = ofind.ObjectIdentifier(self.mIm, filterRadiusLowpass=(1 + fid_scale), filterRadiusHighpass=(3 + fid_scale))
        #else:
        #    self.ofdDr = ofind_xcorr.ObjectIdentifier(self.mIm, self.md.getEntry('PSFFile'), 7, 3e-2)
            
        thres = self.calObjThresh**len(self.driftEstInd)
        self.ofdDr.FindObjects(thres,0, splitter=sfunc, debounceRadius=debounce)
        
        #while len(self.ofdDr) >= 10: #just go for the brightest ones
        #    thres = thres * max(2, len(self.ofdDr)/5)
        #    self.ofdDr.FindObjects(thres,0, splitter=sfunc, debounceRadius=debounce)  
            
    def _displayFoundObjects(self):
        import pylab
        #cm = pylab.cm
        pylab.clf()
        pylab.imshow(self.ofd.filteredData.T, cmap=pylab.cm.hot, hold=False)
        xc = np.array([p.x for p in self.ofd])
        yc = np.array([p.y for p in self.ofd])
        pylab.plot(xc, yc, 'o', mew=2, mec='g', mfc='none', ms=9)

        if self.is_splitter_fit:
            xn, yn = self.__remapSplitterCoords(xc, yc)
            pylab.plot(xn, yn, 'o', mew=2, mec='r', mfc='none', ms=9)


        if self.driftEst:
            pylab.plot([p.x for p in self.ofdDr], [p.y for p in self.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
        #axis('image')
        #gca().set_ylim([255,0])
        pylab.colorbar()
        pylab.show()

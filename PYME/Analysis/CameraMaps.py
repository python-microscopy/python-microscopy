# map reading and processing code
# this should really go in its own module
# currently this typically fails on all but the acquiring machine (because other machines do not have the maps)
import warnings
import os
from PYME.gohlke import tifffile as tif
import numpy
from scipy.signal import convolve2d

defaultBlemishThresholds = {
    'maxReadNoise' : 10.3,
    'maxOffsetDeviation' : 10,
    'maxGainDeviation' : 0.3,
}

# missing functionality:
# - make maps from raw series data
# - check the right gain mode has been used, etc; needs metadata
# - save map in a number of formats
# - io and map creation should use metadata appropriately

# todos:
# - gain should probably be renamed to flatfield
class cmaps:
    blemishVal = 1e7
    def __init__(self,gain=None,offset=None,variance=None,readnoise=None,metadata=None):
        # not sure what has to go here yet
        self.gain = gain
        self.offset = offset
        self.variance = variance
        self.readnoise = readnoise
        self.metadata = metadata
        self.readnoiseB = None
        self.blemishmask = None
        self.thresholds = defaultBlemishThresholds.copy()

        self.updateReadNoise()
        
    def updateReadNoise(self):
        if self.variance is not None:
            if self.metadata is None:
                epc = 0.5 # for now we assume a default, FIXME!
            else:
                epc = self.metadata.getEntry('Camera.ElectronsPerCount')
                self.variance.shape
            self.readnoise = epc*numpy.sqrt(self.variance)
            self.genBlemishes()
        
    @staticmethod
    def _mkROI(map1,md):
        return map1[md.Camera.ROIPosX-1:md.Camera.ROIPosX-1+md.Camera.ROIWidth,
                    md.Camera.ROIPosY-1:md.Camera.ROIPosY-1+md.Camera.ROIHeight]

    def makeROI(self,mdh):
        mapROI = cmaps()
        cmapd = vars(self)
        mROId = vars(mapROI)
        for mp in ['gain','offset','variance','readnoise','readnoiseB']:
            if cmapd[mp] is not None:
                mROId[mp] = self._mkROI(cmapd[mp],mdh)
        mapROI.metadata = mdh # should be a copy! FIXME
        mapROI.updateReadNoise()
        return mapROI

    def genBlemishes(self,maxReadNoise=None,maxOffsetDeviation=None,maxGainDeviation=None):
        if maxReadNoise is not None:
            self.thresholds['maxReadNoise'] = maxReadNoise
        if maxGainDeviation is not None:
            self.thresholds['maxGainDeviation'] = maxGainDeviation
        if maxOffsetDeviation is not None:
            self.thresholds['maxOffsetDeviation'] = maxOffsetDeviation

        offmedian = numpy.median(self.offset)
        if self.readnoise is None:
            self.updateReadNoise()
        rn = self.readnoise.copy()
        rnm,omd,gmd = [self.thresholds[prop] for prop in
                      ['maxReadNoise','maxOffsetDeviation','maxGainDeviation']]
        rn[rn > rnm] = self.blemishVal
        rn[self.offset>offmedian+omd] = self.blemishVal
        rn[self.gain > 1.0+gmd] = self.blemishVal
        rn[self.gain < 1.0-gmd] = self.blemishVal
        self.readnoiseB = rn
        # keep a record of all bad pixels in a mask
        self.blemishmask = self.readnoiseB >= self.blemishVal

    @staticmethod
    def _s8conv(data):
        kernel = numpy.ones((3,3))
        kernel[1,1] = 0
        return convolve2d(data,kernel,mode='same')

    # replace all blemished pixels with the average of their good neighbours in a 3x3 neighbourhood
    def blemishfilter(self,raw):
        # mask: 1 - good pixel, 0 - bad pixel
        mask = 1-1*self.blemishmask
        filtered = (1-mask)*(self._s8conv(mask*raw)/self._s8conv(mask))+mask*raw
        return filtered

def tifReadMaps():
    # need to add some notion of metadata
    mapdir = os.getenv('PYMEZYLAMAPDIR',
                       default='C:/python-microscopy-exeter/PYME/Analysis/FitFactories/')
    maps = {}
    entries = ['offset','variance','gain']
    try:
        for mapname in entries:
            maps[mapname] = tif.imread(os.path.join(mapdir,'%s.tif' % mapname))
        print 'loader V3: loaded Zyla maps'
    except:
        warnings.warn('cannot load Zyla property maps')
        pass
    maps['gain'] /= maps['gain'].mean()
    return cmaps(gain=maps['gain'],offset=maps['offset'],
                 variance=maps['variance'])

def txtReadMaps():
    # need to add some notion of metadata
    mapdir = os.getenv('PYMEZYLAMAPDIR',
                       default='C:/python-microscopy-exeter/PYME/Analysis/FitFactories/')
    maps = {}
    entries = ['offset','variance','gain']
    try:
        for mapname in entries:
            maps[mapname] = numpy.loadtxt(os.path.join(mapdir,'%s.txt' % mapname))
        print 'loader V3: loaded Zyla maps'
    except:
        warnings.warn('cannot load Zyla property maps')
        pass
    maps['gain'] /= maps['gain'].mean()
    return cmaps(gain=maps['gain'],offset=maps['offset'],
                 variance=maps['variance'])

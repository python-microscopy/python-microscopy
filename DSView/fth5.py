import Pyro.core
import sys
#sys.path.append('/home/david/pysmi_simulator/py_fit')
from PYME.Analysis import remFitHDF
import os
from PYME.Analysis import MetaData
from pylab import *

tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')

from PYME.ParallelTasks.relativeFiles import getRelFilename
from PYME.FileUtils.nameUtils import genResultFileName

seriesName = getRelFilename(h5file.filename)

md = MetaData.TIRFDefault

#Guestimate when the laser was turned on
tLon = argmax((diff(ds[:200, :,:].mean(2).mean(1))))
print "tLon = %d" % tLon

#Estimate the offset during the dark time before laser was turned on
#N.B. this will not work if other lights (e.g. room lights, arc lamp etc... are on
md.CCD.ADOffset = ds[:(tLon - 1), :,:].ravel().mean()

#Quick hack for approximate EMGain for gain register settings of 150 & 200
#FIXME to use a proper calibration
if h5file.root.MetaData.Camera._v_attrs.EMGain == 200: #gain register setting
    md.CCD.EMGain = 100 #real gain @ -50C - from curve in performance book - need proper calibration

if h5file.root.MetaData.Camera._v_attrs.EMGain == 150: #gain register setting
    md.CCD.EMGain = 20 #real gain @ -50C - need proper calibration



def pushImages(startingAt=0, detThresh = .9):
    tq.createQueue('HDFResultsTaskQueue', seriesName, genResultFileName(seriesName))
    for i in range(startingAt, ds.shape[0]):
        tq.postTask(remFitHDF.fitTask(seriesName,i, detThresh, md, 'LatGaussFitFR', bgindices=range(max(i-10, tLon),i), SNThreshold=True), queueName=seriesName)


def testFrame(detThresh = 0.9):
    ft = remFitHDF.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', bgindices=range(max(vp.zp-10, tLon),vp.zp), SNThreshold=True)
    return ft(True)



#import fitIO




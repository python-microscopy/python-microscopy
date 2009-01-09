import Pyro.core
import sys
#sys.path.append('/home/david/pysmi_simulator/py_fit')
from PYME.Analysis import remFitBuf
import os
from PYME.Analysis import MetaData
from pylab import *

tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')

from PYME.ParallelTasks.relativeFiles import getRelFilename
#from PYME.FileUtils.nameUtils import genResultFileName

seriesName = getRelFilename(h5file.filename)

md = MetaData.genMetaDataFromHDF(h5file)


def pushImages(startingAt=0, detThresh = .9):
    tq.createQueue('HDFResultsTaskQueue', seriesName, None)
    for i in range(startingAt, ds.shape[0]):
        tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, md, 'LatGaussFitFR', bgindices=range(max(i-10,md.EstimatedLaserOnFrameNo ),i), SNThreshold=True), queueName=seriesName)


def testFrame(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True)
    return ft(True)



#import fitIO




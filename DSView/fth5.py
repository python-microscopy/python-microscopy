import Pyro.core
import sys
#sys.path.append('/home/david/pysmi_simulator/py_fit')
from PYME.Analysis import remFitHDF
import os
from PYME.Analysis import MetaData

tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')

from PYME.ParallelTasks.relativeFiles import getRelFilename

seriesName = getRelFilename(h5file.filename)

def pushImages(startingAt=0, detThresh = .9):
    for i in range(startingAt, ds.shape[0]):
        tq.postTask(remFitHDF.fitTask(seriesName,i, detThresh, MetaData.TIRFDefault, 'LatGaussFitF', bgindices=range(max(i-10, 0),i), SNThreshold=True), queueName=seriesName)


def testFrame(detThresh = 0.9):
    ft = remFitHDF.fitTask(seriesName,vp.zp, detThresh, MetaData.TIRFDefault, 'LatGaussFitF', bgindices=range(max(vp.zp-10, 0),vp.zp), SNThreshold=True)
    return ft(True)

import fitIO




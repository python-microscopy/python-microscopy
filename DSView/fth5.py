import Pyro.core
import sys
#sys.path.append('/home/david/pysmi_simulator/py_fit')
from PYME.Analysis import remFitBuf
import os
from PYME.Analysis import MetaData
from pylab import *
import matplotlib

from PYME.Acquire import ExecTools

ExecTools.execBG("tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')", locals(), globals())

from PYME.ParallelTasks.relativeFiles import getRelFilename
#from PYME.FileUtils.nameUtils import genResultFileName

seriesName = getRelFilename(h5file.filename)

md = MetaData.genMetaDataFromHDF(h5file)

vp.zp = md.EstimatedLaserOnFrameNo

vp.Refresh()

def pushImages(startingAt=0, detThresh = .9):
    tq.createQueue('HDFResultsTaskQueue', seriesName, None)
    for i in range(startingAt, ds.shape[0]):
        tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, md, 'LatGaussFitFR', bgindices=range(max(i-10,md.EstimatedLaserOnFrameNo ),i), SNThreshold=True), queueName=seriesName)


def testFrame(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True)
    return ft(True)

def pushImagesD(startingAt=0, detThresh = .9):
    tq.createQueue('HDFResultsTaskQueue', seriesName, None)
    for i in range(startingAt, ds.shape[0]):
        tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, md, 'LatGaussFitFR', bgindices=range(max(i-10,md.EstimatedLaserOnFrameNo ),i), SNThreshold=True,driftEstInd=range(max(i-5, md.EstimatedLaserOnFrameNo),min(i + 5, ds.shape[0]))), queueName=seriesName)


def testFrameD(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True,driftEstInd=range(max(vp.zp-5, md.EstimatedLaserOnFrameNo),min(vp.zp + 5, ds.shape[0])))
    return ft(True)


def testFrames(detThresh = 0.9, offset = 0):
    close('all')
    matplotlib.interactive(False)
    clf()
    sq = min(md.EstimatedLaserOnFrameNo + 1000, ds.shape[0]/4)
    zps = array(range(md.EstimatedLaserOnFrameNo + 20, md.EstimatedLaserOnFrameNo + 24)  + range(sq, sq + 4) + range(ds.shape[0]/2, ds.shape[0]/2+4))
    zps += offset
    for i in range(12):
        ft = remFitBuf.fitTask(seriesName, zps[i], detThresh, md, 'LatObjFindFR', bgindices=range(max(zps[i] -10, md.EstimatedLaserOnFrameNo), zps[i]), SNThreshold=True)
        res = ft()
        xp = floor(i/4)/3.
        yp = (3 - i%4)/4.
        #print xp, yp
        axes((xp,yp, 1./6,1./4.5))
        d = ds[zps[i], :,:].squeeze().T
        imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
        title('Frame %d' % zps[i])
        xlim(0, d.shape[1])
        ylim(0, d.shape[0])
        xticks([])
        yticks([])
        #print 'i = %d, ft.index = %d' % (i, ft.index)
        #subplot(4,6,2*i+13)
        xp += 1./6
        axes((xp,yp, 1./6,1./4.5))
        d = ft.ofd.filteredData.T
        #d = ft.data.squeeze().T
        imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
        plot([p.x for p in ft.ofd], [p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
        if ft.driftEst:
             plot([p.x for p in ft.ofdDr], [p.y for p in ft.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
        #axis('tight')
        xlim(0, d.shape[1])
        ylim(0, d.shape[0])
        xticks([])
        yticks([])
    show()
    matplotlib.interactive(True)


#import fitIO




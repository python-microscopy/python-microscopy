import Pyro.core
import sys
#sys.path.append('/home/david/pysmi_simulator/py_fit')
from PYME.Analysis import remFitBuf
import os
from PYME.Analysis import MetaData
from PYME.Acquire import MetaDataHandler
from pylab import *
import matplotlib

from PYME.Acquire import ExecTools
#from PYME.Analysis.DataSources.HDFDataSource import DataSource

if not 'tq' in locals():
    ExecTools.execBG("tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')", locals(), globals())

#from PYME.ParallelTasks.relativeFiles import getRelFilename
#from PYME.FileUtils.nameUtils import genResultFileName

#seriesName = getRelFilename(dataSource.h5File.filename)

#if 'MetaData' in h5file.root: #should be true the whole time
#    mdh = MetaDataHandler.HDFMDHandler(h5file)
#else:
#    mdh = None
#    import wx
#    if not None == wx.GetApp():
#        wx.MessageBox("Carrying on with defaults - no gaurantees it'll work well", 'ERROR: No metadata fond in file ...', wx.ERROR|wx.OK)
#    print "ERROR: No metadata fond in file ... Carrying on with defaults - no gaurantees it'll work well"

#dataSource = DataSource(h5file.filename, None)

#md = MetaData.genMetaDataFromSourceAndMDH(dataSource, mdh)

MetaData.fillInBlanks(mdh, dataSource)

md = MetaDataHandler.NestedClassMDHandler(mdh)

if 'Protocol.DataStartsAt' in mdh.getEntyNames():
    vp.zp = mdh.getEntry('Protocol.DataStartsAt')
else:
    vp.zp = mdh.getEntry('EstimatedLaserOnFrameNo')

vp.Refresh()

def pushImages(startingAt=0, detThresh = .9):
    if dataSource.moduleName == 'HDFDataSource':
        pushImagesHDF(startingAt, detThresh)
    else:
        pushImagesQueue(startingAt, detThresh)

def pushImagesHDF(startingAt=0, detThresh = .9):
    tq.createQueue('HDFResultsTaskQueue', seriesName, None)
    mdhQ = MetaDataHandler.QueueMDHandler(tq, seriesName, mdh)
    mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
    for i in range(startingAt, ds.shape[2]):
        tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, md, 'LatGaussFitFR', bgindices=range(max(i-10,md.EstimatedLaserOnFrameNo ),i), SNThreshold=True), queueName=seriesName)

def pushImagesQueue(startingAt=0, detThresh = .9):
    mdh.setEntry('Analysis.DetectionThreshold', detThresh)
    mdh.setEntry('Analysis.FitModule', 'LatGaussFitFR')
    #if not 'Camera.TrueEMGain' in mdh.getEntryNames():
    #    MetaData.fillInBlanks(mdh, dataSource)
    tq.releaseTasks(seriesName, startingAt)
    

def testFrame(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True)
    return ft(True)

def testFrameTQ(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', 'TQDataSource', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True)
    return ft(True, tq)

def pushImagesD(startingAt=0, detThresh = .9):
    tq.createQueue('HDFResultsTaskQueue', seriesName, None)
    mdhQ = MetaDataHandler.QueueMDHandler(tq, seriesName, mdh)
    mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
    for i in range(startingAt, ds.shape[0]):
        tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, md, 'LatGaussFitFR', bgindices=range(max(i-10,md.EstimatedLaserOnFrameNo ),i), SNThreshold=True,driftEstInd=range(max(i-5, md.EstimatedLaserOnFrameNo),min(i + 5, ds.shape[0])), dataSourceModule=dataSource.moduleName), queueName=seriesName)


def testFrameD(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True,driftEstInd=range(max(vp.zp-5, md.EstimatedLaserOnFrameNo),min(vp.zp + 5, ds.shape[0])))
    return ft(True)


def testFrames(detThresh = 0.9, offset = 0):
    close('all')
    matplotlib.interactive(False)
    clf()
    sq = min(md.EstimatedLaserOnFrameNo + 1000, dataSource.getNumSlices()/4)
    zps = array(range(md.EstimatedLaserOnFrameNo + 20, md.EstimatedLaserOnFrameNo + 24)  + range(sq, sq + 4) + range(dataSource.getNumSlices()/2,dataSource.getNumSlices() /2+4))
    zps += offset
    for i in range(12):
        ft = remFitBuf.fitTask(seriesName, zps[i], detThresh, md, 'LatObjFindFR', bgindices=range(max(zps[i] -10, md.EstimatedLaserOnFrameNo), zps[i]), SNThreshold=True)
        res = ft()
        xp = floor(i/4)/3.
        yp = (3 - i%4)/4.
        #print xp, yp
        axes((xp,yp, 1./6,1./4.5))
        #d = ds[zps[i], :,:].squeeze().T
        d = dataSource.getSlice(zps[i]).T
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




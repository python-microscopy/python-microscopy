import os.path
#!/usr/bin/python

##################
# fth5.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Pyro.core
from PYME.Analysis import remFitBuf
import os
import wx
from PYME.Acquire import MetaDataHandler
from pylab import *
from PYME.FileUtils import fileID
from PYME.FileUtils.nameUtils import genResultFileName

if not 'tq' in locals():
    if 'PYME_TASKQUEUENAME' in os.environ.keys():
        taskQueueName = os.environ['PYME_TASKQUEUENAME']
    else:
        taskQueueName = 'taskQueue'
    tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)


def pushImages(startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
    if dataSource.moduleName == 'HDFDataSource':
        pushImagesHDF(startingAt, detThresh, fitFcn)
    else:
        pushImagesQueue(startingAt, detThresh, fitFcn)

#def pushImagesHDF(startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
#    tq.createQueue('HDFResultsTaskQueue', seriesName, None)
#    mdhQ = MetaDataHandler.QueueMDHandler(tq, seriesName, mdh)
#    mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
#    mdhQ.setEntry('Analysis.FitModule', fitFcn)
#    mdhQ.setEntry('Analysis.DataFileID', fileID.genDataSourceID(dataSource))
#    evts = dataSource.getEvents()
#    if len(evts) > 0:
#        tq.addQueueEvents(seriesName, evts)
#    tasks = []
#    for i in range(startingAt, ds.shape[2]):
#        if 'Analysis.BGRange' in mdh.getEntryNames():
#            bgi = range(max(i + mdh.getEntry('Analysis.BGRange')[0],mdh.getEntry('EstimatedLaserOnFrameNo')), max(i + mdh.getEntry('Analysis.BGRange')[1],mdh.getEntry('EstimatedLaserOnFrameNo')))
#        elif 'Analysis.NumBGFrames' in mdh.getEntryNames():
#            bgi = range(max(i - mdh.getEntry('Analysis.NumBGFrames'),mdh.getEntry('EstimatedLaserOnFrameNo')), i)
#        else:
#            bgi = range(max(i - 10,mdh.getEntry('EstimatedLaserOnFrameNo')), i)
#        #tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), fitFcn, bgindices=bgi, SNThreshold=True), queueName=seriesName)
#        tasks.append(remFitBuf.fitTask(seriesName,i, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), fitFcn, bgindices=bgi, SNThreshold=True))
#    tq.postTasks(tasks, queueName=seriesName)

def pushImagesHDF(startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
    global seriesName
    dataFilename = seriesName
    resultsFilename = genResultFileName(seriesName)
    while os.path.exists(resultsFilename):
        di, fn = os.path.split(resultsFilename)
        fdialog = wx.FileDialog(None, 'Analysis file already exists, please select a new filename',
                    wildcard='H5R files|*.h5r', defaultDir=di, defaultFile=os.path.splitext(fn)[0] + '_1.h5r', style=wx.SAVE)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            resultsFilename = fdialog.GetPath().encode()
        else:
            raise RuntimeError('Invalid results file - not running')
        seriesName = resultsFilename
    tq.createQueue('HDFTaskQueue', seriesName, dataFilename = dataFilename, resultsFilename=resultsFilename, startAt = 'notYet')
    mdhQ = MetaDataHandler.QueueMDHandler(tq, seriesName, mdh)
    mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
    mdhQ.setEntry('Analysis.FitModule', fitFcn)
    mdhQ.setEntry('Analysis.DataFileID', fileID.genDataSourceID(dataSource))
    evts = dataSource.getEvents()
    if len(evts) > 0:
        tq.addQueueEvents(seriesName, evts)
#    tasks = []
#    for i in range(startingAt, ds.shape[2]):
#        if 'Analysis.BGRange' in mdh.getEntryNames():
#            bgi = range(max(i + mdh.getEntry('Analysis.BGRange')[0],mdh.getEntry('EstimatedLaserOnFrameNo')), max(i + mdh.getEntry('Analysis.BGRange')[1],mdh.getEntry('EstimatedLaserOnFrameNo')))
#        elif 'Analysis.NumBGFrames' in mdh.getEntryNames():
#            bgi = range(max(i - mdh.getEntry('Analysis.NumBGFrames'),mdh.getEntry('EstimatedLaserOnFrameNo')), i)
#        else:
#            bgi = range(max(i - 10,mdh.getEntry('EstimatedLaserOnFrameNo')), i)
#        #tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), fitFcn, bgindices=bgi, SNThreshold=True), queueName=seriesName)
#        tasks.append(remFitBuf.fitTask(seriesName,i, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), fitFcn, bgindices=bgi, SNThreshold=True))
#    tq.postTasks(tasks, queueName=seriesName)
    tq.releaseTasks(seriesName, startingAt)


def pushImagesQueue(startingAt=0, detThresh = .9, fitFcn='LatGaussFitFR'):
    mdh.setEntry('Analysis.DetectionThreshold', detThresh)
    mdh.setEntry('Analysis.FitModule', fitFcn)
    mdh.setEntry('Analysis.DataFileID', fileID.genDataSourceID(dataSource))
    #if not 'Camera.TrueEMGain' in mdh.getEntryNames():
    #    MetaData.fillInBlanks(mdh, dataSource)
    tq.releaseTasks(seriesName, startingAt)
    

def testFrame(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), cFitType.GetString(cFitType.GetSelection()), bgindices=range(max(vp.zp-10, mdh.getEntry('EstimatedLaserOnFrameNo')),vp.zp), SNThreshold=True)
    return ft(True)

def testFrameTQ(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatGaussFitFR', 'TQDataSource', bgindices=range(max(vp.zp-10, mdh.getEntry('EstimatedLaserOnFrameNo')),vp.zp), SNThreshold=True)
    return ft(True, tq)

def pushImagesD(startingAt=0, detThresh = .9):
    tq.createQueue('HDFResultsTaskQueue', seriesName, None)
    mdhQ = MetaDataHandler.QueueMDHandler(tq, seriesName, mdh)
    mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
    for i in range(startingAt, ds.shape[0]):
        tq.postTask(remFitBuf.fitTask(seriesName,i, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatGaussFitFR', bgindices=range(max(i-10,mdh.getEntry('EstimatedLaserOnFrameNo') ),i), SNThreshold=True,driftEstInd=range(max(i-5, mdh.getEntry('EstimatedLaserOnFrameNo')),min(i + 5, ds.shape[0])), dataSourceModule=dataSource.moduleName), queueName=seriesName)

#def testFrameD(detThresh = 0.9):
#    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, md, 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True,driftEstInd=range(max(vp.zp-5, md.EstimatedLaserOnFrameNo),min(vp.zp + 5, ds.shape[0])))
#    return ft(True)

def testFrameD(detThresh = 0.9):
    ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True,driftEstInd=range(max(vp.zp-5, md.EstimatedLaserOnFrameNo),min(vp.zp + 5, ds.shape[0])))
    return ft(True)

def testFrames(detThresh = 0.9, offset = 0):
    close('all')
    matplotlib.interactive(False)
    clf()
    sq = min(mdh.getEntry('EstimatedLaserOnFrameNo') + 1000, dataSource.getNumSlices()/4)
    zps = array(range(mdh.getEntry('EstimatedLaserOnFrameNo') + 20, mdh.getEntry('EstimatedLaserOnFrameNo') + 24)  + range(sq, sq + 4) + range(dataSource.getNumSlices()/2,dataSource.getNumSlices() /2+4))
    zps += offset
    fitMod = cFitType.GetStringSelection()
    #bgFrames = int(tBackgroundFrames.GetValue())
    bgFrames = [int(v) for v in tBackgroundFrames.GetValue().split(':')]
    for i in range(12):
        #if 'Analysis.NumBGFrames' in md.getEntryNames():
        #bgi = range(max(zps[i] - bgFrames,mdh.getEntry('EstimatedLaserOnFrameNo')), zps[i])
        bgi = range(max(zps[i] + bgFrames[0],mdh.getEntry('EstimatedLaserOnFrameNo')), max(zps[i] + bgFrames[1],mdh.getEntry('EstimatedLaserOnFrameNo')))
        #else:
        #    bgi = range(max(zps[i] - 10,md.EstimatedLaserOnFrameNo), zps[i])
        if 'Splitter' in fitMod:
            ft = remFitBuf.fitTask(seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'SplitterObjFindR', bgindices=bgi, SNThreshold=True)
        else:
            ft = remFitBuf.fitTask(seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatObjFindFR', bgindices=bgi, SNThreshold=True)
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
        if ft.fitModule in remFitBuf.splitterFitModules:
                plot([p.x for p in ft.ofd], [d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
        #axis('tight')
        xlim(0, d.shape[1])
        ylim(0, d.shape[0])
        xticks([])
        yticks([])
    show()
    matplotlib.interactive(True)


#import fitIO




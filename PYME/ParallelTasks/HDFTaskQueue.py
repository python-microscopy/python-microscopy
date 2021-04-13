#!/usr/bin/python

##################
# HDFTaskQueue.py
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

import tables
from taskQueue import *
from PYME.localization.remFitBuf import fitTask

from PYME.Analysis import MetaData
from PYME.IO import MetaDataHandler

import os
import sys
import numpy as np

import logging

import time
try:
    import Queue
except ImportError:
    #py3
    import queue as Queue

from PYME.IO.FileUtils.nameUtils import genResultFileName, getFullFilename, getFullExistingFilename
from PYME.IO import h5rFile
#from PYME.IO.FileUtils.nameUtils import getFullFilename

CHUNKSIZE = 50
MAXCHUNKSIZE = 100 #allow chunk size to be improved to allow better caching

#def genDataFilename(name):
#	fn = os.g

#global lock for all calls into HDF library - on linux you seem to be able to
#get away with locking separately for each file (or maybe not locking at all -
#is linux hdf5 threadsafe?)

#tablesLock = threading.Lock()

#class TaskWatcher(threading.Thread):
#	def __init__(self, tQueue):
#		threading.Thread.__init__(self)
#		self.tQueue = tQueue
#		self.alive = True
#
#	def run(self):
#		while self.alive:
#			self.tQueue.checkTimeouts()
#			#print '%d tasks in queue' % self.tQueue.getNumberOpenTasks()
#			#try:
#                        #        mProfile.report()
#                        #finally:
#                        #        pass
#                        print mProfile.files
#			time.sleep(10)
#
#tw = TaskWatcher(tq)
#    #tw.start()

bufferMisses = 0

class dataBuffer: #buffer our io to avoid decompressing multiple times
    def __init__(self,dataSource, bLen = 1000):
        self.bLen = bLen
        self.buffer = {} #delay creation until we know the dtype
        #self.buffer = numpy.zeros((bLen,) + dataSource.getSliceShape(), 'uint16')
        #self.insertAt = 0
        #self.bufferedSlices = -1*numpy.ones((bLen,), 'i')
        self.dataSource = dataSource

    def getSlice(self,ind):
        global bufferMisses
        #print self.bufferedSlices, self.insertAt, ind
        #return self.dataSource.getSlice(ind)
        if ind in self.bufferedSlices.keys(): #return from buffer
            #print int(numpy.where(self.bufferedSlices == ind)[0])
            return self.buffer[ind]
        else: #get from our data source and store in buffer
            sl = self.dataSource[ind,:,:]
            
            self.buffer[sl] = sl

            bufferMisses += 1
            
            #if bufferMisses % 10 == 0:
            #    print nTasksProcessed, bufferMisses

            return sl

class myLock:
    def __init__(self, lock = threading.Lock()):
        self.lock = lock
        self.owner = None
        self.oowner = None
        self.aqtime = 0

    def acquire(self):
        fr = sys._getframe()
        dt = (time.time() - self.aqtime)
        if self.owner and (dt > 1):    
            logging.info(fr.f_back.f_back.f_code.co_name + ' %d' % fr.f_back.f_back.f_lineno + ' waiting on lock held by %s for %f s' % (self.owner, dt) )
        self.lock.acquire()
        
        self.owner = fr.f_back.f_back.f_code.co_name + ' %d' % fr.f_back.f_back.f_lineno
        dtt = (time.time() - self.aqtime - dt)
        if (dtt > 1):
            logging.info(self.owner + ' succesfully acquired lock held by %s for %f s after waiting %f s' % (self.oowner, dt + dtt, dtt) )
        self.aqtime = time.time()
        #fr = sys._getframe()
        #print 'Acquired Lock - ' + fr.f_back.f_code.co_name + ' %d' % fr.f_back.f_lineno

    def release(self):
        #print 'Released Lock'
        #logging.info(self.owner + ' released lock')
        self.lock.release()
        self.oowner = self.owner
        self.owner = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()

#tablesLock = myLock()

# class readLock(object):
#     def __init__(self, rwlock):
#         self.rwlock = rwlock
#
#     def acquire(self):
#         self.rwlock.acquireRead()
#
#     def release(self):
#         self.rwlock.releaseRead()
#
#     def __enter__(self):
#         self.acquire()
#
#     def __exit__(self, type, value, traceback):
#         self.release()
#

# class writeLock(readLock):
#     def acquire(self):
#         self.rwlock.acquireWrite()
#
#     def release(self):
#         self.rwlock.releaseWrite()
#
#
# class rwlock(object):
#     def __init__(self):
#         self.glock = threading.Lock()
#         self.numReaders = 0
#         self.numWriters = 0
#
#         self.rlock = myLock(readLock(self))
#         self.wlock = myLock(writeLock(self))
#
#     def addReader(self):
#         suc = False
#         with self.glock:
#             if (self.numWriters == 0) and (self.numReaders == 0):
#                 self.numReaders += 1
#                 suc = True
#
#         return suc
#
#     def acquireRead(self):
#         while not self.addReader():
#             time.sleep(.001)
#         #logging.info('Acquired read lock - nr, nw = %d, %d' % (self.numReaders, self.numWriters))
#
#     def releaseRead(self):
#         with self.glock:
#             self.numReaders -= 1
#
#     def addWriter(self):
#         suc = False
#         with self.glock:
#             if (self.numWriters == 0) and (self.numReaders == 0):
#                 self.numWriters += 1
#                 suc = True
#
#         return suc
#
#     def acquireWrite(self):
#         while not self.addWriter():
#             time.sleep(.001)
#         #logging.info('Acquired write lock - nr, nw = %d, %d' % (self.numReaders, self.numWriters))
#
#     def releaseWrite(self):
#         with self.glock:
#             self.numWriters -= 1


class rwlock2(object):
    def __init__(self):
        self.glock = h5rFile.tablesLock #threading.Lock()
        self.numReaders = 0
        self.numWriters = 0
        
        self.rlock = myLock(self.glock)
        self.wlock = self.rlock

tablesLock = rwlock2()


class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)


# class HDFResultsTaskQueue_(TaskQueue):
#     """Task queue which saves it's results to a HDF file"""
#     def __init__(self, name, resultsFilename, initialTasks=[], onEmpty = doNix, fTaskToPop = popZero):
#         """
#         Generate a task queue which saves results to an HDF5 file using pytables
#
#         NOTE: This is only ever used as a base class
#
#         Args:
#             name : string
#                 the queue name by which this set of task is identified
#             resultsFilename: string
#                 the name of the output file
#             initialTasks: list
#                 task to populate with initially - not used in practice
#             onEmpty:
#                 what to do when the list of tasks is empty (nominally for closing output files etc ... but unused)
#             fTaskToPop:
#                 a callback function which decides which task to give a worker. Returns the index of the task to return
#                 based on information about the current worker. An inital attempt at load balancing, which is now not
#                 really used.
#         """
#
#         if resultsFilename is None:
#             #autogenerate a filename if none given
#             resultsFilename = genResultFileName(name)
#
#         if os.path.exists(resultsFilename): #bail if output file already exists
#             raise RuntimeError('Output file already exists: ' + resultsFilename)
#
#         TaskQueue.__init__(self, name, initialTasks, onEmpty, fTaskToPop)
#         self.resultsFilename = resultsFilename
#
#         self.numClosedTasks = 0
#
#         logging.info('Creating results file')
#
#         self.h5ResultsFile = tables.openFile(self.resultsFilename, 'w')
#
#         self.prepResultsFile()
#
#         #self.fileResultsLock = threading.Lock()
#         self.fileResultsLock = tablesLock
#
#         logging.info('Creating results metadata')
#
#         self.resultsMDH = MetaDataHandler.HDFMDHandler(self.h5ResultsFile)
#         self.metaData = MetaDataHandler.NestedClassMDHandler()
#         #self.metaData = None #MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
#         self.metaDataStale = True
#         self.MDHCache = []
#
#         logging.info('Creating results events table')
#         with self.fileResultsLock.wlock:
#             self.resultsEvents = self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))
#
#         logging.info('Events table created')
#
#         self.haveResultsTable = False
#
#         self.resultsQueue = []#Queue.Queue()
#         self.resultsQueueLock = threading.Lock()
#         self.lastResultsQueuePurge = time.time()
#
#         logging.info('Results file initialised')
#
#     def prepResultsFile(self):
#         pass
#
#     def getCompletedTask(self):
#         return None
#
#     def flushMetaData(self):
#         if len(self.MDHCache) > 0:
#             with self.fileResultsLock.wlock:
#                 mdts = list(self.MDHCache)
#                 self.MDHCache = []
#
#                 for mdk, mdv in mdts:
#                     self.resultsMDH.setEntry(mdk, mdv)
#
#     def setQueueMetaData(self, fieldName, value):
#         #with self.fileResultsLock.wlock:
#         self.metaData.setEntry(fieldName, value)
#         self.MDHCache.append((fieldName, value))
#
#     def setQueueMetaDataEntries(self, mdh):
#         with self.fileResultsLock.wlock:
#             self.resultsMDH.copyEntriesFrom(mdh)
#
#         self.metaData.copyEntriesFrom(mdh)
#         #self.MDHCache.append((fieldName, value))
#
#
#     def getQueueMetaData(self, fieldName):
#         #res  = None
#         #with self.fileResultsLock.rlock:
#         #    res = self.resultsMDH.getEntry(fieldName)
#         return self.metaData[fieldName]
#
#         #return res
#
#     def addQueueEvents(self, events):
#         with self.fileResultsLock.wlock:
#             self.resultsEvents.append(events)
#
#
#     def getQueueMetaDataKeys(self):
#         return self.metaData.getEntryNames()
#
#     def getNumberTasksCompleted(self):
#         return self.numClosedTasks
#
#     def purge(self):
#         self.openTasks = []
#         self.numClosedTasks = 0
#         self.tasksInProgress = []
#
#     def cleanup(self):
#         #self.h5DataFile.close()
#         self.h5ResultsFile.close()
#
#
#     def fileResult(self, res):
#         """
#         Called remotely from workers to file / save results
#
#         Adds incoming results to a queue and calls fileResults when enough time has elapsed (5 sec)
#
#         Args:
#             res: a fitResults object, as defined in ParallelTasks.remFitBuf
#
#         Returns:
#
#         """
#         #print res, res.results, res.driftResults, self.h5ResultsFile
#         if res is None:
#             print('res is None')
#
#         if (len(res.results)==0) and (len(res.driftResults) == 0): #if we had a dud frame
#             print 'dud'
#             return
#
#         #print len(res.results), len(res.driftResults)
#
#         rq = None
#         with self.resultsQueueLock:
#             #logging.info('Adding result to queue')
#             self.resultsQueue.append(res)
#             #print 'rq'
#
#             t = time.time()
#             if (t > (self.lastResultsQueuePurge + 5)):# or (len(self.resultsQueue) > 20):
#                 #print 'fr'
#                 self.lastResultsQueuePurge = t
#                 rq = self.resultsQueue
#                 #print(len(rq)), 'r_q'
#                 self.resultsQueue = []
#
#             #print 'rf'
#
#         if rq:
#             #print 'frf'
#             self.fileResults(rq)
#             #print 'rff'
#
#         #logging.info('Result added to queue')
#         #logging.info('Result filed result')
#
#     def fileResults(self, ress):
#         """
#         File/save the results of fitting multiple frames
#
#         Args:
#             ress: list of fit results
#
#         Returns:
#
#         """
#
#         results = []
#         driftResults = []
#
#         for r in ress:
#             #print r, results
#             if not len(r.results) == 0:
#                 results.append(r.results)
#
#             if not len(r.driftResults) == 0:
#                 driftResults.append(r.driftResults)
#
#         #print
#         if (len(results)==0) and (len(driftResults) == 0): #if we had a dud frame
#             return
#
#         #print len(results), len(driftResults)
#
#         with self.fileResultsLock.wlock: #get a lock
#             #logging.info('write lock acquired')
#
#             if not (len(results) == 0):
#                 #print res.results, res.results == []
#                 if not self.haveResultsTable: # self.h5ResultsFile.__contains__('/FitResults'):
#                     #logging.info('creating results table')
#                     self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'FitResults', np.hstack(results), filters=tables.Filters(complevel=5, shuffle=True), expectedrows=500000)
#                     self.haveResultsTable = True
#                 else:
#                     self.h5ResultsFile.root.FitResults.append(np.hstack(results))
#
#             #print 'rs'
#
#             if not (len(driftResults) == 0):
#                 #logging.info('creating drift table')
#                 if not self.h5ResultsFile.__contains__('/DriftResults'):
#                     self.h5ResultsFile.createTable(self.h5ResultsFile.root, 'DriftResults', np.hstack(driftResults), filters=tables.Filters(complevel=5, shuffle=True), expectedrows=500000)
#                 else:
#                     self.h5ResultsFile.root.DriftResults.append(np.hstack(driftResults))
#
#             #self.h5ResultsFile.flush()
#
#         self.numClosedTasks += len(ress)
#
#     def checkTimeouts(self):
#         self.flushMetaData()
#         with self.inProgressLock:
#             curTime = time.clock()
#             for it in self.tasksInProgress:
#                 if 'workerTimeout' in dir(it):
#                     if curTime > it.workerTimeout:
#                         self.openTasks.append(it.index)
#                         self.tasksInProgress.remove(it)
#
#         with self.resultsQueueLock:
#             t = time.time()
#             if (t > (self.lastResultsQueuePurge + 10)):# or (len(self.resultsQueue) > 20):
#                 #print 'fr'
#                 self.lastResultsQueuePurge = t
#                 rq = self.resultsQueue
#                 #print(len(rq)), 'r_q'
#                 self.resultsQueue = []
#                 self.fileResults(rq)
#
#         with self.fileResultsLock.wlock: #get a lock
#             self.h5ResultsFile.flush()
#
#     def getQueueData(self, fieldName, *args):
#         """Get data, defined by fieldName and potntially additional arguments,  ascociated with queue"""
#         if fieldName == 'FitResults':
#             startingAt, = args
#             with self.fileResultsLock.rlock:
#                 if self.h5ResultsFile.__contains__('/FitResults'):
#                     res = self.h5ResultsFile.root.FitResults[startingAt:]
#                 else:
#                     res = []
#
#             return res
#         elif fieldName == 'PSF':
#             #from PYME.ParallelTasks.relativeFiles import getFullExistingFilename
#             res = None
#
#             modName = self.resultsMDH.getEntry('PSFFile')
#             mf = open(getFullExistingFilename(modName), 'rb')
#             res = np.load(mf)
#             mf.close()
#
#             return res
#         elif fieldName == 'MAP':
#             mapName, = args
#             #from PYME.ParallelTasks.relativeFiles import getFullExistingFilename
#             from PYME.IO.image import ImageStack
#
#             print('Serving map: %s' %mapName)
#             fn = getFullExistingFilename(mapName)
#             varmap = ImageStack(filename=fn, haveGUI=False).data[:,:,0].squeeze() #this should handle .tif, .h5, and a few others
#
#             return varmap
#         else:
#             return None


class HDFResultsTaskQueue(TaskQueue):
    """Task queue which saves it's results to a HDF file"""

    def __init__(self, name, resultsFilename, initialTasks=[], onEmpty=doNix, fTaskToPop=popZero):
        """
        Generate a task queue which saves results to an HDF5 file using pytables

        NOTE: This is only ever used as a base class

        Args:
            name : string
                the queue name by which this set of task is identified
            resultsFilename: string
                the name of the output file
            initialTasks: list
                task to populate with initially - not used in practice
            onEmpty:
                what to do when the list of tasks is empty (nominally for closing output files etc ... but unused)
            fTaskToPop:
                a callback function which decides which task to give a worker. Returns the index of the task to return
                based on information about the current worker. An inital attempt at load balancing, which is now not
                really used.
        """

        if resultsFilename is None:
            #autogenerate a filename if none given
            resultsFilename = genResultFileName(name)

        if os.path.exists(resultsFilename): #bail if output file already exists
            raise RuntimeError('Output file already exists: ' + resultsFilename)

        TaskQueue.__init__(self, name, initialTasks, onEmpty, fTaskToPop)
        self.resultsFilename = resultsFilename

        self.numClosedTasks = 0
        #self.fileResultsLock = threading.Lock()
        self.fileResultsLock = tablesLock

        self.metaData = MetaDataHandler.NestedClassMDHandler()
        #self.metaData = None #MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
        self.metaDataStale = True
        self.MDHCache = []

        self.resultsQueue = []#Queue.Queue()
        self.resultsQueueLock = threading.Lock()
        self.lastResultsQueuePurge = time.time()

        logging.info('Results file initialised')

    def prepResultsFile(self):
        pass

    def getCompletedTask(self):
        return None

    def flushMetaData(self):
        if len(self.MDHCache) > 0:
            new_md = dict(self.MDHCache)
            self.MDHCache = []
            with h5rFile.openH5R(self.resultsFilename, 'a') as h5f:
                h5f.updateMetadata(new_md)

    def setQueueMetaData(self, fieldName, value):
        #with self.fileResultsLock.wlock:
        self.metaData.setEntry(fieldName, value)
        self.MDHCache.append((fieldName, value))

    def setQueueMetaDataEntries(self, mdh):
        with h5rFile.openH5R(self.resultsFilename, 'a') as h5f:
            h5f.updateMetadata(mdh)

        self.metaData.update(mdh)

    def getQueueMetaData(self, fieldName):
        return self.metaData[fieldName]

    def addQueueEvents(self, events):
        with h5rFile.openH5R(self.resultsFilename, 'a') as h5f:
            h5f.addEvents(events)

    def getNumQueueEvents(self):
        with h5rFile.openH5R(self.resultsFilename, 'a') as h5f:
            res = len(h5f.events)

        return res

    def getQueueMetaDataKeys(self):
        return self.metaData.getEntryNames()

    def getNumberTasksCompleted(self):
        return self.numClosedTasks

    def purge(self):
        self.openTasks = []
        self.numClosedTasks = 0
        self.tasksInProgress = []

    def cleanup(self):
        pass
        #self.h5DataFile.close()
        #self.h5ResultsFile.close()

    def fileResult(self, res):
        """
        Called remotely from workers to file / save results

        Adds incoming results to a queue and calls fileResults when enough time has elapsed (5 sec)

        Args:
            res: a fitResults object, as defined in ParallelTasks.remFitBuf

        Returns:

        """
        self.fileResults([res,])


    def fileResults(self, ress):
        """
        File/save the results of fitting multiple frames

        Args:
            ress: list of fit results

        Returns:

        """

        with h5rFile.openH5R(self.resultsFilename, 'a') as h5f:
            for res in ress:
                if res is None:
                    logging.warn('got a none result')
                else:
                    if (len(res.results) > 0):
                        h5f.appendToTable('FitResults', res.results)

                    if (len(res.driftResults) > 0):
                        h5f.appendToTable('DriftResults', res.driftResults)

        self.numClosedTasks += len(ress)

    def checkTimeouts(self):
        self.flushMetaData()
        with self.inProgressLock:
            curTime = time.time()
            for it in self.tasksInProgress:
                if 'workerTimeout' in dir(it):
                    if curTime > it.workerTimeout:
                        self.openTasks.append(it.index)
                        self.tasksInProgress.remove(it)

    def getQueueData(self, fieldName, *args):
        """Get data, defined by fieldName and potntially additional arguments,  ascociated with queue"""
        if fieldName == 'FitResults':
            startingAt, = args
            #with self.fileResultsLock.rlock:
            #    if self.h5ResultsFile.__contains__('/FitResults'):
            #        res = self.h5ResultsFile.root.FitResults[startingAt:]
            #    else:
            #        res = []
            with h5rFile.openH5R(self.resultsFilename, 'a') as h5f:
                res = h5f.getTableData('FitResults', slice(startingAt, None))

            return res
        elif fieldName == 'PSF':
            #from PYME.ParallelTasks.relativeFiles import getFullExistingFilename
            from PYME.IO.load_psf import load_psf
            res = None

            modName = self.metaData.getEntry('PSFFile')
           # mf = open(getFullExistingFilename(modName), 'rb')
            #res = np.load(mf)
            #mf.close()
            res = load_psf(getFullExistingFilename(modName))

            return res
        elif fieldName == 'MAP':
            mapName, = args
            #from PYME.ParallelTasks.relativeFiles import getFullExistingFilename
            from PYME.IO.image import ImageStack

            print('Serving map: %s' % mapName)
            fn = getFullExistingFilename(mapName)
            varmap = ImageStack(filename=fn, haveGUI=False).data[:, :,0].squeeze() #this should handle .tif, .h5, and a few others

            return varmap
        else:
            return None


class HDFTaskQueue(HDFResultsTaskQueue):
    """ task queue which, when initialised with an hdf image filename, automatically generates tasks - should also (eventually) include support for dynamically adding to data file for on the fly analysis"""
    def __init__(self, name, dataFilename = None, resultsFilename=None, onEmpty = doNix, fTaskToPop = popZero, startAt = 'guestimate', frameSize=(-1,-1), complevel=6, complib='zlib', resultsURI = None):
        if dataFilename is None:
           self.dataFilename = genDataFilename(name)
        else:
            self.dataFilename = dataFilename

        if resultsFilename is None:
            resultsFilename = genResultFileName(self.dataFilename)
        else:
            resultsFilename = resultsFilename

        ffn = getFullFilename(self.dataFilename)

        self.acceptNewTasks = False
        self.releaseNewTasks = False

        self.postTaskBuffer = []

        initialTasks = []

        self.resultsURI = resultsURI


        if os.path.exists(ffn): #file already exists - read from it
            self.h5DataFile = tables.open_file(ffn, 'r')
            #self.metaData = MetaData.genMetaDataFromHDF(self.h5DataFile)
            self.dataMDH = MetaDataHandler.NestedClassMDHandler(MetaDataHandler.HDFMDHandler(self.h5DataFile))
            #self.dataMDH.mergeEntriesFrom(MetaData.TIRFDefault)
            self.imageData = self.h5DataFile.root.ImageData


            if startAt == 'guestimate': #calculate a suitable starting value
                tLon = self.dataMDH.EstimatedLaserOnFrameNo
                if tLon == 0:
                    startAt = 0
                else:
                    startAt = tLon + 10

            if startAt == 'notYet':
                initialTasks = []
            else:
                initialTasks = list(range(startAt, self.h5DataFile.root.ImageData.shape[0]))

            self.imNum = len(self.imageData)
            self.dataRW = False

        else: #make ourselves a new file
            self.h5DataFile = tables.open_file(ffn, 'w')
            filt = tables.Filters(complevel, complib, shuffle=True)

            self.imageData = self.h5DataFile.create_earray(self.h5DataFile.root, 'ImageData', tables.UInt16Atom(), (0,)+tuple(frameSize), filters=filt, chunkshape=(1,)+tuple(frameSize))
            self.events = self.h5DataFile.create_table(self.h5DataFile.root, 'Events', SpoolEvent,filters=filt)
            self.imNum=0
            self.acceptNewTasks = True

            self.dataMDH = MetaDataHandler.HDFMDHandler(self.h5DataFile)
            self.dataMDH.mergeEntriesFrom(MetaData.TIRFDefault)
            self.dataRW = True

        HDFResultsTaskQueue.__init__(self, name, resultsFilename, initialTasks, onEmpty, fTaskToPop)

        #self.resultsMDH.copyEntriesFrom(self.dataMDH)
        #self.metaData.copyEntriesFrom(self.resultsMDH)
        HDFResultsTaskQueue.setQueueMetaDataEntries(self, self.dataMDH)

        #copy events to results file
        if len (self.h5DataFile.root.Events) > 0:
            HDFResultsTaskQueue.addQueueEvents(self, self.h5DataFile.root.Events[:])
            #self.resultsEvents.append(self.h5DataFile.root.Events[:])

        self.queueID = name

        self.numSlices = self.imageData.shape[0]

        #self.dataFileLock = threading.Lock()
        self.dataFileLock = tablesLock
        #self.getTaskLock = threading.Lock()
        self.lastTaskTime = 0

    def prepResultsFile(self):
        pass

    def postTask(self,task):
        #self.postTaskBuffer = []

        #self.openTasks.append(task)
        #print 'posting tasks not implemented yet'
        if self.acceptNewTasks:
            with self.dataFileLock.wlock:
                self.imageData.append(task)
                self.h5DataFile.flush()
                self.numSlices = self.imageData.shape[0]

            if self.releaseNewTasks:
                self.openTasks.append(self.imNum)
            self.imNum += 1
        else:
            print("can't post new tasks")

    def postTasks(self,tasks):
        #self.openTasks += tasks
        if self.acceptNewTasks:
            with self.dataFileLock.wlock:
                #t1 = time.clock()
                #t_1 = 0
                #t_2 = 0
                for task in tasks:
                    #t1_ = time.clock()
                    #print task.dtype
                    self.imageData.append(task)
                    #self.h5DataFile.flush()
                    #self.dataFileLock.release()
                    #t2_ = time.clock()
                    #t_1 = (t2_ - t1_)
                    if self.releaseNewTasks:
                        self.openTasks.append(self.imNum)
                    self.imNum += 1
                    #t3_ = time.clock()
                    #t_2 = (t3_ - t2_)
                    #print t_1, t_2
                #t2 = time.clock()
                self.h5DataFile.flush()
                #t3 = time.clock()
                self.numSlices = self.imageData.shape[0]
                
            #print len(tasks), t2 - t1, t3 - t2
        else:
            print("can't post new tasks")
        #print 'posting tasks not implemented yet'

    def getNumberOpenTasks(self, exact=True):
        #when doing real time analysis we might not want to tasks out straight
        #away, in order that our caches still work
        nOpen = len(self.openTasks)

        #Answer truthfully if we are being asked for the exact number of tasks, 
        #we have enough tasks to give a full chunk, or if it was a while since 
        #we last gave out any tasks (necessary to make sure all tasks get 
        #processed at the end of the run.
        if exact or nOpen > CHUNKSIZE or (time.time() - self.lastTaskTime) > 2:
            return nOpen
        else: #otherwise lie and make the workers wait
            return 0

    def getTask(self, workerN = 0, NWorkers = 1):
        """get task from front of list, blocks"""
        #print 'Task requested'
        #self.getTaskLock.acquire()
        while len(self.openTasks) < 1:
            time.sleep(0.01)

        #if self.metaDataStale:
#            with self.dataFileLock.rlock:
#                self.metaData = MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
#                self.metaDataStale = False
            
        #patch up old data which doesn't have BGRange in metadata
        if not 'Analysis.BGRange' in self.metaData.getEntryNames():
            nBGFrames = self.metaData.getOrDefault('Analysis.NumBGFrames', 10)

            self.metaData.setEntry('Analysis.BGRange', (-nBGFrames, 0))
        
        taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))
        
        task = fitTask(dataSourceID=self.queueID, frameIndex=taskNum, metadata=self.metaData, dataSourceModule='TQDataSource', resultsURI=self.resultsURI)
        
        task.queueID = self.queueID
        task.initializeWorkerTimeout(time.clock())
        with self.inProgressLock:
            self.tasksInProgress.append(task)
        #self.inProgressLock.release()
        #self.getTaskLock.release()

        self.lastTaskTime = time.time()

        return task

    def getTasks(self, workerN = 0, NWorkers = 1):
        """get task from front of list, blocks"""
        #print 'Task requested'
        #self.getTaskLock.acquire()
        while len(self.openTasks) < 1:
            time.sleep(0.01)

        #if self.metaDataStale:
#            with self.dataFileLock.rlock:
#                self.metaData = MetaDataHandler.NestedClassMDHandler(self.resultsMDH)
#                self.metaDataStale = False

        tasks = []
        
        if not 'Analysis.ChunkSize' in self.metaData.getEntryNames():
            cs = min(max(CHUNKSIZE, min(MAXCHUNKSIZE, len(self.openTasks))),len(self.openTasks))
        else:
            cs = min(self.metaData['Analysis.ChunkSize'], len(self.openTasks))

        for i in range(cs):

            taskNum = self.openTasks.pop(self.fTaskToPop(workerN, NWorkers, len(self.openTasks)))

            task = fitTask(dataSourceID=self.queueID, frameIndex=taskNum, metadata=self.metaData, dataSourceModule='TQDataSource', resultsURI=self.resultsURI)

            task.queueID = self.queueID
            task.initializeWorkerTimeout(time.clock())
            with self.inProgressLock:
                self.tasksInProgress.append(task)
            

            tasks.append(task)

        self.lastTaskTime = time.time()

        return tasks

    def cleanup(self):
        self.h5DataFile.close()
        #self.h5ResultsFile.close()

    def setQueueMetaData(self, fieldName, value):
        self.metaData.setEntry(fieldName, value)
        
        HDFResultsTaskQueue.setQueueMetaData(self, fieldName, value)
        
        if self.dataRW:
            with self.dataFileLock.wlock:
                self.dataMDH.setEntry(fieldName, value)
        self.metaDataStale = True

    def setQueueMetaDataEntries(self, mdh):
        self.metaData.copyEntriesFrom(mdh)
        
        if self.dataRW:
            with self.dataFileLock.wlock:
                self.dataMDH.copyEntriesFrom(mdh)
        
        HDFResultsTaskQueue.setQueueMetaDataEntries(self, mdh)
        self.metaDataStale = True

    def flushMetaData(self):
        with self.fileResultsLock.wlock:
            mdts = dict(self.MDHCache)
            self.MDHCache = []

            if self.dataRW:
                self.dataMDH.update(mdts)

            #if 'resultsMDH' in dir(self):
            #    self.resultsMDH.update(mdts)

        HDFResultsTaskQueue.setQueueMetaDataEntries(self, mdts)


    def getQueueData(self, fieldName, *args):
        """Get data, defined by fieldName and potntially additional arguments,  ascociated with queue"""
        if fieldName == 'ImageShape':
            with self.dataFileLock.rlock:
                res = self.h5DataFile.root.ImageData.shape[1:]
            
            return res
        elif fieldName == 'ImageData':
            sliceNum, = args
            with self.dataFileLock.rlock:
                res = self.h5DataFile.root.ImageData[sliceNum, :,:]
            
            return res
        elif fieldName == 'NumSlices':
            #self.dataFileLock.acquire()
            #res = self.h5DataFile.root.ImageData.shape[0]
            #self.dataFileLock.release()
            #print res, self.numSlices
            #return res
            return self.numSlices
        elif fieldName == 'Events':
            with self.dataFileLock.rlock:
                res = self.h5DataFile.root.Events[:]
            
            return res
#        elif fieldName == 'PSF':
#            from PYME.ParallelTasks.relativeFiles import getFullExistingFilename
#            res = None
#            #self.dataFileLock.acquire()
#            #try:
#                #res = self.h5DataFile.root.PSFData[:]
#            #finally:
#            #    self.dataFileLock.release()
#            #try:
#            modName = self.resultsMDH.getEntry('PSFFile')
#            mf = open(getFullExistingFilename(modName), 'rb')
#            res = np.load(mf)
#            mf.close()
#            #except:
#                #pass
#
#            return res
        else:
            return HDFResultsTaskQueue.getQueueData(self, fieldName, *args)

    def logQueueEvent(self, event):
        eventName, eventDescr, evtTime = event
        with self.dataFileLock.wlock:
            ev = self.events.row

            ev['EventName'] = eventName
            ev['EventDescr'] = eventDescr
            ev['Time'] = evtTime
       
            ev.append()
            self.events.flush()

            ev = self.events[-1:]
        
        HDFResultsTaskQueue.addQueueEvents(self, ev)#[ev,])


    def releaseTasks(self, startingAt = 0):
        self.openTasks += range(startingAt, self.imNum)
        self.releaseNewTasks = True

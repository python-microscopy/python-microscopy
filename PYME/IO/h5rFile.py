import tables
import logging
import threading
import time
from PYME.IO import MetaDataHandler
import numpy as np
import traceback

#global lock across all instances of the H5RFile class as we can have problems across files
tablesLock = threading.Lock()

file_cache = {}


def openH5R(filename, mode='r'):
    key = (filename, mode)
    if key in file_cache and file_cache[key].is_alive:
        return file_cache[key]
    else:
        file_cache[key] = H5RFile(filename, mode)
        return file_cache[key]



KEEP_ALIVE_TIMEOUT = 20 #keep the file open for 20s after the last time it was used

class H5RFile(object):
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode

        logging.debug('pytables open call')
        self._h5file = tables.openFile(filename, mode)
        logging.debug('pytables file open')

        #metadata and events are created on demand
        self._mdh = None
        self._events = None

        # lock for adding things to our queues. This is local to the file and synchronises between the calling thread
        # and our local thread
        self.appendQueueLock = threading.Lock()
        self.appendQueues = {}

        self.keepAliveTimeout = time.time() + KEEP_ALIVE_TIMEOUT
        self.useCount = 0
        self.is_alive = True

        #logging.debug('H5RFile - starting poll thread')
        self._pollThread = threading.Thread(target=self._pollQueues)
        self._pollThread.start()
        #logging.debug('H5RFile - poll thread started')

    def __enter__(self):
        #logging.debug('entering H5RFile context manager')
        with self.appendQueueLock:
            self.useCount += 1

        return self

    def __exit__(self, *args):
        with self.appendQueueLock:
            self.keepAliveTimeout = time.time() + KEEP_ALIVE_TIMEOUT
            self.useCount -= 1


    @property
    def mdh(self):
        if self._mdh is None:
            try:
                self._mdh = MetaDataHandler.HDFMDHandler(self._h5file)
                if self.mode == 'r':
                    self._mdh = MetaDataHandler.NestedClassMDHandler(self._mdh)
            except IOError:
                # our file was opened in read mode and didn't have any metadata to start with
                self._mdh = MetaDataHandler.NestedClassMDHandler()

        return self._mdh

    def updateMetadata(self, mdh):
        """Update the metadata, acquiring the necessary locks"""
        with tablesLock:
            self.mdh.update(mdh)


    @property
    def events(self):
        try:
            return self._h5file.root.Events
        except AttributeError:
            return []

    def addEvents(self, events):
        self.appendToTable('Events', events)

    def _appendToTable(self, tablename, data):
        with tablesLock:
            try:
                table = getattr(self._h5file.root, tablename)
                table.append(data)
            except AttributeError:
                # we don't have a table with that name - create one
                self._h5file.createTable(self._h5file.root, tablename, data,
                                               filters=tables.Filters(complevel=5, shuffle=True),
                                               expectedrows=500000)

    def appendToTable(self, tablename, data):
        #logging.debug('h5rfile - append to table: %s' % tablename)
        with self.appendQueueLock:
            if not tablename in self.appendQueues.keys():
                self.appendQueues[tablename] = []
            self.appendQueues[tablename].append(data)

    def getTableData(self, tablename, _slice):
        with tablesLock:
            try:
                table = getattr(self._h5file.root, tablename)
                res = table[_slice]
            except AttributeError:
                res = []

        return res

    def _pollQueues(self):
        queuesWithData = False

        # logging.debug('h5rfile - poll')

        try:
            while self.useCount > 0 or queuesWithData or time.time() < self.keepAliveTimeout:
                #logging.debug('poll - %s' % time.time())
                with self.appendQueueLock:
                    #find queues with stuff to save
                    tablenames = [k for k, v in self.appendQueues.items() if len(v) > 0]

                queuesWithData = len(tablenames) > 0

                #iterate over the queues
                for tablename in tablenames:
                    with self.appendQueueLock:
                        entries = self.appendQueues[tablename]
                        self.appendQueues[tablename] = []

                    #save the data - note that we can release the lock here, as we are the only ones calling this function.
                    self._appendToTable(tablename, np.hstack(entries))

                self._h5file.flush()
                time.sleep(0.002)

        except:
            traceback.print_exc()
            logging.error(traceback.format_exc())
        finally:
            logging.debug('H5RFile - closing')
            #remove ourselves from the cache
            try:
                file_cache.pop((self.filename, self.mode))
            except KeyError:
                pass

            self.is_alive = False
            #finally, close the file
            self._h5file.close()



    def fileFitResult(self, fitResult):
        """
        Legacy handling for fitResult objects as returned by remFitBuf

        Parameters
        ----------
        fitResult

        Returns
        -------

        """
        if len(fitResult.results) > 0:
            self.appendToTable('FitResults', fitResult.results)

        if len(fitResult.driftResults) > 0:
            self.appendToTable('DriftResults', fitResult.driftResults)

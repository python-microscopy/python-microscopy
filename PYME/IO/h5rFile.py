import tables
import logging
import threading
import time
from PYME.IO import MetaDataHandler
from PYME import config
import numpy as np
import traceback
import collections
import six

#global lock across all instances of the H5RFile class as we can have problems across files
tablesLock = threading.Lock()

file_cache = {}


openLock = threading.Lock()

def openH5R(filename, mode='r', keep_alive_timeout=20.0):
    """
    Open an H5R file in a threadsafe and optimised way. Looks to see if the file is already open in our cache and
    returns the open file if present, otherwise opens the file and adds to the cache. Files are kept open for ``keep_alive_timeout``
    seconds after the last read or write operation to avoid IO overhead in repeatedly opening and closing a file for
    many short writes (such as saving incoming localisation results during analysis) whilst also ensuring that
    files do eventually get closed (the previous strategy had been to leave files open until program exit, but this
    becomes problematic for long-running processes in a high-throughput scenario).
    
    Parameters
    ----------
    filename: string
    mode: string, one of 'r', 'w', 'a'
    keep_alive_timeout: float
        a timeout hint for how long to keep the file open for after the last operation on the file. NOTE: this parameter
        is only respected if the file is not already in the cache - if in the cache the timeout will be the timeout set
        by the openH5R call that initially placed it there.

    Returns
    -------

    """
    key = filename
    
    with openLock:
        if key in file_cache and file_cache[key].is_alive:
            f = file_cache[key]
            if f.mode == 'r' and not mode == 'r':
                raise IOError('File already open in read-only mode, mode %s requested' % mode)
            else:
                return f
        else:
            file_cache[key] = H5RFile(filename, mode, keep_alive_timeout=keep_alive_timeout)
            return file_cache[key]


class H5RFile(object):
    #KEEP_ALIVE_TIMEOUT = 20 #keep the file open for 20s after the last time it was used
    FLUSH_INTERVAL = config.get('h5r-flush_interval', 1)
    
    def __init__(self, filename, mode='r', keep_alive_timeout = 20.0):
        self.filename = filename
        self.mode = mode
        
        self.KEEP_ALIVE_TIMEOUT = keep_alive_timeout

        logging.debug('pytables open call: %s' % filename)
        with tablesLock:
            self._h5file = tables.open_file(filename, mode)
        logging.debug('pytables file open: %s' % filename)

        #metadata and events are created on demand
        self._mdh = None
        self._events = None

        # lock for adding things to our queues. This is local to the file and synchronises between the calling thread
        # and our local thread
        self.appendQueueLock = threading.Lock()
        self.appendQueues = {}
        #self.appendVLQueues = {}

        self.keepAliveTimeout = time.time() + min(self.KEEP_ALIVE_TIMEOUT, 0.1)
        self.useCount = 0
        self.is_alive = True
        
        self._acquired_at_least_once = False

        #logging.debug('H5RFile - starting poll thread')
        self._lastFlushTime = 0
        self._flush_condition = threading.Condition()
        self._pollThread = threading.Thread(target=self._pollQueues)
        self._pollThread.daemon = False #make sure we finish and close the files properly on exit
        self._pollThread.start()
        
        self._pzf_index = None

        #logging.debug('H5RFile - poll thread started')

    def __enter__(self):
        #logging.debug('entering H5RFile context manager')
        with self.appendQueueLock:
            self.useCount += 1
            self._acquired_at_least_once = True

        return self

    def __exit__(self, *args):
        with self.appendQueueLock:
            self.keepAliveTimeout = time.time() + self.KEEP_ALIVE_TIMEOUT
            self.useCount -= 1
            
            if self.KEEP_ALIVE_TIMEOUT == 0:
                #if we have no timeout, wait for data to be written and file closed before returning
                self._pollThread.join(10)


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
                if isinstance(data, bytes):
                    table = self._h5file.create_vlarray(self._h5file.root, tablename, tables.VLStringAtom())
                    table.append(data)
                else:
                    self._h5file.create_table(self._h5file.root, tablename, data,
                                               filters=tables.Filters(complevel=5, shuffle=True),
                                               expectedrows=500000)
                    
            if (tablename == 'PZFImageData'):
                from PYME.IO import PZFFormat
                #special case  for pzf data - also build an index table
                frameNum = PZFFormat.load_header(data)['FrameNum']
                
                #record a mapping from frame number to the row we added
                idx_entry = np.array([frameNum, table.nrows -1], dtype='i4').view(dtype=[('FrameNum', 'i4'), ('Position', 'i4')])
                
                try:
                    index = getattr(self._h5file.root, 'PZFImageIndex')
                    index.append(idx_entry)
                except AttributeError:
                    self._h5file.create_table(self._h5file.root, 'PZFImageIndex', idx_entry,
                                              filters=tables.Filters(complevel=5, shuffle=True),
                                              expectedrows=50000)
                    
                self._pzf_index = None
                    

    def appendToTable(self, tablename, data):
        #logging.debug('h5rfile - append to table: %s' % tablename)
        with self.appendQueueLock:
            if not tablename in self.appendQueues.keys():
                self.appendQueues[tablename] = collections.deque()
            self.appendQueues[tablename].append(data)

    def getTableData(self, tablename, _slice):
        """
        Retrieves a tables.table.Table from the file and returns it as a structured numpy array

        Parameters
        ----------
        tablename: str
            name of table to return
        _slice: slice
            slice object to index the table

        Returns
        -------
        table_part: numpy.ndarray

        """
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
            while self.useCount > 0 or queuesWithData or time.time() < self.keepAliveTimeout or not self._acquired_at_least_once:
                #logging.debug('poll - %s' % time.time())
                with self.appendQueueLock:
                    #find queues with stuff to save
                    tablenames = [k for k, v in self.appendQueues.items() if len(v) > 0]

                queuesWithData = len(tablenames) > 0

                #iterate over the queues
                # for tablename in tablenames:
                #     with self.appendQueueLock:
                #         entries = self.appendQueues[tablename]
                #         self.appendQueues[tablename] = collections.deque()
                #
                #     #save the data - note that we can release the lock here, as we are the only ones calling this function.
                #     rows = np.hstack(entries)
                #     self._appendToTable(tablename, rows)

                #iterate over the queues (in a threadsafe manner)
                for tablename in tablenames:
                    waiting = self.appendQueues[tablename]
                    recs = []
                    try:
                        while len(waiting) > 0:
                            #self._appendToTable(tablename, waiting.popleft())
                            recs.append(waiting.popleft())
                    except IndexError:
                        pass
                    
                    self._appendToTable(tablename, np.hstack(recs))

                curTime = time.time()
                if (curTime - self._lastFlushTime) > self.FLUSH_INTERVAL:
                    with tablesLock:
                        self._h5file.flush()
                    self._lastFlushTime = curTime
                    with self._flush_condition:
                        self._flush_condition.notify_all()

                time.sleep(0.1)

        except:
            traceback.print_exc()
            logging.error(traceback.format_exc())
        finally:
            logging.debug('H5RFile - closing: %s' % self.filename)
            #remove ourselves from the cache
            with openLock:
                try:
                    file_cache.pop((self.filename, self.mode))
                except KeyError:
                    pass
    
                self.is_alive = False
                #finally, close the file
                with tablesLock:
                    self._h5file.close()

            logging.debug('H5RFile - closed: %s' % self.filename)



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
            
    def flush(self):
        """
        Wait until our IO loop has flushed. Currently only used in testing to make sure our data has hit the disk before
        we try and read it back in. Times out after twice our flush interval.

        """
        with self._flush_condition:
            self._flush_condition.wait(timeout=2*self.FLUSH_INTERVAL)
            
    def wait_close(self):
        self.keepAliveTimeout = 0
        self._pollThread.join(1.5*self.KEEP_ALIVE_TIMEOUT)

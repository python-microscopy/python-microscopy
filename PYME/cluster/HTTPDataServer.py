# -*- coding: utf-8 -*-
"""
This is a very simple HTTP server which allows data to be saved to the server using PUT


.. warning::
    Security warning
    ----------------

    The code as it stands lets any client write arbitrary data to the server.
    The only concessions we make to security are:
    
    a) Paths are vetted to make sure that they reside under our server root (i.e. no escape through .. etc)
    b) We enforce a write-once scheme (if a file exists it can't be overridden)
    
    This protects against people accidentally or maliciously altering data, or discovering
    server settings, but leaves us open to denial of service type attacks in which
    a malicious client could fill up our storage.

    **THE CODE AS IT STANDS SHOULD ONLY BE USED ON A TRUSTED NETWORK**

TODO: Add some form of authentication. Needs to be low overhead (e.g. digest based)
"""
from PYME import config
from PYME.misc.computerName import GetComputerName
compName = GetComputerName()
import os
import html
import sys

import errno
def makedirs_safe(dir):
    """
    A safe wrapper for makedirs, which won't throw an error if the directory already exists. This replicates the functionality
    of the exists_ok flag in  the python 3 version of os.makedirs, but should work with both pytion 2 and python 3.

    Parameters
    ----------
    dir : str, directory to be created

    Returns
    -------

    """
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


#make sure we set up our logging before anyone elses does
import logging
import logging.handlers
dataserver_root = config.get('dataserver-root')
# if dataserver_root:
#     log_dir = '%s/LOGS/%s' % (dataserver_root, compName)
#     #if not os.path.exists(log_dir):
#     #    os.makedirs(log_dir)
#     makedirs_safe(log_dir)
#
#     log_file = '%s/LOGS/%s/PYMEDataServer.log' % (dataserver_root, compName)
#
#     #logging.basicConfig(filename =log_file, level=logging.DEBUG, filemode='w')
#     #logger = logging.getLogger('')
#     logger = logging.getLogger('')
#     logger.setLevel(logging.DEBUG)
#     fh = logging.handlers.RotatingFileHandler(filename=log_file, mode='w', maxBytes=1e6, backupCount=1)
#     logger.addHandler(fh)
# else:
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('')
    
#now do all the normal imports
    
#import SimpleHTTPServer
#import BaseHTTPServer
# noinspection PyCompatibility
import http.server
# noinspection PyCompatibility
from socketserver import ThreadingMixIn

from io import StringIO, BytesIO
import shutil
#import urllib
import sys
import ujson as json
import PYME.misc.pyme_zeroconf as pzc
from PYME.misc import sqlite_ns

try:
    # noinspection PyCompatibility
    import urlparse
except ImportError:
    #py3
    # noinspection PyCompatibility
    from urllib import parse as urlparse
    
import requests
import socket
#import fcntl
import threading
import datetime
import time
from PYME.IO import h5File
from PYME.IO.FileUtils.nameUtils import get_service_name
#GPU status functions
try:
    import pynvml
    GPU_STATS=True
except ImportError:
    GPU_STATS = False


compName = GetComputerName()
procName = compName + ' - PID:%d' % os.getpid()

LOG_REQUESTS = False#True
USE_DIR_CACHE = True

startTime = datetime.datetime.now()
#global_status = {}

status = {}
_net = {}
_last_update_time = time.time()
try:
    import psutil
    _net.update(psutil.net_io_counters(True))
except ImportError:
    pass

def updateStatus():
    global _last_update_time
    from PYME.IO.FileUtils.freeSpace import disk_usage

    #status = {}
    #status.update(global_status)

    total, used, free = disk_usage(os.getcwd())
    status['Disk'] = {'total': total, 'used': used, 'free': free}
    status['Uptime'] = str(datetime.datetime.now() - startTime)

    try:
        import psutil
        
        ut = time.time()

        status['CPUUsage'] = psutil.cpu_percent(interval=0, percpu=True)
        status['MemUsage'] = psutil.virtual_memory()._asdict()
        
        nets = psutil.net_io_counters(True)
        dt = ut - _last_update_time
        
        status['Network'] = {iface : {'send' : (nets[iface].bytes_sent - _net[iface].bytes_sent)/dt,
                                      'recv' : (nets[iface].bytes_recv - _net[iface].bytes_recv)/dt} for iface in nets.keys()}
        
        _net.update(nets)
        _last_update_time = ut
    except ImportError:
        pass

    if GPU_STATS:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        gpu_usage = [pynvml.nvmlDeviceGetUtilizationRates(h) for h in handles]
        status['GPUUsage'] = [float(gu.gpu) for gu in gpu_usage]
        status['GPUMem'] = [float(gu.memory) for gu in gpu_usage]

class statusPoller(threading.Thread):
    def __init__(self, *args, **kwargs):
        self.poll = True

        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True

    def run(self):
        while self.poll:
            updateStatus()
            time.sleep(1)

    def stop(self):
        self.poll = False

textfile_locks = {}
def getTextFileLock(filename):
    try:
        return textfile_locks[filename]
    except KeyError:
        textfile_locks[filename] = threading.Lock()
        return textfile_locks[filename]


from collections import OrderedDict

class _LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


_dirCache = _LimitedSizeDict(size_limit=100)
_dirCacheTimeout = 1

#_listDirLock = threading.Lock()

from PYME.IO import clusterListing as cl

class PYMEHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    bandwidthTesting = False
    timeoutTesting = 0
    logrequests = False
    timeout=None
    
    def __init__(self, *args, **kwargs):
        self._path_cache = _LimitedSizeDict(size_limit=1000)
        http.server.SimpleHTTPRequestHandler.__init__(self, *args, **kwargs)


    def _aggregate_txt(self):
        """
        support for results aggregation (append into a file). This 'method' does a dumb append to the file on disk,
        meaning that it will require reasonably carefully formatted input (with, e.g., a trailing newlines on each
        chunk if results are in .csv
        This is mostly intended for output in csv or similar formats, where chunks can be 'cat'ed together.

        NOTE: there is no guarantee of ordering, so the record format should contain enough information to re-order
        the data if necessary
        """
        path = self.translate_path(self.path.lstrip('/')[len('__aggregate_txt'):])

        data = self._get_data()
        
        #if self.headers['Content-Encoding'] == 'gzip':
        #    data = self._gzip_decompress(data)

        dirname = os.path.dirname(path)
        #if not os.path.exists(dirname):
        #    os.makedirs(dirname)
        makedirs_safe(dirname)
        
        if USE_DIR_CACHE and not os.path.exists(path):
            # only update directory cache on initial creation to avoid lock thrashing. Use a placeholder size to indicate file is not complete
            cl.dir_cache.update_cache(path, -1)

        #append the contents of the put request
        with getTextFileLock(path):
            #lock so that we don't corrupt the data by writing from two different threads
            #TODO ?? - keep a cache of open files
            with open(path, 'ab') as f:
                f.write(data)

        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _gzip_decompress(self, data):
        import gzip
        from io import BytesIO
        zbuf = BytesIO(data)
        zfile = gzip.GzipFile(mode='rb', fileobj=zbuf)#, compresslevel=9)
        out = zfile.read()
        zfile.close()
    
        return out
    
    def _get_data(self):
        data = self.rfile.read(int(self.headers['Content-Length']))
    
        if self.headers.get('Content-Encoding') == 'gzip':
            data = self._gzip_decompress(data)
            
        return data
    
    def translate_path(self, path):
        try:
            return self._path_cache[path]
        except KeyError:
            self._path_cache[path] = http.server.SimpleHTTPRequestHandler.translate_path(self, path)
            return self._path_cache[path]

    def _aggregate_h5r(self):
        """
        Support for results aggregation into an HDF5 file, using pytables.
        We treat any path components after the .h5r as locations within the file (ie table names).
        e.g. /path/to/data.h5r/<tablename>
        A few special cases / Table names are accommodated:

        MetaData: assumes we have sent PYME metadata in json format and saves to the file using the appropriate metadatahandler
        No table name: assumes we have a fitResults object (as returned by remFitBuf and saves to the the appropriate tables (as HDF task queue would)
        """
        import numpy as np
        from io import BytesIO
        from six.moves import cPickle
        from PYME.IO import MetaDataHandler
        from PYME.IO import h5rFile

        # path = self.translate_path(self.path.lstrip('/')[len('__aggregate_h5r'):])
        # filename, tablename = path.split('.h5r')
        # filename += '.h5r'

        filename, tablename = self.path.lstrip('/')[len('__aggregate_h5r'):].split('.h5r')
        filename = self.translate_path(filename + '.h5r')

        data = self._get_data()

        if not os.path.exists(filename):
            dirname = os.path.dirname(filename)
            #if not os.path.exists(dirname):
            #    os.makedirs(dirname)
            makedirs_safe(dirname)
        
            if USE_DIR_CACHE:
                # only update directory cache on initial creation to avoid lock thrashing. Use a placeholder size to indicate file is not complete
                cl.dir_cache.update_cache(filename, -1)

        #logging.debug('opening h5r file')
        with h5rFile.openH5R(filename, 'a') as h5f:
            if tablename == '/MetaData':
                mdh_in = MetaDataHandler.CachingMDHandler(json.loads(data))
                h5f.updateMetadata(mdh_in)
            elif tablename == '':
                #legacy fitResults structure
                fitResults = cPickle.loads(data)
                h5f.fileFitResult(fitResults)
            else:
                try:
                    #pickle is much faster than numpy array format (despite the array format being simpler)
                    #reluctanltly use pickles
                    data = np.loads(data)
                except cPickle.UnpicklingError:
                    try:
                        #try to read data as if it was numpy binary formatted
                        data = np.load(BytesIO(data))
                    except IOError:
                        #it's not numpy formatted - try json
                        import pandas as pd
                        #FIXME!! - this will work, but will likely be really slow!
                        data = pd.read_json(data).to_records(False)

                #logging.debug('adding data to table')
                h5f.appendToTable(tablename.lstrip('/'), data)
                #logging.debug('added data to table')


        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return
    
    def _aggregate_h5(self):
        """
        Support for results aggregation into an HDF5 file, using pytables.
        We treat any path components after the .h5r as locations within the file (ie table names).
        e.g. /path/to/data.h5r/<tablename>
        A few special cases / Table names are accommodated:

        MetaData: assumes we have sent PYME metadata in json format and saves to the file using the appropriate metadatahandler
        No table name: assumes we have a fitResults object (as returned by remFitBuf and saves to the the appropriate tables (as HDF task queue would)
        """
        import numpy as np
        from io import BytesIO
        #from six.moves import cPickle
        from PYME.IO import MetaDataHandler
        from PYME.IO import h5File

        #path = self.translate_path()
        filename, tablename = self.path.lstrip('/')[len('__aggregate_h5'):].split('.h5')
        filename = self.translate_path(filename + '.h5')

        data = self._get_data()

        dirname = os.path.dirname(filename)
        #if not os.path.exists(dirname):
        #    os.makedirs(dirname)
        makedirs_safe(dirname)
        
        if USE_DIR_CACHE and not os.path.exists(filename):
            # only update directory cache on initial creation to avoid lock thrashing. Use a placeholder size to indicate file is not complete
            cl.dir_cache.update_cache(filename, -1)

        #logging.debug('opening h5r file')
        with h5File.openH5(filename, 'a') as h5f:
            tablename = tablename.lstrip('/')
            h5f.put_file(tablename, data)
            

        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return


    def _doAggregate(self):
        # TODO - add authentication/checks for aggregation. Files which still allow appends should not be duplicated.
        if self.path.lstrip('/').startswith('__aggregate_txt'):
            self._aggregate_txt()
        elif self.path.lstrip('/').startswith('__aggregate_h5r'):
            self._aggregate_h5r()
        elif self.path.lstrip('/').startswith('__aggregate_h5'):
            self._aggregate_h5()

        return

    def do_PUT(self):
        if self.timeoutTesting:
            #exp = time.time() + float(self.timeoutTesting)
            #while time.time() < exp:
            #    y = pow(5, 7)
            time.sleep(float(self.timeoutTesting)) #wait 10 seconds to force a timeout on the clients
            #print('waited ... ')
            
        if self.bandwidthTesting:
            #just read file and dump contents
            r = self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if self.path.lstrip('/').startswith('__aggregate'):
            #paths starting with __aggregate are special, and trigger appends to an existing file rather than creation
            #of a new file.
            self._doAggregate()
            return



        path = self.translate_path(self.path)

        if os.path.exists(path):
            #Do not overwrite - we use write-once semantics
            self.send_error(405, "File already exists %s" % path)

            #self.end_headers()
            return None
        else:
            dir, file = os.path.split(path)
            #if not os.path.exists(dir):
            #    os.makedirs(dir)
            makedirs_safe(dir)

            query = urlparse.parse_qs(urlparse.urlparse(self.path).query)

            if file == '':
                #we're  just making a directory
                pass
            elif 'MirrorSource' in query.keys():
                #File content is not in message content. This computer should
                #fetch the results from another computer in the cluster instead
                #used for online duplication

                r = requests.get(query['MirrorSource'][0], timeout=.1)

                with open(path, 'wb') as f:
                    f.write(r.content)
                    
                #set the file to read-only (reflecting our write-once semantics
                os.chmod(path, 0o440)

                if USE_DIR_CACHE:
                    cl.dir_cache.update_cache(path, len(r.content))

            else:
                #the standard case - use the contents of the put request
                with open(path, 'wb') as f:
                    #shutil.copyfileobj(self.rfile, f, int(self.headers['Content-Length']))
                    data = self._get_data()
                    f.write(data)

                    #set the file to read-only (reflecting our write-once semantics
                    os.chmod(path, 0o440)
                    
                    if USE_DIR_CACHE:
                        cl.dir_cache.update_cache(path, len(data))

            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

    def do_GET(self):
        """Serve a GET request."""
        if self.timeoutTesting:
            time.sleep(float(self.timeoutTesting)) #wait 10 seconds to force a timeout on the clients
            
        f = self.send_head()
        if f:
            try:
                self.copyfile(f, self.wfile)
            finally:
                f.close()

    def get_status(self):


        f = BytesIO()
        f.write(json.dumps(status).encode())
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        encoding = sys.getfilesystemencoding()
        self.send_header("Content-type", "application/json; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f
    
    def get_glob(self):
        import glob
        query = urlparse.parse_qs(urlparse.urlparse(self.path).query)
        pattern = query['pattern'][0]
        
        logger.debug('glob: pattern = %s' % pattern)
        matches = glob.glob(pattern)

        f = BytesIO()
        f.write(json.dumps(matches).encode())
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        encoding = sys.getfilesystemencoding()
        self.send_header("Content-type", "application/json; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

    def send_head(self):
        """Common code for GET and HEAD commands.

        This sends the response code and MIME headers.

        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.

        """
        path = self.translate_path(self.path)
        f = None
        
        if self.path.lstrip('/') == '__status':
            return self.get_status()
        
        if self.path.lstrip('/').startswith('__glob'):
            return self.get_glob()
        
        if os.path.isdir(path):
            parts = urlparse.urlsplit(self.path)
            if not parts.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(301)
                new_parts = (parts[0], parts[1], parts[2] + '/',
                             parts[3], parts[4])
                new_url = urlparse.urlunsplit(new_parts)
                self.send_header("Content-Length", "0") # we need to set this
                self.send_header("Location", new_url)
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)

        
        
        if '.h5/' in self.path:
            #special case - allow .h5 files to be treated as a directory
            if self.path.endswith('.h5/'):
                return self.list_h5(path)
            else:
                return self.get_h5_part(path)
        elif '.h5r/' in self.path or '.hdf/' in self.path:
            # throw the query back on to our fully resolved path
            return self.get_tabular_part(path + '?' + urlparse.urlparse(self.path).query)

        ctype = self.guess_type(path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            self.send_error(404, "File not found - %s, [%s]" % (self.path, path))
            return None
        try:
            self.send_response(200)
            self.send_header("Content-type", ctype)
            fs = os.fstat(f.fileno())
            self.send_header("Content-Length", str(fs[6]))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except:
            f.close()
            raise
        
    def _string_to_file(self, str):
        f = BytesIO()
        if not isinstance(str, bytes):
            str = str.encode()
        f.write(str)
        length = f.tell()
        f.seek(0)
        
        return f, length

    def list_directory(self, path):
        """Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().

        """
        from PYME.IO import clusterListing as cl
        curTime = time.time()

        with cl.dir_cache.dir_lock(path):
            # if dir is in already in cache, return
            try:
                js_dir, expiry = _dirCache[path]
                if expiry < curTime:
                    try:
                        # remove directory entry from cache as it's expired.
                        _dirCache.pop(path)
                    except KeyError:
                        pass
                    
                    raise KeyError('Entry expired')
            
            except KeyError:
                try:
                    if USE_DIR_CACHE:
                        l2 = cl.dir_cache.list_directory(path)
                    else:
                        l2 = cl.list_directory(path)
                except os.error:
                    self.send_error(404, "No permission to list directory")
                    return None
    
                js_dir = json.dumps(l2)
                _dirCache[path] = (js_dir, time.time() + _dirCacheTimeout)

        f, length = self._string_to_file(js_dir)
        
        self.send_response(200)
        encoding = sys.getfilesystemencoding()
        self.send_header("Content-type", "application/json; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f
    
    def list_h5(self, path):
        from PYME.IO import h5File
        try:
            with h5File.openH5(path.rstrip('/').rstrip('\\')) as h5f:
                f, length = self._string_to_file(json.dumps(h5f.get_listing()))

            self.send_response(200)
            encoding = sys.getfilesystemencoding()
            self.send_header("Content-type", "application/json; charset=%s" % encoding)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            return f

        except IOError:
            self.send_error(404, "File not found - %s, [%s]" % (self.path, path))
            return None
    
    def get_h5_part(self, path):
        from PYME.IO import h5File
        ctype = self.guess_type(path)
        try:
            filename, part = path.split('.h5')
            part = part.lstrip('/').lstrip('\\')
            
            with h5File.openH5(filename + '.h5') as h5f:
                f, length = self._string_to_file(h5f.get_file(part))

                self.send_response(200)
                self.send_header("Content-type", ctype)
                self.send_header("Content-Length", length)
                #self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
                self.end_headers()
                return f
                
            
        except IOError:
            self.send_error(404, "File not found - %s, [%s]" % (self.path, path))
            return None

    def get_tabular_part(self, path):
        """

        Parameters
        ----------
        path: str
            OS-translated path to an hdf or h5r file on the dataserver computer. 
            Append the part of the file to read after the file extension, e.g. 
            .h5r/Events. Return format (for arrays) can additionally be 
            specified, as can slices
            using the following syntax: test.h5r/FitResults.json?from=0&to=100. 
            Supported array formats include json and npy.

        Returns
        -------
        f: BytesIO
            Requested part of the file encoded as bytes

        """
        from PYME.IO import h5rFile, clusterResults

        # parse path
        ext = '.h5r' if '.h5r' in path else '.hdf'
        # TODO - should we just use the the untranslated path?
        filename, details = path.split(ext + os.sep)
        filename = filename + ext  # path to file on dataserver disk
        query = urlparse.urlparse(details).query
        details = details.strip('?' + query)
        if '.' in details:
            part, return_type = details.split('.')
        else:
            part, return_type = details, ''


        try:
            with h5rFile.openH5R(filename) as h5f:
                if part == 'Metadata':
                    wire_data, output_format = clusterResults.format_results(h5f.mdh, return_type)
                else:
                    # figure out if we have any slicing to do
                    query = urlparse.parse_qs(query)
                    start = int(query.get('from', [0])[0])
                    end = None if 'to' not in query.keys() else int(query['to'][0])
                    wire_data, output_format = clusterResults.format_results(h5f.getTableData(part, slice(start, end)),
                                                                             '.' + return_type)

            f, length = self._string_to_file(wire_data)
            self.send_response(200)
            self.send_header("Content-Type", output_format if output_format else 'application/octet-stream')
            self.send_header("Content-Length", length)
            #self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f

        except IOError:
            self.send_error(404, "File not found - %s, [%s]" % (self.path, path))
        
        

    def log_request(self, code='-', size='-'):
        """Log an accepted request.

        This is called by send_response().

        """
        if self.logrequests:
            self.log_message('"%s" %s %s', self.requestline, str(code), str(size))

    def log_error(self, format, *args):
        """Log an error.

        This is called when a request cannot be fulfilled.  By
        default it passes the message on to log_message().

        Arguments are the same as for log_message().

        XXX This should go to the separate error log.

        """
        logger.error("%s - - [%s] %s\n" %
                         (self.client_address[0],
                          self.log_date_time_string(),
                          format % args))

        #self.log_message(format, *args)

    def log_message(self, format, *args):
        """Log an arbitrary message.

        This is used by all other logging functions.  Override
        it if you have specific logging wishes.

        The first argument, FORMAT, is a format string for the
        message to be logged.  If the format string contains
        any % escapes requiring parameters, they should be
        specified as subsequent arguments (it's just like
        printf!).

        The client ip address and current date/time are prefixed to every
        message.

        """

        logger.info("%s - - [%s] %s\n" %
                         (self.client_address[0],
                          self.log_date_time_string(),
                          format % args))

        #sys.stderr.write("%s - - [%s] %s\n" %
        #                 (self.client_address[0],
        #                  self.log_date_time_string(),
        #                  format % args))

    def send_error(self, code, message=None):
        """Send and log an error reply.

        Arguments are the error code, and a detailed message.
        The detailed message defaults to the short entry matching the
        response code.

        This sends an error response (so it must be called before any
        output has been generated), logs the error, and finally sends
        a piece of HTML explaining the error to the user.

        """

        try:
            short, long = self.responses[code]
        except KeyError:
            short, long = '???', '???'
        if message is None:
            message = short
        explain = long
        self.log_error("code %d, message %s", code, message)
        self.send_response(code, message)
        #self.send_header('Connection', 'close')

        # Message body is omitted for cases described in:
        #  - RFC7230: 3.3. 1xx, 204(No Content), 304(Not Modified)
        #  - RFC7231: 6.3.6. 205(Reset Content)
        content = None
        if code >= 200 and code not in (204, 205, 304):
            # HTML encode to prevent Cross Site Scripting attacks
            # (see bug #1100201)
            content = (self.error_message_format % {
                'code': code,
                'message': html.escape(message),
                'explain': explain
            }).encode()
            self.send_header("Content-Type", self.error_content_type)
            self.send_header("Content-Length", str(len(content)))

        self.end_headers()

        if self.command != 'HEAD' and content:
            self.wfile.write(content)


class ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    """Handle requests in a separate thread."""


def main(protocol="HTTP/1.0"):
    global GPU_STATS
    """Test the HTTP request handler class.

    This runs an HTTP server on port 8000 (or the first command line
    argument).

    """
    from optparse import OptionParser

    op = OptionParser(usage='usage: %s [options]' % sys.argv[0])

    #NOTE - currently squatting on port 15348 for testing - TODO can we use an ephemeral port?
    op.add_option('-p', '--port', dest='port', default=config.get('dataserver-port', 15348),
                  help="port number to serve on (default: 15348, see also 'dataserver-port' config entry)")
    op.add_option('-t', '--test', dest='test', help="Set up for bandwidth test (don't save files)", action="store_true", default=False)
    op.add_option('-v', '--protocol', dest='protocol', help="HTTP protocol version", default="1.1")
    op.add_option('-l', '--log-requests', dest='log_requests', help="Display http request info", default=False, action="store_true")
    default_root = config.get('dataserver-root', os.curdir)
    op.add_option('-r', '--root', dest='root', help="Root directory of virtual filesystem (default %s, see also 'dataserver-root' config entry)" % dataserver_root, default=default_root)
    op.add_option('-k', '--profile', dest='profile', help="Enable profiling", default=False, action="store_true")
    op.add_option('--thread-profile', dest='thread_profile', help="Enable thread profiling", default=False, action="store_true")
    default_server_filter = config.get('dataserver-filter', compName)
    op.add_option('-f', '--server-filter', dest='server_filter', help='Add a serverfilter for distinguishing between different clusters', default=default_server_filter)
    op.add_option('--timeout-test', dest='timeout_test', help='deliberately make requests timeout for testing error handling in calling modules', default=0)
    op.add_option('-a', '--advertisements', dest='advertisements', choices=['zeroconf', 'local'], default='zeroconf',
                    help='Optionally restrict advertisements to local machine')


    options, args = op.parse_args()
    if options.profile:
        from PYME.util import mProfile
        mProfile.profileOn(['HTTPDataServer.py','clusterListing.py'])

        profileOutDir = options.root + '/LOGS/%s/mProf' % compName
        
    if options.thread_profile:
        from PYME.util import fProfile
        
        tp = fProfile.ThreadProfiler()
        #tp.profile_on(subs=['PYME/', 'http/server', 'socketserver'],outfile=options.root + '/LOGS/%s/tProf/dataserver.txt' % compName)
        tp.profile_on(subs=['PYME/', ],
                  outfile=options.root + '/LOGS/%s/tProf/dataserver.txt' % compName)

    # setup logging to file
    log_dir = '%s/LOGS/%s' % (options.root, compName)
    makedirs_safe(log_dir)

    log_file = '%s/LOGS/%s/PYMEDataServer.log' % (options.root, compName)
    fh = logging.handlers.RotatingFileHandler(filename=log_file, mode='w', maxBytes=1e6, backupCount=1)
    logger.addHandler(fh)

    
    logger.info('========================================\nPYMEDataServer, running on python %s\n' % sys.version)
    
    #change to the dataserver root if given'
    logger.info('Serving from directory: %s' % options.root)
    os.chdir(options.root)

    if options.advertisements == 'local':
        # preference is to avoid zeroconf on clusterofone due to poor
        # performance on crowded networks
        if config.get('clusterIO-hybridns', True):
            ns = sqlite_ns.getNS('_pyme-http')
        else:
            # if we aren't using the hybridns, we are using zeroconf in clusterIO
            # TODO - warn that we might run into performance issues???
            ns = pzc.getNS('_pyme-http')
        server_address = ('127.0.0.1', int(options.port))
        ip_addr = '127.0.0.1'
    else:
        #default
        ns = pzc.getNS('_pyme-http')
        server_address = ('', int(options.port))
        
        try:
            ip_addr = socket.gethostbyname(socket.gethostname())
        except:
            ip_addr = socket.gethostbyname(socket.gethostname() + '.local')

    PYMEHTTPRequestHandler.protocol_version = 'HTTP/%s' % options.protocol
    PYMEHTTPRequestHandler.bandwidthTesting = options.test
    PYMEHTTPRequestHandler.timeoutTesting = options.timeout_test
    PYMEHTTPRequestHandler.logrequests = options.log_requests

    httpd = ThreadedHTTPServer(server_address, PYMEHTTPRequestHandler)
    #httpd = http.server.HTTPServer(server_address, PYMEHTTPRequestHandler)
    httpd.daemon_threads = True

    #get the actual adress (port) we bound to
    sa = httpd.socket.getsockname()
    service_name = get_service_name('PYMEDataServer [%s]' % options.server_filter)
    ns.register_service(service_name, ip_addr, sa[1])

    status['IPAddress'] = ip_addr
    status['BindAddress'] = server_address
    status['Port'] = sa[1]
    status['Protocol'] = options.protocol
    status['TestMode'] = options.test
    status['ComputerName'] = GetComputerName()

    if GPU_STATS:
        try:
            pynvml.nvmlInit()
        except:
            GPU_STATS = False

    sp = statusPoller()
    sp.start()


    logger.info("Serving HTTP on %s port %d ..." % (ip_addr, sa[1]))
    try:
        httpd.serve_forever()
    finally:
        logger.info('Shutting down ...')
        httpd.shutdown()
        httpd.server_close()
        
        ns.unregister(service_name)

        if options.profile:
            mProfile.report(display=False, profiledir=profileOutDir)
            
        if options.thread_profile:
            tp.profile_off()

        sp.stop()

        if GPU_STATS:
            pynvml.nvmlShutdown()

        try:
            from pytest_cov.embed import cleanup
            cleanup()
        except:
            pass
        
        sys.exit()


if __name__ == '__main__':
    main()

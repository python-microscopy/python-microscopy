# -*- coding: utf-8 -*-
"""
API interface to PYME cluster storage.
--------------------------------------

The cluster storage effectively aggregates storage across a number of cluster nodes. In contrast to a standard file-system,
we only support atomic reads and writes of entire files, and only support a single write to a given file (no modification
or replacement). Under the hood, the file system is implemented as a bunch of HTTP servers, with the `clusterIO` library
simply presenting a unified view of the files on all servers.

.. note::

    PYME cluster storage has a number of design aspects which trade some safety for speed. Importantly it lacks features
    such as locking found in traditional filesystems. This is largely offset by only supporting atomic writes, and being
    write once (no deletion or modification), but does rely on clients being well behaved (i.e. not trying to write to
    the same file concurrently) - see `put_file()` for more details. Directory listings etc .. are also not guaranteed
    to update immediately after file puts, with a latency of up to 2s for directory updates being possible.

The most important functions are:

    | :func:`get_file` which performs an atomic get
    | :func:`put_file` which performs an atomic put to a randomly [#f1]_ chosen server
    | :func:`listdir` which performs a (unified) directory listing
    
There are also a bunch of utility functions, mirroring some of those available in the os, os.path, and glob modules:

    :func:`exists`
    :func:`cglob`
    :func:`walk`
    :func:`isdir`
    :func:`stat`
    
And finally some functions to facilitate efficient (and data local) operations

    | :func:`is_local` which determines if a file is hosted on the local machine
    | :func:`get_local_path` which returns a local path if a file is local
    | :func:`locate_file` which returns a list of http urls where the given file can be found
    | :func:`put_files` which implements a streamed, high performance, put of multiple files
    
There are are also higher level functions in :mod:`PYME.IO.unifiedIO`  which allows files to be accessed in a consistent
way given either a cluster URI or a local path and :mod:`PYME.IO.clusterResults` which helps with saving tabular data to
the cluster.

Cluster identification
----------------------

clusterIO automatically identifies the individual data servers that make up the cluster using the mDNS (zeroconf) protocol.
If you run a (or multiple) server(s) (see :mod:`PYME.cluster.HTTPDataServer`) clusterIO will find them with no additional
configuration. To support multiple clusters on a single network segment, however, we have the concept of a *cluster name*.
Servers will broadcast the name of the cluster they belong to as part of their mDNS advertisement.

On the client side, you can select which cluster you want to talk to with the ``serverfilter`` argument to the
clusterIO functions. Each function will access all servers which include the value of ``serverfilter`` in their name. If
unspecified, the value of the :mod:`PYME.config` configuration option ``dataserver-filter`` is used, which in turn defaults
to the local computer name. This enables the use of a local "cluster of one" for analysis without any additional configuration
and without interfering with any clusters which are already on the network. When setting up a proper, multi-computer cluster,
set the ``dataserver-filter`` config option on all members of the cluster o a common value. Once setup in this manner, it
should not be necessary to actually specify ``serverfilter`` when using clusterIO.
 
 .. note::
 
    The nature of ``serverfilter`` matches is both powerful and requires some care. A value of ``serverfilter=''`` will match
    every data server running on the local network and present them as an aggregated cluster. Similarly ``serverfilter='cluster'``
    will match ``cluster``, ``cluster1``, ``cluster2``, ``3cluster`` etc and prevent them all as an aggregated cluster. This
    cluster aggregation opens up interesting postprocessing and display options, but to avoid unexpected effects it would
    be prudent to follow the following recommendations.
    
    1) Avoid cluster names which are substrings of other cluster names
    2) Always use the full cluster name for ``serverfilter``.
    
    Depending on how useful aggregation actually proves to be
    ``serverfilter`` might change to either requiring an exact match or compiling to a regex at some point in the future.


.. rubric:: Footnotes

.. [#f1] technically we do some basic load balancing


Function Documentation
-----------------------

"""
import six
#import urllib
import socket
import requests
import time
import numpy as np
import threading
from PYME.IO import unifiedIO

if six.PY2:
    import httplib
else:
    import http.client as httplib

import socket
import os

USE_RAW_SOCKETS = True

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()


def to_bytes(input):
    """
    Helper function for python3k to force urls etc to byte strings

    Parameters
    ----------
    input

    Returns
    -------

    """
    
    if isinstance(input, str):
        return input.encode('utf8')
    else:
        return input


from PYME import config
from PYME.misc.computerName import GetComputerName
local_dataroot = (config.get('dataserver-root'))
local_serverfilter = (config.get('dataserver-filter', GetComputerName()))

import sys

try:
    # noinspection PyCompatibility
    from urlparse import urlparse
except ImportError:
    #py3
    # noinspection PyCompatibility
    from urllib.parse import urlparse

import logging

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from .cluster_directory import _LimitedSizeDict, get_ns, DirectoryInfoManager

#_locateCache = _LimitedSizeDict(size_limit=500)
#_dirCache = _LimitedSizeDict(size_limit=100)
#DIR_CACHE_TIME = 1
_fileCache = _LimitedSizeDict(size_limit=100)
_dir_managers = {}

def get_dir_manager(serverfilter):
    try:
        return _dir_managers[serverfilter]
    except KeyError:
        _dir_managers[serverfilter] = DirectoryInfoManager(serverfilter=serverfilter)
    
        return _dir_managers[serverfilter]

#use one session for each server (to allow http keep-alives)
sessions = {}
def _getSession(url):
    if not isinstance(url, bytes):
        url = url.encode()
        
    servinfo = url.split(b'/')[2]
    try:
        session = sessions[servinfo]
    except KeyError:
        session = requests.Session()
        sessions[servinfo] = session

    return session


def locate_file(filename, serverfilter=local_serverfilter, return_first_hit=False):
    """
    Searches the cluster to find which server(s) a given file is stored on

    Parameters
    ----------
    filename : str
        The file name
    serverfilter : str
        The name of our cluster (allows for having multiple clusters on the same network segment)
    return_first_hit : bool
        Whether to try and find all locations, or return when we find the first copy

    Returns
    -------

    """
    
    return get_dir_manager(serverfilter).locate_file(filename, return_first_hit)
    
def listdirectory(dirname, serverfilter=local_serverfilter, timeout=5):
    """Lists the contents of a directory on the cluster.

    Returns a dictionary mapping filenames to clusterListing.FileInfo named tuples.
    """
    return get_dir_manager(serverfilter).listdirectory(dirname, timeout)

def listdir(dirname, serverfilter=local_serverfilter):
    """Lists the contents of a directory on the cluster. Similar to os.listdir,
    but directories are indicated by a trailing slash
    """

    return get_dir_manager(serverfilter).listdir(dirname)

def isdir(name, serverfilter=local_serverfilter):
    """
    Tests if a given path on the cluster is a directory. Analogous to os.path.isdir
    
    Parameters
    ----------
    name
    serverfilter

    Returns
    -------
    
    True or False

    """
    return get_dir_manager(serverfilter).isdir(name)
    
    
def cglob(pattern, serverfilter=local_serverfilter):
    """
    Find files matching a given glob on the cluster. Analogous to the python glob.glob function.
    
    Parameters
    ----------
    pattern : string glob
    serverfilter : cluster name (optional)

    Returns
    -------
    
    a list of files matching the glob

    """
    return get_dir_manager(serverfilter).cglob(pattern)
    

def exists(name, serverfilter=local_serverfilter):
    """
    Test whether a file exists on the cluster. Analogue to os.path.exists for local files.
    
    Parameters
    ----------
    name : string, file path
    serverfilter : name of the cluster (optional)

    Returns
    -------
    
    True if file exists, else False

    """
    return get_dir_manager(serverfilter).exists(name)

class _StatResult(object):
    def __init__(self, file_info):
        self.st_size = file_info.size
        self.size = self.st_size
        self.st_ctime = time.time()
        self.st_atime = self.st_ctime
        self.st_mtime = self.st_ctime
        self.type = file_info.type

def stat(name, serverfilter=local_serverfilter):
    """
    Cluster analog to os.stat
    
    Parameters
    ----------
    name
    serverfilter

    Returns
    -------

    """
    from . import clusterListing as cl

    name = (name)
    serverfilter = (serverfilter)
    
    rname = name.rstrip('/')
    dirname, fname = os.path.split(rname)
    
    listing = listdirectory(dirname, serverfilter)
    #print name, listing
    
    if fname == '':
        #special case for the root directory
        return _StatResult(cl.FileInfo(cl.FILETYPE_DIRECTORY, 0))
    
    try:
        r = _StatResult(listing[fname])
    except KeyError:
        try:
            r = _StatResult(listing[fname + '/'])
        except:
            logger.exception('error stating: %s' % name)
            #print dirname, fname
            #print listing.keys()
            raise
    
    return r


def walk(top, topdown=True, on_error=None, followlinks=False, serverfilter=local_serverfilter):
    """Directory tree generator. Adapted from the os.walk 
    function in the python std library.

    see docs for os.walk for usage details

    """

    top = (top)
    serverfilter = (serverfilter)

    # islink, join, isdir = path.islink, path.join, path.isdir
    def islink(name):
        # cluster does not currently have the concept of symlinks
        return False

    def join(*args):
        j = '/'.join(args)
        return j.replace('//', '/')

    def isdir(name):
        return name.endswith('/')

    # We may not have read permission for top, in which case we can't
    # get a list of the files the directory contains.  os.path.walk
    # always suppressed the exception then, rather than blow up for a
    # minor reason when (say) a thousand readable directories are still
    # left to visit.  That logic is copied here.
    try:
        # Note that listdir and error are globals in this module due
        # to earlier import-*.
        names = listdir(top, serverfilter=serverfilter)
    except Exception as err:
        if on_error is not None:
            on_error(err)
        return

    dirs, nondirs = [], []
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    if topdown:
        yield top, dirs, nondirs
    for name in dirs:
        new_path = join(top, name)
        if followlinks or not islink(new_path):
            for x in walk(new_path, topdown, on_error, followlinks):
                yield x
    if not topdown:
        yield top, dirs, nondirs


def _chooseLocation(locs):
    """Chose the location to load the file from

    default to choosing the "closest" (currently the one with the shortest response
    time to our directory query)

    """

    #logger.debug('choosing location from: %s' % locs)
    cost = np.array([l[1] for l in locs])

    return locs[cost.argmin()][0]


def parseURL(URL):
    URL = (URL)
    
    scheme, body = URL.split('://')
    parts = body.split('/')

    serverfilter = parts[0]
    filename = '/'.join(parts[1:])

    return filename, serverfilter


def is_local(filename, serverfilter):
    """
    Test to see if a file is held on the local computer.
    
    This is used in the distributed scheduler to score tasks.
    
    Parameters
    ----------
    filename
    serverfilter

    Returns
    -------

    """
    filename = (filename)
    serverfilter = (serverfilter)
    
    if serverfilter == local_serverfilter and local_dataroot:
        #look for the file in the local server folder (short-circuit the server)
        localpath = os.path.join(local_dataroot, filename)
        return os.path.exists(localpath)
    else:
        return False

def get_local_path(filename, serverfilter):
    """
    Get a local path, if available based on a cluster filename
    
    Parameters
    ----------
    filename
    serverfilter

    Returns
    -------

    """
    filename = (filename)
    serverfilter = (serverfilter)
    
    if (serverfilter == local_serverfilter or serverfilter == '') and local_dataroot:
        #look for the file in the local server folder (short-circuit the server)
        localpath = os.path.join(local_dataroot, filename)
        if os.path.exists(localpath):
            return localpath

def get_file(filename, serverfilter=local_serverfilter, numRetries=3, use_file_cache=True, local_short_circuit=True, timeout=5):
    """
    Get a file from the cluster.
    
    Parameters
    ----------
    filename : string
        filename relative to cluster root
    serverfilter : string
        A filter to use when finding servers - used to facilitate the operation for multiple clusters on the one network
        segment. Note that this is still not fully supported.
    numRetries : int
        The number of times to retry on failure
    use_file_cache : bool
        By default we cache the last 100 files requested locally. This cache never expires, although entries are dropped
        when we get over 100 entries. Under our working assumption that data on the cluster is immutable, this is generally
        safe, with the exception of log files and files streamed using the _aggregate functionality. We can optionally
        request a non-cached version of the file.
    local_short_circuit: bool
        if file exists locally, load/read/return contents directly in this thread unless this flag is False in which case
        we will get the contents through the dataserver over the network.

    Returns
    -------

    """
    #filename = (filename)
    #serverfilter = (serverfilter)
    
    if use_file_cache:
        try:
            return _fileCache[(filename, serverfilter)]
        except KeyError:
            pass

    #look for the file in the local server folder (short-circuit the server)
    localpath = get_local_path(filename, serverfilter) if local_short_circuit else None
    if localpath:
        with open(localpath, 'rb') as f:
            return f.read()
    
    locs = locate_file(filename, serverfilter, return_first_hit=True)

    nTries = 1
    while nTries < numRetries and len(locs) == 0:
        #retry, giving a little bit of time for the data servers to come up
        logger.debug('Could not find %s, retrying ...' % filename)
        time.sleep(1)
        nTries += 1
        locs = locate_file(filename, serverfilter, return_first_hit=True)


    if (len(locs) == 0):
        # we did not find the file
        logger.debug('could not find file %s on cluster: cluster nodes = %s' % (filename, get_ns().list()))
        raise IOError("Specified file could not be found: %s" % filename)


    url = _chooseLocation(locs)#.encode()
    haveResult = False
    nTries = 0
    while nTries < numRetries and not haveResult:
        try:
            nTries += 1
            s = _getSession(url)
            t = time.time()
            r = s.get(url, timeout=timeout)
            dt = time.time() - t
            if dt > 1:
                logger.warning('get_file(%s) took > 1s (%3.2fs)' % (url, dt))
            haveResult = True
        except (requests.Timeout, requests.ConnectionError):
            # s.get sometimes raises ConnectionError instead of ReadTimeoutError
            # see https://github.com/requests/requests/issues/2392
            logger.exception('Timeout on get file %s' % url)
            logger.info('%d retries left' % (numRetries - nTries))
            if nTries == numRetries:
                raise

    #s = _getSession(url)
    #r = s.get(url, timeout=.5)

    if not r.status_code == 200:
        msg = 'Request for %s failed with error: %d' % (url, r.status_code)
        logger.error(msg)
        raise RuntimeError(msg)

    content = r.content

    if len(content) < 1000000:
        #cache small files
        _fileCache[(filename, serverfilter)] = content

    return content


_last_access_time = {}
_lastwritespeed = {}
#_diskfreespace = {}


def _netloc(info):
    return ('%s:%s' % (socket.inet_ntoa(info.address), info.port))


_choose_server_lock = threading.Lock()
def _chooseServer(serverfilter=local_serverfilter, exclude_netlocs=[]):
    """chose a server to save to by minimizing a cost function

    currently takes the server which has been waiting longest

    TODO: add free disk space and improve metrics/weightings

    """
    serv_candidates = [((k), (v)) for k, v in get_ns().get_advertised_services() if
                       (serverfilter in (k)) and not (_netloc(v) in exclude_netlocs)]

    with _choose_server_lock:
        t = time.time()
    
        costs = []
        for k, v in serv_candidates:
            try:
                cost = _last_access_time[k] - t
            except KeyError:
                cost = -100
                
    #        try:
    #            cost -= 0*_lastwritespeed[k]/1e3
    #        except KeyError:
    #            pass
            
            #try:
            #    cost -= _lastwritespeed[k]
            #except KeyError:
            #    pass
    
            costs.append(cost)  # + .1*np.random.normal())
            
        name, info = serv_candidates[np.argmin(costs)]
    
        #t = time.time()
        _last_access_time[name] = t
    
        return name, info


def mirror_file(filename, serverfilter=local_serverfilter):
    """Copies a given file to another server on the cluster (chosen by algorithm)

    The actual copy is performed peer to peer.
    
    This is used in cluster duplication and should not (usually) be called by end-user code
    """
    filename = (filename)
    serverfilter = (serverfilter)
    
    locs = locate_file(filename, serverfilter)

    # where is the data currently located - exclude these from destinations
    currentCopyNetlocs = [urlparse(l[0]).netloc for l in locs]

    # choose a server to mirror onto
    destName, destInfo = _chooseServer(serverfilter, exclude_netlocs=currentCopyNetlocs)

    # and a source to copy from
    sourceUrl = _chooseLocation(locs)

    url = 'http://%s:%d/%s?MirrorSource=%s' % (socket.inet_ntoa(destInfo.address), destInfo.port, filename, sourceUrl)
    url = url.encode()
    s = _getSession(url)
    r = s.put(url, timeout=1)

    if not r.status_code == 200:
        raise RuntimeError('Mirror failed with %d: %s' % (r.status_code, r.content))

    r.close()


def put_file(filename, data, serverfilter=local_serverfilter, timeout=10):
    """
    Put a file to the cluster. The server on which the file resides is chosen by a crude load-balancing algorithm
    designed to uniformly distribute data across the servers within the cluster. The target file must not exist.
    
    .. warning::
        Putting a file is not strictly safe when run from multiple processes, and might result in unexpected behaviour
        if puts with identical filenames are made concurrently (within ~2s). It is up to the calling code to ensure that such
        filename collisions cannot occur. In practice this is reasonably easy to achieve when machine generated filenames
        are used, but implies that interfaces which allow the user to specify arbitrary filenames should run through a
        single user interface with external locking (e.g. clusterUI), particularly if there is any chance that multiple
        users will be creating files simultaneously.
    
    Parameters
    ----------
    filename : string
        path to new file, which much not exist
    data : bytes
        the data to put
    serverfilter : string
        the cluster name (optional)
    timeout: float
        timeout in seconds for http operations. **Warning:** alter from the default setting of 1s only with extreme care.
        If operations are timing out it is usually an indication that something else is going wrong and you should usually
        fix this first. The serverless and lockless architecture depends on having low latency.

    Returns
    -------

    """

    from . import clusterListing as cl

    if not isinstance(data, bytes):
        raise TypeError('data should be bytes (not a unicode string)')
    unifiedIO.assert_name_ok(filename)
    
    success = False
    nAttempts = 0
    
    while not success and nAttempts < 3:
        nAttempts +=1
        name, info = _chooseServer(serverfilter)
    
        url = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, filename)
        print(repr(url))
    
        t = time.time()
    
        #url = str(url) # force tp string (not unicode)
        try:
            s = _getSession(url)
            r = s.put(url, data=data, timeout=timeout)
            dt = time.time() - t
            #print r.status_code
            if not r.status_code == 200:
                raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))

            _lastwritespeed[name] = len(data) / (dt + .001)
            
            if dt > 1:
                logger.warning('put_file(%s) on %s took more than 1s (%3.2f s)' % (filename, url, dt))
            
            success = True

            # add file to location cache
            get_dir_manager(serverfilter).register_file(filename, url, len(data))
            
        except requests.ConnectTimeout:
            if nAttempts >= 3:
                logger.error('Timeout attempting to put file: %s, after 3 retries, aborting' % url)
                raise
            else:
                logger.warn('Timeout attempting to put file: %s, retrying' % url)
        finally:
            try:
                r.close()
            except:
                pass
    
        

if USE_RAW_SOCKETS:
    def _read_status(fp):
        line = fp.readline()
        #print('STATUS: %s' % line)
        try:
            [version, status, reason] = line.split(None, 2)
        except ValueError:
            [version, status] = line.split(None, 1)
            reason = ""
    
        status = int(status)
        
        return version, status, reason
    
    _MAXLINE = 65536
    
    def _parse_response(fp):
        '''A striped down version of httplib.HttpResponse.begin'''
        
        headers = {}
        # read until we get a non-100 response
        while True:
            version, status, reason = _read_status(fp)
            if status != httplib.CONTINUE:
                break
            # skip the header from the 100 response
            while True:
                skip = fp.readline(_MAXLINE + 1)
                if len(skip) > _MAXLINE:
                    raise httplib.LineTooLong("header line")
                skip = skip.strip()
                if not skip:
                    break
            
        # read the rest of the headers
        n_headers = 0
        while True:
            line = fp.readline(_MAXLINE +1)
            if len(line) > _MAXLINE:
                raise httplib.LineTooLong("header line")
            line = line.strip()
            #print(line)
            
            if line in (b'\r\n', b'\n', b''):
                break
                
            n_headers += 1
            if n_headers > httplib._MAXHEADERS:
                raise httplib.HTTPException("got more than %d headers" % httplib._MAXHEADERS)
            
            key, value = line.split(b': ')
            
            headers[key.strip().lower()] = value.strip()
                
                
        #print(headers)
                    
        #if six.PY2:
        #    msg = httplib.HTTPMessage(fp, 0)
        #    msg.fp = None #force the message to relinquish it's file pointer
        #    length = msg.getheader('content-length')
        #else:
        #    # noting that httplib here is actually http.client
        #    msg = httplib.parse_headers(fp)
            
        length = int(headers.get(b'content-length', 0))
        
        
        if length > 0:
            data = fp.read(length)
        else:
            data = b''
        
        return status, reason, data
                
    
    def put_files(files, serverfilter=local_serverfilter, timeout=30):
        """
        Put a bunch of files to a single server in the cluster (chosen by algorithm)
        
        This uses a long-lived http2 session with keep-alive to avoid the connection overhead in creating a new
        session for each file, and puts files before waiting for a response to the last put. This function exists to
        facilitate fast streaming
        
        As it reads the replies *after* attempting to put all the files, this is currently not as safe as put_file (in
        handling failures we assume that no attempts were successful after the first failed file).
        
        Parameters
        ----------
        files : list of tuple
            a list of tuples of the form (<string> filepath, <bytes> data) for the files to be uploaded
            
        serverfilter: str
            the cluster name (optional), to select a specific cluster

        Returns
        -------

        """
        

        files = [(f) for f in files]
        serverfilter = (serverfilter)
        
        nRetries = 0
        nChunksRemaining = len(files)

        dir_manager = get_dir_manager(serverfilter)
        
        while nRetries < 3 and nChunksRemaining > 0:
            name, info = _chooseServer(serverfilter)
            #logger.debug('Chose server: %s:%d' % (name, info.port))
            try:
                t = time.time()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(30)
        
                #conect to the server
                s.connect((socket.inet_ntoa(info.address), info.port))
        
        
                datalen = 0
                      
                #_last_access_time[name] = t
                
                url_ = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)
        
                rs = []
        
                #nChunksRemaining = len(files)
                connection = b'keep-alive'
                #pipeline the sends
                
                nChunksSpooled = 0
                while nChunksRemaining > 0:
                    filename, data = files[-nChunksRemaining]
                    unifiedIO.assert_name_ok(filename)
                    dl = len(data)
                    if nChunksRemaining <= 1:
                        connection = b'close'
        
                    
                    header = b'PUT /%s HTTP/1.1\r\nConnection: %s\r\nContent-Length: %d\r\n\r\n' % (filename.encode(), connection, dl)
                    s.sendall(header)
                    s.sendall(data)
                    
                    # register file now (TODO - wait until we get spooling confirmation?)
                    url = url_ + filename
                    dir_manager.register_file(filename, url, dl)
        
                    datalen += dl
                    nChunksSpooled += 1
                    nChunksRemaining -= 1
                    
                # # TODO - FIXME so that reading replies is fast enough
                # for i in range(nChunksSpooled):
                #     #read all our replies
                #     #print(i, files[i][0])
                #     resp = httplib.HTTPResponse(s, buffering=False)
                #     resp.begin()
                #     status = resp.status
                #     msg = resp.read()
                #     if not status == 200:
                #         logging.debug(('Response %d - status: %d' % (i,status)) + ' msg: ' + msg)
                #         raise RuntimeError('Error spooling chunk %d: status: %d, msg: %s' % (i, status, msg))

                fp = s.makefile('rb', 65536)
                try:
                    for i in range(nChunksSpooled):
                        status, reason, msg = _parse_response(fp)
                        if not status == 200:
                            logging.error(('Response %d - status: %d' % (i, status)) + ' msg: ' + str(msg))
                            raise RuntimeError('Error spooling chunk %d: status: %d, msg: %s' % (i, status, str(msg)))
                finally:
                    fp.close()

                dt = time.time() - t
                _lastwritespeed[name] = datalen / (dt + .001)
                    
            
            except socket.timeout:
                if nRetries < 2:
                    nRetries += 1
                    logger.error('Timeout writing to %s, trying another server for %d remaining files' % (socket.inet_ntoa(info.address), nChunksRemaining))
                else:
                    logger.exception('Timeout writing to %s after 3 retries, aborting - DATA WILL BE LOST' % socket.inet_ntoa(info.address))
                    raise
                
            except socket.error:
                if nRetries < 2:
                    nRetries += 1
                    logger.exception('Error writing to %s, trying another server for %d remaining files' % (socket.inet_ntoa(info.address), nChunksRemaining))
                else:
                    logger.exception('Error writing to %s after 3 retries, aborting - DATA WILL BE LOST' % socket.inet_ntoa(info.address))
                    raise
                
            finally:
                # this causes the far end to close the connection after sending all the replies
                # it is important for the connection to close, otherwise the subsequent recieves will block forever
                # TODO: This is probably a bug/feature of SimpleHTTPServer. The correct way of doing this is probably to send
                # a "Connection: close" header in the last request.
                # s.sendall('\r\n')
        
                # try:
                #     #perform all the recieves at once
                #     resp = s.recv(4096)
                #     while len(resp) > 0:
                #         resp = s.recv(4096)
                # except:
                #     logger.error('Failure to read from server %s' % socket.inet_ntoa(info.address))
                #     s.close()
                #     raise
                #print resp
                #TODO: Parse responses
                s.close()
        
                

            #r.close()
else:
    def put_files(files, serverfilter=local_serverfilter):
        """
        Put a bunch of files to a single server in the cluster (chosen by algorithm)
        
        This version does not normally get called, but is between put_file and the raw sockets version of put_files in speed.
        
        Parameters
        ----------
        files : list of tuple
            a list of tuples of the form (<string> filepath, <bytes> data) for the files to be uploaded
            
        serverfilter: str
            the cluster name (optional), to select a specific cluster

        Returns
        -------

        """
        files = [(f) for f in files]
        serverfilter = (serverfilter)
        
        name, info = _chooseServer(serverfilter)
        dir_manager = get_dir_manager(serverfilter)

        for filename, data in files:
            unifiedIO.assert_name_ok(filename)
            url = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, filename)

            t = time.time()
            #_last_access_time[name] = t
            #url = str(url) #force to string on py2
            s = _getSession(url)
            r = s.put(url, data=data, timeout=1)
            dt = time.time() - t
            #print r.status_code
            if not r.status_code == 200:
                raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))

            dir_manager.register_file(filename, url, len(data))
            _lastwritespeed[name] = len(data) / (dt + .001)

            r.close()

_cached_status = None
_cached_status_expiry = 0
_status_lock = threading.Lock()
def get_status(serverfilter=local_serverfilter):
    """
    Get status of cluster servers (currently only used in the clusterIO web service)
    
    
    Parameters
    ----------
    serverfilter: str
            the cluster name (optional), to select a specific cluster

    Returns
    -------
    status_list: list
        a status dictionary for each node. See PYME.cluster.HTTPDataServer.updateStatus
            Disk: dict
                total: int
                    storage on the node [bytes]
                used: int
                    used storage on the node [bytes]
                free: int
                    available storage on the node [bytes]
            CPUUsage: float
                cpu usage as a percentile
            MemUsage: dict
                total: int
                    total RAM [bytes]
                available: int
                    free RAM [bytes]
                percent: float
                    percent usage
                used: int
                    used RAM [bytes], calculated differently depending on platform
                free: int
                    RAM which is zero'd and ready to go [bytes]
                [other]:
                    more platform-specific fields
            Network: dict
                send: int
                    bytes sent per second since the last status update
                recv: int
                    bytes received per second since the last status update
            GPUUsage: list of float
                [optional] returned for NVIDIA GPUs only. Should be compute usage per gpu as percent?
            GPUMem: list of float
                [optional] returned for NVIDIA GPUs only. Should be memory usage per gpu as percent?


    """
    global _cached_status, _cached_status_expiry

    serverfilter = (serverfilter)

    with _status_lock:
        t = time.time()
        if t > _cached_status_expiry:
            status = []
        
            for name, info in get_ns().get_advertised_services():
                if serverfilter in name:
                    surl = 'http://%s:%d/__status' % (socket.inet_ntoa(info.address), info.port)
                    url = surl.encode()
                    s = _getSession(url)
                    try:
                        r = s.get(url, timeout=.5)
                        st = r.json()
                        st['Responsive'] = True
                        status.append(st)
                    except (requests.Timeout, requests.ConnectionError):
                        # s.get sometimes raises ConnectionError instead of ReadTimeoutError
                        # see https://github.com/requests/requests/issues/2392
                        status.append({"IPAddress":socket.inet_ntoa(info.address), 'Port':info.port, 'Responsive' : False})
    
            _cached_status = status
            _cached_status_expiry = time.time() + 1.0

    return _cached_status




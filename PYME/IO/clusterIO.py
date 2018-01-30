# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 10:07:11 2016

@author: david
"""

import PYME.misc.pyme_zeroconf as pzc
#import urllib
import socket
import requests
import time
import numpy as np
import threading

import socket
import os

USE_RAW_SOCKETS = True

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()

from PYME import config
local_dataroot = config.get('dataserver-root')
local_serverfilter = config.get('dataserver-filter', '')


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

_ns = None
_ns_lock = threading.Lock()

def get_ns():
    global _ns
    with _ns_lock:
        if _ns is None:
            #stagger query times
            time.sleep(3*np.random.rand())
            _ns = pzc.getNS('_pyme-http')
            #wait for replies
            time.sleep(5)

    return _ns
            

if not 'sphinx' in sys.modules.keys():
    # do not start zeroconf if running under sphinx
    # otherwise, spin up a thread to go and find the ns and get on with the rest of our initialization
    
    threading.Thread(target=get_ns).start()

from collections import OrderedDict

class _LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)

        self._lock = threading.Lock()
        self._check_size_limit()

    def __setitem__(self, key, value):
        with self._lock:
            OrderedDict.__setitem__(self, key, value)
            self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


_locateCache = _LimitedSizeDict(size_limit=500)
_dirCache = _LimitedSizeDict(size_limit=100)
DIR_CACHE_TIME = 1
_fileCache = _LimitedSizeDict(size_limit=100)

#use one session for each server (to allow http keep-alives)
sessions = {}
def _getSession(url):
    servinfo = url.split('/')[2]
    try:
        session = sessions[servinfo]
    except KeyError:
        session = requests.Session()
        sessions[servinfo] = session

    return session


def _listSingleDir(dirurl, nRetries=3):
    t = time.time()

    try:
        dirL, rt, dt = _dirCache[dirurl]
        if (t - rt) > DIR_CACHE_TIME:
            raise RuntimeError('key is expired')
        logger.debug('dir cache hit')
    except (KeyError, RuntimeError):
        logger.debug('dir cache miss')
        # t = time.time()
        url = dirurl.encode()
        haveResult = False
        nTries = 0
        while nTries < nRetries and not haveResult:
            try:
                nTries += 1
                s = _getSession(url)
                r = s.get(url, timeout=1)
                haveResult = True
            except (requests.Timeout, requests.ConnectionError) as e:
                # s.get sometimes raises ConnectionError instead of ReadTimeoutError
                # see https://github.com/requests/requests/issues/2392
                logger.exception('Timeout on listing directory')
                logger.info('%d retries left' % (nRetries - nTries))
                if nTries == nRetries:
                    raise

        dt = time.time() - t
        if not r.status_code == 200:
            logger.debug('Request failed with error: %d' % r.status_code)

            #make sure we read a reply so that the far end doesn't hold the connection open
            dump = r.content
            return {}, dt
        try:
            dirL = r.json()
        except ValueError:
            # directory doesn't exist
            return {}, dt

        _dirCache[dirurl] = (dirL, t, dt)

    return dirL, dt


def locateFile(filename, serverfilter='', return_first_hit=False):
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
    cache_key = serverfilter + '::' + filename
    try:
        locs, t = _locateCache[cache_key]
        return locs
    except KeyError:
        locs = []

        dirname = '/'.join(filename.split('/')[:-1])
        fn = filename.split('/')[-1]
        if (len(dirname) >= 1):
            dirname += '/'

        servers = []
        localServers = []

        # print ns.advertised_services.keys()
        services = get_ns().get_advertised_services()
        for name, info in services:
            if serverfilter in name:
                if info is None or info.address is None or info.port is None:
                    # handle the case where zeroconf gives us bad name info. This  is a result of a race condition within
                    # zeroconf, which should probably be fixed instead, but hopefully this workaround is enough.
                    # FIXME - fix zeroconf module race condition on info update.
                    logger.error('''Zeroconf gave us NULL info, ignoring and hoping for the best ...
                    Node with bogus info was: %s
                    Total number of nodes: %d
                    ''' % (name, len(services)))
                else:
                    dirurl = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, dirname)
    
                    if compName in name:
                        localServers.append(dirurl)
                    else:
                        servers.append(dirurl)

        #try data servers on the local machine first
        for dirurl in localServers:
            try:
                dirL, rt, dt = _dirCache[dirurl]
                cached = True
            except KeyError:
                cached = False

            if cached and fn in dirL: #note we're using short-circuit evaluation here
                locs.append((dirurl + fn, dt))

            else:
                dirList, dt = _listSingleDir(dirurl)

                if fn in dirList.keys():
                    locs.append((dirurl + fn, dt))

            if return_first_hit and len(locs) > 0:
                return locs

        #now try data remote servers
        for dirurl in servers:
            try:
                dirL, rt, dt = _dirCache[dirurl]
                cached = True
            except KeyError:
                cached = False

            if cached and fn in dirL: #note we're using short-circuit evaluation here
                locs.append((dirurl + fn, dt))

            else:
                dirList, dt = _listSingleDir(dirurl)

                if fn in dirList.keys():
                    locs.append((dirurl + fn, dt))

            if return_first_hit and len(locs) > 0:
                return locs

        if len(locs) > 0:
            #cache if we found something (this is safe due to write-once nature of fs)
            _locateCache[cache_key] = (locs, time.time())

        return locs

_pool = None

_list_dir_lock = threading.Lock()
def listdirectory(dirname, serverfilter=''):
    """Lists the contents of a directory on the cluster.

    Returns a dictionary mapping filenames to clusterListing.FileInfo named tuples.
    """
    global _pool
    from . import clusterListing as cl
    from multiprocessing.pool import ThreadPool

    with _list_dir_lock:
        if _pool is None:
            _pool = ThreadPool(10)
    
        dirlist = dict()
    
        urls = []
    
        dirname = dirname.lstrip('/')
    
        if not dirname.endswith('/'):
            dirname = dirname + '/'
    
        services = get_ns().get_advertised_services()
        for name, info in services:
            if serverfilter in name:
                if info is None or info.address is None or info.port is None:
                    # handle the case where zeroconf gives us bad name info. This  is a result of a race condition within
                    # zeroconf, which should probably be fixed instead, but hopefully this workaround is enough.
                    # FIXME - fix zeroconf module race condition on info update.
                    logger.error('''Zeroconf gave us NULL info, ignoring and hoping for the best ...
                    Node with bogus info was: %s
                    Total number of nodes: %d
                    ''' % (name, len(services)))
                    
                else:
                    urls.append('http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, dirname))
    
        listings = _pool.map(_listSingleDir, urls)
    
        for dirL, dt in listings:
            cl.aggregate_dirlisting(dirlist, dirL)
    
        return dirlist

def listdir(dirname, serverfilter=''):
    """Lists the contents of a directory on the cluster. Similar to os.listdir,
    but directories are indicated by a trailing slash
    """

    return sorted(listdirectory(dirname, serverfilter).keys())

def isdir(name, serverfilter=''):
    return len(listdir(name, serverfilter)) > 0


def exists(name, serverfilter=''):
#    if name.endswith('/'):
#        name = name[:-1]
#        trailing = '/'
#    else:
#        trailing = ''
#
#    dirname = '/'.join(name.split('/')[:-1])
#    fname = name.split('/')[-1] + trailing
    #return fname in listdir(dirname, serverfilter)
    return (len(locateFile(name, serverfilter, True)) > 0) or isdir(name, serverfilter)


def walk(top, topdown=True, on_error=None, followlinks=False, serverfilter=''):
    """Directory tree generator. Adapted from the os.walk 
    function in the python std library.

    see docs for os.walk for usage details

    """

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

    cost = np.array([l[1] for l in locs])

    return locs[cost.argmin()][0]


def parseURL(URL):
    scheme, body = URL.split('://')
    parts = body.split('/')

    serverfilter = parts[0]
    filename = '/'.join(parts[1:])

    return filename, serverfilter


def isLocal(filename, serverfilter):
    if serverfilter == local_serverfilter and local_dataroot:
        #look for the file in the local server folder (short-circuit the server)
        localpath = os.path.join(local_dataroot, filename)
        return os.path.exists(localpath)
    else:
        return False

def get_local_path(filename, serverfilter):
    if serverfilter == local_serverfilter and local_dataroot:
        #look for the file in the local server folder (short-circuit the server)
        localpath = os.path.join(local_dataroot, filename)
        if os.path.exists(localpath):
            return localpath

def getFile(filename, serverfilter='', numRetries=3, use_file_cache=True):
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

    Returns
    -------

    """
    
    if use_file_cache:
        try:
            return _fileCache[(filename, serverfilter)]
        except KeyError:
            pass

    #look for the file in the local server folder (short-circuit the server)
    localpath = get_local_path(filename, serverfilter)
    if localpath:
        with open(localpath, 'rb') as f:
            return f.read()
    
    locs = locateFile(filename, serverfilter, return_first_hit=True)

    nTries = 1
    while nTries < numRetries and len(locs) == 0:
        #retry, giving a little bit of time for the data servers to come up
        logger.debug('Could not find file, retrying ...')
        time.sleep(1)
        nTries += 1
        locs = locateFile(filename, serverfilter, return_first_hit=True)


    if (len(locs) == 0):
        # we did not find the file
        logger.debug('could not find file %s on cluster: cluster nodes = %s' % (filename, get_ns().list()))
        raise IOError("Specified file could not be found: %s" % filename)


    url = _chooseLocation(locs).encode()
    haveResult = False
    nTries = 0
    while nTries < numRetries and not haveResult:
        try:
            nTries += 1
            s = _getSession(url)
            r = s.get(url, timeout=.5)
            haveResult = True
        except (requests.Timeout, requests.ConnectionError) as e:
            # s.get sometimes raises ConnectionError instead of ReadTimeoutError
            # see https://github.com/requests/requests/issues/2392
            logger.exception('Timeout on get file')
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
    return '%s:%s' % (socket.inet_ntoa(info.address), info.port)


_choose_server_lock = threading.Lock()
def _chooseServer(serverfilter='', exclude_netlocs=[]):
    """chose a server to save to by minimizing a cost function

    currently takes the server which has been waiting longest

    TODO: add free disk space and improve metrics/weightings

    """
    serv_candidates = [(k, v) for k, v in get_ns().get_advertised_services() if
                       (serverfilter in k) and not (_netloc(v) in exclude_netlocs)]

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


def mirrorFile(filename, serverfilter=''):
    """Copies a given file to another server on the cluster (chosen by algorithm)

    The actual copy is performed peer to peer.
    """

    locs = locateFile(filename, serverfilter)

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


def putFile(filename, data, serverfilter=''):
    """put a file to a server in the cluster (chosen by algorithm)

    TODO - Add retry with a different server on failure
    """
    success = False
    nAttempts = 0
    
    while not success and nAttempts < 3:
        nAttempts +=1
        name, info = _chooseServer(serverfilter)
    
        url = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, filename)
    
        t = time.time()
    
        url = url.encode()
        try:
            s = _getSession(url)
            r = s.put(url, data=data, timeout=1)
            dt = time.time() - t
            #print r.status_code
            if not r.status_code == 200:
                raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))

            _lastwritespeed[name] = len(data) / (dt + .001)
            
            success = True
            
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
    def putFiles(files, serverfilter=''):
        """put a bunch of files to a single server in the cluster (chosen by algorithm)

        TODO - Add retry with a different server on failure
        """
        
        nRetries = 0
        nChunksRemaining = len(files)
        
        while nRetries < 3 and nChunksRemaining > 0:
            name, info = _chooseServer(serverfilter)
            #logger.debug('Chose server: %s:%d' % (info.address, info.port))
    
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(5.0)
    
            #conect to the server
            s.connect((socket.inet_ntoa(info.address), info.port))
    
    
            datalen = 0
    
            t = time.time()
            #_last_access_time[name] = t
    
            rs = []
    
            #nChunksRemaining = len(files)
            connection = 'keep-alive'
            #pipeline the sends
            
            try:
                while nChunksRemaining > 0:
                    filename, data = files[-nChunksRemaining]
                    dl = len(data)
                    if nChunksRemaining <= 1:
                        connection = 'close'
        
                    
                    header = 'PUT /%s HTTP/1.1\r\nConnection: %s\r\nContent-Length: %d\r\n\r\n' % (filename, connection, dl)
                    s.sendall(header)
                    s.sendall(data)
        
                    datalen += dl
                    nChunksRemaining -= 1

            except socket.timeout:
                if nRetries < 3:
                    nRetries += 1
                    logger.error('Timeout writing to %s, trying another server for %d remaining files' % (socket.inet_ntoa(info.address), nChunksRemaining))
                else:
                    logger.error('Timeout writing to %s after 3 retries, aborting - DATA WILL BE LOST' % socket.inet_ntoa(info.address))
                    raise
                
            finally:
                # this causes the far end to close the connection after sending all the replies
                # it is important for the connection to close, otherwise the subsequent recieves will block forever
                # TODO: This is probably a bug/feature of SimpleHTTPServer. The correct way of doing this is probably to send
                # a "Connection: close" header in the last request.
                # s.sendall('\r\n')
        
                try:
                    #perform all the recieves at once
                    resp = s.recv(4096)
                    while len(resp) > 0:
                        resp = s.recv(4096)
                except:
                    logger.error('Failure to read from server %s' % socket.inet_ntoa(info.address))
                    s.close()
                    raise
                #print resp
                #TODO: Parse responses
                s.close()
        
                dt = time.time() - t
                _lastwritespeed[name] = datalen / (dt + .001)

            #r.close()
else:
    def putFiles(files, serverfilter=''):
        """put a bunch of files to a single server in the cluster (chosen by algorithm)

        TODO - Add retry with a different server on failure
        """
        name, info = _chooseServer(serverfilter)

        for filename, data in files:
            url = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, filename)

            t = time.time()
            #_last_access_time[name] = t
            url = url.encode()
            s = _getSession(url)
            r = s.put(url, data=data, timeout=1)
            dt = time.time() - t
            #print r.status_code
            if not r.status_code == 200:
                raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))

            _lastwritespeed[name] = len(data) / (dt + .001)

            r.close()

_cached_status = None
_cached_status_expiry = 0
def getStatus(serverfilter=''):
    """Lists the contents of a directory on the cluster. Similar to os.listdir,
        but directories are indicated by a trailing slash
        """
    import json
    global _cached_status, _cached_status_expiry

    t = time.time()
    if t < _cached_status_expiry:
        return _cached_status

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
    _cached_status_expiry = time.time() + .5

    return status




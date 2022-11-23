"""
Work in progress ... refactor some of the listing stuff out of clusterIO to

a) reduce the size of clusterIO
b) reduce duplication between, e.g. locate and listdirectory
c) ultimately allow pluggable directory management / caching, e.g. to enable a central directory server (unclear if this
   would solve current performance issues, but potentially worth a try).

"""
import time
import socket
import threading
import requests
import posixpath
import sys
import numpy as np

from PYME import config

from multiprocessing.pool import ThreadPool

from PYME.misc.computerName import GetComputerName
local_computer_name = GetComputerName()

import logging
logger = logging.getLogger(__name__)

SERVICE_CACHE_LIFETIME = 1 #seconds
DIR_CACHE_TIME = 1 #seconds

import PYME.misc.pyme_zeroconf as pzc
from PYME.misc import hybrid_ns

_ns = None
_ns_lock = threading.Lock()

if config.get('clusterIO-hybridns', True):
    def get_ns():
        global _ns
        with _ns_lock:
            if _ns is None:
                #stagger query times
                time.sleep(3 * np.random.rand())
                #_ns = pzc.getNS('_pyme-http')
                _ns = hybrid_ns.getNS('_pyme-http')
                #wait for replies
                time.sleep(5)
        
        return _ns
else:
    def get_ns():
        global _ns
        with _ns_lock:
            if _ns is None:
                #stagger query times
                time.sleep(3 * np.random.rand())
                #_ns = pzc.getNS('_pyme-http')
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

class DirectoryInfoManager(object):
    def __init__(self, ns=None, serverfilter=''):
        self._dirCache = _LimitedSizeDict(size_limit=100)
        self._locateCache = _LimitedSizeDict(size_limit=500)
        
        if not ns:
            self._ns = get_ns()
        else:
            self._ns = ns
            
        self.serverfilter = serverfilter
        
        self._service_expiry_time = 0
        self._cached_servers = None

        #use one session for each server (to allow http keep-alives)
        self._sessions = {}
        self._pool = ThreadPool(10)
        
        self._list_dir_lock = threading.Lock()

    def _getSession(self, url):
        if not isinstance(url, bytes):
            url = url.encode()

        servinfo = url.split(b'/')[2]
        try:
            session = self._sessions[servinfo]
        except KeyError:
            session = requests.Session()
            session.trust_env = False
            self._sessions[servinfo] = session

        return session
        
        
    @property
    def dataservers(self):
        """
        Find all the data servers belonging to the cluster, caching the results

        """
        t = time.time()
        if (t > self._service_expiry_time):
            self._cached_servers = []
            services = self._ns.get_advertised_services()

            for name, info in services:
                if self.serverfilter in name:
                    # only look at servers associated with our cluster
                    if info is None or info.address is None or info.port is None:
                        # handle the case where zeroconf gives us bad name info. This  is a result of a race condition within
                        # zeroconf, which should probably be fixed instead, but hopefully this workaround is enough.
                        # FIXME - fix zeroconf module race condition on info update.
                        logger.error('''Zeroconf gave us NULL info, ignoring and hoping for the best ...
                        Node with bogus info was: %s
                        Total number of nodes: %d
                        ''' % (name, len(services)))
                    else:
                        serverurl = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)
            
                        if local_computer_name in name:
                            self._cached_servers.insert(0, serverurl)
                        else:
                            self._cached_servers.append(serverurl)
            
            self._service_expiry_time = t + SERVICE_CACHE_LIFETIME
        
        if len(self._cached_servers) < 1:
            # check that there is a cluster running
            raise IOError('No data servers found, is the cluster running?')
        
        return self._cached_servers
            
            
    def list_single_node_dir(self, dirurl, nRetries = 1, timeout = 10, strict_caching=False):
        """
        List the directory on a single node
        
        Parameters
        ----------
        dirurl
        nRetries
        timeout

        Returns
        -------

        """
        t = time.time()
    
        try:
            dirL, rt, dt = self._dirCache[dirurl]
            time_in_cache = t - rt
            if time_in_cache > DIR_CACHE_TIME:
                # use cached value if the last request took longer than the length of time we've been expired as it
                # doesn't make sense to hit the server again
                if strict_caching or (time_in_cache > (DIR_CACHE_TIME + dt)):
                    raise RuntimeError('key is expired')
                else:
                    logger.warning('Using expired entry from directory cache on %s as previous request took too long' % dirurl)
        except (KeyError, RuntimeError):
            #logger.debug('dir cache miss')
            # t = time.time()
            url = dirurl.encode()
            haveResult = False
            nTries = 0
            while nTries < nRetries and not haveResult:
                try:
                    nTries += 1
                    s = self._getSession(url)
                    r = s.get(url, timeout=timeout)
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
                logger.debug('Request for %s failed with error: %d' % (url, r.status_code))
            
                #make sure we read a reply so that the far end doesn't hold the connection open
                dump = r.content
                return {}, dt
            try:
                dirL = r.json()
            except ValueError:
                # directory doesn't exist
                # TODO - cache empty directories???
                return {}, dt
        
            self._dirCache[dirurl] = (dirL, t, dt)
    
        return dirL, dt

    def listdirectory(self, dirname, timeout=5):
        """Lists the contents of a directory on the cluster.

        Returns a dictionary mapping filenames to clusterListing.FileInfo named tuples.
        """
        from . import clusterListing as cl
        dirlist = dict()
        dirname = dirname.lstrip('/')

        if not dirname.endswith('/'):
            dirname = dirname + '/'
  
        urls = [s + dirname for s in self.dataservers]
        
        with self._list_dir_lock:
            # TODO - is lock needed?
            listings = self._pool.map(self.list_single_node_dir, urls)

        for dirL, dt in listings:
            cl.aggregate_dirlisting(dirlist, dirL)

        return dirlist

    def listdir(self, dirname):
        """Lists the contents of a directory on the cluster. Similar to os.listdir,
        but directories are indicated by a trailing slash
        """
    
        return sorted(self.listdirectory(dirname).keys())

    def locate_file(self, filename, return_first_hit=False):
        """
        Searches the cluster to find which server(s) a given file is stored on

        Parameters
        ----------
        filename : str
            The file name
        return_first_hit : bool
            Whether to try and find all locations, or return when we find the first copy

        Returns
        -------

        """
    
        try:
            locs, t = self._locateCache[filename]
            return locs
        except KeyError:
            locs = []
            
            dirname, fn = posixpath.split(filename)
            if not dirname.endswith('/'):
                dirname += '/'
            
            for s in self.dataservers:
                #note: servers are ordered so we try those on the local machine first
                dirurl = s + dirname
                try:
                    dirL, rt, dt = self._dirCache[dirurl]
                    if dirL[fn]: # this will raise a KeyError if filename is not present
                        locs.append((dirurl + fn, dt))
                except KeyError:
                    dirL, dt = self.list_single_node_dir(dirurl)
                
                    if dirL.get(fn, None) is not None:
                        locs.append((dirurl + fn, dt))
            
                if return_first_hit and len(locs) > 0:
                    return locs
        
            if len(locs) > 0:
                #cache if we found something (this is safe due to write-once nature of fs)
                self._locateCache[filename] = (locs, time.time())
        
            return locs

    def isdir(self, name):
        """
        Tests if a given path on the cluster is a directory. Analogous to os.path.isdir
        """
        import os
        
        name = name.rstrip('/')
        if name in ['/', '']:
            #special case for root dir
            return True
    
        pn, n = posixpath.split(name)
        try:
            d = self.listdirectory(pn)[n + '/']
        except KeyError:
            return False
    
        return d.type > 0

    def _cglob(self, url, timeout=2, nRetries=1):
        url = url.encode()
        haveResult = False
        nTries = 0
        while nTries < nRetries and not haveResult:
            try:
                nTries += 1
                s = self._getSession(url)
                r = s.get(url, timeout=timeout)
                haveResult = True
            except (requests.Timeout, requests.ConnectionError) as e:
                # s.get sometimes raises ConnectionError instead of ReadTimeoutError
                # see https://github.com/requests/requests/issues/2392
                logger.exception('Timeout on listing directory')
                logger.info('%d retries left' % (nRetries - nTries))
                if nTries == nRetries:
                    raise
    
        if not r.status_code == 200:
            logger.debug('Request failed with error: %d' % r.status_code)
        
            #make sure we read a reply so that the far end doesn't hold the connection open
            dump = r.content
            return []
    
        try:
            matches = r.json()
        except ValueError:
            # directory doesn't exist
            return []
    
        return list(matches)

    def cglob(self, pattern):
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
        urls = [s + '__glob?pattern=%s' % pattern for s in self.dataservers]
        
        with self._list_dir_lock:
            matches = self._pool.map(self._cglob, urls)
    
        #concatenate lists
        matches = sum(matches, [])
        return list(set(matches))

    def exists(self, name):
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
        return (len(self.locate_file(name, True)) > 0) or self.isdir(name)
    
    def register_file(self, filename, url, size):
        """
        Call after uploading a new file so we can update our caches

        """
        from . import clusterListing as cl
        t = time.time()
        self._locateCache[filename] = ([(url, .1), ], t)
        try:
            dirurl, fn = posixpath.split(url)
            dirurl = dirurl + '/'
            dirL, rt, dt = self._dirCache[dirurl]
            if (t - rt) > DIR_CACHE_TIME:
                pass #cache entry is expired
            else:
                dirL[fn] = cl.FileInfo(cl.FILETYPE_NORMAL, size)
                self._dirCache[dirurl] = (dirL, rt, dt)
    
        except KeyError:
            pass
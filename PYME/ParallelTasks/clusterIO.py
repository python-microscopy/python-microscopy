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

import urlparse

import logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


ns = pzc.getNS('_pyme-http')
time.sleep(1.5) #wait for ns resonses

from collections import OrderedDict
class LimitedSizeDict(OrderedDict):
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

_locateCache = LimitedSizeDict(size_limit=500)
_dirCache = LimitedSizeDict(size_limit=100)
DIR_CACHE_TIME = 20

def _listSingleDir(dirurl):
    t = time.time()
    
    try:
        dirL, rt, dt = _dirCache[dirurl]
        if (t-rt) > DIR_CACHE_TIME:
            raise RuntimeError('key is expired')
    except (KeyError, RuntimeError):
        #t = time.time()
        r = requests.get(dirurl.encode(), timeout=.1)
        dt = time.time() - t
        try:
            dirL = r.json()
        except ValueError:
            #directory doesn't exist
            dirL = []
        _dirCache[dirurl] = (dirL, t, dt)
        
    return dirL, dt
        

def locateFile(filename, serverfilter=''):
    cache_key = serverfilter + '::' + filename
    try: 
        locs, t = _locateCache[cache_key]
        return locs
    except KeyError:
        locs = []
        
        dirname = '/'.join(filename.split('/')[:-1])
        fn = filename.split('/')[-1]
        if (len(dirname) >=1):
            dirname = dirname + '/'
        
        for name, info in ns.advertised_services.items():
            if serverfilter in name:
                dirurl = 'http://%s:%d/%s' %(socket.inet_ntoa(info.address), info.port, dirname) 
                dirList, dt = _listSingleDir(dirurl)
                
                if fn in dirList:
                    locs.append((dirurl + fn, dt))
                    
        _locateCache[cache_key] = (locs, time.time())
                    
        return locs
    
def listdir(dirname, serverfilter=''):
    '''Lists the contents of a directory on the cluster. Similar to os.listdir,
    but directories are indicated by a trailing slash
    '''
    dirlist = set()
    
    for name, info in ns.advertised_services.items():
        if serverfilter in name:
            dirurl = 'http://%s:%d/%s' %(socket.inet_ntoa(info.address), info.port, dirname) 
            #print dirurl
            dirL, dt = _listSingleDir(dirurl)  

            dirlist.update(dirL)
                
    return list(dirlist)
    
def exists(name, serverfilter=''):
    if name.endswith('/'):
        name = name[:-1]
        trailing='/'
    else:
        trailing= ''
        
    dirname = '/'.join(name.split('/')[:-1])
    fname = name.split('/')[-1] + trailing
    
    return fname in listdir(dirname, serverfilter)
    
def walk(top, topdown=True, onerror=None, followlinks=False, serverfilter=''):
    """Directory tree generator. Adapted from the os.walk 
    function in the python std library.

    see docs for os.walk for usage details

    """

    #islink, join, isdir = path.islink, path.join, path.isdir
    def islink(name):
        #cluster does not currently have the concept of symlinks
        return False
        
    def join(*args):
        j =  '/'.join(args)
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
    except Exception, err:
        if onerror is not None:
            onerror(err)
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
            for x in walk(new_path, topdown, onerror, followlinks):
                yield x
    if not topdown:
        yield top, dirs, nondirs
    

def _chooseLocation(locs):
    '''Chose the location to load the file from

    default to choosing the "closest" (currently the one with the shortest response 
    time to our directory query)
    
    '''
    
    cost = np.array([l[1] for l in locs])
    
    return locs[cost.argmin()][0]
    
def getFile(filename, serverfilter=''):
    locs = locateFile(filename)
    
    if (len(locs) ==0):
        #we did not find the file
        raise IOError("Specified file could not be found")
    else:
        url = _chooseLocation(locs)
        r = requests.get(url.encode(), timeout=.1)
        
        return r.content

_lastwritetime = {}
_lastwritespeed = {}
#_diskfreespace = {}

def _netloc(info):
    return '%s:%s' % (socket.inet_ntoa(info.address), info.port)

def _chooseServer(serverfilter='', exclude_netlocs=[]):
    '''chose a server to save to by minimizing a cost function
    
    currently takes the server which has been waiting longest
    
    TODO: add free disk space and improve metrics/weightings
    
    '''
    serv_candidates = [(k, v) for k, v in ns.advertised_services.items() if (serverfilter in k) and not (_netloc(v) in exclude_netlocs)]

    t = time.time()    
    
    costs = []
    for k, v in serv_candidates:
        try:
            cost = _lastwritetime[k] - t
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
            
        costs.append(cost)# + .1*np.random.normal())
        
    return serv_candidates[np.argmin(costs)]
    
def mirrorFile(filename, serverfilter=''):
    '''Copies a given file to another server on the cluster (chosen by algorithm)
    
    The actual copy is performed peer to peer.
    '''
    
    locs = locateFile(filename, serverfilter)
    
    #where is the data currently located - exclude these from destinations
    currentCopyNetlocs = [urlparse.urlparse(l[0]).netloc for l in locs]
    
    #choose a server to mirror onto
    destName, destInfo = _chooseServer(serverfilter, exclude_netlocs=currentCopyNetlocs)
    
    #and a source to copy from
    sourceUrl = _chooseLocation(locs)
    
    url = 'http://%s:%d/%s?MirrorSource=%s' %(socket.inet_ntoa(destInfo.address), destInfo.port, filename, sourceUrl)
    
    r = requests.put(url.encode(), timeout=1)

    if not r.status_code == 200:
        raise RuntimeError('Mirror failed with %d: %s' % (r.status_code, r.content))
    
    r.close()
    
        
def putFile(filename, data, serverfilter=''):
    '''put a file to a server in the cluster (chosen by algorithm)
    
    TODO - Add retry with a different server on failure
    '''
    name, info = _chooseServer(serverfilter)
    
    url = 'http://%s:%d/%s' %(socket.inet_ntoa(info.address), info.port, filename)
    
    t = time.time()
    _lastwritetime[name] = t
    r = requests.put(url.encode(), data=data, timeout=1)
    dt = time.time() - t
    #print r.status_code
    if not r.status_code == 200:
        raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))
    
    r.close()
        
    
    _lastwritespeed[name] = len(data)/dt
        
def putFiles(files, serverfilter=''):
    '''put a bunch of files to a single server in the cluster (chosen by algorithm)
    
    TODO - Add retry with a different server on failure
    '''
    name, info = _chooseServer(serverfilter)
    
    for filename, data in files:    
        url = 'http://%s:%d/%s' %(socket.inet_ntoa(info.address), info.port, filename)
        
        t = time.time()
        _lastwritetime[name] = t
        r = requests.put(url.encode(), data=data, timeout=1)
        dt = time.time() - t
        #print r.status_code
        if not r.status_code == 200:
            raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))
            
        
        _lastwritespeed[name] = len(data)/dt
        
    r.close()
    
    
        
    
        
    
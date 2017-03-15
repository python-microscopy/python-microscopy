"""
This file provides utility functions for creating and interpreting clusterIO directory listings.
"""
from collections import namedtuple
import os
import six

#flags (bitfields) for file type specification
FILETYPE_NORMAL = 0
FILETYPE_DIRECTORY = 1 << 0
FILETYPE_SERIES =  1 << 1 # a directory which is actually a series - treat specially
FILETYPE_SERIES_COMPLETE = 1 << 2 # a series which has finished spooling

""" Information about a file in a directory listing.

type: is a bitfield consisting of a combination of filetype bits.

size: is the size of the file in bytes, OR the number of entries in a directory

"""
FileInfo = namedtuple('FileInfo', 'type, size')

try:
    from PYME.IO.countdir import dirsize, file_info

    def _file_info(path, fn):
        fpath = os.path.join(path, fn)

        finfo = file_info(fpath)

        if finfo[0] & FILETYPE_DIRECTORY:
            fn = fn + '/'

        return (fn, finfo)
except ImportError: # coundir module is posix only, fall back to more naive methods on windows
    def dirsize(path):
        return len(os.listdir(path))

    def _file_info(path, fn):
        fpath = os.path.join(path, fn)
        if os.path.isdir(fpath):
            ftype = FILETYPE_DIRECTORY

            if os.path.exists(fpath + '/metadata.json'):
                #if there is a metadata.json, set the series flag
                ftype |= FILETYPE_SERIES

            if os.path.exists(fpath + '/events.json'):
                #if there is a metadata.json, set the series flag
                ftype |= FILETYPE_SERIES_COMPLETE

            return (fn + '/',  FileInfo(ftype, dirsize(fpath)))
        else:
            return (fn,  FileInfo(FILETYPE_NORMAL, os.path.getsize(fpath)))


def aggregate_dirlisting(dir_list, single_dir):
    """
    aggregate / add file info in a dir listing

    Parameters
    ----------
    dir_list : dict
        a dictionary mapping filenames to FileInfo.
    single_dir : dict
        a dictionary mapping filenames to tuples, as returned by a json load of a single directory listing.

    Returns
    -------

    """

    for k, v in six.iteritems(single_dir):
        type, size = v

        fi = dir_list.get(k, None)
        if not fi is None:
            type |= fi.type

            if type & FILETYPE_DIRECTORY:
                size += fi.size

        dir_list[k] = FileInfo(type, size)

import threading
import time
import os

class DirCache(object):
    def __init__(self, cache_size = 1000, lifetime_s=(30*60)):
        self._cache_size = cache_size
        self._cache = {}
        self._purge_list = []
        self._n = 0
        self._lifetime_s = lifetime_s
        
        self._lock = threading.Lock()
        
    def update_cache(self, filename, filesize):
        dirname, fname = os.path.split(filename)
        with self._lock:
            try:
                try:
                    fs = self._cache[dirname][filename].size
                except KeyError:
                    fs = 0
                    
                self._cache[dirname][filename] = FileInfo(FILETYPE_NORMAL, fs + filesize)
            except KeyError:
                pass
        
    
    def _add_entry(self, dirname, listing):
        with self._lock:
            if self._n >= self._cache_size:
                # overflowing - remove an entry before adding
        
                to_remove = self._purge_list.pop(0)
                self._cache.pop(to_remove)
                self._n -= 1
            
            self._cache[dirname] = (listing, time.time() + self._lifetime_s)
            self._purge_list.append(dirname)
            self._n += 1
    
    def list_directory(self, dirname):
        try:
            listing, expiry = self._cache[dirname]
            if expiry < time.time():
                raise RuntimeError('Cache entry expired')
        except (KeyError, RuntimeError):
            list = os.listdir(dirname)
    
            list.sort(key=lambda a: a.lower())
    
            #l2 = dict(map(lambda fn : _file_info(path, fn), list))
            listing = dict([_file_info(dirname, fn) for fn in list])
            self._add_entry(dirname, listing)
            
        return listing
    
            
dir_cache = DirCache()
        


def list_directory(path):
    list = os.listdir(path)

    list.sort(key=lambda a: a.lower())

    #l2 = dict(map(lambda fn : _file_info(path, fn), list))
    l2 = dict([_file_info(path, fn) for fn in list])

    return l2

_pool = None

def list_directory_p(path):
    global _pool
    from multiprocessing.pool import ThreadPool
    list = os.listdir(path)

    list.sort(key=lambda a: a.lower())

    if _pool is None:
        _pool = ThreadPool(10)

    l2 = dict(_pool.map(lambda fn : _file_info(path, fn), list))

    return l2

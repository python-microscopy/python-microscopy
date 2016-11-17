"""
This file provides utility functions for creating and interpreting clusterIO directory listings.
"""
from collections import namedtuple
import os
from PYME.IO.countdir import dirsize

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

def dirsize_(path):
    return len(os.listdir(path))

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

    for k, v in single_dir.iteritems():
        type, size = v

        fi = dir_list.get(k, None)
        if not fi is None:
            type |= fi.type

            if type & FILETYPE_DIRECTORY:
                size += fi.size

        dir_list[k] = FileInfo(type, size)

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

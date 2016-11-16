"""
This file provides utility functions for creating and interpreting clusterIO directory listings.
"""
from collections import namedtuple

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

"""Filename globbing utility.

Stolen and adapted from the python standard library glob module
"""

import sys
import os
import re
import fnmatch

from . import clusterIO

try:
    _unicode = unicode
except NameError:
    # If Python is built without Unicode support, the unicode type
    # will not exist. Fake one.
    class _unicode(object):
        pass

__all__ = ["glob", "iglob"]

def glob(pathname, serverfilter=clusterIO.local_serverfilter, include_scheme=False):
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    """
    return list(iglob(pathname, serverfilter, include_scheme))

def iglob(pathname, serverfilter=clusterIO.local_serverfilter, include_scheme=False):
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    """

    scheme = 'pyme-cluster://' + serverfilter if include_scheme else ''
    dirname, basename = os.path.split(pathname)
    if not has_magic(pathname):
        if basename:
            if clusterIO.exists(pathname, serverfilter):
                yield scheme + pathname
        else:
            # Patterns ending with a slash should match only directories
            if clusterIO.isdir(dirname, serverfilter):
                yield scheme + pathname
        return
    if not dirname:
        for name in glob1('/', basename, serverfilter):
            yield scheme + name
        return
    # `os.path.split()` returns the argument itself as a dirname if it is a
    # drive or UNC path.  Prevent an infinite recursion if a drive or UNC path
    # contains magic characters (i.e. r'\\?\C:').
    if dirname != pathname and has_magic(dirname):
        dirs = iglob(dirname, serverfilter)
    else:
        dirs = [dirname]
    if has_magic(basename):
        glob_in_dir = glob1
    else:
        glob_in_dir = glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename, serverfilter):
            yield scheme + '/'.join([dirname, name])

# These 2 helper functions non-recursively glob inside a literal directory.
# They return a list of basenames. `glob1` accepts a pattern while `glob0`
# takes a literal basename (so it only has to check for its existence).

def glob1(dirname, pattern, serverfilter=clusterIO.local_serverfilter):
    if not dirname:
        dirname = '/'
    #if isinstance(pattern, _unicode) and not isinstance(dirname, unicode):
    #    dirname = unicode(dirname, sys.getfilesystemencoding() or
    #                               sys.getdefaultencoding())
    
    try:
        names = [n.rstrip('/') for n in clusterIO.listdir(dirname, serverfilter)]
    except os.error:
        return []
    if pattern[0] != '.':
        names = filter(lambda x: x[0] != '.', names)
    return fnmatch.filter(names, pattern)

def glob0(dirname, basename, serverfilter=clusterIO.local_serverfilter):
    if basename == '':
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator.  'q*x/' should match only directories.
        if clusterIO.isdir(dirname, serverfilter):
            return [basename]
    else:
        if clusterIO.exists('/'.join([dirname, basename]), serverfilter):
            return [basename]
    return []


magic_check = re.compile('[*?[]')

def has_magic(s):
    return magic_check.search(s) is not None

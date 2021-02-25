from PYME.IO.FileUtils import nameUtils
import os
from io import BytesIO
from contextlib import contextmanager
import tempfile
import re
try:  # py3
    from urllib.parse import quote, urlencode
except ImportError:  # py2
    from urllib import quote, urlencode
import logging
logger = logging.getLogger(__name__)


alpha_regex = re.compile(r'^[\w/\.\-]+$')

def check_name(name):
    """
    Check if filename / url is OK to use with, e.g. clusterIO.

    Parameters
    ----------
    name: str
        The filename or url to query

    Returns
    -------
    ok: bool
        True if filename/url is fully compatible

    """
    return bool(alpha_regex.match(name))
    #return name == fix_name(name)

def check_uri(name):
    return check_name(name.split('://')[1])

def assert_name_ok(name):
    """
    Raise if name contains reserved/invalid characters for use with, e.g. clusterIO.

    Parameters
    ----------
    name: str or bytes
        The filename or url to query

    """
    try:
        assert check_name(name) == True
    except AssertionError:
        raise AssertionError('Name "%s" is invalid. Names must only include alphanumeric characters and underscore' % name)

def assert_uri_ok(name):
    """
    Raise if name contains reserved/invalid characters for use with, e.g. clusterIO.

    Parameters
    ----------
    name: str or bytes
        The filename or url to query

    """
    try:
        assert check_uri(name) == True
    except AssertionError:
        raise AssertionError('Name "%s" is invalid. Names must only include alphanumeric characters and underscore' % name)

def fix_name(name):
    """
    Cleans filename / url for use with, e.g. clusterIO, by replacing spaces with underscores and removing all
    percent-encoded characters other than ':' and '/'.

    Parameters
    ----------
    name: str
        The filename or url to query

    Returns
    -------
    fixed_name: str
        The cleaned file name

    """
    return re.sub('%..', '', quote(name.replace(' ', '_')).replace('%3A', ':'))

def verbose_fix_name(name):
    """
    Wrapper for fix_name which sacrifices performance in order to complain.
    Parameters
    ----------
    name: str
        The filename or url to query

    Returns
    -------
    fixed_name: str
        The cleaned file name

    """
    try:
        assert_name_ok(name)
    except AssertionError as e:
        logger.error(str(e))
    return fix_name(name)

def is_cluster_uri(url):
    """
    Checks whether the supplied uri/filename is a pyme-cluster uri
    
    Parameters
    ----------
    url

    Returns
    -------
    
    bool

    """
    return (url.startswith('pyme-cluster') or url.startswith('PYME-CLUSTER'))

def split_cluster_url(url):
    if not is_cluster_uri(url):
        raise RuntimeError('Not a cluster URL')

    clusterfilter = url.split('://')[1].split('/')[0]
    sequenceName = url.split('://%s/' % clusterfilter)[1]

    return sequenceName, clusterfilter

def dirname(url):
    try:
        return os.path.dirname(split_cluster_url(url)[0])
    except RuntimeError:
        return os.path.dirname(nameUtils.getFullExistingFilename(url))

@contextmanager
def local_or_temp_filename(url):
    """
    Gets a local filename for a given url.

    The purpose is to let us load files from the cluster using IO libraries (e.g. pytables, tifffile) which need a
    filename, not a file handle.



    Does the following (in order).

    * checks to see if the url refers to a local file, if so return the filename
    * if the url points to a cluster file, check to see if it happens to be stored locally. If so, return the local path
    * download file to a temporary file and return the filename of that temporary file.

    NB: This should be used as a context manager (ie in a with statement) so that the temporary file gets cleaned up properly

    Parameters
    ----------
    url : basestring

    Returns
    -------

    filename : basestring

    Notes
    -----

    TODO - there is potential for optimization by using memmaps rather than files on disk.

    """
    filename = nameUtils.getFullExistingFilename(url)

    if os.path.exists(filename):
        yield filename
    elif is_cluster_uri(url):
        from .import clusterIO

        sequenceName, clusterfilter = split_cluster_url(filename)

        localpath = clusterIO.get_local_path(sequenceName, clusterfilter)
        if localpath:
            yield localpath
        else:
            ext = os.path.splitext(sequenceName)[-1]

            with tempfile.NamedTemporaryFile(mode='w+b', suffix=ext) as outf:
                s = clusterIO.get_file(sequenceName, clusterfilter)
                outf.write(s)
                outf.flush()

                yield outf.name

    else:
        raise IOError('Path "%s" could not be found' % url)


def openFile(filename, mode='rb'):
    filename = nameUtils.getFullExistingFilename(filename)

    if os.path.exists(filename):
        return open(filename, mode)

    elif is_cluster_uri(filename):
        #TODO - add short-circuiting for local files
        from . import clusterIO

        sequenceName, clusterfilter = split_cluster_url(filename)

        s = clusterIO.get_file(sequenceName, clusterfilter)
        return BytesIO(s)
    else:
        raise IOError('File does not exist or URI not understood: %s' % filename)

def read(filename):
    filename = nameUtils.getFullExistingFilename(filename)

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            s = f.read()
        return s

    elif is_cluster_uri(filename):
        from . import clusterIO

        sequenceName, clusterfilter = split_cluster_url(filename)

        s = clusterIO.get_file(sequenceName, clusterfilter)
        return s
    else:
        raise IOError('File does not exist or URI not understood: %s' % filename)
    
def write(filename, data):
    if is_cluster_uri(filename):
        from . import clusterIO
        sequenceName, clusterfilter = split_cluster_url(filename)
        clusterIO.put_file(sequenceName, data, serverfilter=clusterfilter)
    else:
        with open(filename, 'wb') as f:
            f.write(data)
    
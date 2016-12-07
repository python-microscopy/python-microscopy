from PYME.IO.FileUtils import nameUtils
import os
import cStringIO
from contextlib import contextmanager
import tempfile

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
    elif filename.startswith('pyme-cluster') or filename.startswith('PYME-CLUSTER'):
        import clusterIO

        clusterfilter = filename.split('://')[1].split('/')[0]
        sequenceName = filename.split('://%s/' % clusterfilter)[1]

        localpath = clusterIO.get_local_path(sequenceName, clusterfilter)
        if localpath:
            yield localpath
        else:
            ext = os.path.splitext(sequenceName)[-1]

            with tempfile.NamedTemporaryFile(mode='w+b', suffix=ext) as outf:
                s = clusterIO.getFile(sequenceName, clusterfilter)
                outf.write(s)
                outf.flush()

                yield outf.name

    else:
        raise IOError('Path "%s" could not be found' % url)


def openFile(filename, mode='rb'):
    filename = nameUtils.getFullExistingFilename(filename)

    if os.path.exists(filename):
        return open(filename, mode)

    elif filename.startswith('pyme-cluster') or filename.startswith('PYME-CLUSTER'):
        import clusterIO

        clusterfilter = filename.split('://')[1].split('/')[0]
        sequenceName = filename.split('://%s/' % clusterfilter)[1]

        s = clusterIO.getFile(sequenceName, clusterfilter)
        return cStringIO.StringIO(s)
    else:
        raise IOError('File does not exist or URI not understood')

def read(filename):
    filename = nameUtils.getFullExistingFilename(filename)

    if os.path.exists(filename):
        with open(filename) as f:
            s = f.read()
        return s

    elif filename.startswith('pyme-cluster') or filename.startswith('PYME-CLUSTER'):
        import clusterIO

        clusterfilter = filename.split('://')[1].split('/')[0]
        sequenceName = filename.split('://%s/' % clusterfilter)[1]

        s = clusterIO.getFile(sequenceName, clusterfilter)
        return s
    else:
        raise IOError('File does not exist or URI not understood')
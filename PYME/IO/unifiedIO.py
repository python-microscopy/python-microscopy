from PYME.IO.FileUtils import nameUtils
import os
import cStringIO

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
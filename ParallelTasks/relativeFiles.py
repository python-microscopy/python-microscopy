import os
import string

def translateSeparators(filename):
    return string.translate(filename, string.maketrans('\\/', os.sep + os.sep))

def getFullFilename(relFilename):
    ''' returns a fully resolved filename given a filename relative to 
    the environment variable PYMEDATADIR. If environment variable not defined,
    assumes path is absolute.'''
    relFilename = translateSeparators(relFilename)

    if 'PYMEDATADIR' in os.environ.keys():
        return os.path.join(os.environ['PYMEDATADIR'], relFilename)
    else:
        return relFilename


def getRelFilename(filename):
    '''returns the tail of filename'''
    filename = translateSeparators(filename)
    
    #first make sure we have an absolute path
    filename = os.path.expanduser(filename)
    if not os.path.isabs(filename):
        filename= os.path.abspath(filename)

    if 'PYMEDATADIR' in os.environ.keys():
        dataDir = os.environ['PYMEDATADIR']
        if not dataDir[-1] in [os.sep, os.altsep]:
            dataDir = dataDir + os.sep

        if filename.startswith(dataDir): #if we've selected something which isn't under our data directory we're going to have to stick with an absolute path
            return filename[len(dataDir):]

    return filename


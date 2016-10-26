#!/usr/bin/python

##################
# nameUtils.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################
"""A selection of tools for automatically generating paths to either save or find
data"""

import os
import re
import datetime
import sys
import string

seps = re.compile('[\\\\/*]')

def getUsername():
    """
    Returns the user name in a platform independant way
    """
    if sys.platform == 'win32':
        import win32api
        return '_'.join(win32api.GetUserName().split(' '))
    if sys.platform.startswith('darwin'):
        return os.environ['USER']
    else: #linux
        #return os.getlogin() #broken when not runing from command line
        return os.environ.get('USER', 'nobody')


dtn = datetime.datetime.now()

homedir = os.path.expanduser('~') #unix & possibly others ...
if 'USERPROFILE' in os.environ.keys(): #windows
    homedir = os.environ['USERPROFILE']

datadir = os.path.join(homedir, 'PYMEData')
if 'PYMEDATADIR' in os.environ.keys() and os.access(os.environ['PYMEDATADIR'], os.W_OK):
    datadir = os.environ['PYMEDATADIR']

dirSuffix=''
if 'PYMEDIRSUFFIX' in os.environ.keys():
    dirSuffix = '_' + os.environ['PYMEDIRSUFFIX']

        

dateDict = {'username' : getUsername(), 'day' : dtn.day, 'month' : dtn.month, 'year':dtn.year, 'sep' : os.sep, 'dataDir' : datadir, 'homeDir': homedir, 'dirSuffix': dirSuffix}


#\\ / and * will be replaced with os dependant separator
datadirPattern = '%(dataDir)s/%(username)s/%(year)d_%(month)d_%(day)d%(dirSuffix)s'
clusterDirPattern = '%(username)s/%(year)d_%(month)d_%(day)d%(dirSuffix)s'
filePattern = '%(day)d_%(month)d_series'

#resultsdirPattern = '%(homeDir)s/analysis/%(dday)d-%(dmonth)d-%(dyear)d'
#resultsdirPatternShort = '%(homeDir)s/analysis/'

resultsdirPattern = '%(dataDir)s/%(username)s/analysis/%(dday)d-%(dmonth)d-%(dyear)d'
resultsdirPatternShort = '%(dataDir)s/%(username)s/analysis/'

calibrationdirPattern = '%(dataDir)s/CALIBRATION/%(serialNum)s/'

def getCalibrationDir(serialNum, create=True):
    """Returns the default directory where we would expect to find calibration
    data - e.g. sCMOS calibration maps"""
    p =  os.path.join(*seps.split(calibrationdirPattern)) % {'dataDir':datadir, 'serialNum':serialNum}
    if create and not os.path.exists(p): #create the necessary directories
        os.makedirs(p)

    return os.path.normpath(p)

def genHDFDataFilepath(create=True):
    """Generate a default path for saving HDF formatted raw data on the local
    hard drive"""
    p =  os.path.join(*seps.split(datadirPattern)) % dateDict
    if create and not os.path.exists(p): #create the necessary directories
        os.makedirs(p)

    return os.path.normpath(p)
    
def genClusterDataFilepath():
    """Generates a default path for saving raw data on the cluster"""
    return clusterDirPattern % dateDict

def genResultFileName(dataFileName, create=True):
    """Generates a filename for saving fit results based on the original image
    filename"""
    fn, ext = os.path.splitext(dataFileName) #remove extension
    fn = fn.replace(':', '/')
    #print os.path.join(*seps.split(resultsdirPatternShort)) % dateDict
    p = os.path.join(*(seps.split(resultsdirPatternShort) + seps.split(fn)[-2:])) %dateDict

    if create and not os.path.exists(os.path.split(p)[0]): #create the necessary directories
        os.makedirs(os.path.split(p)[0])

    return p + '.h5r'

def genClusterResultFileName(dataFileName, create=True):
    """Generates a filename for saving fit results based on the original image
    filename"""
    fn, ext = os.path.splitext(dataFileName) #remove extension

    clusterfilter = fn.split('://')[1].split('/')[0]
    rel_name = fn.split('://%s/' % clusterfilter)[1]

    #fn = fn.replace(':', '/')
    #print os.path.join(*seps.split(resultsdirPatternShort)) % dateDict
    #p = os.path.join(*(seps.split(resultsdirPatternShort) + seps.split(fn)[-2:])) %dateDict

    #if create and not os.path.exists(os.path.split(p)[0]): #create the necessary directories
    #    os.makedirs(os.path.split(p)[0])

    return rel_name + '.h5r'

def genResultDirectoryPath():
    """Returns the default destination for saving fit reults"""
    
    #print os.path.join(*seps.split(resultsdirPatternShort)) % dateDict
    p = os.path.join(*(seps.split(resultsdirPatternShort) )) %dateDict

    return p 


def genShiftFieldDirectoryPath():
    """Returns the default directory for shiftmaps"""
    return os.path.join(datadir, 'shiftmaps')

def baseconvert(number,todigits):
    """Converts a number to an arbtrary base.
    
    Parameters
    ----------
    number : int
        The number to convert
    todigits : iterable or string
        The digits of the base e.g. '0123456' (base 7) 
        or 'ABCDEFGHIJK' (non-numeric base 11)
    """
    x = number

    # create the result in base 'len(todigits)'
    res=""

    if x == 0:
        res=todigits[0]
    
    while x>0:
        digit = x % len(todigits)
        res = todigits[digit] + res
        x /= len(todigits)

    return res
    
def numToAlpha(num):
    """Convert a number to an alphabetic code"""
    return baseconvert(num, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
########################
#Previous relativeFiles
#from PYME.IO.FileUtils import nameUtils 

seps2 = re.compile('[\\\\/]')

def translateSeparators(filename):
    """Convert a filename which might use mixed separators / slashes to os
    native form.
    """
    
    #return string.translate(filename, string.maketrans('\\/', os.sep + os.sep))
    #print seps.split(filename)
    #fn = os.path.join(*seps.split(filename))
    #if filename[0] == '/': #replace leading /, if present
    #    fn = '/' + fn
    fn = string.join(seps2.split(filename), os.sep)

    return fn

def getFullFilename(relFilename):
    """ returns a fully resolved filename given a filename relative to
    the environment variable PYMEDATADIR. If environment variable not defined,
    assumes path is absolute."""
    relFilename = translateSeparators(relFilename)

    #if 'PYMEDATADIR' in os.environ.keys():
    #    return os.path.join(os.environ['PYMEDATADIR'], relFilename)
    #else:
    #    return relFilename
    return os.path.join(datadir, relFilename)

def getFullExistingFilename(relFilename):
    """ returns a fully resolved filename given a filename relative to
    the environment variable PYMEDATADIR. If environment variable not defined,
    or the absolute path exists, assumes path is absolute."""

    if os.path.exists(relFilename) or relFilename.startswith('PYME-CLUSTER://') or relFilename.startswith('pyme-cluster://'):
        return relFilename
    else:
        if relFilename.startswith('d:\\') or relFilename.startswith('D:\\'):
            relFilename = relFilename[3:]

        return getFullFilename(relFilename)


def getRelFilename(filename):
    """returns the tail of filename - ie that portion which is underneath the
    PYMEDATADIR directory"""
    filename = translateSeparators(filename)
    #print filename
    
    #first make sure we have an absolute path
    filename = os.path.expanduser(filename)
    if not os.path.isabs(filename):
        filename= os.path.abspath(filename)

    #print filename
    #if 'PYMEDATADIR' in os.environ.keys():
    dataDir = datadir
    if not dataDir[-1] in [os.sep, os.altsep]:
        dataDir = dataDir + os.sep

    if filename.startswith(dataDir): #if we've selected something which isn't under our data directory we're going to have to stick with an absolute path
        return filename[len(dataDir):]

    return filename

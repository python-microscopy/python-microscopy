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

import os
import re
import datetime
import sys

seps = re.compile('[\\\\/*]')

def getUsername():
    '''
    Returns the user name in a platform dependant way
    '''
    if sys.platform == 'win32':
        import win32api
        return '_'.join(win32api.GetUserName().split(' '))
    if sys.platform.startswith('darwin'):
        return os.environ['USER']
    else: #linux
        #return os.getlogin() #broken when not runing from command line
        return os.environ['USER']


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
filePattern = '%(day)d_%(month)d_series'

#resultsdirPattern = '%(homeDir)s/analysis/%(dday)d-%(dmonth)d-%(dyear)d'
#resultsdirPatternShort = '%(homeDir)s/analysis/'

resultsdirPattern = '%(dataDir)s/%(username)s/analysis/%(dday)d-%(dmonth)d-%(dyear)d'
resultsdirPatternShort = '%(dataDir)s/%(username)s/analysis/'

def genHDFDataFilepath(create=True):
    p =  os.path.join(*seps.split(datadirPattern)) % dateDict
    if create and not os.path.exists(p): #create the necessary directories
        os.makedirs(p)

    return os.path.normpath(p)

def genResultFileName(dataFileName, create=True):
    fn, ext = os.path.splitext(dataFileName) #remove extension
    fn = fn.replace(':', '/')
    #print os.path.join(*seps.split(resultsdirPatternShort)) % dateDict
    p = os.path.join(*(seps.split(resultsdirPatternShort) + seps.split(fn)[-2:])) %dateDict

    if create and not os.path.exists(os.path.split(p)[0]): #create the necessary directories
        os.makedirs(os.path.split(p)[0])

    return p + '.h5r'

def genResultDirectoryPath():
    
    #print os.path.join(*seps.split(resultsdirPatternShort)) % dateDict
    p = os.path.join(*(seps.split(resultsdirPatternShort) )) %dateDict

    return p 


def genShiftFieldDirectoryPath():
    return os.path.join(datadir, 'shiftmaps')

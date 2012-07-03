#!/usr/bin/python

##################
# relativeFiles.py
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
import string
import re

seps = re.compile('[\\\\/]')

def translateSeparators(filename):
    #return string.translate(filename, string.maketrans('\\/', os.sep + os.sep))
    #print seps.split(filename)
    #fn = os.path.join(*seps.split(filename))
    #if filename[0] == '/': #replace leading /, if present
    #    fn = '/' + fn
    fn = string.join(seps.split(filename), os.sep)

    return fn

def getFullFilename(relFilename):
    ''' returns a fully resolved filename given a filename relative to 
    the environment variable PYMEDATADIR. If environment variable not defined,
    assumes path is absolute.'''
    relFilename = translateSeparators(relFilename)

    if 'PYMEDATADIR' in os.environ.keys():
        return os.path.join(os.environ['PYMEDATADIR'], relFilename)
    else:
        return relFilename

def getFullExistingFilename(relFilename):
    ''' returns a fully resolved filename given a filename relative to
    the environment variable PYMEDATADIR. If environment variable not defined,
    assumes path is absolute.'''

    if os.path.exists(relFilename):
        return relFilename
    else:
        if relFilename.startswith('d:\\') or relFilename.startswith('D:\\'):
            relFilename = relFilename[3:]

        return getFullFilename(relFilename)


def getRelFilename(filename):
    '''returns the tail of filename'''
    filename = translateSeparators(filename)
    #print filename
    
    #first make sure we have an absolute path
    filename = os.path.expanduser(filename)
    if not os.path.isabs(filename):
        filename= os.path.abspath(filename)

    #print filename
    if 'PYMEDATADIR' in os.environ.keys():
        dataDir = os.environ['PYMEDATADIR']
        if not dataDir[-1] in [os.sep, os.altsep]:
            dataDir = dataDir + os.sep

        if filename.startswith(dataDir): #if we've selected something which isn't under our data directory we're going to have to stick with an absolute path
            return filename[len(dataDir):]

    return filename


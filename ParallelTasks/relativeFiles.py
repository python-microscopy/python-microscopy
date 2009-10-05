#!/usr/bin/python

##################
# relativeFiles.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
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


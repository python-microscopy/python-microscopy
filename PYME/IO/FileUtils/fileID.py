import os.path
#!/usr/bin/python

##################
# fileID.py
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

import tables
import os
import numpy as np
from PYME.IO import MetaDataHandler

from PYME.misc.hash32 import hashString32

def genDataFileID(filename):
#    h5f = tables.openFile(filename)
#
#    ds = h5f.root.ImageData[0, :33, 0].ravel().astype('i')
#
#    h5f.close()
#    #print ds
#
#    ds = np.diff(np.diff(ds))
#
#    #print ds
#
#    ds = ds > 0
#    #print ds.shape
#    print ds.sum()
#
#    return ((2*ds)**np.arange(31)).sum()
    h5f = tables.open_file(filename)
    ds = h5f.root.ImageData[0, :, :]
    h5f.close()
    ret  =  genFrameID(ds)

    return ret

def genFrameID(frame):
    #h5f = tables.openFile(filename)

    ds = frame.ravel()[:33].astype('i')

    #h5f.close()
    #print ds.shape

    ds = np.diff(np.diff(ds))

    #print ds

    ds = ds > 0
    #print ds.shape
    #print ds.sum()

    return ((2*ds)**np.arange(31)).sum()


def genDataSourceID(datasource):
#    ds = datasource.getSlice(0)[:33, 0].ravel().astype('i')
#
#    ds = np.diff(np.diff(ds)) > 0
#    #print ds.shape
#
#    return ((2*ds)**np.arange(31)).sum()
    return genFrameID(datasource.getSlice(0))

def genResultsFileID(filename):
    h5f = tables.open_file(filename)

    ds = str(h5f.root.FitResults[0].data)

    h5f.close()

    return  hashString32(ds)

def genFileID(filename):
    """generate database ids for files. Where we know about the file type an ID
    is generated from the data which should be persistant over copies of the file,
    otherwise a hash of the filename is used.
    """

    if os.path.exists(filename):
        ext = os.path.splitext(filename)[1]
        try:
            if ext == '.h5':
                return genDataFileID(filename)
            elif ext == '.h5r':
                return genResultsFileID(filename)
        except:
            pass
    
    return hashString32(filename)


def genImageID(filename, guess=False):
    ext = os.path.splitext(filename)[1]
    #print ext

    try:
        if ext == '.h5':
            return genDataFileID(filename)
        elif ext == '.h5r':
            h5f = tables.open_file(filename)
            md = MetaDataHandler.HDFMDHandler(h5f)

            if 'Analysis.DataFileID' in md.getEntryNames():
                ret = md.getEntry('Analysis.DataFileID')
            elif guess:
                ret = guessH5RImageID(filename)
                #print ret
            else:
                ret = None
            #print guess, ret

            h5f.close()
            return ret
        else:
            return hashString32(filename)
    except:
        return hashString32(filename)

def genImageTime(filename):
    ext = os.path.splitext(filename)[1]
    #print ext

    try:
        if ext in ['.h5', '.h5r']:
            h5f = tables.open_file(filename)
            md = MetaDataHandler.HDFMDHandler(h5f)

            ret = md.getEntry('StartTime')
            #print guess, ret

            h5f.close()
            return ret
        else:
            return 0
    except:
        return 0

def guessH5RImageID(filename):
    #try and find the original data
    fns = filename.split(os.path.sep)
    cand = os.path.sep.join(fns[:-3]  + fns[-2:])[:-1]
    #print cand
    if os.path.exists(cand):
        #print 'Found Analysis'
        return genDataFileID(cand)
    else:
        return None

def guessUserID(filename):
    fns = filename.split(os.path.sep)

    ext = os.path.splitext(filename)[1]
    if ext == '.h5':
        return fns[-3]
    elif ext == '.h5r':
        return fns[-4]

def getFileMetadata(filename):
    ext = os.path.splitext(filename)[1]
    #print ext

    mdh = MetaDataHandler.NestedClassMDHandler()

    try:
        if ext in ['.h5', '.h5r']:
            h5f = tables.open_file(filename)
            md = MetaDataHandler.HDFMDHandler(h5f)

            mdh = MetaDataHandler.NestedClassMDHandler(md)
            #print guess, ret

            h5f.close()
    except:
        pass

    return mdh

def getImageTags(filename):
    ext = os.path.splitext(filename)[1]
    #print ext

    tags = []

    try:
        if ext in ['.h5', '.h5r']:
            h5f = tables.open_file(filename)
            if 'Events' in dir(h5f.root):
                events = h5f.root.Events[:]

                evKeyNames = set()
                for e in events:
                    evKeyNames.add(e['EventName'])

                if b'ProtocolFocus' in evKeyNames:
                    tags.append('Z-Stack')

            md = MetaDataHandler.HDFMDHandler(h5f)

            if 'Protocol.Filename' in md.getEntryNames():
                tags.append('Protocol_%s' % md.getEntry('Protocol.Filename'))
            

            h5f.close()
    except:
        pass

    return tags


def getFileTags(filename, md):
    ext = os.path.splitext(filename)[1]
    #print ext

    tags = []

    if ext in ['.h5r']:
        if 'Analysis.FitModule' in md.getEntryNames():
            tags.append('FitModule_%s' % md.getEntry('Analysis.FitModule'))



    return tags





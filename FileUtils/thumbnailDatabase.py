#!/usr/bin/python

##################
# thumbnailDatabase.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import sqlite3
import cPickle as pickle
import os
import zlib
from numpy import ndarray

THUMBSIZE=200
thumbnailers = {'.h5':'h5-thumbnailer', '.h5r':'h5r-thumbnailer', '.kdf':'kdf-thumbnailer'}

#teach sqlite about numpy arrays
def adapt_numarray(array):
    return sqlite3.Binary(zlib.compress(array.dumps()))

def convert_numarray(s):
    return pickle.loads(zlib.decompress(s))

sqlite3.register_adapter(ndarray, adapt_numarray)
sqlite3.register_converter("ndarray", convert_numarray)


thumbDB = sqlite3.connect('/home/david/PYME/PYME/SampleDB/PYMEThumbnails.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
thumbDB.isolation_level = None

#see what tables we've got defined
tableNames = [a[0] for a in thumbDB.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]

#if we haven't already got a thumbnail database, create one
if not 'Thumbnails' in tableNames:
    thumbDB.execute("CREATE TABLE Thumbnails (filename text, thumbnail ndarray)")




def getThumbnail(filename):
    #strip extra leading / (inserted for some reason by test server)
    if filename.startswith('//'):
        filename = filename[1:]

    ret = thumbDB.execute("SELECT thumbnail FROM Thumbnails WHERE filename=?", (filename,)).fetchone()

    if ret == None:
        #cludge around old entries with extra /
        ret = thumbDB.execute("SELECT thumbnail FROM Thumbnails WHERE filename=?", ('/' + filename,)).fetchone()


    if ret == None:
        ext = os.path.splitext(filename)[-1]

        if ext in thumbnailers.keys():
            try:
                #thumbMod = __import__(thumbnailers[ext])
                thumbMod = __import__('PYME.FileUtils.' + thumbnailers[ext], fromlist=['PYME', 'FileUtils'])
                ret = thumbMod.generateThumbnail(filename, THUMBSIZE)

                thumbDB.execute("INSERT INTO Thumbnails VALUES (?, ?)", (filename, ret))
                thumbDB.commit()
            except:
                pass
    else:
        ret = ret[0]

    return ret

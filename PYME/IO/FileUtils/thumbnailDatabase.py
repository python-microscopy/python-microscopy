#!/usr/bin/python

##################
# thumbnailDatabase.py
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

import sqlite3
try:
    import cPickle as pickle
except ImportError:
    #py3
    import pickle
    
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

dbOpen = False
thumbDB = None

def openDB():
    global dbOpen, thumbDB
    
    if not dbOpen:
        dbOpen = True
        thumbDB = sqlite3.connect('/srv/PYME/PYMEThumbnails.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        thumbDB.isolation_level = None
        
        #see what tables we've got defined
        tableNames = [a[0] for a in thumbDB.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
        
        #if we haven't already got a thumbnail database, create one
        if not 'Thumbnails' in tableNames:
            thumbDB.execute("CREATE TABLE Thumbnails (filename text, thumbnail ndarray)")




def getThumbnail(filename):
    openDB()
    #strip extra leading / (inserted for some reason by test server)
    if filename.startswith('//'):
        filename = filename[1:]

    #if missing leading /, add
    if not filename.startswith('/'):
        filename = '/' + filename

    ret = thumbDB.execute("SELECT thumbnail FROM Thumbnails WHERE filename=?", (filename,)).fetchone()

    if ret is None:
        #cludge around old entries with extra /
        ret = thumbDB.execute("SELECT thumbnail FROM Thumbnails WHERE filename=?", ('/' + filename,)).fetchone()


    if ret is None:
        ext = os.path.splitext(filename)[-1]

        if ext in thumbnailers.keys():
            #print ext
            #try:
            #thumbMod = __import__(thumbnailers[ext])
            thumbMod = __import__('PYME.IO.FileUtils.' + thumbnailers[ext], fromlist=['PYME', 'io', 'FileUtils'])

            ret = thumbMod.generateThumbnail(filename, THUMBSIZE)

            thumbDB.execute("INSERT INTO Thumbnails VALUES (?, ?)", (filename, ret))
            thumbDB.commit()
            #except:
            #    pass
    else:
        ret = ret[0]

    return ret

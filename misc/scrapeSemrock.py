#!/usr/bin/python
##################
# scrapeSemrock.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import sqlite3
import urllib2
import numpy as np
from numpy import ndarray
import re
import zlib
import pickle

#teach sqlite about numpy arrays
def adapt_numarray(array):
    return sqlite3.Binary(zlib.compress(array.dumps()))

def convert_numarray(s):
    return pickle.loads(zlib.decompress(s))

sqlite3.register_adapter(ndarray, adapt_numarray)
sqlite3.register_converter("ndarray", convert_numarray)

#we are going to buffer our requests in a local database - to make it both faster and
#to avoid loading the semrock server
dbconn = sqlite3.connect('semrockFilters', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

#see what tables we've got defined
tableNames = [a[0] for a in dbconn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]

#if we haven't already got a spectra table, create one
if not 'Spectra' in tableNames:
    dbconn.execute("CREATE TABLE Spectra (name text, spectrum ndarray)")


def _scrapeFilterNames():
    webcon = urllib2.urlopen('http://www.semrock.com/filters.aspx?page=1&so=0&recs=1000')
    d = webcon.read()
    webcon.close()

    return list(set([f[0] for f in re.findall(r'FilterDetails.aspx\?id=(\w*-(\w|/)*)(\"|-25\"|-12.5\"|-25x36\"|-13x15x0.5\")', d)]))


def getFilterNames(refresh = False):
    c = dbconn.cursor()

    foundInBuffer = False

    if not refresh:
        try:
            names = [n[0] for n in c.execute('select name from FilterNames')]
            #c.close()
            foundInBuffer = True
        except:
            pass

    if not foundInBuffer:# we don't have any buffered names, or we want to refresh them
        names = _scrapeFilterNames()

        #get rid of any previous entries if there
        try:
            c2 = c.execute('drop table FilterNames')
            dbconn.commit()
            #c2.close()
        except:
            pass
        
        c.execute('create table FilterNames (name text)')
        
        c.executemany('insert into FilterNames values (?)', [(nm, ) for nm in names])

        #c2.close()
        dbconn.commit()
        
    c.close()
    
    return(names)

def _scrapeSpectrum(filterName):
    expName = filterName.replace('/', '_')

    r2 = urllib2.urlopen('http://www.semrock.com/_ProductData/Spectra/%s_spectrum.txt' % expName)
    a = np.fromregex(r2, r'(\d[0-9eE\-\+\.]*)\t(\d[0-9eE\-\+\.]*)', [('lambda', 'f4'), ('T', 'f4')])

    return a

def getFilterSpectrum(filterName):
    c = dbconn.cursor()

    r = c.execute('select spectrum from Spectra where name=?', (filterName,)).fetchone()

    if r:
        c.close()
        return r[0]
    else:
        spect = _scrapeSpectrum(filterName)

        c.execute('insert into Spectra values (?, ?)', (filterName, spect))
        dbconn.commit()

        c.close()
        return spect









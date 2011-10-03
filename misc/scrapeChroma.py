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
import httplib
import numpy as np
from numpy import ndarray
import re
import zlib
import pickle
import os

#teach sqlite about numpy arrays
def adapt_numarray(array):
    return sqlite3.Binary(zlib.compress(array.dumps()))

def convert_numarray(s):
    return pickle.loads(zlib.decompress(s))

sqlite3.register_adapter(ndarray, adapt_numarray)
sqlite3.register_converter("ndarray", convert_numarray)

#we are going to buffer our requests in a local database - to make it both faster and
#to avoid loading the semrock server
dbconn = sqlite3.connect(os.path.join(os.path.split(__file__)[0],'chromaFilters'), detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

#see what tables we've got defined
tableNames = [a[0] for a in dbconn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]

#if we haven't already got a spectra table, create one
if not 'Spectra' in tableNames:
    dbconn.execute("CREATE TABLE Spectra (name text, spectrum ndarray)")

import threading
import Queue
import sys

def _scrapeFilterNames(numFilters=1000, nThreads=20): #get rid of ugly hard-coded filter number
    taskQueue = Queue.Queue()
    for i in range(numFilters):
        taskQueue.put(i)
        
    resultsQueue = Queue.Queue()

    def doOpenUrl():
        conn = httplib.HTTPConnection('www.chroma.com')
        while 1:
            try:
                id = taskQueue.get(block=False)
            except Queue.Empty:
                conn.close()
                sys.exit(1)

            #print id
            try:
                #webcon = urllib2.urlopen('http://www.chroma.com/products/part/%d/' % id)
                #d = webcon.read()
                #webcon.close()
                conn.request('GET', '/products/part/%d/'%id)
                d = conn.getresponse().read()

                #print id

                partName = re.search(r'Part Number:</td>\s+<\w+\s\w+="value">((\w|\d|/)+)<', d).groups()[0]
                specNum = int(re.search(r'products/spectra/(\d+)/ascii/',d).groups()[0])

                resultsQueue.put((partName, specNum))
            except AttributeError:
                pass

    threadList = []
    for i in range(nThreads):
        t = threading.Thread(target=doOpenUrl)
        t.start()
        threadList.append(t)



    for t in threadList:
        t.join()

    filts = []
    while not resultsQueue.empty():
        filts.append(resultsQueue.get())


    return filts


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
        namesandnums = _scrapeFilterNames()

        #get rid of any previous entries if there
        try:
            c2 = c.execute('drop table FilterNames')
            dbconn.commit()
            #c2.close()
        except:
            pass
        
        c.execute('create table FilterNames (name text, num int)')
        
        c.executemany('insert into FilterNames values (?, ?)', [nm for nm in namesandnums])

        #c2.close()
        dbconn.commit()

        names = [n[0] for n in c.execute('select name from FilterNames')]
        
    c.close()
    
    return(names)

def _scrapeSpectrum(specNum):
    #expName = filterName.replace('/', '_')

    r2 = urllib2.urlopen('http://www.chroma.com/products/spectra/%d/ascii' % specNum)
    a = np.fromregex(r2, r'\n(\d[0-9eE\-\+\.]*)\t(\d[0-9eE\-\+\.]*)', [('lambda', 'f4'), ('T', 'f4')])

    return a


def getFilterSpectrum(filterName):
    c = dbconn.cursor()

    r = c.execute('select spectrum from Spectra where name=?', (filterName,)).fetchone()

    if r:
        c.close()
        return r[0]
    else:
        prodID = c.execute('select num from FilterNames where name=?', (filterName,)).fetchone()[0]
        spect = _scrapeSpectrum(prodID)

        c.execute('insert into Spectra values (?, ?)', (filterName, spect))
        dbconn.commit()

        c.close()
        return spect






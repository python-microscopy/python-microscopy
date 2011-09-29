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
import json
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
dbconn = sqlite3.connect(os.path.join(os.path.split(__file__)[0],'omegaFilters'), detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

#see what tables we've got defined
tableNames = [a[0] for a in dbconn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]

#if we haven't already got a spectra table, create one
if not 'Spectra' in tableNames:
    dbconn.execute("CREATE TABLE Spectra (name text, spectrum ndarray)")

if not 'DyeSpectra' in tableNames:
    dbconn.execute("CREATE TABLE DyeSpectra (name text, emission ndarray, excitation ndarray)")


def _scrapeNames():
    webcon = urllib2.urlopen('http://www.omegafilters.com/Products/Curvomatic')
    d = unicode(webcon.read(), 'utf8')
    webcon.close()

    filterNames = [(f[1], f[2].split()[0]) for f in re.findall(r'<option productType="(\w*)"  set="false"  canUseFluorophores="\w+" productId="(\d+)" value="\d+" overstock="\d">((\s|\w|\-)+)<', d)]

    dyeNames = [(f[0], f[1].replace(u'\u2122', '').replace(u'\u00AE', '')) for f in re.findall(ur'<option productId="(\d+)" value="\d+">((\s|\w|\-|\u2122|\u00AE)+)<', d)]

    return filterNames, dyeNames

def _updateNamesDatabase():
    c = dbconn.cursor()
    
    nameAndID, dyeNameAndID = _scrapeNames()

    #get rid of any previous entries if there
    try:
        c.execute('drop table FilterNames')
        c.execute('drop table DyeNames')
        dbconn.commit()
        #c2.close()
    except:
        pass

    c.execute('create table FilterNames (prodID int, name text)')
    c.execute('create table DyeNames (prodID int, name text)')

    c.executemany('insert into FilterNames values (?, ?)', [nm for nm in nameAndID])
    c.executemany('insert into DyeNames values (?, ?)', [nm for nm in dyeNameAndID])
    
    c.close()
    dbconn.commit()

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
        _updateNamesDatabase()

        names = [n[0] for n in c.execute('select name from FilterNames')]

        #c2.close()
        #dbconn.commit()
        
    c.close()
    
    return(names)

def getDyeNames(refresh = False):
    c = dbconn.cursor()

    foundInBuffer = False

    if not refresh:
        try:
            names = [n[0] for n in c.execute('select name from DyeNames')]
            #c.close()
            foundInBuffer = True
        except:
            pass

    if not foundInBuffer:# we don't have any buffered names, or we want to refresh them
        _updateNamesDatabase()

        names = [n[0] for n in c.execute('select name from DyeNames')]

        #c2.close()
        #dbconn.commit()

    c.close()

    return(names)

def _scrapeFilterSpectrum(prodID):
    #expName = filterName.replace('/', '_')

    r2 = urllib2.urlopen('http://www.omegafilters.com/Modules/curvomatic/comproxy.cfc?method=AddFilter&productID=%d' % prodID)

    d = json.load(r2)
    p = d['TRANSMISSIONDATAPOINTS']

    return np.array(p, 'f4').view([('lambda', 'f4'), ('T', 'f4')]).squeeze()

def _scrapeDyeSpectrum(prodID):
    #expName = filterName.replace('/', '_')

    r2 = urllib2.urlopen('http://www.omegafilters.com/Modules/curvomatic/comproxy.cfc?method=AddFluorophore&productID=%d' % prodID)

    d = json.load(r2)
    emissionSpectrum = np.array(d['DATAPOINTS1'], 'f4').view([('lambda', 'f4'), ('T', 'f4')]).squeeze()

    if 'DATAPOINTS2' in d.keys():
        excitationSpectrum = np.array(d['DATAPOINTS2'], 'f4').view([('lambda', 'f4'), ('T', 'f4')]).squeeze()
    else:
        excitationSpectrum = np.empty(dtype=[('lambda', 'f4'), ('T', 'f4')])

    return emissionSpectrum, excitationSpectrum

def getFilterSpectrum(filterName):
    c = dbconn.cursor()

    r = c.execute('select spectrum from Spectra where name=?', (filterName,)).fetchone()

    if r:
        c.close()
        return r[0]
    else:
        prodID = c.execute('select prodID from FilterNames where name=?', (filterName,)).fetchone()[0]
        spect = _scrapeFilterSpectrum(prodID)

        c.execute('insert into Spectra values (?, ?)', (filterName, spect))
        dbconn.commit()

        c.close()
        return spect

def getDyeSpectrum(dyeName):
    c = dbconn.cursor()

    r = c.execute('select emission, excitation from DyeSpectra where name=?', (dyeName,)).fetchone()

    if r:
        c.close()
        return r
    else:
        prodID = c.execute('select prodID from DyeNames where name=?', (dyeName,)).fetchone()[0]
        emSpect, exSpect = _scrapeDyeSpectrum(prodID)

        c.execute('insert into DyeSpectra values (?, ?, ?)', (dyeName, emSpect, exSpect))
        dbconn.commit()

        c.close()
        return emSpect, exSpect









#!/usr/bin/python

##################
# inpFilt.py
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

''' import filters for localisation microscopy results. These masquerade as 
dictionaries which can be looked up to yield the desired data. The visualisation
routines expect at least 'x' and 'y' to be defined as keys, and may also 
understand additional values, e.g. 'error_x' 
'''
import types
import numpy as np

from numpy import * #to allow the use of sin cos etc in mappings
from PYME.Analysis.piecewise import * #allow piecewise linear mappings

import tables

class inputFilter(object):
    def toDataFrame(self, keys=None):
        import pandas as pd
        if keys == None:
            keys = self.keys()
        
        d = {k: self.__getitem__(k) for k in keys}
        
        return pd.DataFrame(d)
    

class randomSource(inputFilter):
    _name = "Random Source"
    def __init__(self, xmax, ymax, nsamps):
        '''Uniform random source, for testing and as an example'''
        self.x = xmax*np.random.rand(nsamps)
        self.y = ymax*np.random.rand(nsamps)

        self._keys = ['x', 'y']

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if not key in self._keys:
            raise RuntimeError('Key not defined')
        
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y

    def getInfo(self):
        return 'Random Data Source\n\n %d points' % len(self.x)


def unNestNames(nameList, parent=''):
    unList = []
    for n in nameList:
        if n.__class__ == str:
            unList.append(parent + n)
        else:
            unList += unNestNames(n[1], parent + n[0] + '_')
    return unList

def unNestDtype(descr, parent=''):
    unList = []
    for n in descr:
        #print n, n.__class__, len(n)
        if n.__class__ == tuple and len(n) == 2 and n[1].__class__ == str:
            unList.append(parent + n[0])
        else:
            unList += unNestDtype(n[1], parent + n[0] + '_')
    return unList

class fitResultsSource(inputFilter):
    _name = "recarrayfi Source"
    def __init__(self, fitResults):
        self.setResults(fitResults)
        
    def setResults(self, fitResults):
        self.fitResults = fitResults

        #sort by time
        self.fitResults.sort(order='tIndex')

        #allow access using unnested original names
        self._keys = unNestDtype(self.fitResults.dtype.descr)
        #or shorter aliases
        self.transkeys = {'A' : 'fitResults_A', 'x' : 'fitResults_x0',
                          'y' : 'fitResults_y0', 'sig' : 'fitResults_sigma',
                          'error_x' : 'fitError_x0', 'error_y' : 'fitError_y0','t':'tIndex'}

        for k in self.transkeys.keys():
            if not self.transkeys[k] in self._keys:
                self.transkeys.pop(k)


    def keys(self):
        return self._keys + self.transkeys.keys()

    def __getitem__(self, key):
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise RuntimeError('Key not found')

        k = key.split('_')

        if len(k) == 1:
            return self.fitResults[k[0]].astype('f')
        elif len(k) == 2:
            return self.fitResults[k[0]][k[1]].astype('f')
        elif len(k) == 3:
            return self.fitResults[k[0]][k[1]][k[2]].astype('f')
        else:
            raise RuntimeError("Don't know about deeper nesting yet")


    def close(self):
        self.h5f.close()

    def getInfo(self):
        return 'PYME h5r Data Source\n\n %d points' % self.fitResults.shape[0]


class h5rSource(inputFilter):
    _name = "h5r Data Source"
    def __init__(self, h5fFile):
        ''' Data source for use with h5r files as saved by the PYME analysis 
        component. Takes either an open h5r file or a string filename to be
        opened.'''

        if type(h5fFile) == tables.file.File:
            self.h5f = h5fFile
        else:
            self.h5f = tables.openFile(h5fFile)
        
        if not 'FitResults' in dir(self.h5f.root):
            raise RuntimeError('Was expecting to find a "FitResults" table')

        self.fitResults = self.h5f.root.FitResults[:]

        #sort by time
        self.fitResults.sort(order='tIndex')

        #allow access using unnested original names
        self._keys = unNestNames(self.h5f.root.FitResults.description._v_nestedNames)
        #or shorter aliases
        self.transkeys = {'A' : 'fitResults_A', 'x' : 'fitResults_x0',
                          'y' : 'fitResults_y0', 'sig' : 'fitResults_sigma', 
                          'error_x' : 'fitError_x0', 'error_y' : 'fitError_y0', 't':'tIndex'}

        for k in self.transkeys.keys():
            if not self.transkeys[k] in self._keys:
                self.transkeys.pop(k)


    def keys(self):
        return self._keys + self.transkeys.keys()

    def __getitem__(self, key):
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise RuntimeError('Key not found')

        k = key.split('_')
        
        if len(k) == 1:
            return self.fitResults[k[0]].astype('f')
        elif len(k) == 2:
            return self.fitResults[k[0]][k[1]].astype('f')
        elif len(k) == 3:
            return self.fitResults[k[0]][k[1]][k[2]].astype('f')
        else:
            raise RuntimeError("Don't know about deeper nesting yet")
        

    def close(self):
        self.h5f.close()

    def getInfo(self):
        return 'PYME h5r Data Source\n\n %d points' % self.h5f.root.FitResults.shape[0]


class h5rDSource(inputFilter):
    _name = "h5r Drift Source"
    def __init__(self, h5fFile):
        ''' Data source for use with h5r files as saved by the PYME analysis 
        component'''
        
        if type(h5fFile) == tables.file.File:
            self.h5f = h5fFile
        else:
            self.h5f = tables.openFile(h5fFile)

        if not 'DriftResults' in dir(self.h5f.root):
            raise RuntimeError('Was expecting to find a "DriftResults" table')

        #allow access using unnested original names
        self._keys = unNestNames(self.h5f.root.DriftResults.description._v_nestedNames)
        #or shorter aliases
        self.transkeys = {'A' : 'fitResults_A', 'x' : 'fitResults_x0',
                          'y' : 'fitResults_y0', 'sig' : 'fitResults_sigma', 
                          'error_x' : 'fitError_x0', 'error_y' : 'fitError_y0', 't':'tIndex'}


    def keys(self):
        return self._keys + self.transkeys.keys()

    def __getitem__(self, key):
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise RuntimeError('Key not found')

        k = key.split('_')
        
        if len(k) == 1:
            return self.h5f.root.DriftResults[:][k[0]].astype('f')
        elif len(k) == 2:
            return self.h5f.root.DriftResults[:][k[0]][k[1]].astype('f')
        elif len(k) == 3:
            return self.h5f.root.DriftResults[:][k[0]][k[1]][k[2]].astype('f')
        else:
            raise RuntimeError("Don't know about deeper nesting yet")
        

    def close(self):
        self.h5f.close()

    def getInfo(self):
        return 'PYME h5r Drift Data Source\n\n %d points' % self.h5f.root.DriftResults.shape[0]

class textfileSource(inputFilter):
    _name = "Text File Source"
    def __init__(self, filename, columnnames, delimiter=None, skiprows=0):
        ''' Input filter for use with delimited text data. Defaults
        to whitespace delimiter. need to provide a list of variable names
        in the order that they appear in the file. Using 'x', 'y' and 'error_x'
        for the position data and it's error should ensure that this functions
        with the visualisation backends'''

        self.res = np.loadtxt(filename, dtype={'names' : columnnames, 
                                               'formats' :  ['f4' for i in range(len(columnnames))]}, delimiter = delimiter, skiprows=skiprows)
        
        self._keys = list(columnnames)
       


    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if not key in self._keys:
            raise RuntimeError('Key not found')

       
        return self.res[key]

    
    def getInfo(self):
        return 'Text Data Source\n\n %d points' % len(self.res['x'])

class matfileSource(inputFilter):
    _name = "Matlab Source"
    def __init__(self, filename, columnnames, varName='Orte'):
        ''' Input filter for use with matlab data. Need to provide a variable name
        and a list of column names
        in the order that they appear in the file. Using 'x', 'y' and 'error_x'
        for the position data and it's error should ensure that this functions
        with the visualisation backends'''

        import scipy.io

        self.res = scipy.io.loadmat(filename)[varName].astype('f4')
        
        self.res = np.rec.fromarrays(self.res.T, dtype={'names' : columnnames,  'formats' :  ['f4' for i in range(len(columnnames))]})

        self._keys = list(columnnames)



    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if not key in self._keys:
            raise RuntimeError('Key not found')


        return self.res[key]


    def getInfo(self):
        return 'Text Data Source\n\n %d points' % len(self.res['x'])
        

class resultsFilter(inputFilter):
    _name = "Results Filter"
    def __init__(self, resultsSource, **kwargs):
        '''Class to permit filtering of fit results - masquarades 
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with 
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source.'''

        self.resultsSource = resultsSource

        #by default select everything
        self.Index = np.ones(self.resultsSource[resultsSource.keys()[0]].shape) >  0.5

        for k in kwargs.keys():
            if not k in self.resultsSource.keys():
                raise RuntimeError('Requested key not present: ' + k)

            range = kwargs[k]
            if not len(range) == 2:
                raise RuntimeError('Expected an iterable of length 2')

            self.Index *= (self.resultsSource[k] > range[0])*(self.resultsSource[k] < range[1])
                

    def __getitem__(self, key):
        return self.resultsSource[key][self.Index]

    def keys(self):
        return self.resultsSource.keys()

class cachingResultsFilter(inputFilter):
    _name = "Caching Results Filter"
    def __init__(self, resultsSource, **kwargs):
        '''Class to permit filtering of fit results - masquarades
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source.'''

        self.resultsSource = resultsSource
        self.cache = {}

        #by default select everything
        self.Index = np.ones(self.resultsSource[resultsSource.keys()[0]].shape) >  0.5

        for k in kwargs.keys():
            if not k in self.resultsSource.keys():
                raise RuntimeError('Requested key not present: ' + k)

            range = kwargs[k]
            if not len(range) == 2:
                raise RuntimeError('Expected an iterable of length 2')

            self.Index *= (self.resultsSource[k] > range[0])*(self.resultsSource[k] < range[1])


    def __getitem__(self, key):
        if key in self.cache.keys():
            return self.cache[key]
        else:
            res = self.resultsSource[key][self.Index]
            self.cache[key] = res
            return res

    def keys(self):
        return self.resultsSource.keys()


class mappingFilter(inputFilter):
    _name = "Mapping Filter"
    def __init__(self, resultsSource, **kwargs):
        '''Class to permit transformations (e.g. drift correction) of fit results
        - masquarades as a dictionary. Takes mappings as keyword arguments, eg:
        f = resultsFliter(source, xp='x + a*tIndex', yp=compile('y + b*tIndex', '/tmp/test1', 'eval'), a=1, b=2)
        will return an object that behaves like source, but has additional members
        xp and yp.

        the mappings should either be code objects, strings (which will be compiled into code objects),
        or something else (which will be turned into a local variable - eg constants in above example)

        '''

        self.resultsSource = resultsSource

        self.mappings = {}

        for k in kwargs.keys():
            v = kwargs[k]
            self.setMapping(k,v)


    def __getitem__(self, key):
        if key in self.mappings.keys():
            return self.getMappedResults(key)
        else:
            return self.resultsSource[key]

    def keys(self):
        return list(self.resultsSource.keys()) + self.mappings.keys()

    def setMapping(self, key, mapping):
        if type(mapping) == types.CodeType:
            self.mappings[key] = mapping
        elif type(mapping) == types.StringType:
            self.mappings[key] = compile(mapping, '/tmp/test1', 'eval')
        else:
            self.__dict__[key] = mapping

    def getMappedResults(self, key):
        map = self.mappings[key]

        #get all the variables needed for evaluation into local namespace
        varnames = map.co_names
        for vname in varnames:
            if vname in globals():
                pass
            if vname in self.resultsSource.keys(): #look at original results first
                locals()[vname] = self.resultsSource[vname]
            elif vname in dir(self): #look for constants
                locals()[vname] = self.__dict__[vname]
            elif vname in self.mappings.keys(): #finally try other mappings
                #try to prevent infinite recursion here if mappings have circular references
                if not vname == key and not key in self.mappings[vname].co_names:
                    locals()[vname] = self.getMappedResults(vname)
                else:
                    raise RuntimeError('Circular reference detected in mapping')

        return eval(map)

class colourFilter(inputFilter):
    _name = "Colour Filter"
    def __init__(self, resultsSource, visFr, currentColour=None):
        '''Class to permit filtering by colour
        '''

        self.resultsSource = resultsSource
        self.currentColour = currentColour
        #self.shifts = {}

        self.visFr = visFr


    def __getitem__(self, key):
        colChans = self.getColourChans()

        if not self.currentColour in colChans:
            return self.resultsSource[key]
        else:
            p_dye = self.resultsSource['p_%s' % self.currentColour]

            p_other = 0*p_dye
            p_tot = self.visFr.t_p_background*self.resultsSource['ColourNorm']

            for k in colChans:
                p_tot  += self.resultsSource['p_%s' % k]
                if not self.currentColour == k:
                    p_other = np.maximum(p_other, self.resultsSource['p_%s' % k])

            p_dye = p_dye/p_tot
            p_other = p_other/p_tot

            ind = (p_dye > self.visFr.t_p_dye)*(p_other < self.visFr.t_p_other)

            #chromatic shift correction
            #print self.currentColour
            if self.currentColour in self.visFr.chromaticShifts.keys() and key in self.visFr.chromaticShifts[self.currentColour].keys():
                return self.resultsSource[key][ind] + self.visFr.chromaticShifts[self.currentColour][key]
            else:
                return self.resultsSource[key][ind]

    def getColourChans(self):
        return [k[2:] for k in self.keys() if k.startswith('p_')]

    def setColour(self, colour):
        self.currentColour = colour

    def keys(self):
        return self.resultsSource.keys()

    
    
class cloneSource(inputFilter):
    _name = "Cloned Source"
    def __init__(self, resultsSource):
        '''Creates an in memory copy of a (filtered) data source'''

        resultsSource
        self.cache = {}

        for k in resultsSource.keys():
            self.cache[k] = resultsSource[k]



    def __getitem__(self, key):
        return self.cache[key]

    def keys(self):
        return self.cache.keys()
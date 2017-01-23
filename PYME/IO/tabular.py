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

""" import filters for localisation microscopy results. These masquerade as
dictionaries which can be looked up to yield the desired data. The visualisation
routines expect at least 'x' and 'y' to be defined as keys, and may also 
understand additional values, e.g. 'error_x' 
"""
import types
import numpy as np

from numpy import * #to allow the use of sin cos etc in mappings
from PYME.Analysis.piecewise import * #allow piecewise linear mappings

import tables

class TabularBase(object):
    def toDataFrame(self, keys=None):
        import pandas as pd
        if keys is None:
            keys = self.keys()
        
        d = {k: self.__getitem__(k) for k in keys}
        
        return pd.DataFrame(d)
        
    def _getKeySlice(self, keys):
        if isinstance(keys, tuple) and len(keys) > 1:
            key = keys[0]
            sl = keys[1]
        else:
            key = keys
            sl = slice(None)

        #print key, sl
            
        return key, sl

    def to_recarray(self, keys=None):
        from numpy.core import records
        if keys is None:
            keys = self.keys()

        return records.fromarrays([self.__getitem__(k) for k in keys], names = keys)

    def to_hdf(self, filename, tablename='Data', keys=None, metadata=None):
        from PYME.IO import h5rFile

        with h5rFile.H5RFile(filename, 'a') as f:
            f.appendToTable(tablename, self.to_recarray(keys))

            if metadata is not None:
                f.updateMetadata(metadata)
        
    

class randomSource(TabularBase):
    _name = "Random Source"
    def __init__(self, xmax, ymax, nsamps):
        """Uniform random source, for testing and as an example"""
        self.x = xmax*np.random.rand(nsamps)
        self.y = ymax*np.random.rand(nsamps)

        self._keys = ['x', 'y']

    def keys(self):
        return self._keys

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        
        if not key in self._keys:
            raise KeyError('Key (%s) not defined' % key)
        
        if key == 'x':
            return self.x[sl]
        elif key == 'y':
            return self.y[sl]

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

class fitResultsSource(TabularBase):
    _name = "recarrayfi Source"
    def __init__(self, fitResults, sort=True):
        self.setResults(fitResults, sort=sort)
        
    def setResults(self, fitResults, sort=True):
        self.fitResults = fitResults

        if sort:
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

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
            
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise KeyError('Key  (%s) not found' % key)

        k = key.split('_')

        if len(k) == 1:  # TODO: evaluate why these are cast as floats
            return self.fitResults[k[0]].astype('f')[sl]
        elif len(k) == 2:
            return self.fitResults[k[0]][k[1]].astype('f')[sl]
        elif len(k) == 3:
            return self.fitResults[k[0]][k[1]][k[2]].astype('f')[sl]
        else:
            raise KeyError("Don't know about deeper nesting yet")


    def close(self):
        self.h5f.close()

    def getInfo(self):
        return 'PYME h5r Data Source\n\n %d points' % self.fitResults.shape[0]


class h5rSource(TabularBase):
    _name = "h5r Data Source"
    def __init__(self, h5fFile, tablename='FitResults'):
        """ Data source for use with h5r files as saved by the PYME analysis
        component. Takes either an open h5r file or a string filename to be
        opened."""
        self.tablename = tablename

        if type(h5fFile) == tables.file.File:
            self.h5f = h5fFile
        else:
            self.h5f = tables.open_file(h5fFile)
        
        if not tablename in dir(self.h5f.root):
            raise RuntimeError('Was expecting to find a "%s" table' % tablename)

        self.fitResults = getattr(self.h5f.root, tablename)[:]

        #allow access using unnested original names
        self._keys = unNestNames(getattr(self.h5f.root, tablename).description._v_nested_names)
        #or shorter aliases
        self.transkeys = {'A' : 'fitResults_A', 'x' : 'fitResults_x0',
                          'y' : 'fitResults_y0', 'sig' : 'fitResults_sigma', 
                          'error_x' : 'fitError_x0', 'error_y' : 'fitError_y0', 't':'tIndex'}

        for k in self.transkeys.keys():
            if not self.transkeys[k] in self._keys:
                self.transkeys.pop(k)

        #sort by time
        if 'tIndex' in self._keys:
            self.fitResults.sort(order='tIndex')


    def keys(self):
        return self._keys + self.transkeys.keys()

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
            
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise KeyError('Key not found - %s' % key)

        k = key.split('_')
        
        if len(k) == 1:
            return self.fitResults[k[0]][sl]
        elif len(k) == 2:
            return self.fitResults[k[0]][k[1]][sl]
        elif len(k) == 3:
            return self.fitResults[k[0]][k[1]][k[2]][sl]
        else:
            raise KeyError("Don't know about deeper nesting yet")
        

    def close(self):
        self.h5f.close()

    def getInfo(self):
        return 'PYME h5r Data Source\n\n %d points' % self.fitResults.shape[0]


class h5rDSource(h5rSource):
    _name = "h5r Drift Source"

    def __init__(self, h5fFile):
        """ Data source for use with h5r files as saved by the PYME analysis
        component"""

        h5rSource.__init__(self, h5fFile, 'DriftResults')

    def getInfo(self):
        return 'PYME h5r Drift Data Source\n\n %d points' % self.fitResults.shape[0]

class hdfSource(h5rSource):
    _name = "hdf Data Source"

    def __init__(self, h5fFile, tablename='FitResults'):
        """ Data source for use with h5r files as saved by the PYME analysis
        component. Takes either an open h5r file or a string filename to be
        opened."""
        self.tablename = tablename

        if type(h5fFile) == tables.file.File:
            self.h5f = h5fFile
        else:
            self.h5f = tables.open_file(h5fFile)

        if not tablename in dir(self.h5f.root):
            raise RuntimeError('Was expecting to find a "%s" table' % tablename)

        self.fitResults = getattr(self.h5f.root, tablename)[:]

        #allow access using unnested original names
        self._keys = unNestNames(getattr(self.h5f.root, tablename).description._v_nested_names)
        #or shorter aliases

        #sort by time
        if 'tIndex' in self._keys:
            self.fitResults.sort(order='tIndex')

    def keys(self):
        return self._keys #+ self.transkeys.keys()

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)

        if not key in self._keys:
            raise KeyError('Key (%s) not found' % key)

        return self.fitResults[key][sl]


    def close(self):
        self.h5f.close()

    def getInfo(self):
        return 'PYME hdf Data Source\n\n %d points' % self.fitResults.shape[0]

# class h5rDSource(inputFilter):
#     _name = "h5r Drift Source"
#     def __init__(self, h5fFile):
#         """ Data source for use with h5r files as saved by the PYME analysis
#         component"""
#
#         if type(h5fFile) == tables.file.File:
#             self.h5f = h5fFile
#         else:
#             self.h5f = tables.openFile(h5fFile)
#
#         if not 'DriftResults' in dir(self.h5f.root):
#             raise RuntimeError('Was expecting to find a "DriftResults" table')
#
#         self.driftResults = self.h5f.root.DriftResults[:]
#
#         #sort by time
#         self.driftResults.sort(order='tIndex')
#
#         #allow access using unnested original names
#         self._keys = unNestNames(self.h5f.root.DriftResults.description._v_nestedNames)
#         #or shorter aliases
#         self.transkeys = {'A' : 'fitResults_A', 'x' : 'fitResults_x0',
#                           'y' : 'fitResults_y0', 'sig' : 'fitResults_sigma',
#                           'error_x' : 'fitError_x0', 'error_y' : 'fitError_y0', 't':'tIndex'}
#
#
#     def keys(self):
#         return self._keys + self.transkeys.keys()
#
#     def __getitem__(self, keys):
#         key, sl = self._getKeySlice(keys)
#
#         #if we're using an alias replace with actual key
#         if key in self.transkeys.keys():
#             key = self.transkeys[key]
#
#         if not key in self._keys:
#             raise RuntimeError('Key not found')
#
#         k = key.split('_')
#
#         if len(k) == 1:
#             return self.driftResults[sl][k[0]].astype('f')
#         elif len(k) == 2:
#             return self.driftResults[sl][k[0]][k[1]].astype('f')
#         elif len(k) == 3:
#             return self.driftResults[sl][k[0]][k[1]][k[2]].astype('f')
#         else:
#             raise RuntimeError("Don't know about deeper nesting yet")
#
#
#     def close(self):
#         self.h5f.close()
#
#     def getInfo(self):
#         return 'PYME h5r Drift Data Source\n\n %d points' % self.h5f.root.DriftResults.shape[0]

class textfileSource(TabularBase):
    _name = "Text File Source"
    def __init__(self, filename, columnnames, delimiter=None, skiprows=0):
        """ Input filter for use with delimited text data. Defaults
        to whitespace delimiter. need to provide a list of variable names
        in the order that they appear in the file. Using 'x', 'y' and 'error_x'
        for the position data and it's error should ensure that this functions
        with the visualisation backends"""

        self.res = np.loadtxt(filename, dtype={'names' : columnnames,  # TODO: evaluate why these are cast as floats
                                               'formats' :  ['f4' for i in range(len(columnnames))]}, delimiter = delimiter, skiprows=skiprows)
        
        self._keys = list(columnnames)
       


    def keys(self):
        return self._keys

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        
        if not key in self._keys:
            raise KeyError('Key (%s) not found' % key)

       
        return self.res[key][sl]

    
    def getInfo(self):
        return 'Text Data Source\n\n %d points' % len(self.res['x'])

class matfileSource(TabularBase):
    _name = "Matlab Source"
    def __init__(self, filename, columnnames, varName='Orte'):
        """ Input filter for use with matlab data. Need to provide a variable name
        and a list of column names
        in the order that they appear in the file. Using 'x', 'y' and 'error_x'
        for the position data and it's error should ensure that this functions
        with the visualisation backends"""

        import scipy.io

        self.res = scipy.io.loadmat(filename)[varName].astype('f4')  # TODO: evaluate why these are cast as floats
        
        self.res = np.rec.fromarrays(self.res.T, dtype={'names' : columnnames,  'formats' :  ['f4' for i in range(len(columnnames))]})

        self._keys = list(columnnames)



    def keys(self):
        return self._keys

    def __getitem__(self, key):
        key, sl = self._getKeySlice(key)
        if not key in self._keys:
            raise KeyError('Key (%s) not found' % key)


        return self.res[key][sl]


    def getInfo(self):
        return 'Text Data Source\n\n %d points' % len(self.res['x'])
        

class resultsFilter(TabularBase):
    _name = "Results Filter"
    def __init__(self, resultsSource, **kwargs):
        """Class to permit filtering of fit results - masquarades
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with 
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""

        self.resultsSource = resultsSource

        #by default select everything
        self.Index = np.ones(self.resultsSource[resultsSource.keys()[0]].shape) >  0.5

        for k in kwargs.keys():
            if not k in self.resultsSource.keys():
                raise KeyError('Requested key not present: ' + k)

            range = kwargs[k]
            if not len(range) == 2:
                raise RuntimeError('Expected an iterable of length 2')

            self.Index *= (self.resultsSource[k] > range[0])*(self.resultsSource[k] < range[1])
                

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        return self.resultsSource[key][self.Index][sl]

    def keys(self):
        return self.resultsSource.keys()


class concatenateFilter(TabularBase):
    _name = "Concatenation Filter"

    def __init__(self, source0, source1):
        """Class which concatenates two tabular data sources. The data sources should have the same keys.

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""

        self.source0 = source0
        self.source1 = source1


    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        if key == 'concatSource':
            return np.hstack((np.zeros(len(self.source0[self.source0.keys()[0]])), np.ones(len(self.source1[self.source1.keys()[0]]))))
        else:
            return np.hstack((self.source0[key], self.source1[key]))[sl]

    def keys(self):
        s1_keys = self.source1.keys()
        return list(set(['concatSource', ] + [k for k in self.source0.keys() if k in s1_keys]))

class cachingResultsFilter(TabularBase):
    _name = "Caching Results Filter"
    def __init__(self, resultsSource, **kwargs):
        """Class to permit filtering of fit results - masquarades
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""

        self.resultsSource = resultsSource
        self.cache = {}

        #by default select everything
        self.Index = np.ones(self.resultsSource[resultsSource.keys()[0]].shape) >  0.5

        for k in kwargs.keys():
            if not k in self.resultsSource.keys():
                raise KeyError('Requested key not present: ' + k)

            range = kwargs[k]
            if not len(range) == 2:
                raise RuntimeError('Expected an iterable of length 2')

            self.Index *= (self.resultsSource[k] > range[0])*(self.resultsSource[k] < range[1])


    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        if key in self.cache.keys():
            return self.cache[key][sl]
        else:
            res = np.array(self.resultsSource[key])[self.Index]
            self.cache[key] = res
            return res[sl]

    def keys(self):
        return self.resultsSource.keys()


class mappingFilter(TabularBase):
    _name = "Mapping Filter"
    def __init__(self, resultsSource, **kwargs):
        """Class to permit transformations (e.g. drift correction) of fit results
        - masquarades as a dictionary. Takes mappings as keyword arguments, eg:
        f = resultsFliter(source, xp='x + a*tIndex', yp=compile('y + b*tIndex', '/tmp/test1', 'eval'), a=1, b=2)
        will return an object that behaves like source, but has additional members
        xp and yp.

        the mappings should either be code objects, strings (which will be compiled into code objects),
        or something else (which will be turned into a local variable - eg constants in above example)

        """

        self.resultsSource = resultsSource

        self.mappings = {}
        self.new_columns = {}
        self.variables = {}

        for k in kwargs.keys():
            v = kwargs[k]
            self.setMapping(k,v)


    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        if key in self.mappings.keys():
            return self.getMappedResults(key, sl)
        elif key in self.new_columns.keys():
            return self.new_columns[key][sl]
        else:
            return self.resultsSource[keys]

    def keys(self):
        return list(set(list(self.resultsSource.keys()) + self.mappings.keys() + self.new_columns.keys()))

    def addVariable(self, name, value):
        """
        Adds a scalar variable to the mapping object. This will be accessible from mappings. An example usage might
        be to define a scaling parameter for one of our column variables.

        Parameters
        ----------
        name : string
            The name we want to be able to access the variable by
        value : float (or something which can be cast to a float)
            The value

        """

        #insert into our __dict__ object (for backwards compatibility - TODO change me to something less hacky)
        #setattr(self, name, float(value))

        self.variables[name] = float(value)

    def addColumn(self, name, values):
        """
        Adds a column of values to the mapping.

        Parameters
        ----------
        name : str
            The new column name
        values : array-like
            The values. This should be the same length as the existing columns.

        """

        #force to be an array
        values = np.array(values)

        if not len(values) == len(self.resultsSource[self.resultsSource.keys()[0]]):
            raise RuntimeError('New column does not match the length of existing columns')

        #insert into our __dict__ object (for backwards compatibility - TODO change me to something less hacky)
        #setattr(self, name, values)

        self.new_columns[name] = values


    def setMapping(self, key, mapping):
        if type(mapping) == types.CodeType:
            self.mappings[key] = mapping
        elif type(mapping) == types.StringType:
            self.mappings[key] = compile(mapping, '/tmp/test1', 'eval')
        else:
            self.__dict__[key] = mapping

    def getMappedResults(self, key, sl):
        map = self.mappings[key]

        #get all the variables needed for evaluation into local namespace
        varnames = map.co_names
        for vname in varnames:
            if vname in globals():
                pass
            if vname in self.resultsSource.keys(): #look at original results first
                locals()[vname] = self.resultsSource[vname][sl]
            elif vname in self.new_columns.keys():
                locals()[vname] = self.new_columns[vname][sl]
            elif vname in self.variables.keys():
                locals()[vname] = self.variables[vname]
            elif vname in dir(self): #look for constants
                locals()[vname] = self.__dict__[vname]
            elif vname in self.mappings.keys(): #finally try other mappings
                #try to prevent infinite recursion here if mappings have circular references
                if not vname == key and not key in self.mappings[vname].co_names:
                    locals()[vname] = self.getMappedResults(vname, sl)
                else:
                    raise RuntimeError('Circular reference detected in mapping')

        return eval(map)

class colourFilter(TabularBase):
    _name = "Colour Filter"
    def __init__(self, resultsSource, currentColour=None):
        """Class to permit filtering by colour
        """

        self.resultsSource = resultsSource
        self.currentColour = currentColour
        self.chromaticShifts = {}

        self.t_p_dye = 0.1
        self.t_p_other = 0.1
        self.t_p_background = .01

    @property
    def index(self):
        colChans = self.getColourChans()

        if not self.currentColour in colChans:
            return np.ones(len(self.resultsSource[self.resultsSource.keys()[0]]), 'bool')
        else:
            p_dye = self.resultsSource['p_%s' % self.currentColour]

            p_other = 0 * p_dye
            p_tot = self.t_p_background * self.resultsSource['ColourNorm']

            for k in colChans:
                p_tot += self.resultsSource['p_%s' % k]
                if not self.currentColour == k:
                    p_other = np.maximum(p_other, self.resultsSource['p_%s' % k])

            p_dye = p_dye / p_tot
            p_other = p_other / p_tot

            return (p_dye > self.t_p_dye) * (p_other < self.t_p_other)


    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        colChans = self.getColourChans()

        if not self.currentColour in colChans:
            return self.resultsSource[keys]
        else:
            #chromatic shift correction
            #print self.currentColour
            if  self.currentColour in self.chromaticShifts.keys() and key in self.chromaticShifts[self.currentColour].keys():
                return self.resultsSource[key][self.index][sl] + self.chromaticShifts[self.currentColour][key]
            else:
                return self.resultsSource[key][self.index][sl]

    @classmethod
    def get_colour_chans(cls, resultsSource):
        return [k[2:] for k in resultsSource.keys() if k.startswith('p_')]

    def getColourChans(self):
        return self.get_colour_chans(self)

    def setColour(self, colour):
        self.currentColour = colour

    def keys(self):
        return self.resultsSource.keys()

    
    
class cloneSource(TabularBase):
    _name = "Cloned Source"
    def __init__(self, resultsSource, keys=None):
        """Creates an in memory copy of a (filtered) data source"""
        #resultsSource
        self.cache = {}

        if not keys:
            klist = resultsSource.keys()
        else:
            klist = keys
        for k in klist:
            self.cache[k] = resultsSource[k]



    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        return self.cache[key][sl]

    def keys(self):
        return self.cache.keys()

class recArrayInput(TabularBase):
    _name = 'RecArray Source'
    def __init__(self, recordArray):
        self.recArray = recordArray
        self._keys = self.recArray.dtype.names

    def keys(self):
        return self._keys

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)

        if not key in self._keys:
            raise KeyError('Key (%s) not found' % key)

        return self.recArray[key][sl]

    def getInfo(self):
        return 'Record Array Source\n\n %d points' % len(self.recArray['x'])
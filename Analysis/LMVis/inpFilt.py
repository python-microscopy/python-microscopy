''' import filters for localisation microscopy results. These masquerade as 
dictionaries which can be looked up to yield the desired data. The visualisation
routines expect at least 'x' and 'y' to be defined as keys, and may also 
understand additional values, e.g. 'error_x' 
'''
import numpy as np
import tables

class randomSource:
    def __init__(self, xmax, ymax, nsamps):
        '''Uniform random source, for testing and as an example'''
        self.x = xmax*np.rand(nsamps)
        self.y = ymax*np.rand(nsamps)

        self.keys = ['x', 'y']

    def keys(self):
        return self.keys

    def __getitem__(self, key):
        if not key in self.keys:
            raise 'Key not defined'
        
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y


def unNestNames(nameList, parent=''):
    unList = []
    for n in nameList:
        if n.__class__ == str:
            unList.append(parent + n)
        else:
            unList += unNestNames(n[1], parent + n[0] + '_')
    return unList


class h5rSource:
    def __init__(self, h5fFilename):
        ''' Data source for use with h5r files as saved by the PYME analysis 
        component'''
        
        self.h5f = tables.openFile(h5fFilename)
        
        if not 'FitResults' in dir(self.h5f.root):
            raise 'Was expecting to find a "FitResults" table'

        #allow access using unnested original names
        self._keys = unNestNames(self.h5f.root.FitResults.description._v_nestedNames)
        #or shorter aliases
        self.transkeys = {'A' : 'fitResults_A', 'x' : 'fitResults_x0',
                          'y' : 'fitResults_y0', 'sig' : 'fitResults_sigma', 
                          'error_x' : 'fitError_x0'}


    def keys(self):
        return self._keys + self.transkeys.keys()

    def __getitem__(self, key):
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise 'Key not found'

        k = key.split('_')
        
        if len(k) == 1:
            return self.h5f.root.FitResults[:][k[0]]
        elif len(k) == 2:
            return self.h5f.root.FitResults[:][k[0]][k[1]]
        elif len(k) == 3:
            return self.h5f.root.FitResults[:][k[0]][k[1]][k[2]]
        else:
            raise "Don't know about deeper nesting yet"
        

    def close(self):
        self.h5f.close()


class h5rDSource:
    def __init__(self, h5fFilename):
        ''' Data source for use with h5r files as saved by the PYME analysis 
        component'''
        
        self.h5f = tables.openFile(h5fFilename)
        
        if not 'DriftResults' in dir(self.h5f.root):
            raise 'Was expecting to find a "DriftResults" table'

        #allow access using unnested original names
        self._keys = unNestNames(self.h5f.root.DriftResults.description._v_nestedNames)
        #or shorter aliases
        self.transkeys = {'A' : 'fitResults_A', 'x' : 'fitResults_x0',
                          'y' : 'fitResults_y0', 'sig' : 'fitResults_sigma', 
                          'error_x' : 'fitError_x0'}


    def keys(self):
        return self._keys + self.transkeys.keys()

    def __getitem__(self, key):
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise 'Key not found'

        k = key.split('_')
        
        if len(k) == 1:
            return self.h5f.root.DriftResults[:][k[0]]
        elif len(k) == 2:
            return self.h5f.root.DriftResults[:][k[0]][k[1]]
        elif len(k) == 3:
            return self.h5f.root.DriftResults[:][k[0]][k[1]][k[2]]
        else:
            raise "Don't know about deeper nesting yet"
        

    def close(self):
        self.h5f.close()

class textfileSource:
    def __init__(self, filename, columnnames, delimiter=None):
        ''' Input filter for use with delimited text data. Defaults
        to whitespace delimiter. need to provide a list of variable names
        in the order that they appear in the file. Using 'x', 'y' and 'error_x'
        for the position data and it's error should ensure that this functions
        with the visualisation backends'''

        self.res = np.loadtxt(filename, dtype={'names' : columnnames, 
                                               'formats' :  ['f4' for i in range(len(columnnames))]}, delimiter = delimiter)
        
        self._keys = columnnames
       


    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if not key in self._keys:
            raise 'Key not found'

       
        return self.res[key]
       
        

class resultsFilter:
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
        self.Index = np.ones(self.resultsSource[kwargs.keys()[0]].shape) >  0.5

        for k in kwargs.keys():
            if not k in self.resultsSource.keys():
                raise 'Requested key not present'

            range = kwargs[k]
            if not len(range) == 2:
                raise 'Expected an iterable of length 2'

            self.Index *= (self.resultsSource[k] > range[0])*(self.resultsSource[k] < range[1])
                

    def __getitem__(self, key):
        return self.resultsSource[key][self.Index]

    def keys(self):
        return self.resultsSource.keys()
    

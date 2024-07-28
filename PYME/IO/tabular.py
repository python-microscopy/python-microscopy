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
import six
import warnings
import numpy as np

from numpy import * #to allow the use of sin cos etc in mappings
from PYME.Analysis.piecewise import * #allow piecewise linear mappings

import tables
import logging

logger = logging.getLogger(__name__)

#helper function for renaming classes

def deprecated_name(name):
    def _dec(cls):
        
        def _dep_name(*args, **kwargs):
            warnings.warn(VisibleDeprecationWarning('%s is deprecated, use %s instead' % (name, cls.__name__)))
            return cls(*args, **kwargs)
        
        globals()[name] = _dep_name
        
        return cls
    
    return _dec

class TabularBase(object):
    def toDataFrame(self, keys=None):
        warnings.warn('toDataFrame is deprecated, use to_pandas instead', DeprecationWarning)
        self.to_pandas(keys)

    def to_pandas(self, keys=None):
        """
        Convert tabular data to a pandas DataFrame
        """
        import pandas as pd
        if keys is None:
            keys = self.keys()
        
        d = {}
        
        for k in keys:
            v = self.__getitem__(k)
            if np.ndim(v) == 1:
                d[k] = v
            else:
                v1 = np.empty(len(v), 'O')
                for i in range(len(v)):
                    v1[i] = v[i]
                d[k] = v1
        
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
        """
        Converts tabular data types into record arrays, which is useful for e.g. saving as an hdf table. In order to be
        converted, the tabular data source must be able to be flattened.

        Parameters
        ----------
        keys : list of fields to be copied into output. Defaults to all existing keys.

        Returns
        -------
        numpy recarray version of self

        """
        from numpy.core import records
        if keys is None:
            keys = list(self.keys())

        columns = [self.__getitem__(k) for k in keys]
        
        filtered_cols = [i for i, v in enumerate(columns) if not v.dtype == 'O']
        
        cols = [columns[i] for i in filtered_cols]
        keys_ = [keys[i] for i in filtered_cols]
        
        
        dt = [(k, v.dtype, v.shape[1:]) for k, v in zip(keys_, cols)]
        
        #print(dt)
        return records.fromarrays(cols, names=keys_, dtype=dt)

    def to_hdf(self, filename, tablename='Data', keys=None, metadata=None,
               keep_alive_timeout=0):
        """
        Writes data to a table in an HDF5 file
        
        Parameters
        ----------
        
        filename: string
            the name of the file to save to
        tablename: string [optional]
            the name of the table within the file to save to. Defaults to "Data"
        keys: list [optional]
            a list of column names to save (if keys == None, all columns are saved)
        metadata: a MetaDataHandler instance [optional]
            associated metadata to write to the file
        keep_alive_timeout: float
            a timeout in seconds. If non-zero, the file is held open after we have finished writing to it until the
            timeout elapses. Useful as a performance optimisation when making multiple writes to a single file,
            potentially across multiple threads. NOTE: the keep_alive_timeout is not garuanteed to be observed - it
            gets set by the first open call of a given session, so if the file is already open due to a previous openH5R
            call, the timeout requested by that call will be used.
            
        """
        from PYME.IO import h5rFile

        with h5rFile.openH5R(filename, 'a', keep_alive_timeout=keep_alive_timeout) as f:
            f.appendToTable(tablename, self.to_recarray(keys))

            if metadata is not None:
                f.updateMetadata(metadata)
                
            #wait until data is written
            f.flush()

    def to_csv(self, outFile, keys=None):
        if outFile.endswith('.csv'):
            delim = ', '
        else:
            delim = '\t'
    
        if keys is None:
            keys = self.keys()
    
        #nRecords = len(ds[keys[0]])
        
        def fmt(d):
            if np.isscalar(d):
                if isinstance(d, six.string_types):
                    return str(d)
                else:
                    return '%e' % d
            else:
                # try to make sure odd objects don't break file formatting
                return ('"%s"' % str(d).replace(delim,' ').replace('\n',' '))
    
        of = open(outFile, 'w')
    
        of.write('# ' + delim.join(['%s' % k for k in keys]) + '\n')
    
        for row in zip(*[self[k] for k in keys]):
            of.write(delim.join([fmt(c) for c in row]) + '\n')
    
        of.close()
            
                
    def keys(self):
        raise NotImplementedError('Should be over-ridden in derived class')
    
    def __getitem__(self, keys):
        raise NotImplementedError('Should be over-ridden in derived class')
                
    def __len__(self):
        return len(self[list(self.keys())[0]])
    
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError("'%s' has not attribute '%s'" % (self.__class__, item))
        
    def __dir__(self):
        return list(self.keys()) + list(self.__dict__.keys()) + list(dir(type(self)))

    def to_JSON(self, keys=None, return_slice=slice(None)):
        # TODO - do we actually use the slice argument? Should the signature better match to_hdf, to_recarray, toDataFrame?
        # TODO - is this redundant when compared to .toDataFrame().to_json()
        import json

        d= {}
        keys = keys if keys != None else self.keys()
        for k in keys:
            d[k] = self[(k, return_slice)].tolist()
        return json.dumps(d)


# Data sources (File IO, or adapters to other data formats - e.g. recarrays
###########################################################################

@deprecated_name('randomSource')
class RandomSource(TabularBase):
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
        if isinstance(n, six.string_types):
            unList.append(parent + n)
        else:
            unList += unNestNames(n[1], parent + n[0] + '_')
    return unList

def unNestDtype(descr, parent=''):
    unList = []
    for n in descr:
        #print n, n.__class__, len(n)
        if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1],six.string_types):
            unList.append(parent + n[0])
        else:
            unList += unNestDtype(n[1], parent + n[0] + '_')
    return unList

def unnest_dtype(dtype, parent=''):
    if isinstance(dtype, np.dtype):
        descr = dtype.descr
    else:
        descr = dtype
        
    dt = []
    for node in descr:
        if isinstance(node, tuple):# and len(node) == 2:
            name, t = node[:2]
            if isinstance(t, str):
                dt.append((parent + name, ) + node[1:])
            elif len(node) == 2:
                dt += unnest_dtype(node[1], parent=parent + name + '_')
            else:
                raise RuntimeError('unexpected dtype descr: %s, node: %s' % (descr, node))
        else:
            raise RuntimeError('unexpected dtype descr: %s, node: %s' % (descr, node))
    
    if parent == '':
        #cast to a numpy dtype if we are at the top recursion level
        return np.dtype(dt)
    else:
        # otherwise just return the description
        return dt

@deprecated_name('fitResultsSource')
class FitResultsSource(TabularBase):
    _name = "recarrayfi Source"
    def __init__(self, fitResults, sort=True):
        self.setResults(fitResults, sort=sort)
        
    def _set_transkeys(self):
        self.transkeys = {'A': 'fitResults_A', 'x': 'fitResults_x0',
                          'y': 'fitResults_y0', 'sig': 'fitResults_sigma',
                          'error_x': 'fitError_x0', 'error_y': 'fitError_y0', 'error_z': 'fitError_z0', 't': 'tIndex'}
    
        for k in list(self.transkeys.keys()):
            if not self.transkeys[k] in self._keys:
                self.transkeys.pop(k)
        
    def setResults(self, fitResults, sort=True):
        self.fitResults = fitResults

        if sort:
            #sort by time
            self.fitResults.sort(order='tIndex')

        #allow access using unnested original names
        # TODO???? - replace key translation with a np.view call?
        #self._keys = unNestDtype(self.fitResults.dtype.descr)
        self._keys = list(unnest_dtype(self.fitResults.dtype).names)
        
        #or shorter aliases
        self._set_transkeys()
        


    def keys(self):
        return self._keys + list(self.transkeys.keys())

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
            
        #if we're using an alias replace with actual key
        if key in self.transkeys.keys():
            key = self.transkeys[key]

        if not key in self._keys:
            raise KeyError('Key  (%s) not found' % key)

        k = key.split('_')

        if len(k) == 1:  # TODO: evaluate why these are cast as floats
            return self.fitResults[k[0]][sl]
        elif len(k) == 2:
            return self.fitResults[k[0]][k[1]][sl]
        elif len(k) == 3:
            return self.fitResults[k[0]][k[1]][k[2]][sl]
        else:
            raise KeyError("Don't know about deeper nesting yet")

    def getInfo(self):
        return 'PYME h5r Data Source\n\n %d points' % self.fitResults.shape[0]



class _BaseHDFSource(FitResultsSource):
    ''' Copy of the original BaseHDFSource which used pytables directly rather than h5rFile
        Currently unused, but kept for historical reasons.
    '''
    def __init__(self, h5fFile, tablename='FitResults'):
        """ Data source for use with h5r files as saved by the PYME analysis
        component. Takes either an open h5r file or a string filename to be
        opened."""
        self.tablename = tablename
        
        if type(h5fFile) == tables.file.File:
            h5f = h5fFile
            self._own_file = False #did we open the file
        else:
            h5f = tables.open_file(h5fFile)
            self.h5f = h5f
            self._own_file = True
        
        #if not tablename in dir(self.h5f.root):
        
        
        try:
            self.fitResults = getattr(h5f.root, tablename)[:]
        except (AttributeError, tables.NoSuchNodeError):
            logger.exception('Was expecting to find a "%s" table' % tablename)
            raise
        
        #allow access using unnested original names
        self._keys = unNestNames(getattr(h5f.root, tablename).description._v_nested_names)
        
        #close the hdf file (if we opened it)
        #if self._own_file:
        #    h5f.close()
        
        #or shorter aliases
        
    def close(self):
        if self._own_file:
            try:
                self.h5f.close()
            except:
                pass
            
    def __del__(self):
        self.close()


class BaseHDFSource(FitResultsSource):
    def __init__(self, h5fFile, tablename='FitResults'):
        """ Data source for use with h5r files as saved by the PYME analysis
        component. Takes either an open h5r file or a string filename to be
        opened."""
        from PYME.IO import h5rFile
        self.tablename = tablename
        
        if isinstance(h5fFile, tables.file.File):
            try:
                fr = getattr(h5fFile.root, tablename)
                self.fitResults = fr[:]
                
                #allow access using unnested original names
                self._keys = unNestNames(fr.description._v_nested_names)

            except (AttributeError, tables.NoSuchNodeError):
                logger.exception('Was expecting to find a "%s" table' % tablename)
                raise
    
            
        
        else:
            if isinstance(h5fFile, h5rFile.H5RFile):
                h5f = h5fFile
            else:
                h5f = h5rFile.openH5R(h5fFile)
        
            with h5f:
                self.fitResults = h5f.getTableData(tablename, slice(None))
                if (len(self.fitResults) == 0):
                    raise RuntimeError('Was expecting to find a "%s" table' % tablename)
                
                #allow access using unnested original names
                self._keys = unNestNames(getattr(h5f._h5file.root, tablename).description._v_nested_names)
            
        #close the hdf file (if we opened it)
        #if self._own_file:
        #    h5f.close()
        
        #or shorter aliases
    
    def close(self):
        pass


@deprecated_name('h5rSource')
class H5RSource(BaseHDFSource):
    _name = "h5r Data Source"
    def __init__(self, h5fFile, tablename='FitResults'):
        BaseHDFSource.__init__(self, h5fFile, tablename)
        
        # set up column aliases
        self._set_transkeys()

        #sort by time
        if 'tIndex' in self._keys:
            I = self.fitResults['tIndex'].argsort()
            self.fitResults = self.fitResults[I]
            #self.fitResults.sort(order='tIndex')
        

    def getInfo(self):
        return 'PYME h5r Data Source\n\n %d points' % self.fitResults.shape[0]


@deprecated_name('h5rDSource')
class H5RDSource(H5RSource):
    _name = "h5r Drift Source"

    def __init__(self, h5fFile):
        """ Data source for use with h5r files as saved by the PYME analysis
        component"""

        H5RSource.__init__(self, h5fFile, 'DriftResults')

    def getInfo(self):
        return 'PYME h5r Drift Data Source\n\n %d points' % self.fitResults.shape[0]

@deprecated_name('hdfSource')
class HDFSource(H5RSource):
    _name = "hdf Data Source"

    def __init__(self, h5fFile, tablename='FitResults'):
        BaseHDFSource.__init__(self, h5fFile, tablename)
        #or shorter aliases

        #sort by time
        if 'tIndex' in self._keys:
            I = self.fitResults['tIndex'].argsort()
            self.fitResults = self.fitResults[I]

    def keys(self):
        return self._keys #+ self.transkeys.keys()

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)

        if not key in self._keys:
            raise KeyError('Key (%s) not found' % key)

        return self.fitResults[key][sl]


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
@deprecated_name('textfileSource')
class TextfileSource(TabularBase):
    _name = "Text File Source"
    def __init__(self, filename, columnnames, delimiter=None, skiprows=0, skip_footer=0, invalid_raise=True, comments='#'):
        """ Input filter for use with delimited text data. Defaults
        to whitespace delimiter. Need to provide a list of variable names
        in the order that they appear in the file. Using 'x', 'y' and 'error_x'
        for the position data and it's error should ensure that this functions
        with the visualisation backends"""

        #self.res = np.loadtxt(filename, dtype={'names' : columnnames,  # TODO: evaluate why these are cast as floats
        #                                       'formats' :  ['f4' for i in range(len(columnnames))]}, delimiter = delimiter, skiprows=skiprows)

        from PYME import config
        
        #print('invalid_raise:', invalid_raise)

        if config.get('TextFileSource-use_pandas',False):
            logger.info('Opening "%s" using pandas (set TextFileSource-use_pandas: False in config.yaml to use legacy np.genfromtxt instead)' % filename)
            import pandas as pd
            self.res = pd.read_csv(filename,
                comment=comments,
                delimiter=delimiter,
                skiprows=skiprows,
                skipfooter=skip_footer,
                names=columnnames,
                dtype='f4',
                skipinitialspace=True,
                on_bad_lines='error' if invalid_raise else 'warn,'
                ).to_records(index=False)

        else:
            logger.info('Opening %s using np.genfromtxt (set TextFileSource-use_pandas: True in config.yaml to use pandas instead)' % filename)
            self.res = np.genfromtxt(filename,
                             comments=comments,
                             delimiter=delimiter,
                             skip_header=skiprows,
                             skip_footer=skip_footer,
                             names=columnnames, 
                             dtype='f4', replace_space='_',
                             missing_values=None, filling_values=np.nan, # use NaN to flag missing values
                             invalid_raise=invalid_raise,
                             encoding='latin-1') # Zeiss Elyra bombs unless we go for latin-1 encoding, maybe make flavour specific?

        # check for missing values:
        # TODO - is this needed/helpful, or should we propagate missing values further?
        # cast to a non-structured dtype and use the fact than the sum will be NaN if any of the individual values is NaN
        r = self.res.view(('f4', len(columnnames))).sum(1)
        if np.any(np.isnan(r)):
            logger.warning('Text file contains missing values, discarding lines with missing values')
            self.res = self.res[~np.isnan(r)]

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


@deprecated_name('matfileSource')
class MatfileSource(TabularBase):
    _name = "Matlab Source"
    def __init__(self, filename, columnnames, varName='Orte'):
        """ Input filter for use with matlab data where all variables are in in one 2D array (variable).
        Need to provide a variable name within the matfile to and a list of column names
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

@deprecated_name('matfileColumnSource')
class MatfileColumnSource(TabularBase):
    _name = "Matlab Column Source"
    
    def __init__(self, filename):
        """ Input filter for use with matlab data where the each column is in a separate variable.
        Relies on variables having suitable column names - columns named x, y, z, t, and probe (if multi-colour) should
        be present.
        """
        
        import scipy.io
        
        self.res = scipy.io.loadmat(filename)  # TODO: evaluate why these are cast as floats
        
        self._keys = [k for k in self.res.keys() if not k.startswith('_')]
    
    def keys(self):
        return self._keys
    
    def __getitem__(self, key):
        key, sl = self._getKeySlice(key)
        if not key in self._keys:
            raise KeyError('Key (%s) not found' % key)
        
        return self.res[key][sl].astype('f4').squeeze()
    
    def getInfo(self):
        return 'Text Data Source\n\n %d points' % len(self.res['x'])

class MatfileMultiColumnSource(MatfileColumnSource):
    def __init__(self, filename):
        MatfileColumnSource.__init__(self, filename)
        
        # Unwrap multiple channels in self.res
        tmp_res = {}
        for k in self._keys:
            tmp_res[k] = np.vstack(self.res[k][0]).squeeze()
        n_channels = self.res[self._keys[0]][0].shape[0]
        tmp_res['probe'] = np.vstack(self.res[self._keys[0]][0]*np.zeros(n_channels)+np.arange(n_channels)).squeeze()
        self._keys.append('probe')

        self.res = tmp_res

@deprecated_name('recArrayInput')
class RecArraySource(TabularBase):
    _name = 'RecArray Source'
    def __init__(self, recordArray):
        self.recArray = recordArray
        self._keys = list(self.recArray.dtype.names)

    def keys(self):
        return self._keys

    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)

        if not key in self._keys:
            raise KeyError('Key (%s) not found' % key)

        return self.recArray[key][sl]

    def getInfo(self):
        return 'Record Array Source\n\n %d points' % len(self.recArray['x'])


class DictSource(TabularBase):
    _name = 'Dict Source'
    
    def __init__(self, source):
        """
        Create a data source from a dictionary of numpy arrays, where each entry is a column.
        
        Parameters
        ----------
        source
        
        """
        self._verify(source)
        self._source = source
        
    def _verify(self, source):
        L = len(source[list(source.keys())[0]])
        
        for k, v in source.items():
            if not isinstance(v, np.ndarray):
                raise TypeError('Column "%s" is not a numpy array' % k)
            
            if not len(v) == L:
                raise ValueError('Columns are different lengths')

    def addColumn(self, name, values):
        if not isinstance(values, np.ndarray):
            raise TypeError('New column "%s" is not a numpy array' % name)

        if not len(values) == len(self):
            raise ValueError('Columns are different lengths')
        
    def keys(self):
        return list(self._source.keys())
    
    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        return self._source[key][sl]

class ColumnSource(DictSource):
    _name = 'Column Source'
    def __init__(self, **kwargs):
        """
        Create a datasource from columns specified by kwargs. Each column should be a numpy array.
        
        Parameters
        ----------
        kwargs
        """
        DictSource.__init__(self, dict(kwargs))

def scalar_dict_source(d):
    """
    Creates a tabular data source of length 1 from a scalar data
    """
    
    return DictSource({k : np.asarray([v]) for k, v in d.items()})

def scalar_column_source(**kwargs):
    """
    Creates a tabular data source of length 1 from a scalar data
    """
    
    return ColumnSource(**{k : np.asarray([v]) for k, v in kwargs.items()})

# Filters (which remap existing data sources)
#############################################


class SelectionFilter(TabularBase):
    _name = "Selection Filter"
    
    def __init__(self, resultsSource, index):
        """ A filter which relies on a supplied index (either integer or boolean)"""
        
        self.resultsSource = resultsSource
        
        self.Index = index
    
    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        return self.resultsSource[key][self.Index][sl]
    
    def keys(self):
        return list(self.resultsSource.keys())

@deprecated_name('resultsFilter')
class ResultsFilter(SelectionFilter):
    _name = "Results Filter"
    def __init__(self, resultsSource, **kwargs):
        """Class to permit filtering of fit results - masquarades
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with 
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""
        
        if not isinstance(resultsSource, TabularBase):
            raise TypeError('Expecting a tabular object for resultsSource')

        self.resultsSource = resultsSource

        #by default select everything
        #self.Index = np.ones(self.resultsSource[list(resultsSource.keys())[0]].shape[0]) >  0.5
        self.Index = np.ones(len(self.resultsSource), dtype=bool)

        for k in kwargs.keys():
            if not k in self.resultsSource.keys():
                raise KeyError('Requested key not present: ' + k)

            range = kwargs[k]
            if not len(range) == 2:
                raise RuntimeError('Expected an iterable of length 2')

            self.Index *= (self.resultsSource[k] > range[0])*(self.resultsSource[k] < range[1])
    

@deprecated_name('randomSelectionFilter')
class RandomSelectionFilter(SelectionFilter):
    _name = "Random Selection Filter"
    
    def __init__(self, resultsSource, num_Samples):
        """Class to permit filtering of fit results - masquarades
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""
        
        if not isinstance(resultsSource, TabularBase):
            raise TypeError('Expecting a tabular object for resultsSource')
        
        self.resultsSource = resultsSource
        
        #by default select everything
        self.Index = np.random.choice(len(self.resultsSource), num_Samples, replace=False)


@deprecated_name('idFilter')
class IdFilter(SelectionFilter):
    _name = "Id Filter"
    
    def __init__(self, resultsSource, id_column, valid_ids):
        """Class to permit filtering of fit results - masquarades
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""
        
        if not isinstance(resultsSource, TabularBase):
            raise TypeError('Expecting a tabular object for resultsSource')
        
        self.resultsSource = resultsSource
        self.id_column = id_column
        self.valid_ids = valid_ids
        
        #self.Index = np.zeros(self.resultsSource[list(resultsSource.keys())[0]].shape)
        self.Index = np.zeros(len(self.resultsSource))
        
        for id in valid_ids:
            self.Index += (self.resultsSource[id_column] == id)
            
            
            
        self.Index = self.Index > 0.5


@deprecated_name('concatenateFilter')
class ConcatenateFilter(TabularBase):
    _name = "Concatenation Filter"

    def __init__(self, source0, source1, *args, concatKey='concatSource'):
        """Class which concatenates two (or more) tabular data sources. The data sources should have the same keys.

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""

        self.source0 = source0
        #self.source1 = source1

        self._sources = [source0, source1, ] + list(args)
        self._concat_key = concatKey


    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        if key == self._concat_key:
            return np.hstack([i* np.ones(len(s[list(s.keys())[0]])) for i, s in enumerate(self._sources)])
            #return np.hstack((np.zeros(len(self.source0[list(self.source0.keys())[0]])), np.ones(len(self.source1[list(self.source1.keys())[0]]))))
        else:
            return np.hstack([s[key] for s in self._sources])[sl]

    def keys(self):
        return set(self.source0.keys()).intersection(*[s.keys() for s in self._sources]).union([self._concat_key,])
        

@deprecated_name('cachingResultsFilter')
class CachingResultsFilter(TabularBase):
    _name = "Caching Results Filter"
    def __init__(self, resultsSource, **kwargs):
        """Class to permit filtering of fit results - masquarades
        as a dictionary. Takes item ranges as keyword arguments, eg:
        f = resultsFliter(source, x=[0,10], error_x=[0,5]) will return
        an object that behaves like source, but with only those points with
        an x value in the range [0, 10] and a x error in the range [0, 5].

        The filter class does not have any explicit knowledge of the keys
        supported by the underlying data source."""

        if not isinstance(resultsSource, TabularBase):
            raise TypeError('Expecting a tabular object for resultsSource')
        
        self.resultsSource = resultsSource
        self.cache = {}

        #by default select everything
        self.Index = np.ones(len(self.resultsSource), dtype=bool)

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
        return list(self.resultsSource.keys())
    
    @property
    def mdh(self):
        return self.resultsSource.mdh

@deprecated_name('mappingFilter')
class MappingFilter(TabularBase):
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
        
        if not isinstance(resultsSource, TabularBase):
            warnings.warn(VisibleDeprecationWarning('Mapping filter created with something that is not a tabular object. This will be unsupported in a future release. Consider DictSource or ColumnSource instead'))

        self.resultsSource = resultsSource

        self.mappings = {}
        self.new_columns = {}
        self.variables = {}
        self.hidden_columns = []

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
        keys = list(dict.fromkeys(list(self.resultsSource.keys()) + list(self.mappings.keys()) + list(self.new_columns.keys())))
        for k in self.hidden_columns:
            keys.remove(k)
            
        return keys

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
        
    def set_variables(self, **kwargs):
        for k, v in kwargs.items():
            self.variables[k] = float(v)

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

        if not len(values) == len(self.resultsSource[list(self.resultsSource.keys())[0]]):
            raise RuntimeError('New column does not match the length of existing columns')

        #insert into our __dict__ object (for backwards compatibility - TODO change me to something less hacky)
        #setattr(self, name, values)

        self.new_columns[name] = values


    def setMapping(self, key, mapping):
        if type(mapping) == types.CodeType:
            self.mappings[key] = mapping
        elif isinstance(mapping, six.string_types):
            self.mappings[key] = compile(mapping, '/tmp/test1', 'eval')
        else:
            warnings.warn('setMapping should not be used to add a variable/data column', DeprecationWarning)
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
            elif vname in self.__dict__.keys(): #look for constants
                #FIXME - do we still need this now we have variables
                locals()[vname] = self.__dict__[vname]
            elif vname in self.mappings.keys(): #finally try other mappings
                #try to prevent infinite recursion here if mappings have circular references
                if not vname == key and not key in self.mappings[vname].co_names:
                    locals()[vname] = self.getMappedResults(vname, sl)
                else:
                    raise RuntimeError('Circular reference detected in mapping')

        return eval(map)

class _ChannelFilter(TabularBase):
    def __init__(self, colour_filter, channel):
        self.colour_filter = colour_filter
        self.channel = channel
        
    def __getitem__(self, keys):
        return self.colour_filter.get_channel_column(self.channel, keys)
    
    def keys(self):
        return list(self.colour_filter.keys())

@deprecated_name('colourFilter')
class ColourFilter(TabularBase):
    _name = "Colour Filter"
    def __init__(self, resultsSource, currentColour=None):
        """Class to permit filtering by colour
        """
        
        if not isinstance(resultsSource, TabularBase):
            raise TypeError('Expecting a tabular object for resultsSource')

        self.resultsSource = resultsSource
        self.currentColour = currentColour
        self.chromaticShifts = {}

        self.t_p_dye = 0.1
        self.t_p_other = 0.1
        self.t_p_background = .01

    @property
    def index(self):
        return self._index(self.currentColour)
        
    def _index(self, channel):
        colChans = self.getColourChans()
        if not channel in colChans:
            return np.ones(len(self.resultsSource[list(self.resultsSource.keys())[0]]), 'bool')
        else:
            p_dye = self.resultsSource['p_%s' % channel]

            p_other = 0 * p_dye
            p_tot = self.t_p_background * self.resultsSource['ColourNorm']

            for k in colChans:
                p_tot += self.resultsSource['p_%s' % k]
                if not channel == k:
                    p_other = np.maximum(p_other, self.resultsSource['p_%s' % k])

            p_dye = p_dye / p_tot
            p_other = p_other / p_tot

            return (p_dye > self.t_p_dye) * (p_other < self.t_p_other)


    def __getitem__(self, keys):
        return self.get_channel_column(self.currentColour, keys)
            
    def get_channel_column(self, chan, keys):
        key, sl = self._getKeySlice(keys)
        colChans = self.getColourChans()
    
        if not chan in colChans:
            return self.resultsSource[keys]
        else:
            #chromatic shift correction
            #print self.currentColour
            if chan in self.chromaticShifts.keys() and key in self.chromaticShifts[chan].keys():
                return self.resultsSource[key][self._index(chan)][sl] + self.chromaticShifts[chan][key]
            else:
                return self.resultsSource[key][self._index(chan)][sl]
            
    def get_channel_ds(self, chan):
        return _ChannelFilter(self, chan)
        

    @classmethod
    def get_colour_chans(cls, resultsSource):
        return [k[2:] for k in resultsSource.keys() if k.startswith('p_')]

    def getColourChans(self):
        return self.get_colour_chans(self)

    def setColour(self, colour):
        self.currentColour = colour

    def keys(self):
        return list(self.resultsSource.keys())

    
@deprecated_name('cloneSource')
class CloneSource(TabularBase):
    _name = "Cloned Source"
    def __init__(self, resultsSource, keys=None):
        """Creates an in memory copy of a (filtered) data source"""
        #resultsSource
        self.cache = {}
        
        if not isinstance(resultsSource, TabularBase):
            raise TypeError('Expecting a tabular object for resultsSource')

        klist = resultsSource.keys() if not keys else keys

        for k in klist:
            self.cache[k] = resultsSource[k]



    def __getitem__(self, keys):
        key, sl = self._getKeySlice(keys)
        return self.cache[key][sl]

    def keys(self):
        return list(self.cache.keys())

class AnndataSource(TabularBase):
    """Interfaces with anndata/scanpy/squidpy packages for spatialomics."""
    _name = "Anndata Source"

    def __init__(self, filename):
        try:
            import anndata as ad
        except ModuleNotFoundError:
            raise ModuleNotFoundError("No module named anndata. Please install anndata to use this source.")
        
        # backed='r' keeps us from loading the full file into memory
        # TODO: Is this the behaviour we want?
        self.res = ad.read(filename, backed='r')

        # Flatten the anndata structure as best we can to fit our tabular structure
        self._keys = list(self.res.var_names) + list(self.res.obs.keys())
        self._keys_attrs = {**{k: 'X' for k in self.res.var_names}, **{k: 'obs' for k in self.res.obs_keys()}}
        for k, v in self.res.obsm.items():
            n_var = v.shape[1]
            if k == "spatial" or k == "spatial3d":
                if n_var == 2:
                    proposed_keys = ['x', 'y']
                elif n_var == 3:
                    proposed_keys = ['x', 'y', 'z']
                else:
                    print(f"Could not load spatial data. Unsupported number of dimensions {n_var}.")
                    proposed_keys = None
                
                for i, kk in enumerate(proposed_keys):
                    count = 0
                    while proposed_keys[i] in self._keys:
                        proposed_keys[i] = f"{kk}{count}"
                        count += 1
                    self._keys += [proposed_keys[i]]
                    self._keys_attrs.update({proposed_keys[i]: {'obsm': {k: i}}})
            else:
                self._keys += [f"{k}_{j}" for j in range(n_var)]
                self._keys_attrs.update({f"{k}_{j}": {'obsm': {k: j}} for j in range(n_var)})

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        key, sl = self._getKeySlice(key)
        if key not in self._keys:
            raise KeyError('Key (%s) not found' % key)
            
        print(f"Getting {key}")
        
        if self._keys_attrs[key] == "X":
            x = self.res[sl, key].X
            if isinstance(x, np.ndarray):
                return x.squeeze()
            return x.toarray().squeeze()
        elif isinstance(self._keys_attrs[key], dict):
            first_key = next(iter(self._keys_attrs[key]))
            if first_key == "obsm":
                d = self._keys_attrs[key][first_key]
                dkey = next(iter(d))
                return getattr(self.res, first_key)[dkey][:,d[dkey]]
            else:
                raise KeyError(f"Unknown subkeys starting from {first_key}.")
        else:
            return getattr(self.res, self._keys_attrs[key])[key][sl].to_numpy().squeeze()

    def getInfo(self):
        return f"Anndata Data Source\n\n {self.res.X.shape[0]} points"

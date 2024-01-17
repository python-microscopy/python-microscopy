
import six

class RaggedBase(object):
    """Base class for ragged or free-form objects in recipes. Looks like a read-only list of
    arbitrary objects. Objects should be json serializable (ie simple combinations of base types)."""
    
    def __init__(self, mdh=None):
        self.mdh = mdh
        
    def __getitem__(self, item):
        """Implement this in derived classes to mimic list semantics"""
        raise NotImplementedError
    
    def __len__(self):
        """Implement this in derived classes to mimic list semantics"""
        raise NotImplementedError

    def _jsify(self, obj):
        import json
        import numpy as np
        """call a custom to_JSON method, if available"""
        #if isinstance(obj, np.integer):
        #    return int(obj)
        #elif isinstance(obj, np.number):
        #    return float(obj)
        if isinstance(obj, np.generic):
            return obj.tolist()
    
        try:
            return obj.to_JSON()
        except AttributeError:
            return obj
    
    def to_json(self):
        #TODO - write me
        import json
        return json.dumps([self._jsify(o) for o in self])
    
    def to_hdf(self, filename, tablename, mode='w', metadata=None):
        #TODO - write me / re-evaluate. This should be cluster aware and use h5r file. Ragged array logic belongs in h5rfile
        from PYME.IO import h5rFile
        import json
        
        with h5rFile.H5RFile(filename, mode) as h5f:
            for item in self:
                item_js = self._jsify(item)
                if not isinstance(item_js, str):
                    item_js = json.dumps(item_js)
                h5f.appendToTable(tablename, item_js.encode())

            # handle metadata
            if metadata is not None:
                h5f.updateMetadata(metadata)

class RaggedCache(RaggedBase):
    def __init__(self, iterable=None, mdh=None):
        if iterable:
            self._data = list(iterable)
        else:
            self._data = list()
            
        RaggedBase.__init__(self, mdh)
        
    def __getitem__(self, item):
        return self._data[item]
    
    def __len__(self):
        return len(self._data)


class RaggedJSON(RaggedCache):
    def __init__(self, filename, mdh=None):
        import json
        
        with open(filename, 'r') as f:
            data = list(json.loads(f.read()))
        
        RaggedCache.__init__(self, data, mdh)
     
    
class RaggedVLArray(RaggedBase):
    def __init__(self, h5f, tablename, mdh=None, copy=False):
        """
        Ragged type which wraps an HDF table variable-length array
        
        Parameters
        ----------
        h5f : HDF table or str
            Either an open HDF table instance, or a str of the filepath to open one
        tablename : str
            Name of the table to open and wrap (beyond root, i.e. h5f.root.tablename
        mdh : PYME.MetaDataHandler.MDHandlerBase or derived class
            Metadata to initialize RaggedVLArray with. If None, will be initialized with blank metadata
            
        copy: load entire data into memory so we can close the original file
            
        """
        RaggedBase.__init__(self, mdh)

        if isinstance(h5f, six.string_types):
            import tables
            h5f = tables.open_file(h5f)
            self._h5file = h5f  # if we open it, grab a reference so we can close it later
            self._own_hdf = True
        else:
            self._own_hdf = False

        self._data = h5f.get_node(h5f.root, tablename)
        
        if copy:
            self._data = self._data[:]
            if self._own_hdf:
                self._h5file.close()
                

    def __del__(self):
        """
        Make sure we close our h5 file if we opened one. If it was created outside the scope of this class, it should be
        handled elsewhere.

        """
        try:
            if self._own_hdf:
                self._h5file.close()
        except AttributeError:
            pass
    
    def __getitem__(self, item):
        import json
        
        return json.loads(self._data[item].decode())
    
    def __len__(self):
        return len(self._data)
        

    
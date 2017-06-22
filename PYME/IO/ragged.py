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
    
    def to_hdf(self, filename, tablename):
        #TODO - write me / re-evaluate. This should be cluster aware and use h5r file. Ragged array logic belomgs in h5rfile
        from PYME.IO import h5rFile
        
        with h5rFile.H5RFile(filename, 'w') as h5f:
            for item in self:
                h5f.appendToTable(tablename, self._jsify(item))

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
    def __init__(self, filename, tablename):
        RaggedBase.__init__(self)
        
        import tables
        
        self._h5f = tables.open_file(filename)
        
        self._data = self._h5f.get_node(self._h5f.root, tablename)
        
        #raise NotImplementedError
    
    def __getitem__(self, item):
        import json
        
        return json.loads(self._data[item])
    
    def __len__(self):
        return len(self._data)
        

    
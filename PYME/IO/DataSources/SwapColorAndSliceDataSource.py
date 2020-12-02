from .BaseDataSource import BaseDataSource, DefaultList
import numpy as np


class DataSource(BaseDataSource):
    moduleName = 'SwapColorAndSliceDataSource'
    def __init__(self, datasource):
        if len(datasource.shape) < 4:
            raise ValueError('no')
        self._datasource = datasource
        self.sizeC = self._datasource.shape[3]
        
    def _getNumFrames(self):
        self._datasource.shape[3]
    
    @property
    def shape(self):
        return DefaultList(self.getSliceShape() + (self.sizeC, int(self.getNumSlices()/self.sizeC)))
    
    def getSlice(self, ind):
        return self[:,:,ind % self.shape[2], ind // self.shape[2]].squeeze()
    
    def __getitem__(self, keys):
        keys = list(keys)
        if len(keys) == 3:
            keys.insert(2, slice(None))
        elif len(keys) == 4:  # swap
            keys[3], keys[2] = keys[2], keys[3]
        
        raw = self._datasource[keys]

        try:
            return np.swapaxes(raw, 2, 3)
        except np.AxisError:
            return raw

    
    def getSliceShape(self):
        return self._datasource.getSliceShape()
        
    def getNumSlices(self):
        """supposedly this is the product of c and t, so doesn't change

        Returns
        -------
        [type]
            [description]
        """
        return self._datasource.getNumSlices()

    def getEvents(self):
        return self._datasource.getEvents()

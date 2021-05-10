from .BaseDataSource import XYTCDataSource
import numpy as np

class DataSource(XYTCDataSource):
    moduleName = 'RandomDataSource'

    def __init__(self, pix_r, pix_c, length=100):
        self.pix_r = pix_r
        self.pix_c = pix_c
        self.length = length
        self.data = np.random.randint(1, np.iinfo(np.int16).max, (self.pix_r, self.pix_c, self.length), np.int16)

    def getSlice(self, ind):
        return self.data[:, :, ind]

    def getSliceShape(self):
        return (self.pix_r, self.pix_c)

    def getNumSlices(self):
        return self.length

    def getEvents(self):
        return []

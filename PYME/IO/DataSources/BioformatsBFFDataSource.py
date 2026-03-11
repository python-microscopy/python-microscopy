
import numpy as np
from bffile import BioFile
from .BaseDataSource import XYTCDataSource
import logging

logger = logging.getLogger(__name__)


class DataSource(XYTCDataSource):
    moduleName = 'BioformatsBFFDataSource'

    def __init__(self, filename, series=0):
        self.filename = filename
        self.series_num = series

        # Keep the file open for the lifetime of the DataSource so the lazy
        # array can read
        self._bf = BioFile(filename)
        self._bf.open()
        self._arr = self._bf.as_array(series=series)  # LazyBioArray (T,C,Z,Y,X), squeezed

        # Use OME metadata for dimension sizes (avoids squeezing ambiguity)
        pixels = self._bf.ome_metadata.images[series].pixels
        self.sizeX = pixels.size_x
        self.sizeY = pixels.size_y
        self.sizeZ = pixels.size_z
        self.sizeT = pixels.size_t
        self.sizeC = pixels.size_c

        self.additionalDims = 'TC'

    def _tczyxs_index(self, t, c, z):
        """Build an index tuple for the lazy array, skipping squeezed dimensions.

        ``LazyBioArray.dims`` contains only the non-squeezed dimension names in
        TCZYXS order.  We include an integer index for each of T, C, Z that is
        actually present; Y and X are left as full slices so the result is a 2-D
        view.
        """
        dim_to_idx = {'T': t, 'C': c, 'Z': z}
        return tuple(dim_to_idx[d] for d in self._arr.dims if d in dim_to_idx)

    # ------------------------------------------------------------------
    # XYTCDataSource interface
    # ------------------------------------------------------------------

    def getSlice(self, ind):
        c = int(ind // (self.sizeZ * self.sizeT))
        t = int((ind // self.sizeZ) % self.sizeT)
        z = int(ind % self.sizeZ)

        key = self._tczyxs_index(t, c, z)
        plane = self._arr[key] if key else self._arr
        return np.asarray(plane).squeeze()

    def getSliceShape(self):
        return (self.sizeY, self.sizeX)

    def getNumSlices(self):
        return self.sizeC * self.sizeT * self.sizeZ

    def getEvents(self):
        return []

    def release(self):
        try:
            self._bf.close()
        except Exception:
            pass

    def __del__(self):
        self.release()


import numpy as np
from ._base import Plugin

class DriftCorrector(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        dsviewer.AddMenuItem('Processing', 'Sub-pixel Drift Correction', self.OnSubPixelDriftCorrect)


    def OnSubPixelDriftCorrect(self, event):
        """sub-pixel lateral drift correction based on the drift tracking events
        in the transmitted-light channel
        """
        #from scipy import ndimage
        #from PYME.IO.DataSources import ArrayDataSource
        #from PYME.IO.DataSources import BaseDataSource
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        from PYME.LMVis import pipeline
        from PYME.IO.DataSources import DriftCorrectDataSource

        # read lateral drift values from events
        ev_mappings, _ = pipeline._processEvents(self.image.data_xyztc, self.image.events, self.image.mdh)
        driftx = ev_mappings['driftx']
        drifty = ev_mappings['drifty']
        #dx = driftx(np.arange(1, self.image.data_xyztc.shape[3]+1))   # in pixel unit
        #dy = drifty(np.arange(1, self.image.data_xyztc.shape[3]+1))   # in pixel unit

        """
        # correct lateral drift
        drift_corrected = np.ones_like(self.image.data_xyztc[:,:,:,:,:].squeeze())
        for t in range(self.image.data_xyztc.shape[3]):
            corrected = ndimage.shift(self.image.data_xyztc.getSlice(t), shift=[-dx[t], -dy[t]], order=3, mode='nearest')
            drift_corrected[:,:,t] *= corrected

        im = BaseDataSource.XYZTCWrapper(ArrayDataSource.ArrayDataSource(drift_corrected), 'XYZTC', self.image.data_xyztc.shape[2], drift_corrected.shape[2], 1)
        im = ImageStack(im)
        im.mdh.copyEntriesFrom(self.image.mdh)
        """

        ds = DriftCorrectDataSource.XYZTCDriftCorrectSource(self.image.data_xyztc, driftx, drifty, x_scale=1.0, y_scale=1.0)
        im = ImageStack(ds[:,:,:,:,:], mdh=ds.mdh)
        ViewIm3D(im, mode=self.dsviewer.mode, glCanvas=self.dsviewer.glCanvas)


def Plug(dsviewer):
    return DriftCorrector(dsviewer)


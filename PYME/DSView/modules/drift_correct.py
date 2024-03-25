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
        from scipy import ndimage
        from PYME.Analysis import piecewiseMapping as piecewise_mapping
        from PYME.IO.DataSources import ArrayDataSource
        from PYME.IO.DataSources import BaseDataSource
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        ShiftMeasure_positions = []

        if self.image.mdh.ActiveCamera == 'PcoEdge42LT':
            print('oidic correction')
        # image from single shot mode camera, cannot use piecewiseMapping.times_to_frames
        # to match ShiftMeasure positions to the number of frames
        #FIXME
            for event in self.image.events:
                if event['EventName'] == b'PYME2ShiftMeasure':
                    positions = event['EventDescr'].decode('ascii').split(', ')
                    ShiftMeasure_positions.append(np.array([float(pos) for pos in positions]))
            ShiftMeasure_frames = np.arange(1, self.image.data_xyztc.shape[3]+1)
            print(ShiftMeasure_frames)

        else:
            print('smlm correction')
            ShiftMeasure_times= []
            for event in self.image.events:
                if event['EventName'] == b'ShiftMeasure':
                    ShiftMeasure_times.append(float(event['Time']))
                    positions = event['EventDescr'].decode('ascii').split(', ')
                    ShiftMeasure_positions.append(np.array([float(pos) for pos in positions]))
            ShiftMeasure_times = np.asarray(ShiftMeasure_times)
            ShiftMeasure_frames = piecewise_mapping.times_to_frames(ShiftMeasure_times, 
                                                                self.image.events, self.image.mdh).astype(int)
            #ShiftMeasure_frames = ShiftMeasure_frames[0: ShiftMeasure_frames.shape[0]-1]
            #print(np.shape(ShiftMeasure_positions))
            #print(ShiftMeasure_frames.shape)
    
        X, Y, Z = np.mgrid[0:self.image.data_xyztc.shape[0], 0:self.image.data_xyztc.shape[1], 0:self.image.data_xyztc.shape[2]]
    
        vx, vy, vz = self.image.voxelsize
        if vz == 0:
            vz = 1

        x0, y0, z0 = self.image.origin

        Xnm = X * vx + x0
        Ynm = Y * vy + y0
        Znm = Z * vz + z0

        print('Start correcting')
        
        drift_corrected = []
        idx = 0
        for t in ShiftMeasure_frames:
            if t <= self.image.data_xyztc.shape[3]-1:
                print(t)
                corrected = ndimage.map_coordinates(np.atleast_3d(self.image.data_xyztc[:,:,:,t,0]), 
                                          [(Xnm + ShiftMeasure_positions[idx][0]*vx - x0)/vx, 
                                           (Ynm + ShiftMeasure_positions[idx][1]*vy - y0)/vy, 
                                           (Znm + ShiftMeasure_positions[idx][2]*1000 - z0)/vz],
                                   mode='nearest', order=3)
                drift_corrected.append(corrected)
                idx += 1

        print('End correcting')

        drift_corrected = np.asarray(drift_corrected)
        drift_corrected.shape = (self.image.data_xyztc.shape[0], self.image.data_xyztc.shape[1], self.image.data_xyztc.shape[2], idx, 1)
        #print(drift_corrected.shape)
        im = BaseDataSource.XYZTCWrapper(ArrayDataSource.ArrayDataSource(drift_corrected), 'XYZTC', self.image.data_xyztc.shape[2], drift_corrected.shape[3], 1)
        im = ImageStack(im)
        #print(im.data_xyztc.shape)
        im.mdh.copyEntriesFrom(self.image.mdh)

        print('Start plotting')
        ViewIm3D(im, mode=self.dsviewer.mode, glCanvas=self.dsviewer.glCanvas)


def Plug(dsviewer):
    return DriftCorrector(dsviewer)


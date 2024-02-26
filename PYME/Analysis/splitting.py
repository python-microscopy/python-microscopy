""" Takes the analysis part of spltter.py from PYMEAcquire.

Functions for unmixing and unsplitting multichannel data that has been acquired 
using an image splitting device which splits the channels onto a single camera.
"""
import os
import numpy as np

import logging
logger=logging.getLogger(__name__)

def _get_supported_sub_roi(x, w, x0, iw, flip=0):
    '''
    Find the largest sub-ROI which is fully covered in all
    channenls for a given acquisition ROI

    Parameters
    ----------
    x : array_like, int
        The origin of the splitter channel ROIs (0-indexed)
        relative to the uncropped sensor.
    w : array_like (or constant), int
        The width of the splitter channel ROIs.
    x0 : int
        The origin of the acquisition ROI (0-indexed).  
    iw : int
        The width of the acquisition ROI / the acquired image.
    flip : array_like (or constant), int
        Whether the channel was flipped by the splitter. 
        Int masquerading as a boolean. 
    '''
    _rx0 = np.maximum(x0-x, 0)
    _rx1 = (np.minimum(x+w, iw + x0) - (x))

    #print(rx0, rx1, flip, w)
    rx0 = (1-flip)*_rx0 + flip*(w-_rx1)
    rx1 = (1-flip)*_rx1 + flip*(w-_rx0)
    #print(rx0, rx1)
    return rx0.max(), rx1.min()


class SplittingInfo(object):
    """ Common class which captures all the splitter ROI information
    
    Uses the metadata if available, otherwise falls back to legacy defaults.
    Gives priority to Splitter.Channel0ROI and Splitter.Channel1ROI, but uses
    Multiview metadata if present.
    """
    def __init__(self, mdh=None, data_shape=None, ROI=None, chanROIs = None, flip=False) -> None:
        """
        Note that the flip parameter is ignored if there is metadata
        """
        if mdh is None or data_shape is None:
            # assume we have been passed an ROI and chanROIS
            if ROI is None or chanROIs is None:
                raise RuntimeError('SplittingInfo needs either mdh and data_shape or ROI and chanROIs')
            
            self.chanROIs = chanROIs
            self.camera_roi_origin = ROI[:2]
            self.data_shape = np.array(ROI[2:]) - np.array(ROI[:2])
            self._flip = flip
        else:
            # assume we have been passed an mdh and data_shape  
            from PYME.IO.MetaDataHandler import get_camera_roi_origin
            self.mdh = mdh
            self.data_shape = data_shape

            self.camera_roi_origin = get_camera_roi_origin(mdh)
            self._flip = flip 
            self.chanROIs = None

            if 'Splitter.Channel0ROI' in mdh.getEntryNames():
                self._flip = self.mdh.get('Splitter.Flip', True)
                self.chanROIs = [mdh['Splitter.Channel0ROI'], mdh['Splitter.Channel1ROI']]

            elif 'Multiview.NumROIs' in mdh.getEntryNames():
                # we have more than 2 ROIs
                numROIs = mdh['Multiview.NumROIs']
                w, h = mdh['Multiview.ROISize']

                #print self.image.data.shape, w, h, numROIs
                self._flip = False

                if data_shape[0] == numROIs*w:
                    # ROIS have already been extracted ....
                    # we need to fudge things
                    h_ = min(h, int(data_shape[1]))

                    self.chanROIs = []
                    for i in range(numROIs):
                        x0, y0 = (i * w, 0)
                        self.chanROIs.append((x0, y0, w, h_))

                    # as we've already extracted the ROIs, the origin metadata is no longer valid
                    # so fudge a 0 offset
                    # FIXME - this will likely also bite us elsewhere
                    self.camera_roi_origin = (0,0)

                else:
                    #raw data - do the extraction ourselves
                    raise RuntimeError("data is full frame / ROIed multiview, we can't handle this at present")
                    self.chanROIs = []
                    for i in range(numROIs):
                        x0, y0 = mdh['Multiview.ROISize']
                        self.chanROIs.append((x0, y0, w, h))
            else:
                #default to legacy splitting (i.e. 2 channels split up and down on the chip, each half the chip height)
                self._flip = self.mdh.get('Splitter.Flip', True)

                w = self.mdh.get('Camera.SensorWidth', 512) #If we get here we are probably dealing with really old data which can be assumed to come from a 512x512 sensor
                h = self.mdh.get('Camera.SensorHeight', 512)
                
                self.chanROIs = [
                    (0, 0, w, h / 2),
                    (0, h / 2, 2, h / 2)
                ]
            
        self.flip_y = np.zeros(len(self.chanROIs), dtype=int)
        if self._flip:
            self.flip_y[1] = 1

        self.data_slicesx, self.data_slicesy = self._get_common_rois()

    @property
    def channel_shape(self):
        return self.data_slicesx[0].stop - self.data_slicesx[0].start, self.data_slicesy[0].stop - self.data_slicesy[0].start
        
    
    @property
    def channel_rois(self):
        return self.chanROIs
    
    @property
    def num_chans(self):
        return len(self.chanROIs)
    
    def _get_common_rois(self):
        """Get max-size common ROIs for the channels relative to the data origin"""

        x0, y0 = self.camera_roi_origin

        xs, ys, ws, hs = np.array([np.array(r) for r in self.chanROIs]).T
        rx0, rx1 = _get_supported_sub_roi(xs, ws, x0, self.data_shape[0])
        ry0, ry1 = _get_supported_sub_roi(ys, hs, y0, self.data_shape[1], self.flip_y)

        xslices = [slice(int(x + rx0 - x0), int(x + rx1 - x0), 1) for x in xs]
        yslices = [slice(int(y + h - y0 - ry0 -1), int(y + h - y0 - ry1 -1), -1) if f else slice(int(y + ry0 - y0), int(y + ry1 - y0), 1) for y, f, h in zip(ys, self.flip_y, hs)]

        return xslices, yslices

def LoadShiftField(filename = None):
    if not filename:
        import wx
        fdialog = wx.FileDialog(None, 'Select shift field',
                wildcard='*.sf;*.h5;*.h5r', style=wx.FD_OPEN)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            filename = fdialog.GetPath()
        else:
            return None

    ext = os.path.splitext(filename)[1]

    if ext in ['sf']:
        return np.load(filename)
    else:
        #try and extract shiftfield from h5 / h5r file
        try:
            import tables
            from PYME.IO.MetaDataHandler import HDFMDHandler
            
            h5file = tables.open_file(filename)
            mdh = HDFMDHandler(h5file)

            dx = mdh.getEntry('chroma.dx')
            dy = mdh.getEntry('chroma.dy')

            return [dx,dy]
        except:
            return None



def get_channel(data, splitting_info : SplittingInfo, chan=0):
    '''
    Extract a channel from an image which has been recorded using an image splitting device.

    Parameters
    ----------
    data : array_like (2D)
        The image data to extract the channel from.
    ROI : array_like (4 elements)
        The region of interest which was used to acquire the data. Channel ROIs are speficied with respect to
        the full chip, so we need to correct for any acquisition ROI.
    flip : One of False, 'left_right',  or 'up_down'
        Whether, and in which direction to flip the channel data. This is necessary if the channel was flipped by the splitter.
    chanROIs : array_like (Nx4 elements)
        The ROIs on the unclipped sensor corresponding to the individual channels. Each ROI is a 4-tuple of (x,y,w,h) where 
        x and y are the ROI origin (0-indexed) and w, h are the width and height of the ROI.
    chan : int
        The index of the channel to extract.
    chanShape : array_like (2 elements)
        Used to specify the shape of the extracted data if it should be smaller than provided for by the chanROIs spec. 
        This can be indicated if an acquisition ROI has cropped one or more of the channels (i.e. to specify the shape of the 
        largest common ROI between channels). TODO - move the logic for determining this in here?? 

    '''
    data = data.squeeze()

    return data[splitting_info.data_slicesx[chan], splitting_info.data_slicesy[chan]]


    # if ROI is None:
    #     logger.warning('No ROI specified - assuming full chip (deprecated)')
    #     ROI = [0,0,data.shape[0], data.shape[1]]

    # if chanROIs:
    #     x, y, w, h = chanROIs[chan]
    #     x1 = x -(ROI[0])
    #     y1 = y -(ROI[1])
        
    #     x = max(x1, 0)
    #     y = max(y1, 0)

    #     if chanShape:
    #         chanShape = (min(w, chanShape[0]), min(h, chanShape[1]))
    #     else:
    #         chanShape = (w, h)
        
    #     if flip == 'up_down':
    #         #print y, h + min(y1, 0), self.sliceShape
    #         y = y + h + min(y1, 0) - chanShape[1]

    #     w, h = chanShape
        
        
    # else:
    #     logger.warning('No channel ROIs specified - using deprecated backwards compatible behaviour')
        
    #     if chan == 0:
    #         x, y, w, h = 0,0, data.shape[0], int(data.shape[1] / 2)
    #     else:
    #         x, y, w, h = 0, int(data.shape[1] / 2), data.shape[0], int(data.shape[1] / 2)
    #         #print x, y
    #         return data[x:(x+w), y:(y+h)]
    
    # c = data[x:(x+w), y:(y+h)]
            
    # if flip == 'left_right':
    #     c = np.flipud(c)
    # elif flip == 'up_down':
    #     c = np.fliplf(c)


    # return c


class ShiftCorrector(object):
    '''
    Nearest pixel shift correction for a single channel.
    '''
    def __init__(self, shiftfield=None):
       self.set_shiftfield(shiftfield)
       self._idx_cache = {} 

    def set_shiftfield(self, shiftfield):
        if isinstance(shiftfield, str):
            shiftfield = LoadShiftField(shiftfield)
        
        self.shiftfield = shiftfield

        if shiftfield:
            self.spx, self.spy = shiftfield

        # clear the cache
        self._idx_cache = {}

    def correct(self, data, voxelsize, origin_nm=None):
        ''' Correct a single channel of data for chromatic shift.
        '''
        if self.shiftfield:
            X, Y = self._idx(data.shape, voxelsize, origin_nm)

            return data[X,Y]
        else:
            # no shiftfield, so no correction
            return data

    def __idx(self, data_shape, voxelsize, origin_nm=None):
        ''' Calculate the indices for a shift correction.
        '''
        if origin_nm is None:
            logger.warning('No origin specified - assuming (0,0)')
            origin_nm = (0,0)

        X, Y = np.ogrid[:data_shape[0], :data_shape[1]]
        x_ = X * voxelsize[0] + origin_nm[0]
        y_ = Y * voxelsize[1] + origin_nm[1]

        dx = self.spx.ev(x_, y_)
        dy = self.spy.ev(x_, y_)

        X = np.clip(np.round(X - dx/voxelsize[0]).astype('i'), 0, data_shape[0]-1)
        Y = np.clip(np.round(Y - dy/voxelsize[1]).astype('i'), 0, data_shape[1]-1)

        return X, Y
        
    def _idx(self, data_shape, voxelsize, origin_nm=None):
        ''' Access a cached version of the indices calculated by __idx.
        '''
        cache_key = (data_shape, voxelsize, origin_nm)
        try:
            return self._idx_cache[cache_key]
        except KeyError:
            self._idx_cache[cache_key] = self.__idx(data_shape, voxelsize, origin_nm)
            return self._idx_cache[cache_key]

    

    




class Unmixer(object):
    def __init__(self, shiftfield=None, pixelsize=70., flip=True, axis='up_down', chanROIs=None):
        self.pixelsize = pixelsize
        self.axis = axis

        # patch flip to be consistent with splitter.get_channel
        # TODO - change calling code
        if flip and axis == 'up_down':
            self.flip='up_down'
        elif flip and axis == 'left_right':
            self.flip='left_right'
        else:
            self.flip=False

        self.chanROIs = chanROIs

        self.SetShiftField(shiftfield)
        

    def SetShiftField(self, shiftField, scope=None):
       self.shift_corr = ShiftCorrector(shiftField)


    def Unmix(self, data, mixMatrix, offset, ROI=[0,0,512, 512]):
        ''' Extract channels and do linear unmixing.
        
        TODO - separate out the channel extraction, shift correction and unmixing steps.
        '''
        import scipy.linalg

        umm = scipy.linalg.inv(mixMatrix)

        dsa = data.squeeze() - offset

        si = SplittingInfo(ROI=ROI, chanROIs=self.chanROIs, flip=self.flip)

        g_ = get_channel(dsa, si, chan=0)
        r_ = get_channel(dsa, si, chan=1)
        r_ = self.shift_corr.correct(r_, (self.pixelsize, self.pixelsize), origin_nm=self.pixelsize*np.array(ROI[:2]))
        

        g = umm[0,0]*g_ + umm[0,1]*r_
        r = umm[1,0]*g_ + umm[1,1]*r_

        g = g*(g > 0)
        r = r*(r > 0)

        return [r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))]

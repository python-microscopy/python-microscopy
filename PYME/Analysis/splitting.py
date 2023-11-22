""" Takes the analysis part of spltter.py from PYMEAcquire.

Functions for unmixing and unsplitting multichannel data that has been acquired 
using an image splitting device which splits the channels onto a single camera.
"""
import os
import numpy as np

import logging
logger=logging.getLogger(__name__)

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



def get_channel(data, ROI=None, flip=False, chanROIs=None, chan=0, chanShape=None):
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

    if ROI is None:
        logger.warning('No ROI specified - assuming full chip (deprecated)')
        ROI = [0,0,data.shape[0], data.shape[1]]

    if chanROIs:
        x, y, w, h = chanROIs[chan]
        x1 = x -(ROI[0])
        y1 = y -(ROI[1])
        
        x = max(x1, 0)
        y = max(y1, 0)

        if chanShape:
            chanShape = (min(w, chanShape[0]), min(h, chanShape[1]))
        else:
            chanShape = (w, h)
        
        if flip == 'up_down':
            #print y, h + min(y1, 0), self.sliceShape
            y = y + h + min(y1, 0) - chanShape[1]

        w, h = chanShape
        
        
    else:
        logger.warning('No channel ROIs specified - using deprecated backwards compatible behaviour')
        
        if chan == 0:
            x, y, w, h = 0,0, data.shape[0], int(data.shape[1] / 2)
        else:
            x, y, w, h = 0, int(data.shape[1] / 2), data.shape[0], int(data.shape[1] / 2)
            #print x, y
            return data[x:(x+w), y:(y+h)]
    
    c = data[x:(x+w), y:(y+h)]
            
    if flip == 'left_right':
        c = np.flipud(c)
    elif flip == 'up_down':
        c = np.fliplf(c)


    return c


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

        g_ = get_channel(dsa, ROI, False, self.chanROIs, chan=0)
        r_ = get_channel(dsa, ROI, self.flip, self.chanROIs, chan=1)
        r_ = self.shift_corr.correct(r_, (self.pixelsize, self.pixelsize), origin_nm=self.pixelsize*np.array(ROI[:2]))
        

        g = umm[0,0]*g_ + umm[0,1]*r_
        r = umm[1,0]*g_ + umm[1,1]*r_

        g = g*(g > 0)
        r = r*(r > 0)

        return [r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))]
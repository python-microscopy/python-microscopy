""" Takes the analysis part of spltter.py from PYMEAcquire.

Functions for unmixing and unsplitting multichannel data that has been acquired 
using an image splitting device which splits the channels onto a single camera.
"""
import os
import numpy as np

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




class Unmixer(object):
    def __init__(self, shiftfield=None, pixelsize=70., flip=True, axis='up_down'):
        self.pixelsize = pixelsize
        self.flip = flip
        self.axis = axis
        if shiftfield:
            self.SetShiftField(shiftfield)

    def SetShiftField(self, shiftField, scope):
        #self.shiftField = shiftField
        #self.shiftFieldname = sfname
    
        if self.axis == 'up_down':
            X, Y = np.ogrid[:512, :256]
        else:
            X, Y = np.ogrid[:scope.cam.GetPicWidth()/2, :scope.cam.GetPicHeight()]

        self.X2 = np.round(X - shiftField[0](X*70., Y*70.)/70.).astype('i')
        self.Y2 = np.round(Y - shiftField[1](X*70., Y*70.)/70.).astype('i')

    def _deshift(self, red_chan, ROI=[0,0,512, 512]):
        if 'X2' in dir(self):
            x1, y1, x2, y2 = ROI

            #print self.X2.shape

            if self.axis == 'up_down':
                Xn = self.X2[x1:x2, y1:(y1 + red_chan.shape[1])] - x1
                Yn = self.Y2[x1:x2, y1:(y1 + red_chan.shape[1])] - y1
            else:
                Xn = self.X2[x1:(x1 + red_chan.shape[0]), y1:y2-1] - x1
                Yn = self.Y2[x1:(x1 + red_chan.shape[0]), y1:y2-1] - y1

            #print Xn.shape

            Xn = np.maximum(np.minimum(Xn, red_chan.shape[0]-1), 0)
            Yn = np.maximum(np.minimum(Yn, red_chan.shape[1]-1), 0)

            return red_chan[Xn, Yn]

        else:
            return red_chan


    def Unmix_(self, data, mixMatrix, offset, ROI=[0,0,512, 512]):
        import scipy.linalg
        from PYME.localisation import splitting
        #from pylab import *
        #from PYME.DSView.dsviewer_npy import View3D

        umm = scipy.linalg.inv(mixMatrix)

        dsa = data.squeeze() - offset
        g_, r_ = [dsa[roi[0]:roi[2], roi[1]:roi[3]] for roi in rois]
        
        if self.flip:
            if self.axis == 'up_down':
                r_ = np.fliplr(r_)
            else:
                r_ = np.flipud(r_)

            r_ = self._deshift(r_, ROI)

        #print g_.shape, r_.shape

        g = umm[0,0]*g_ + umm[0,1]*r_
        r = umm[1,0]*g_ + umm[1,1]*r_

        g = g*(g > 0)
        r = r*(r > 0)

#        figure()
#        subplot(211)
#        imshow(g.T, cmap=cm.hot)
#
#        subplot(212)
#        imshow(r.T, cmap=cm.hot)

        #View3D([r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))])
        return [r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))]

    def Unmix(self, data, mixMatrix, offset, ROI=[0,0,512, 512]):
        import scipy.linalg
        
        #from pylab import *
        #from PYME.DSView.dsviewer_npy import View3D

        umm = scipy.linalg.inv(mixMatrix)

        dsa = data.squeeze() - offset
        
        if self.axis == 'up_down':
            g_ = dsa[:, :int(dsa.shape[1]/2)]
            r_ = dsa[:, int(dsa.shape[1]/2):]
            if self.flip:
                r_ = np.fliplr(r_)
            r_ = self._deshift(r_, ROI)
        else:
            g_ = dsa[:int(dsa.shape[0]/2), :]
            r_ = dsa[int(dsa.shape[0]/2):, :]
            if self.flip:
                r_ = np.flipud(r_)
            r_ = self._deshift(r_, ROI)

        #print g_.shape, r_.shape

        g = umm[0,0]*g_ + umm[0,1]*r_
        r = umm[1,0]*g_ + umm[1,1]*r_

        g = g*(g > 0)
        r = r*(r > 0)

#        figure()
#        subplot(211)
#        imshow(g.T, cmap=cm.hot)
#
#        subplot(212)
#        imshow(r.T, cmap=cm.hot)

        #View3D([r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))])
        return [r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))]
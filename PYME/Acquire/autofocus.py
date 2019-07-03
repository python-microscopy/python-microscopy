#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from scipy import ndimage
from simple_pid import PID
import numpy as np
from scipy import optimize
import logging
logger = logging.getLogger(__name__)
import time
try:
    # get monotonic time to ensure that time deltas are always positive
    _current_time = time.monotonic
except AttributeError:
    # time.monotonic() not available (using python < 3.3), fallback to time.time()
    _current_time = time.time

class AutoFocus(object):
    def __init__(self, scope, increment=0.5):
        self.scope = scope
        self.incr = increment
        self.lastMax =0
        self.lastMaxPos = 0
        
        self.lastStep = .5
        
    def OnFrameGroup(self, **kwargs):
        im_f = self.scope.frameWrangler.currentFrame.astype('f')
        self.im_d = ndimage.gaussian_filter(im_f, 1) - ndimage.gaussian_filter(im_f, 5)
        m = self.im_d.std()#self.scope.frameWrangler.currentFrame.std()
        #m = im_f.std()
        if m > self.lastMax:
            #continue
            self.lastMax = m
            self.lastMaxPos = self.scope.state['Positioning.z']
            
        else:
            if self.incr > 0:
                #reverse direction
                self.incr = -self.incr
            else:
                #already runing backwards
                self.scope.state['Positioning.z']=self.lastMaxPos
                #self.scope.frameWrangler.WantFrameGroupNotification.remove(self.tick)
                self.scope.frameWrangler.onFrameGroup.disconnect(self.OnFrameGroup)

                print('af_done')
            
        #self.scope.SetPos(z=self.lastMaxPos + self.incr)
        self.scope.state['Positioning.z'] = self.lastMaxPos + self.incr
        
        print('af, %s' % m)
        
    def af(self, incr=0.5):
        self.lastMax = 0
        self.incr = incr
        #self.scope.frameWrangler.WantFrameGroupNotification.append(self.tick)
        self.scope.frameWrangler.onFrameGroup.connect(self.OnFrameGroup)

class LineGaussFitter(object):
    def _model_function(self, parameters, position):
        """
        1D gaussian
        Parameters
        ----------
        parameters : tuple
            fit model parameters
        distance : ndarray
            1D position array [pixel]
        Returns
        -------
        model : ndarray
        """
        amplitude, center, sigma, bx, b = parameters
        return amplitude * np.exp(-((position - center) ** 2) / (2 * sigma ** 2)) + bx * position + b

    def _error_function(self, parameters, position, data):
        """
        """
        return data - self._model_function(parameters, position)

    def _calc_guess(self, position, data):
        offset = data.min()
        amplitude = data.max() - offset
        max_ind = np.argmax(data)
        fwhm = np.sum(data > offset + 0.5 * amplitude)
        # amplitude, center, sigma, bx, b = parameters
        return amplitude, position[max_ind], fwhm / 2.355, (data[-1] - data[0]) / (position[-1] - data[0]), offset

    def fit(self, position, data):
        guess = self._calc_guess(position, data)

        (res, cov_x, infodict, mesg, resCode) = optimize.leastsq(self._error_function, guess, args=(position, data),
                                                                 full_output=1)

        # estimate uncertainties
        residuals = infodict['fvec']  # note that fvec is error function evaluation, or (data - model_function)
        try:
            # calculate residual variance, a.k.a. reduced chi-squared
            residual_variance = np.sum(residuals ** 2) / (len(position) - len(guess))
            # multiply cov by residual variance for estimating parameter variance
            errors = np.sqrt(np.diag(residual_variance * cov_x))
        except (
                TypeError,
                ValueError) as e:  # cov_x is None for singular matrices -> ~no curvature along at least one dimension
            print(str(e))
            errors = -1 * np.ones_like(res)

        return tuple(res.astype('f')), tuple(errors.astype('f'))

class FocusLockPID(PID):
    def __init__(self, scope, piezo, p=1., i=0.1, d=0.05, sample_time=0.01):
        """

        Parameters
        ----------
        scope: PYME.Acquire.Hardware.microscope.microscope
        piezo: PYME.Acquire.Hardware.Piezos.offsetPiezoREST.OffsetPiezoClient
        p: float
        i: float
        d: float
        sample_time: float
            See simple_pid.PID, but this servo does not have a tolerence on the lock position, but rather a dead-time
            of-sorts by only updating at ~regular time intervals. The correction is only changed once per sample_time.
        """
        self.scope = scope
        self.piezo = piezo

        self._fitter = LineGaussFitter()
        self.fit_roi_size = 30


        self.peak_position = 512  # default to half of the camera size

        PID.__init__(self, p, i, d, setpoint=self.peak_position, auto_mode=False)

    @property
    def lock_enabled(self):
        return self.auto_mode

    @lock_enabled.setter
    def lock_enabled(self, enable):
        self.auto_mode(enable)

    def register(self):
        self.scope.frameWrangler.onFrameGroup.connect(self.on_frame)
        self.tracking = True

    def deregister(self):
        self.scope.frameWrangler.onFrameGroup.disconnect(self.on_frame)
        self.tracking = False

    def ToggleLock(self, enable=None):
        if enable is None:
            self.set_auto_mode(~self.lock_enabled)
        else:
            self.set_auto_mode(enable)

    def ChangeSetpoint(self, setpoint=None):
        if setpoint is None:
            self.setpoint = self.peak_position
        else:
            self.setpoint = setpoint

    @property
    def fit_roi_size(self):
        return self._fit_roi_size

    @fit_roi_size.setter
    def fit_roi_size(self, roi_size):
        self._fit_roi_size = roi_size
        self._roi_position = np.arange(roi_size)

    def find_peak(self, profile):
        crop_start = np.argmax(profile) - int(0.5 * self._fit_roi_size)
        results, errors = self._fitter.fit(self._roi_position, profile[crop_start:crop_start + self._fit_roi_size])
        return results[1] + crop_start

    def on_frame(self, **kwargs):
        # get focus position
        profile = self.scope.frameWrangler.currentFrame.squeeze().sum(axis=0).astype(float)
        self.peak_position = self.find_peak(profile)

        # calculate correction
        elapsed_time =_current_time() - self._last_time
        correction = self(self.peak_position)
        # note that correction only updates if elapsed_time is larger than sample time - don't apply same correction 2x.
        if self.lock_enabled and elapsed_time > self.sample_time:
            # logger.debug('current: %f, setpoint: %f, correction :%f, time elapsed: %f' % (self.peak_position,
            #                                                                               self.setpoint, correction,
            #                                                                               elapsed_time))
            self.piezo.MoveRel(0, correction)

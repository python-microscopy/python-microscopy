
# This file is for focus locks which involve some sort of reflection off of the coverslip.
from simple_pid import PID
import numpy as np
from scipy import optimize
from PYME.util import webframework
import requests
import threading
import logging
logger = logging.getLogger(__name__)
import time
try:
    # get monotonic time to ensure that time deltas are always positive
    _current_time = time.monotonic
except AttributeError:
    # time.monotonic() not available (using python < 3.3), fallback to time.time()
    _current_time = time.time


class GaussFitter1D(object):
    """
    1D gaussian fitter for use with focus locks which either have line-cameras, or whose frames are summed alone one
    direction to create a line profile, the peak position of which indicates the current focal position.
    """
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

        (res, cov_x, infodict, mesg, res_code) = optimize.leastsq(self._error_function, guess, args=(position, data),
                                                                 full_output=1)
        if res_code < 1 or res_code > 4:
            # fit error
            logger.debug('Focus lock fit error')
            return tuple(np.zeros(5)), tuple(np.zeros(5))
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

class ReflectedLinePIDFocusLock(PID):
    """
    The hardware implementation of this focus lock is an NIR laser piped into the objective straight, but slightly off-
    center, such that some of it is reflected off of the coverslip-to-(sample immersion media) interface and collected
    by the objective. This light then passes through a beamsplitter and is imaged as a stripe onto a camera using a
    cylindrical lens. The stripe's (in our case, vertical) position on the camera therefore indicates the distance
    betweem the objective and the sample.

    This class takes camera frames, sums them along the direction parallel to the stripe, crops a part of the resulting
    1D line profile and fits it using the GaussianFitter1D class above. The center position of the fitted Gaussian is
    fed into the PID servo to determine an appropriate correction. We use an "offsetPiezo" so that we can image fields
    of view with axial coordinates relative to the coverslip rather than absolute axial position (which is influenced by
    coverslips sagging from media, etc.).
    """
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

        self._fitter = GaussFitter1D()
        self.fit_roi_size = 30


        self.peak_position = 512  # default to half of the camera size

        PID.__init__(self, p, i, d, setpoint=self.peak_position, auto_mode=False)

    @webframework.register_endpoint('/GetPeakPosition', output_is_json=False)
    def GetPeakPosition(self):
        return self.peak_position

    # @property
    # def lock_enabled(self):
    #     return self.auto_mode
    #
    # @lock_enabled.setter
    # def lock_enabled(self, enable):
    #     self.auto_mode = enable

    @webframework.register_endpoint('/LockEnabled', output_is_json=False)
    def LockEnabled(self):
        return self.auto_mode

    @webframework.register_endpoint('/EnableLock', output_is_json=False)
    def EnableLock(self):
        logger.debug('Enabling focus lock')
        self.set_auto_mode(True)

    @webframework.register_endpoint('/DisableLock', output_is_json=False)
    def DisableLock(self):
        logger.debug('Disabling focus lock')
        self.set_auto_mode(False)

    def register(self):
        self.scope.frameWrangler.onFrameGroup.connect(self.on_frame)
        self.tracking = True

    def deregister(self):
        self.scope.frameWrangler.onFrameGroup.disconnect(self.on_frame)
        self.tracking = False

    @webframework.register_endpoint('/ToggleLock', output_is_json=False)
    def ToggleLock(self):
        self.set_auto_mode(~self.auto_mode)

    @webframework.register_endpoint('/ChangeSetpoint', output_is_json=False)
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
        """

        Parameters
        ----------
        profile: np.ndarray
            1D array of the pixel intensities after summing along direction orthogonal to axis the focus-lock beam moves
            along

        Returns
        -------

        """
        crop_start = np.argmax(profile) - int(0.5 * self._fit_roi_size)
        start, stop = max(crop_start, 0), min(crop_start + self.fit_roi_size, profile.shape[0])
        results, errors = self._fitter.fit(self._roi_position[:stop - start], profile[start:stop])
        return results[1] + start

    def on_frame(self, **kwargs):
        # get focus position
        profile = self.scope.frameWrangler.currentFrame.squeeze().sum(axis=0).astype(float)
        self.peak_position = self.find_peak(profile)

        # calculate correction
        elapsed_time =_current_time() - self._last_time
        correction = self(self.peak_position)
        # note that correction only updates if elapsed_time is larger than sample time - don't apply same correction 2x.
        if self.auto_mode and elapsed_time > self.sample_time:
            # logger.debug('current: %f, setpoint: %f, correction :%f, time elapsed: %f' % (self.peak_position,
            #                                                                               self.setpoint, correction,
            #                                                                               elapsed_time))
            # self.piezo.MoveRel(0, correction)
            self.piezo.SetOffset(self.piezo.GetOffset() + correction)


class RLPIDFocusLockClient(object):
    def __init__(self, host='127.0.0.1', port=9798, name='focus_lock'):
        self.host = host
        self.port = port
        self.name = name

        self.base_url = 'http://%s:%d' % (host, port)

    @property
    def lock_enabled(self):
        return self.LockEnabled()

    def LockEnabled(self):
        response = requests.get(self.base_url + '/LockEnabled')
        return bool(response.json())

    def EnableLock(self):
        return requests.get(self.base_url + '/EnableLock')

    def DisableLock(self):
        return requests.get(self.base_url + '/DisableLock')

    def GetPeakPosition(self):
        response = requests.get(self.base_url + '/GetPeakPosition')
        return float(response.json())

    def ChangeSetpoint(self, setpoint=None):
        if setpoint is None:
            setpoint = self.GetPeakPosition()
        return requests.get(self.base_url + '/ChangeSetpoint?setpoint=%3.3f' % (setpoint,))

    def ToggleLock(self):
        if self.lock_enabled:
            self.DisableLock()
        else:
            self.EnableLock()


class RLPIDFocusLockServer(webframework.APIHTTPServer, ReflectedLinePIDFocusLock):
    def __init__(self, scope, piezo, port=9798, **kwargs):
        ReflectedLinePIDFocusLock.__init__(self, scope, piezo, **kwargs)

        server_address = ('127.0.0.1', port)
        self.port = port


        webframework.APIHTTPServer.__init__(self, server_address)
        self.daemon_threads = True

        self._server_thread = threading.Thread(target=self._thread_target)
        self._server_thread.daemon_threads = True

        self._server_thread.start()

    @property
    def lock_enabled(self):
        return self.auto_mode

    def _thread_target(self):
        try:
            logger.info('Starting piezo on 127.0.0.1:%d' % (self.port,))
            self.serve_forever()
        finally:
            logger.info('Shutting down ...')
            self.shutdown()
            self.server_close()

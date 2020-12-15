
# This file is for focus locks which involve some sort of reflection off of the coverslip.
from simple_pid import PID
import os
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
    def __init__(self, maxfev=200, min_amp=0, max_sigma=np.finfo(float).max):
        """

        Parameters
        ----------
        maxfev: int
            see scipy.optimize.leastsq argument by the same name
        min_amp : float
            minimum fit result amplitude which we are willing to accept as a
            successful fit.
        max_sigma : float
            maximum fit result sigma which we are willing to accept as a
            successful fit.
        """
        self.maxfev = maxfev
        self._min_amp = min_amp
        self._max_sigma = max_sigma

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
                                                                 full_output=True, maxfev=self.maxfev)

        success = res_code > 0 and res_code < 5 and res[0] > self._min_amp and res[2] < self._max_sigma
        return tuple(res.astype('f')), success
        
        # # estimate uncertainties
        # residuals = infodict['fvec']  # note that fvec is error function evaluation, or (data - model_function)
        # try:
        #     # calculate residual variance, a.k.a. reduced chi-squared
        #     residual_variance = np.sum(residuals ** 2) / (len(position) - len(guess))
        #     # multiply cov by residual variance for estimating parameter variance
        #     errors = np.sqrt(np.diag(residual_variance * cov_x))
        # except (
        #         TypeError,
        #         ValueError) as e:  # cov_x is None for singular matrices -> ~no curvature along at least one dimension
        #     print(str(e))
        #     errors = -1 * np.ones_like(res)
        #
        # return tuple(res.astype('f')), tuple(errors.astype('f'))

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
    def __init__(self, scope, piezo, p=1., i=0.1, d=0.05, sample_time=0.01, 
                 mode='frame', fit_roi_size=75, min_amp=0, 
                 max_sigma=np.finfo(float).max):
        """

        Parameters
        ----------
        scope: PYME.Acquire.Hardware.microscope.microscope
        piezo: PYME.Acquire.Hardware.Piezos.offsetPiezoREST.OffsetPiezoClient
        p: float
        i: float
        d: float
        sample_time: float
            See simple_pid.PID, but this servo does not have a tolerance on the lock position, but rather a dead-time
            of-sorts by only updating at ~regular time intervals. The correction is only changed once per sample_time.
        mode: str
            flag, where 'frame' queries on each frame and 'time' queries at fixed times by polling at sample_time
        fit_roi_size : int
            size of profile to crop about the peak for actual fitting, allowing
            us to fit partial profiles and save some time
        min_amp : float
            minimum fit result amplitude which we are willing to accept as a
            valid peak measurement we can use to correct the focus.
        max_sigma : float
            maximum fit result sigma which we are willing to accept as a valid
            peak measurement we can use to correct the focus.
        
        """
        self.scope = scope
        self.piezo = piezo
        # self._last_offset = self.piezo.GetOffset()

        self._lock_ok = False
        self._ok_tolerance = 5

        self.fit_roi_size = fit_roi_size
        self._fitter = GaussFitter1D(min_amp=min_amp, max_sigma=max_sigma)
        
        self.peak_position = self.scope.frameWrangler.currentFrame.shape[1] * 0.5  # default to half of the camera size
        self.subtraction_profile = None

        PID.__init__(self, p, i, d, setpoint=self.peak_position, auto_mode=False, sample_time=sample_time)

        self._mode = mode
        self._polling = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, update_mode):
        self.deregister()
        self._mode = update_mode
        self.register()

    def _poll(self):
        while self._polling:
            t = time.time()
            self.on_frame()
            # throttle this to whatever we set cycle time to
            time.sleep(max(self.sample_time - time.time() - t, 0))


    def StartPolling(self):
        self._polling = True
        self._thread = threading.Thread(target=self._poll)
        self._thread.start()

    def StopPolling(self):
        self._polling = False
        self._thread.join()

    @webframework.register_endpoint('/GetPeakPosition', output_is_json=False)
    def GetPeakPosition(self):
        return self.peak_position

    @webframework.register_endpoint('/LockEnabled', output_is_json=False)
    def LockEnabled(self):
        return self.auto_mode

    @webframework.register_endpoint('/EnableLock', output_is_json=False)
    def EnableLock(self):
        """
        Returns offset to last-known offset before enabling the lock.

        The servo is generally more robust to changing its setpoint when it is running than when you toggle it off, move
        off the setpoint, and then slam it on again. This just makes the slam small.
        """
        # make sure piezo is ready
        retry = 0
        while not self.piezo.OnTarget() and retry < 3:
            logger.debug('waiting for piezo to stop moving')
            time.sleep(0.1)
            retry += 1
        logger.debug('Enabling focus lock')
        self.set_auto_mode(True)

    @webframework.register_endpoint('/DisableLock', output_is_json=False)
    def DisableLock(self):
        self.set_auto_mode(False)
        logger.debug('Disabling focus lock')
        self.piezo.LogFocusCorrection(self.piezo.GetOffset())

    def register(self):
        if self.mode == 'time':
            self.StartPolling()
        else:
            self.scope.frameWrangler.onFrameGroup.connect(self.on_frame)

    def deregister(self):
        if self.mode == 'time':
            self.StopPolling()
        else:
            self.scope.frameWrangler.onFrameGroup.disconnect(self.on_frame)

    @webframework.register_endpoint('/ToggleLock', output_is_json=False)
    def ToggleLock(self):
        self.set_auto_mode(~self.auto_mode)

    @webframework.register_endpoint('/ChangeSetpoint', output_is_json=False)
    def ChangeSetpoint(self, setpoint=None):
        if setpoint is None:
            self.setpoint = self.peak_position
        else:
            self.setpoint = float(setpoint)

    @webframework.register_endpoint('/SetSubtractionProfile', output_is_json=False)
    def SetSubtractionProfile(self):
        """
        Set a profile to subtract before any fitting is performed on each frame
        Returns
        -------

        """
        self.subtraction_profile = self.scope.frameWrangler.currentFrame.squeeze().sum(axis=0).astype(float)


    @property
    def fit_roi_size(self):
        return self._fit_roi_size

    @fit_roi_size.setter
    def fit_roi_size(self, roi_size):
        self._fit_roi_size = roi_size
        self._roi_position = np.arange(roi_size)
    
    @webframework.register_endpoint('/LockOK', output_is_json=False)
    def LockOK(self):
        """Check whether the lock is enabled, and the lock is on target

        Returns
        -------
        bool
            lock is enabled and on-target
        """
        return self.LockEnabled() and self.lockable()
    
    def lockable(self, tolerance=None):
        """check whether the profile is being fit OK and within tolerance of the
        setpoint

        Parameters
        ----------
        tolerance : float, optional
            maximum deviation, in pixels, to be considered 'lockable', by 
            default None

        Returns
        -------
        bool
            whether the focus lock _could_ lock easily if we enabled it
        """
        return self._lock_ok and self.on_target(tolerance)
    
    def on_target(self, tolerance=None):
        """check whether the focus lock profile is within a target tolerance of
        the setpoint

        Parameters
        ----------
        tolerance : float, optional
            maximum deviation, in pixels, to be considered 'lockable', by 
            default None

        Returns
        -------
        bool
            whether the last-updated focus lock profile was sufficiently close 
            to where we want it
        """
        if tolerance == None:
            tolerance = self._ok_tolerance
        return bool(abs(self.peak_position - self.setpoint) < tolerance)
    
    @webframework.register_endpoint('/ReacquireLock', output_is_json=False)
    def ReacquireLock(self, step_size=3.):
        """Routine to call if we've lost the lock. The lock is disabled,
        objective is moved to its lowest position, and we step upwards gradually
        until we get decent fits on the profile and the profile is sufficiently
        close to where we think it should be, then we renabled. Objective will
        be sent back to its lowest position if we cannot reacquire the lock.

        Parameters
        ----------
        step_size : float, optional
            number of microns to step the objective position by when searching, 
            by default 3.
        """
        step_size = float(step_size)
        logger.debug('reacquiring lock')
        self.DisableLock()

        min_offset = self.piezo.GetMinOffset()
        max_offset = self.piezo.GetMaxOffset()

        scan_positions = np.arange(min_offset, max_offset + step_size, 
                                   step_size)
        assert len(scan_positions) > 0

        for pos in scan_positions:
            logger.debug('looking for focus, offset: %.1f' % pos)
            
            self.piezo.SetOffset(pos)
            
            time.sleep(0.3)
            if self.lockable(self._ok_tolerance):
                logger.debug('found focus, offset %.1f' % pos)
                self.EnableLock()
                return
        
        logger.debug('failed to find focus, lowering objective')
        self.piezo.SetOffset(min_offset)
    
    @webframework.register_endpoint('/DisableLockAfterAcquiring', 
                                    output_is_json=False)
    def DisableLockAfterAcquiring(self):
        self.EnableLock()  # make sure we have the lock on
        if not self.LockOK():
            import time
            logger.debug('lock not OK, pausing for 5 s')
            time.sleep(5)
            if not self.LockOK():
                logger.debug('still not OK, starting pause/reacquire sequence')
                time.sleep(5)
                self.ReacquireLock()
            else:
                logger.debug('lock OK')
        
        self.DisableLock()
    
    @webframework.register_endpoint('/DisableLockAfterAcquiringIfEnabled', 
                                    output_is_json=False)
    def DisableLockAfterAcquiringIfEnabled(self):
        """
        Helper function to allow protocols used in automated workflows to make 
        sure they have the right focal plane without barring that protocols
        use for manual imaging without the focus lock on/set up
        """
        if self.LockEnabled():
            self.DisableLockAfterAcquiring()

    def find_peak(self, profile):
        """

        Parameters
        ----------
        profile: np.ndarray
            1D array of the pixel intensities after summing along direction orthogonal to axis the focus-lock beam moves
            along

        Returns
        -------
        success : bool
            Whether the fit converged
        peak_position : float
            center position of the reflection on the camera [pix]

        """
        crop_start = np.argmax(profile) - int(0.5 * self._fit_roi_size)
        start, stop = max(crop_start, 0), min(crop_start + self.fit_roi_size, profile.shape[0])
        self._fit_results, success = self._fitter.fit(self._roi_position[:stop - start], profile[start:stop])
        return self._fit_results[1] + start, success

    def on_frame(self, **kwargs):
        # get focus position
        profile = self.scope.frameWrangler.currentFrame.squeeze().sum(axis=0).astype(float)
        if self.subtraction_profile is not None:
            peak_position, success = self.find_peak(profile - self.subtraction_profile)
        else:
            peak_position, success = self.find_peak(profile)

        if not success:
            self._lock_ok = False
            # restart the integration / derivatives so we don't go wild when we
            # eventually get a good fit again
            self.reset()
            return
        
        self.peak_position = peak_position
        self._lock_ok = True

        # calculate correction
        elapsed_time =_current_time() - self._last_time
        correction = self(self.peak_position)
        # note that correction only updates if elapsed_time is larger than sample time - don't apply same correction 2x.
        if self.auto_mode and elapsed_time > self.sample_time:
            # logger.debug('Correction: %.2f' % correction)
            # logger.debug('components %s' % (self.components,))
            self.piezo.CorrectOffset(correction)


class RLPIDFocusLockClient(object):
    def __init__(self, host='127.0.0.1', port=9798, name='focus_lock'):
        self.host = host
        self.port = port
        self.name = name

        self.base_url = 'http://%s:%d' % (host, port)
        self._session = requests.Session()

    @property
    def lock_enabled(self):
        return self.LockEnabled()

    def LockEnabled(self):
        response = self._session.get(self.base_url + '/LockEnabled')
        return bool(response.json())
    
    def LockOK(self):
        response = self._session.get(self.base_url + '/LockOK')
        return bool(response.json())

    def EnableLock(self):
        return self._session.get(self.base_url + '/EnableLock')

    def DisableLock(self):
        return self._session.get(self.base_url + '/DisableLock')

    def GetPeakPosition(self):
        response = self._session.get(self.base_url + '/GetPeakPosition')
        return float(response.json())

    def ChangeSetpoint(self, setpoint=None):
        if setpoint is None:
            setpoint = self.GetPeakPosition()
        return self._session.get(self.base_url + '/ChangeSetpoint?setpoint=%3.3f' % (setpoint,))

    def ToggleLock(self):
        if self.lock_enabled:
            self.DisableLock()
        else:
            self.EnableLock()

    def SetSubtractionProfile(self):
        return self._session.get(self.base_url + '/SetSubtractionProfile')
    
    @webframework.register_endpoint('/ReacquireLock', output_is_json=False)
    def ReacquireLock(self, step_size=3.):
        return self._session.get(self.base_url + '/ReacquireLock?step_size=%3.3f' % (step_size,))
    
    def DisableLockAfterAcquiring(self):
        return self._session.get(self.base_url + '/DisableLockAfterAcquiring')
    
    def DisableLockAfterAcquiringIfEnabled(self):
        return self._session.get(self.base_url + '/DisableLockAfterAcquiringIfEnabled')


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


class FocusLogger(object):
    _dtype = [('time', '<f4'), ('focus', '<f4')]

    def __init__(self, position_handle, log_interval=1.0):
        """
        Logs focus position (or really any float return by the function passed
        to this initialization) to and hdf file at a specified interval.

        Parameters
        ----------
        position_handle : function
            function handle to call to get position to log.
        log_interval : float, optional
            approximate time between successive logs, in seconds, by default 1.0
        """
        self._position_handle = position_handle
        self._log_file = None
        self._log_interval = log_interval
        self._poll_thread = None
        self._logging = False
        self._start_time = 0
    
    def set_interval(self, log_interval):
        self._log_interval = log_interval
    
    def ensure_stopped(self):
        """
        Stop any current logging. Note that we let the h5rFile poll thread
        do the hdf file closing
        """
        self._logging = False
        try:
            self._poll_thread.join()
        except AttributeError:
            pass
    
    def start_logging(self, log_file, log_interval=None):
        """
        Create a log file and start storing focus position values at a set time
        interval.

        Parameters
        ----------
        log_file : str
            path to create hdf file storing contents in `focus_log` table.
        log_interval : float, optional
            approximate time between successive logs, in seconds, by default 1.
        """
        from PYME.IO.h5rFile import H5RFile

        self.ensure_stopped()
        if log_interval != None:
            self.set_interval(log_interval)
            
        log_dir, log_stub = os.path.split(log_file)
        os.makedirs(log_dir, exist_ok=True)
        log_stub, ext = os.path.splitext(log_file)
        if ext != '.hdf':
            log_file = os.path.join(log_dir, log_stub + '.hdf')
        
        self._log_file = H5RFile(log_file, mode='a', 
                                 keep_alive_timeout=max(20.0, 
                                                        self._log_interval))
        self._logging = True
        self._poll_thread = threading.Thread(target=self._poll)
        logger.debug('starting focus logger')
        self._start_time = _current_time()
        self._poll_thread.start()
    
    def _poll(self):
        while self._logging:
            d = np.array([(_current_time() - self._start_time, 
                           self._position_handle())],
                         dtype=self._dtype)
            
            self._log_file.appendToTable('focus_log', d)

            time.sleep(self._log_interval)

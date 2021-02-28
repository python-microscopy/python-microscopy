
from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float
import numpy as np
from PYME.IO import tabular, MetaDataHandler
import logging

logger = logging.getLogger(__name__)


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
        amplitude, center, sigma, b = parameters
        return amplitude * np.exp(-((position - center) ** 2) / (2 * sigma ** 2)) + b

    def _error_function(self, parameters, position, data):
        """
        """
        return data - self._model_function(parameters, position)

    def _calc_guess(self, position, data):
        offset = data.min()
        p95 = np.percentile(data, 95)
        amplitude = p95 - offset
        max_ind = np.argmin(np.abs(data - p95))
        fwhm = np.sum(data > offset + 0.5 * amplitude)
        # amplitude, center, sigma, bx, b = parameters
        return amplitude, position[max_ind], fwhm / 2.355, offset

    def fit(self, position, data):
        from scipy import optimize

        guess = self._calc_guess(position, data)

        (res, cov_x, infodict, mesg, res_code) = optimize.leastsq(self._error_function, guess, args=(position, data),
                                                                 full_output=True, maxfev=self.maxfev)

        success = res_code > 0 and res_code < 5 and res[0] > self._min_amp and res[2] < self._max_sigma
        return tuple(res.astype('f')), success
        

@register_module('EstimateStackSettings')
class EstimateStackSettings(ModuleBase):
    """
    
    """
    input_stack = Input('input')
    z_bottom = Float(49.0)
    min_range = Float(3)
    step_size = Float(0)
    z_max = Float(58)
    output = Output('with_stack_settings')
    
    def execute(self, namespace):
        from scipy.ndimage import laplace
        from PYME.Analysis.piezo_movement_correction import correct_target_positions
        from PYME.recipes.processing import Threshold

        im = namespace[self.input_stack]

        z = correct_target_positions(np.arange(im.data.shape[2]), im.events, im.mdh)
        if 'Multiview.ActiveViews' in im.mdh:
            # dodge striping in the middle
            from PYME.recipes.multiview import ExtractMultiviewChannel
            lps = []
            for view in im.mdh['Multiview.ActiveViews']:
                chan = ExtractMultiviewChannel(view_number=view).apply_simple(im)
                lps.append(np.stack([laplace(chan.data[:,:,ind,0].squeeze()) for ind in range(chan.data.shape[2])], axis=2))
            lp = np.concatenate(lps, axis=0)
        else:
            lp = np.stack([laplace(im.data[:,:,ind,0].squeeze()) for ind in range(im.data.shape[2])], axis=2)

        
        otsu = Threshold(method='otsu').apply_simple(im)
        masked = otsu.data[:,:,:,0] * lp
        metric = np.sum(masked ** 2, axis=(0, 1))

        step_size = im.mdh['StackSettings.StepSize']
        next_stack = np.argmin(np.abs(z - z[z < z.min() + 0.5 * step_size][0])) + 1

        fitter = GaussFitter1D()
        res, success = fitter.fit(z[:next_stack], metric[:next_stack])
        if not success:
            raise RuntimeError('Fit did not converge')

        if z.min() > self.z_bottom:
            logger.error('stack does not go below bottom z, going to be some guesswork')
        zz = np.linspace(min(z.min(), self.z_bottom), max(z.max(), self.z_max), 100)
        fitted = fitter._model_function(res, zz)
        # plt.plot(zz, fitted)

        target_lap = fitter._model_function(res, self.z_bottom)
        top = zz[fitted > target_lap][-1]
        if top - self.z_bottom < self.min_range:
            logger.error('found focus smaller than minimum range, increasing')
            top = self.z_bottom + self.min_range

        mdh = MetaDataHandler.DictMDHandler({
            'StackSettings.StartPos': self.z_bottom,
            'StackSettings.EndPos': top
        })
        logger.debug('StartPos %.3f, EndPos %.3f' % (self.z_bottom, top))
        if self.step_size != 0:
            mdh['StackSettings.StepSize'] = self.step_size
        for k in im.mdh.keys():
            if k.startswith('Sample'):
                mdh[k] = im.mdh[k]

        vx, vy, _ = im.mdh.voxelsize_nm
        
        roi_position = tabular.DictSource({
            # set position to be center of the field of view imaged
            'x': np.asarray([1e3 * im.mdh['Positioning.x'] + vx * (im.mdh['Multiview.ROI0Origin'][0] + 0.5 * im.mdh['Multiview.ROISize'][0])]),
            'y': np.asarray([1e3 * im.mdh['Positioning.y'] + vy * (im.mdh['Multiview.ROI0Origin'][1] + 0.5 * im.mdh['Multiview.ROISize'][1])])
        })

        roi_position.mdh = mdh

        namespace[self.output] = roi_position

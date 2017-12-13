import numpy as np
from math import log
from scipy.interpolate import SmoothBivariateSpline, Rbf
from PYME.Deconv import dec  # base deconvolution classes
import matplotlib.pyplot as plt
reload(dec)
np.seterr(divide='ignore', invalid='ignore')

#https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


class DeconvEmpiricalHist(dec.dec):
    def set_size(self, data_size):
        self.height = data_size[0]
        self.width = data_size[1]
        # self.depth  = data_size[2]

        self.shape = data_size

    def startGuess(self, data):
        return data + np.random.uniform(size=data.shape)
        return np.ones_like(data) * data.mean()

    def Afunc(self, f):
        # in this case, A is the identity
        return f

    def Ahfunc(self, f):
        # transpose of identiy is also the identity
        return f

    def Lfunc(self, f):
        # likelihood - use a smoothness constraint
        fs = f.reshape(self.height, self.width)

        # do finite difference based gradient
        # NB: we do standard finite differences along the y(intensity) axis, but
        # modify the gradient calculation slightly such that we do not
        # calculate the horizontal gradient between the 0th and 1st columns
        # (time steps) this allows the discontinuity that is due to the fact
        # that the first time bin contains everything from 0 up to the
        # frame-rate to be reconstructed in the output. A similar approach could
        # be used with the last time bin in the off-time histogram to allow a
        #  discontinuity here related to bleaching
        a = np.zeros_like(fs)

        a[:-1, :] += (fs[:-1, :] - fs[1:, :])
        a[1:, :] += (fs[1:, :] - fs[:-1, :])
        a[:, 1:-1] += (fs[:, 1:-1] - fs[:, 2:])
        a[:, 2:] += (fs[:, 2:] - fs[:, 1:-1])

        return a.astype('f').ravel()

    def Lhfunc(self, f):
        return self.Lfunc(f)

class EmpiricalHist:
    """Stores histograms of real dye kinetics data.

        Most of the data in this class comes from a JSON file containing a root key, usually named after the dye to
        which the file corresponds, with subkeys

        Parameters
        ----------
        on   :
            Bivariate (laser power and time) histogram of on times for flurophores
        off  :
            Bivariate (laser power and time) histogram of off times for flurophores

        plog :
            Boolean array determining if power dimensions for off and on are log-scaled
        tlog :
            Boolean array determining if time dimensions for off and on are log-scaled
        pmin :
            Minimum laser powers (kW/cm^2)_for on and off
        pmax :
            Maximum laser powers (kW/cm^2) for on and off
        tmin :
            Minimum on and off times
        tmax :
            Maximum on and off times

        Attributes
        ----------
        _data : JSON object
            Real dye kinetics data passed in JSON format.

        """
    def __init__(self, **kwargs):
        hist_list = ['on', 'off']
        self._plog = dict(zip(hist_list, kwargs.get('plog')))  # log2
        self._tlog = dict(zip(hist_list, kwargs.get('tlog')))  # log
        self._pmin = dict(zip(hist_list, kwargs.get('pmin')))
        self._pmax = dict(zip(hist_list, kwargs.get('pmax')))
        self._tmin = dict(zip(hist_list, kwargs.get('tmin')))
        self._tmax = dict(zip(hist_list, kwargs.get('tmax')))
        self._hist = {}
        self._hist['on'] = np.array(kwargs.get('on')).astype(float)
        self._hist['off'] = np.array(kwargs.get('off')).astype(float)
        # self._spline = {}
        # self._spline['on'] = None
        # self._spline['off'] = None

        # Sanitize histograms
        self._hist['on'][np.isinf(self._hist['on'])] = 0
        self._hist['off'][np.isinf(self._hist['off'])] = 0

        # Estimate smooth histograms
        self.hist = {}
        self.hist['on'] = self.estimate_hist('on')
        self.hist['off'] = self.estimate_hist('off')

        self.cumhist = {}
        # self.cumhist['on'] = np.cumsum(self.hist['on'], axis=0)/\
        #                      np.sum(self.hist['on'], axis=0)
        # self.cumhist['off'] = np.cumsum(self.hist['off'], axis=0)/\
        #                       np.sum(self.hist['off'], axis=0)

    def estimate_hist(self, key):
        hist = self._hist[key]

        # estimate true lambda (assuming Poisson statistics and a single value,
        # x, the expectation value for underlying population mean, \lambda is
        # E(\lambda | x) = x + 1
        est_hist = hist + 1

        # estimate the standard deviation (assuming Poisson statistics,
        # this is equal to sqrt(\lambda)
        est_sig = np.sqrt(est_hist)

        # try correcting for the number of observations at a given intensity
        # this is simply done by dividing by the sum of the counts on the
        # row. The +1 on the denominator keeps this defined in the case where
        #  we get a completely blank row. It *might* be able to be justified
        # by the same logic as is used to set p_on_est = p_on_obs + 1,
        # but this is unclear. It is also unclear whether we should be
        # summing p_on_est, or p_on_obs in the denominator.
        est_corr = est_hist / (hist.sum(1)[:, None] + 1.0)

        # correct the std. deviations in the same way as we do the actual observations
        est_corr_sig = est_sig / (hist.sum(1)[:, None] + 1.0)
        data = est_corr
        weights = 1.0 / est_corr_sig ** 2

        dc = DeconvEmpiricalHist()
        dc.set_size(data.shape)
        dec_res = dc.deconv(data, .5, 90, weights.ravel())
        return dec_res

    def trange(self, key):
        if self._tlog[key]:
            return np.exp(np.linspace(log(self._tmin[key]),
                                      log(self._tmax[key]),
                                      self._hist[key].shape[1]))
        return np.linspace(self._tmin[key], self._tmax[key],
                           self._hist[key].shape[1])

    def prange(self, key):
        if self._plog[key]:
            return np.power(2, np.linspace(log(self._pmin[key])/log(2),
                                           log(self._pmax[key])/log(2),
                                           self._hist[key].shape[0]))
        return np.linspace(self._pmin[key], self._pmax[key],
                           self._hist[key].shape[0])

    def get_time(self, pow, prob, key):
        """
        Return time estimate for a given power and cumulative probability.
        """

        # Which power are we looking at?
        p_int = np.digitize(pow, self.prange())

        # Get the time bin
        t_int = np.digitize(prob, self.cumhist[key][p_int, :])

        # now return the time
        return self.trange[t_int]

    # def at(self, p, t, key):
    #     p_int = np.digitize(p, self.prange())
    #     t_int = np.digitize(t, self.trange())
    #
    #     return self.cumhist[key][p_int, t_int]

    # def generate_spline(self, key):
    #     # Grab the desired histogram
    #     hist = self.hist[key]
    #     # Calculate cumulative histogram
    #     cumhist = np.cumsum(hist, axis=0)/np.sum(hist, axis=0)
    #     # Sanitize again
    #     cumhist[np.isnan(cumhist)] = 1
    #     # Create a spline fit of the real data in a 1x1 box
    #     x1 = np.linspace(self._pmin[key], self._pmax[key], cumhist.shape[0])
    #     y1 = np.linspace(self._tmin[key], self._tmax[key], cumhist.shape[1])
    #     x, y = np.meshgrid(x1, y1)
    #     #spline = Rbf(x.flatten(), y.flatten(), cumhist.flatten(), function='cubic', smooth=0)
    #     c = cumhist.flatten()
    #     weights = 1./np.sqrt(c+1)
    #     spline = SmoothBivariateSpline(x.flatten(), y.flatten(), c, w=weights)
    #     #spline = RectBivariateSpline(x1,y1,cumhist)
    #
    #     self._spline[key] = spline
    #
    # def at(self, p, t, key):
    #     spline = self._spline[key]
    #     if spline is None:
    #         self.generate_spline(key)
    #         spline = self._spline[key]
    #
    #     if not (is_number(p) or is_number(t) or len(p) == len(t)):
    #         res = np.empty(shape=(len(p), len(t)))
    #         for ip in p:
    #             res[:, p == ip] = spline.ev(ip, t)
    #         return res
    #
    #     return spline.ev(p, t)
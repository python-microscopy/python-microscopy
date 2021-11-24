"""
Code fo fitting signals from vesicle fusion assay. We extract profiles over time at a range of different radii from
the fusion event. This allows us to independently estimate pore openness, TIRF enhancement, diffusion, and bleaching.

maths below ...

Like the original paper, we use the diffusion greens function to calculate our model function. Unlike the original paper, we can't make the assumption that we are on axis (r=0), which is arguably a poor assumption an any case.

The 2D diffusion greens function is:

$$G(x,y, t) = e^{-\frac{x^2 + y^2}{4 \pi D t}}$$

this can be written as a function of r, the radius giving:

$$G(r, t) = e^{-\frac{r^2}{4 \pi D t}}$$

At this point, before doing the convolution with the release function, we generate a pseudo-Greens function for the intensity contained within a circle of a given radius:

$$
\begin{aligned}
G_{circ}(r^\prime, t) &= \int_0^{r^\prime} \int_0^{2 \pi} G(r)  d\theta dr\\
 &= \int_0^{r^\prime} 2 \pi r e^{-\frac{r^2}{4 \pi D t}} dr\\
 &= \left[\frac{e^{-\frac{r^2}{4 D t}}}{4 \pi D t}\right]_0^{r^\prime}\\
 &= 1 - \frac{e^{-\frac{{r^\prime}^2}{4 D t}}}{4 \pi D t}
\end{aligned}
$$

Because we're looking at different radii, we should also take the shape of the PSF into account, approximating this with a Gaussian we get:

$$ PSF(r) = \frac{1}{2\pi\sigma^2} e^{-\frac{r^2}{2\sigma^2}}$$

The integral of this within a disk follows similar maths to above, leaving us with:

$$ PSF(r') = 1 - e^{-\frac{r'^2}{2\sigma^2}} $$

Knowing that the convolution of two Gaussians is another Gaussian with the sum of the variances, we can get:

$$ G_{image}(r') = 1 - \frac{e^{-\frac{{r^\prime}^2}{4 D t + 2\sigma^2}}}{4 \pi D t}$$

NB: This step is a little fudged. It'll be approximately right, but we should really have done this convolution before we did the integration

At this point we're ready to convolve with our release function, $H(t)$. We'll assume a given pore permeability which is roughly constant throughout the course of the event (this could be a small pore, a rapidly flickering pore, or a large pore, we just assume a constant average permeability). *Note that I'm not sure we can realistically distinguish between a large pore and instant vesicle collapse (full fusion), so I've just used the one model here*. Under these assumptions, we expect exponential release, giving us:

$$ H(t) = \frac{A}{\tau_{release}} e^{-\frac{t}{\tau_{release}}}$$

where $A$ is the total amount of dye being released. Our signal would thus be:

$$ I(r^\prime, t) = \int_0^\tau G_{image}(r', \tau) H(t - \tau) d\tau$$

Unfortunately this integral isn't easily solved analytically - it was possible on axis by invoking various approximations in the last paper, but this time round I'll do it numerically instead.
"""
import numpy as np
from scipy.special import erf, jn
from PYME.Analysis.Tracking import trackUtils
from PYME.Analysis import _fithelpers
from PYME.Analysis import PSFGen
from scipy import ndimage
#reload(_fithelpers)

#####
# Helper functions for computing the radial shape of the PSF for super-critical fluorescence

def _rad_dist(d):
    X, Y = np.mgrid[0.0:d.shape[0], 0.0:d.shape[1]]
    X = X - X.mean()
    Y = Y - Y.mean()
    R = np.sqrt(X * X + Y * Y)

    out = []

    for r in range(int(np.floor(X.max()))):
        out.append((d * (R < r)).sum())

    return np.array(out)

def _genPSF(pixelsize, wavelength, NA, SCA_enhancement=1.0, vesc_size=0):
    """
    Generates a PSF using the Gibson-Lanni model.

    To approximate (very roughly) the effect of super-critical angle fluorescence, we amplify the outer 10% of the pupil
    by an enhancement factor.

    Parameters
    ----------
    pixelsize : float
        The pixel size in nm
    wavelength : float
        The wavelength in nm
    NA : float
        The numerical aperture
    SCA_enhancement : float
        How much to enhance the outer 20% of the pupil. A value of one gives a standard widefield PSF

    Returns
    -------

    """
    nyquist_pixel_size = wavelength/(2*2.3*NA)
    if pixelsize > nyquist_pixel_size:
        #data is undersampled, but we need to generate a properly sampled PSF when simulating.
        mag_factor = int(np.ceil(pixelsize/nyquist_pixel_size))
        sim_pixel_size = pixelsize/mag_factor
    else:
        mag_factor = 1
        sim_pixel_size = pixelsize

    X = sim_pixel_size * np.arange(-mag_factor*20., mag_factor*20)
    P = np.linspace(0, 1, 500)
    Z = np.array([0])

    W = np.ones_like(P)
    W[-50:] *= SCA_enhancement
    W = W / W.sum()
    
    if vesc_size > 0:
        W *= np.exp(-((P*np.pi)**2)*(vesc_size/(2.35*sim_pixel_size))**2)
        
    W= W/W.sum()

    im = PSFGen.genWidefieldPSFW(X, X, Z, P, W, k=2 * np.pi / wavelength, NA=NA, depthInSample=0)
    if mag_factor > 1:
        im = ndimage.uniform_filter(im.squeeze() * mag_factor**2, mag_factor)[::mag_factor, ::mag_factor]

    return im.squeeze()

_radCurveCache = {}
def _getPSFRadialCurve(pixelsize=260, wavelength=640, NA=1.45, SCA_enhancement=1.0, vesc_size=0):
    key = (pixelsize, wavelength, NA, SCA_enhancement, vesc_size)

    try:
        return _radCurveCache[key]
    except KeyError:
        rc = _rad_dist(_genPSF(pixelsize,wavelength,NA, SCA_enhancement, vesc_size))
        _radCurveCache[key] = rc
        return rc

def getPSFRadialValue(r, r_cal=11, SCA_enhancement=1.0, **kwargs):
    #linearly interpolate between 2 values
    sca_0 = np.floor(SCA_enhancement*10.)/10.
    sca_f = SCA_enhancement - sca_0

    rc0 = _getPSFRadialCurve(SCA_enhancement=sca_0, **kwargs)
    rc0 = rc0[r.astype('i')]/rc0[int(r_cal)]

    rc1 = _getPSFRadialCurve(SCA_enhancement=(sca_0 + .1), **kwargs)
    rc1 = rc1[r.astype('i')] / rc1[int(r_cal)]

    return sca_f*rc0 + (1-sca_f)*rc1

def docked_vesc_SCA(r, sca, **kwargs):
    return getPSFRadialValue(r, SCA_enhancement=sca, **kwargs)

def docked_vesc_(r, sig):
    return (1. - np.exp(-(r * r) / (2. * sig * sig)))

def docked_vesc__(r, sig):
    x = r*2.2/sig
    return 1.0 -jn(0, x)**2-jn(1, x)**2

def docked_vesc_l(r, sig):
    #Lorentizian
    return (2/np.pi)*np.arctan(.7*r/sig)


def diffuse_greens(t, r, D):
    t = np.maximum(t, 1e-9)
    return (t > 1e-9) * np.exp(-r ** 2 / (4 * D * (t))) / (4 * np.pi * D * (t))


def diffuse_greens_ring(t, r, D):
    return diffuse_greens_circ(t, r, D) - diffuse_greens_circ(t, r - 1, D)


def diffuse_greens_circ(t, r, D, sig=0.):
    t = np.maximum(t, 1e-9)
    #plot(exp(-(r**2)/4*D*t))
    #return (t>1e-9)*erf(r/(2*sqrt(D*t)))/(4*sqrt(pi*D*t))
    return (t > 1e-9) * (1. - np.exp(-(r ** 2) / (4. * D * t + 2 * sig ** 2)))

def diffuse_greens_circ_SCA(t, r, D, sca=1.0, sig=0., **kwargs):
    t = np.maximum(t, 1e-9)
    #plot(exp(-(r**2)/4*D*t))
    #return (t>1e-9)*erf(r/(2*sqrt(D*t)))/(4*sqrt(pi*D*t))
    vesc = docked_vesc_SCA(r, sca, **kwargs)
    diff = (1. - np.exp(-(r ** 2) / (4. * D * t + 2 * sig ** 2)))
    
    #print diff.shape, vesc.sum()
    diff_3 = (1. - np.exp(-(3 ** 2) / (4. * D * t + 2 * sig ** 2)))
    
    v_d = diff + (r > 3)*diff_3*vesc

    return (t > 1e-9) * v_d #np.minimum(vesc, diff)


def diffModel(params, t, r, sig=1.):
    A, t0, D, tau, enh, tau_bl, tdocked, background = params

    t_ = np.arange(0.0, 5 * tau)
    #print(t_)

    docked = docked_vesc(r, sig) * np.minimum(1.0, np.exp(-(t - t0) / tau))
    release = 0 * t
    for ti in t_:
        mult = np.exp(-ti / tau) - np.exp(-(ti + 1) / tau)
        #print ti, mult, diffuse_greens_circ(t-t0-ti, r, D)
        release += mult * diffuse_greens_circ(t - t0 - ti, r, D, sig)

    return (t > tdocked) * A * docked * np.exp(-(t - tdocked) / tau_bl) + A * enh * release * np.exp(
        -(t - t0) * enh / tau_bl) + background * np.pi * r ** 2


def diffMultiModel(params, t, sig=1., radii=[1., 2., 3., 5., 10, 1.0]):
    A, t0, D, tau, enh, tau_bl, tdocked, background, SCA= params
    tau = tau**2
    enh = enh**2
    tau_bl = 100 * tau_bl**2
    SCA = 1 + SCA**2

    rref = radii[-1]

    #pre-calculate weights for the convolution with the release function
    t_ = np.linspace(0.0, 5 * tau)
    dt_ = t_[1] - t_[0]
    conv_weights = np.exp(-t_ / tau) - np.exp(-(t_ + dt_) / tau)
    
    print(conv_weights.sum())

    #pre-calculate temporal component to the docked vesicle signal
    #this has 3 components - a unit step when the vesicle docks, an expoential decay due to release, and a second
    #decay due to bleaching
    docked_t = A * (t > tdocked) * np.minimum(1.0, np.exp(-(t - t0) / tau)) * np.exp(-(t - tdocked) / tau_bl)

    #precalculate bleaching component of release trace
    release_t = A* np.exp(-(t0 - tdocked)/tau_bl) * enh *np.exp(-(t - t0) * enh / tau_bl)

    #allocate an output array
    out = np.zeros([len(radii), len(t)], 'f')

    for i, r in enumerate(radii):
        #the signal for a docked vesicle (2D diffraction limited spot)
        #docked = docked_t * docked_vesc(r, sig)
        docked = docked_t * docked_vesc_SCA(r, SCA, r_cal=rref)

        out[i, :] = docked + background * np.pi * r ** 2

        #now add the release signal. We approximate the convolution with a sum
        for ti, cw in zip(t_, conv_weights):
            out[i, :] = out[i, :] + cw * diffuse_greens_circ(t - t0 - ti, r, D, sig) * release_t
            #out[i, :] = out[i, :] + cw * diffuse_greens_circ_SCA(t - t0 - ti, r, D, 3, sig) * release_t

        #out[i, :] = out[i, :]*(r-1)/rref

    return out

def docked_model(p, r, r_cal):
    return docked_vesc_SCA(r, p[0], r_cal=r_cal)

def fitModelToClump(clump, radii = [1., 2., 3., 5., 10], numLeadFrames=10, numFollowFrames=50, sig=2.0, radii_corr = 0, fig = None):
    import matplotlib.pyplot as plt

    #sp = [1, 10., 2, 3, 2., 6., 0, .7]

    t = clump['t'].astype('f')
    t = t - t[0] - numLeadFrames

    numDockedFrames = len(t) - numLeadFrames - numFollowFrames

    data = np.zeros([len(radii), len(t)], 'f')
    for i, r in enumerate(radii):
        d_i = clump['r%d' % r]
        data[i, :] = d_i - d_i[:(numLeadFrames-2)].mean()

    weights = np.ones_like(data)/data[:,:(numLeadFrames-2)].std(1)[:,None]

    radii = radii - radii_corr

    bg = 0# data[0, :numLeadFrames].mean()
    A = data[-1, numLeadFrames:(numLeadFrames+numDockedFrames)].mean()

    A -= bg
    bg /= (2*np.pi*radii[0]**2)

    sp = [1, t[-1] - numFollowFrames-1, .5, 1., 2., 6., -1, bg/A, 1.0]

    res = _fithelpers.FitModelFixed(diffMultiModel, sp, [1, 0, 1, 1, 1, 1, 0, 0, 0], data/A, t, sig, radii, eps=.1, weights=weights)

    def fprint(x):
        print(np.array2string(x, formatter={'float_kind': lambda x: "%.2f" % x}))

    fprint(np.array(sp))
    fprint(res[0])

    fits = diffMultiModel(res[0], t, sig, radii)

    if fig is None:
        plt.figure(figsize=(10, 7))

    for i, r in enumerate(radii):
        #plt.subplot(len(radii), 1, i + 1)
        c = 0.8*np.array(plt.cm.hsv(float(i)/len(radii)))
        plt.plot(t, .5*i + 1 * data[i, :] / A, 'x-', c=np.array([0.5, 0.5, 0.5, 1])*c, label='r=%d' % r)
        plt.plot(t, .5*i + fits[i, :], c=c, lw=2)

        #plot(t, fitsp[i,:])

    plt.grid()
    plt.legend()
    plt.ylabel('Normalized sum intensity')
    plt.xlabel('Time [frames]')

    A, t0, D, tau, enh, tau_bl, tdocked, background, sca = res[0]
    tau = tau ** 2
    tau_bl = 100 * tau_bl**2
    enh = enh**2
    plt.title(r'D=%3.2f, $\tau_{rel}$=%3.2f, G=%3.2f, $\tau_{bleach}$=%3.2f' % (D, tau, enh, tau_bl))


def cached_property(f):
    """returns a cached property that is calculated by function f"""

    def get(self):
        try:
            return self._property_cache[f]
        except AttributeError:
            self._property_cache = {}
            x = self._property_cache[f] = f(self)
            return x
        except KeyError:
            x = self._property_cache[f] = f(self)
            return x

    return property(get)


class FusionTrack(trackUtils.Track):
    def __init__(self, track_or_pipeline, clumpID=None, numLeadFrames=10, numFollowFrames=50, sig=1.5, startParams = {}, fitWhich = {}):
        if isinstance(track_or_pipeline, trackUtils.Track):
            #copy the track
            self.__dict__.update(track_or_pipeline.__dict__)
        else:
            #create a new track
            trackUtils.Track.__init__(self, track_or_pipeline, clumpID)

        self.numLeadFrames = numLeadFrames
        self.numFollowFrames = numFollowFrames
        self.sig = sig
        self._I = 0

        self.startParams = {'A' : 1.0, 't_fusion' : None, 'D' : .5, "tau_rel" : .1, 'G' : 5., 'tau_bleach' : 3600.,
                            't_docked' : -1.0, 'background' : 0.0, 'sca': 3.0}
        self.startParams.update(startParams)

        self.fitWhich =  {'A' : True, 't_fusion' : False, 'D' : True, "tau_rel" : True, 'G' : True, 'tau_bleach' : True,
                            't_docked' : False, 'background' : False, 'sca' : False}
        self.fitWhich.update(fitWhich)


    @cached_property
    def radii(self):
        radii = []
        for k in self.keys():
            if k.startswith('r'):
                try:
                    radii.append(int(k[1:]))
                except ValueError:
                    # the key didn't match the 'r%d' format expected
                    pass

        return np.array(sorted(radii))

    @cached_property
    def _fusion_data(self):
        t = self['t'].astype('f')
        t = t - t[0] - self.numLeadFrames

        numDockedFrames = len(t) - self.numLeadFrames - self.numFollowFrames

        radii = self.radii

        data = np.zeros([len(radii), len(t)], 'f')
        for i, r in enumerate(radii):
            d_i = self['r%d' % r]
            data[i, :] = d_i - d_i[:(self.numLeadFrames - 2)].mean()

        #we have already subtracted the background
        bg = 0# data[0, :numLeadFrames].mean()
        A = data[-1, self.numLeadFrames:(self.numLeadFrames + numDockedFrames)].mean()
        A -= bg
        
        self._I = A

        data = data/A

        return t, data

    @cached_property
    def fusion_fit(self):
        t, data = self._fusion_data

        weights = np.ones_like(data) / data[:, :(self.numLeadFrames - 2)].std(1)[:, None]

        if self.startParams['t_fusion'] is None:
            self.startParams['t_fusion'] = t[-1] - self.numFollowFrames - 1

        #sp = [1, t[-1] - self.numFollowFrames - 1, .5, 1., 2., 6., -1, 0]
        sp = self._unpack_params(self.startParams)
        fitWhich = self._unpack_fitwhich(self.fitWhich)

        #fit super-critical angle component to the docked stage only
        if fitWhich[-1]:
            numDockedFrames = len(t) - self.numLeadFrames - self.numFollowFrames
            i_mean = data[:, (self.numLeadFrames + 1):(self.numLeadFrames + numDockedFrames - 1)].mean(1)

            # sca = _fithelpers.FitModelFixed(lambda p, r: docked_vesc_SCA(r, p[0], r_cal=self.radii[-1]),
            #                                 [sp[-1],], [True,], i_mean, self.radii, eps=.1)[0][0]

            sca_ = sp[-1]**2 + 1.0

            sca = _fithelpers.FitModelFixed(docked_model, [sca_,], [True,], i_mean, self.radii.astype('i'), int(self.radii[-1]), eps=.3)[0][0]

            sp[-1] = np.sqrt(sca - 1)
            fitWhich[-1] = False

        #print sp, fitWhich, self.startParams, self.fitWhich

        res = _fithelpers.FitModelFixed(diffMultiModel, sp, fitWhich, data, t, self.sig, self.radii, eps=.1,
                                        weights=weights)

        A, t0, D, tau, enh, tau_bl, tdocked, background, sca = res[0]
        tau = tau ** 2
        tau_bl = 100 * tau_bl ** 2
        enh = enh ** 2
        sca = 1 + sca**2

        return {'A' : A, 't_fusion' : t0, 'D' : D, "tau_rel" : tau, 'G' : enh, 'tau_bleach' : tau_bl, 't_docked' : tdocked,
                'background' : background, 'sca':sca, 'I': self._I}

    def _unpack_params(self, params):
        out = [params['A'],
              params['t_fusion'],
              params['D'],
              np.sqrt(params['tau_rel']),
              np.sqrt(params['G']),
              np.sqrt(params['tau_bleach'] / 100),
              params['t_docked'],
              params['background'],
              np.sqrt(params['sca'] - 1)]

        return out

    def _unpack_fitwhich(self, params):
        out = [params['A'],
               params['t_fusion'],
               params['D'],
               params['tau_rel'],
               params['G'],
               params['tau_bleach'],
               params['t_docked'],
               params['background'],
               params['sca']]

        return out


    def plot_fusion_fit(self, fig):
        import matplotlib.pyplot as plt


        fitResults = self.fusion_fit

        params = self._unpack_params(fitResults)
        t, data = self._fusion_data
        t_ = np.arange(t[0], t[-1], .1)

        fits = diffMultiModel(params, t_, self.sig, self.radii)

        if fig is None:
            fig = plt.figure(figsize=(10, 7))

        ax = fig.gca()

        for i, r in enumerate(self.radii):
            #plt.subplot(len(radii), 1, i + 1)
            c = 0.8 * np.array(plt.cm.hsv(float(i) / len(self.radii)))
            ax.plot(t, .5 * i + 1 * data[i, :], 'x-', c=np.array([0.5, 0.5, 0.5, 1]) * c, label='r=%d' % r)
            ax.plot(t_, .5 * i + fits[i, :], c=c, lw=2)

            #plot(t, fitsp[i,:])

        ax.grid()
        ax.legend()
        ax.set_ylabel('Normalized sum intensity')
        ax.set_xlabel('Time [frames]')

        ax.set_title(r'D=%(D)3.2f, $\tau_{rel}$=%(tau_rel)3.2f, G=%(G)3.2f, $\tau_{bleach}$=%(tau_bleach)3.2f' % fitResults)

    @property
    def fusion_plot(self):
        import matplotlib.pyplot as plt
        import mpld3
        import warnings
        if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
            #mpld3 has messed with warnings - undo
            warnings.filters.pop(0)

        plt.ioff()

        fig = plt.figure(figsize=(10, 7))
        self.plot_fusion_fit(fig)

        ret = mpld3.fig_to_html(fig)
        plt.ion()
        plt.close(fig)
        return ret







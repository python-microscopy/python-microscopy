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
from PYME.Analysis import _fithelpers
reload(_fithelpers)


def docked_vesc_(r, sig):
    return (1. - np.exp(-(r * r) / (2. * sig * sig)))

def docked_vesc__(r, sig):
    x = r*2.2/sig
    return 1.0 -jn(0, x)**2-jn(1, x)**2

def docked_vesc(r, sig):
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


def diffMultiModel(params, t, sig=1., radii=[1., 2., 3., 5., 10]):
    A, t0, D, tau, enh, tau_bl, tdocked, background = params
    tau = tau**2
    enh = enh**2
    tau_bl = 100 * tau_bl**2

    rref = radii[-1]

    #pre-calculate weights for the convolution with the release function
    t_ = np.arange(0.0, 5 * tau)
    conv_weights = np.exp(-t_ / tau) - np.exp(-(t_ + 1) / tau)

    #pre-calculate temporal component to the docked vesicle signal
    #this has 3 components - a unit step when the vesicle docks, an expoential decay due to release, and a second
    #decay due to bleaching
    docked_t = A * (t > tdocked) * np.minimum(1.0, np.exp(-(t - t0) / tau)) * np.exp(-(t - tdocked) / tau_bl)

    #precalculate bleaching component of release trace
    release_t = A * enh * np.exp(-(t - t0) * enh / tau_bl)

    #allocate an output array
    out = np.zeros([len(radii), len(t)], 'f')

    for i, r in enumerate(radii):
        #the signal for a docked vesicle (2D diffraction limited spot)
        docked = docked_t * docked_vesc(r, sig)

        out[i, :] = docked + background * np.pi * r ** 2

        #now add the release signal. We approximate the convolution with a sum
        for ti, cw in zip(t_, conv_weights):
            out[i, :] = out[i, :] + cw * diffuse_greens_circ(t - t0 - ti, r, D, sig) * release_t

        #out[i, :] = out[i, :]*(r-1)/rref

    return out

def fitModelToClump(clump, radii = [1., 2., 3., 5., 10], numLeadFrames=10, numFollowFrames=50, sig=2.0, radii_corr = 0, fig = None):
    import matplotlib.pyplot as plt

    #sp = [1, 10., 2, 3, 2., 6., 0, .7]

    t = clump['t']
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

    sp = [1, t[-1] - numFollowFrames-1, .5, 1., 2., 6., -1, bg/A]

    res = _fithelpers.FitModelFixed(diffMultiModel, sp, [1, 0, 1, 1, 1, 1, 0, 0], data/A, t, sig, radii, eps=.1, weights=weights)

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
        plt.plot(t, .5*i + 1 * data[i, :] / A, 'x-', c=c, label='r=%d' % r)
        plt.plot(t, .5*i + fits[i, :], c=c, lw=2)

        #plot(t, fitsp[i,:])

    plt.grid()
    plt.legend()
    plt.ylabel('Normalized sum intensity')
    plt.xlabel('Time [frames]')

    A, t0, D, tau, enh, tau_bl, tdocked, background = res[0]
    tau = tau ** 2
    tau_bl = 100 * tau_bl**2
    enh = enh**2
    plt.title(r'D=%3.2f, $\tau_{rel}$=%3.2f, G=%3.2f, $\tau_{bleach}$=%3.2f' % (D, tau, enh, tau_bl))
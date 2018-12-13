"""
A variety of routines for fitting (and testing) if a bunch of points might represent a ring
"""

import numpy as np
from scipy.optimize import fmin
from PYME.simulation import locify

##########
# Simulation


def ring(radius=30, n_handles=18):
    t = np.linspace(0, 2 * np.pi, n_handles + 1)[:-1]
    
    x = radius * np.sin(t)
    y = radius * np.cos(t)
    
    return x, y


def ring_paint(radius=30, n_handles=18, labeling_prob=0.5, meanIntensity=1500, meanDuration=3, backGroundIntensity=100,
               meanEventNumber=2):
    from PYME.simulation import locify
    
    x, y = ring(radius, n_handles)
    
    mask = np.random.uniform(size=len(x)) < labeling_prob
    x = x[mask]
    y = y[mask]
    
    #print len(x), mask
    
    return locify.eventify2(x, y, meanIntensity, meanDuration, backGroundIntensity, meanEventNumber), len(x)


def gaussian_blob_paint(fwhm=30, n_fluors=18, meanIntensity=1500, meanDuration=3, backGroundIntensity=100,
                        meanEventNumber=2):
    from PYME.simulation import locify
    
    nf = np.random.poisson(n_fluors) + 1
    
    x = (fwhm / 2.35) * np.random.normal(size=nf)
    y = (fwhm / 2.35) * np.random.normal(size=nf)
    
    return locify.eventify2(x, y, meanIntensity, meanDuration, backGroundIntensity, meanEventNumber), len(x)


def uniform_blob_paint(radius=30, n_fluors=18, meanIntensity=1500, meanDuration=3, backGroundIntensity=100,
                       meanEventNumber=2):
    from PYME.simulation import locify
    
    nf = np.random.poisson(4 * n_fluors / np.pi) + 1
    
    x = radius * (2 * np.random.uniform(size=nf) - 1)
    y = radius * (2 * np.random.uniform(size=nf) - 1)
    
    ind = (x * x + y * y) < radius * radius
    
    x = x[ind]
    y = y[ind]
    
    return locify.eventify2(x, y, meanIntensity, meanDuration, backGroundIntensity, meanEventNumber), len(x)

# End Simulation functions
###########################

############################
# Theoretical models for radial distribution functions

from scipy.special import erf
from scipy.integrate import quad, simps


def _unif_pos(r, R, sig=5.0):
    s2s = 2 * sig * sig
    r2sig = np.sqrt(2.0) * sig
    return (1 / (np.sqrt(2 * np.pi) * R ** 2)) * (
    2 * sig * (np.exp(-r ** 2 / s2s) - np.exp(-(r - R) ** 2 / s2s)) + r * np.sqrt(2 * np.pi) * (
    erf((R - r) / r2sig) + erf(r / r2sig)))


def uniform_theory(r, R, sig=5.0):
    #s2s = 2*sig*sig
    #r2sig = np.sqrt(2.0)*sig
    
    return _unif_pos(r, R, sig) + _unif_pos(-r - 1, R, sig)
    
    #return (1/(np.sqrt(2*np.pi)*R**2))*(2*sig*(np.exp(-r**2/s2s) - np.exp(-(r-R)**2/s2s)) + r*np.sqrt(2*np.pi)*(erf((R-r)/r2sig) + erf(r/r2sig)))
    #return (1/(np.sqrt(2*np.pi)*R**2))*(2*sig*(np.exp(-r**2/s2s) + np.exp(-(-r-1)**2/s2s) - np.exp(-(r-R)**2/s2s)- np.exp(-(-r - 1 -R)**2/s2s)) +
    #                                    r*np.sqrt(2*np.pi)*(erf((R-r)/r2sig) + erf(r/r2sig) - erf((R+r - 1)/r2sig) - erf(-(r-1)/r2sig)))


def uniform_theory_cdf(r, R, sig=5.0):
    r_ = np.arange(0, 50.)
    ucd = np.cumsum(uniform_theory(r_, R, sig))
    return np.interp(r, r_, ucd)
    #return quad(uniform_theory, 0.0, float(r), (float(R), sig))


def ring_theory(r, R, sig=5.0):
    s2s = 2 * sig * sig
    
    return (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-(r - R) ** 2 / s2s)


def ring_theory_cdf(r, R, sig=5.0):
    #s2s = 2*sig*sig
    
    return (1 + erf((r - R) / (sig * np.sqrt(2)))) / 2
    #return (erf(sig*(r-R)/np.sqrt(2)) + erf(R*sig/sqrt(2)))/s2s


# end radial distribution fuctions
##################################


###################
# Ring fitting goal function

def ring_corr(x, y, x0, y0, r, sig_r=5.0):
    dr = r - np.sqrt((x-x0)**2 + (y-y0)**2)
    return np.exp(-dr**2/(2*sig_r**2)).sum()

def ring_corr_goal(p, x, y, r=30, sig_r=5):
    x0, y0 = p
    return -ring_corr(x, y, x0, y0, r, sig_r)



##############################
# Ring testing.
#
# concept is to test radial distibution against both ring and uniform CDFs

from scipy.stats import kstest

def _do_radial_analysis(x, y, error_x=5.0, disp='simple', roi_size=50):
    #find centre of mass
    xm = x.mean()
    ym = y.mean()

    
    
    #radius of points wrt center of mass
    r_com = np.sqrt((x - xm) ** 2 + (y - ym) ** 2)
    
    #print r_com.mean(), np.median(r_com)
    
    ring_radius = r_com.mean()
    uniform_radius = 1.5 * r_com.mean()
    
    #find centre of best fitting ring
    xc, yc = fmin(ring_corr_goal, [xm, ym], (x, y, ring_radius), disp=0)
    
    #radius of points with respect to the best fitting ring centre
    r_ring = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    
    sig = np.mean(error_x)
    
    #calculate probabilites
    ring_radius = r_ring.mean()
    p_ring_ring = kstest(r_ring, lambda x: ring_theory_cdf(x, ring_radius, sig))
    p_uniform_ring = kstest(r_ring, lambda x: uniform_theory_cdf(x, uniform_radius, sig))
    
    ring_radius = r_com.mean()
    p_ring_com = kstest(r_com, lambda x: ring_theory_cdf(x, ring_radius, sig))
    p_uniform_com = kstest(r_com, lambda x: uniform_theory_cdf(x, uniform_radius, sig))
    
    if disp =='full':
        import matplotlib.pyplot as plt
        r__ = np.arange(0, 50)
        #plotting
        plt.figure(figsize=(15, 3))
        plt.subplot(151)
        plt.plot(x, y, '.')
        plt.axis('equal')

        x_r, y_r = ring(radius=ring_radius, n_handles=50)

        plt.plot(x_r + xm, y_r + ym)
        plt.plot(x_r + xc, y_r + yc)
        plt.plot(x_r, y_r, ':')

        plt.axis('equal')
        plt.xlim(-roi_size, roi_size)
        plt.ylim(-roi_size, roi_size)
        plt.title('Localizations')

        plt.subplot(152)
        plt.hist(r_ring, np.linspace(0, 60, 20), alpha=0.5)
        plt.title('Best fit ring')

        plt.subplot(154)
        plt.hist(r_com, np.linspace(0, 60, 20), alpha=0.5)
        plt.title('Centre of mass')
        
        #calculate theoretical distributions for plotting
        rtc = ring_theory_cdf(r__, ring_radius, sig)
        ufc = uniform_theory_cdf(r__, uniform_radius, sig)

        plt.subplot(153)
        rs = np.sort(r_ring)
        plt.plot(rs, np.linspace(0, 1, len(rs)))

        plt.plot(r__, rtc, ':')
        plt.plot(r__, ufc, ':')

        plt.text(25, 0.45, 'p_ring=%1.1e\np_uniform=%1.1e' % (p_ring_ring[1], p_uniform_ring[1]),
             bbox=dict(facecolor='white', alpha=1))

        plt.grid()
        plt.title('Best fit ring')

        plt.subplot(155)
        rs = np.sort(r_com)
        plt.plot(rs, np.linspace(0, 1, len(rs)))

        plt.plot(r__, rtc, ':')
        plt.plot(r__, ufc, ':')

        plt.text(25, 0.45, 'p_ring=%1.1e\np_uniform=%1.1e' % (p_ring_com[1], p_uniform_com[1]),
             bbox=dict(facecolor='white', alpha=1))
        plt.grid()
        plt.title('Centre of mass')

        plt.legend(['c', 'ring', 'uniform'])
        
    elif disp =='simple':
        import matplotlib.pyplot as plt
        r__ = np.arange(0, 50)
        #plotting
        plt.figure(figsize=(15, 3))
        plt.subplot(141)
        plt.plot(x, y, '.')
        plt.axis('equal')
    
        x_r, y_r = ring(radius=ring_radius, n_handles=50)
    
        plt.plot(x_r + xm, y_r + ym)
        plt.plot(x_r + xc, y_r + yc)
        plt.plot(x_r, y_r, ':')
    
        plt.axis('equal')
        plt.xlim(-roi_size, roi_size)
        plt.ylim(-roi_size, roi_size)
        plt.title('Localizations')
    
        #plt.subplot(152)
        #plt.hist(r_ring, np.linspace(0, 60, 20), alpha=0.5)
        #plt.title('Best fit ring')
    
        #plt.subplot(154)
        #plt.hist(r_com, np.linspace(0, 60, 20), alpha=0.5)
        #plt.title('Centre of mass')
    
        #calculate theoretical distributions for plotting
        rtc = ring_theory_cdf(r__, ring_radius, sig)
        ufc = uniform_theory_cdf(r__, uniform_radius, sig)
    
        plt.subplot(142)
        rs = np.sort(r_ring)
        plt.plot(rs, np.linspace(0, 1, len(rs)))
    
        plt.plot(r__, rtc, ':')
        #plt.plot(r__, ufc, ':')
    
        plt.text(25, 0.45, 'p_ring=%1.1e' % (p_ring_ring[1],),
                 bbox=dict(facecolor='white', alpha=1))
    
        plt.grid()
        plt.title('Best fit ring')
        plt.xlabel('Radius [nm]')
        plt.ylabel('Cumulative freq')
    
        plt.subplot(143)
        rs = np.sort(r_com)
        plt.plot(rs, np.linspace(0, 1, len(rs)))
    
        #plt.plot(r__, rtc, ':')
        plt.plot(r__, ufc, ':')
    
        plt.text(25, 0.45, 'p_uniform=%1.1e' % (p_uniform_com[1]),
                 bbox=dict(facecolor='white', alpha=1))
        plt.grid()
        plt.title('Centre of mass')
        plt.xlabel('Radius [nm]')
        plt.ylabel('Cumulative freq')
    
        plt.legend(['c', 'uniform'])
        
        plt.subplot(144)
        p_r = p_ring_ring[1]/p_uniform_com[1]
        c = plt.cm.RdYlGn((np.log10(p_r) + 3) / 6)
        plt.text(0, .5, 'p_ring/p_uniform=%1.1e' % (p_r), color=c)
        plt.axis('off')
    
    return p_ring_ring[1], p_uniform_ring[1], p_ring_com[1], p_uniform_com[1]


def do_radial_anal(res, disp='simple', error_cutoff=50.):
    ind = res['fitError']['x0'] < error_cutoff
    x = res['fitResults']['x0'][ind]
    y = res['fitResults']['y0'][ind]
    error_x = res['fitError']['x0'][ind]
    
    return _do_radial_analysis(x, y, error_x, disp=disp)


def do_summary_plots(res):
    import matplotlib.pyplot as plt
    p_ring = np.sort(res[:, 0].copy())
    ratio = np.sort(res[:, 0] / res[:, 3])
    
    plt.figure()
    plt.plot(p_ring, np.linspace(0, 1, len(p_ring)))
    plt.grid()
    
    plt.ylabel('Cumulative Frequency')
    plt.xlabel('p_value for rejecting a ring')
    
    plt.figure()
    plt.plot(ratio, np.linspace(0, 1, len(p_ring)))
    plt.semilogx()
    plt.grid()
    plt.xlim(10**-3, 10**3)

    plt.ylabel('Cumulative Frequency')
    plt.xlabel('Ratio of p_ring to p_uniform')
    


def radial_analysis_tabular(data, label_key, disp=False, error_multiplier=1.0):
    labels = data[label_key]
    unique_labels = np.unique(labels[labels>0])
    
    res = []

    point_p_ring = np.zeros(len(labels), 'f')
    point_ratio = np.zeros(len(labels), 'f')
    
    for i, l in enumerate(unique_labels):
        ind = (labels == l)
        
        x = data['x'][ind]
        y = data['y'][ind]
        x = x - x.mean()
        y = y - y.mean()
        error_x = error_multiplier*data['error_x'][ind]
        
        if i < 20:
            #only display the first 20 fits
            r1 = _do_radial_analysis(x, y, error_x, disp=disp)
        else:
            r1 = _do_radial_analysis(x, y, error_x, disp=False)
            
        res.append(r1)

        p_ring = r1[0]
        ratio = r1[0]/r1[3]
        
        point_p_ring[ind] = p_ring
        point_ratio[ind] = ratio
        
    
    res = np.array(res)
        
    
    return res, point_p_ring, point_ratio
        



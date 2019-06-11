import numpy as np
import PYME.Analysis._fithelpers as fh

def get_offtime_cdf(frame_num, envelope='upper'):
    ts = np.sort(frame_num)
    
    #find the gap lengths. -1 as otherwise two consecutive frames would give a gap length of 1
    dt = np.diff(ts) - 1
    
    #just take the non-zero gaps and sort to give x values of CDF
    dt = np.sort(dt[dt>0])
    
    #y-values of CDF
    cum_frac = np.linspace(0,1-1.0/len(dt), len(dt))
    
    #this gives a stepped CDF - we want to find the envelope. Do this by finding where the x values are repeated
    if envelope == 'upper':
        ddt = np.diff(np.hstack([dt, 10000000]))
    elif envelope == 'lower':
        ddt = np.diff(np.hstack([0, dt]))
    
    x_vals = dt[ddt>0]
    y_vals = cum_frac[ddt>0]
    
    return x_vals, y_vals

def qpaint2mod(p, t):
    a, t1, t2 = p**2 #squared to enforce positivity
    
    return 1 - a*np.exp(-t/t1) - (1-a)*np.exp(-t/t2)


def qpaint1mod(p, t):
    t1 = p ** 2 #squared to enforce positivity
    
    return 1 - np.exp(-t / t1)


def fit(frame_nums, model='q2'):
    t, c = get_offtime_cdf(frame_nums)

    if (model == 'q2'):
        res = fh.FitModel(qpaint2mod, np.sqrt([0.5, 3.5, 500.]), c, t)
    
    else:
        res = fh.FitModel(qpaint1mod, np.sqrt([500., ]), c, t)
    
    return res[0]**2


def plot_and_fit(frame_nums, model='q2'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    
    t, c = get_offtime_cdf(frame_nums)
    
    plt.plot(t, c, 'x')

    t_ = np.logspace(0, np.log10(t.max()))
    
    if (model == 'q2'):
        res = fh.FitModel(qpaint2mod, np.sqrt([0.5, 3.5, 500.]), c, t)
        
        f = qpaint2mod(res[0], t_)
    else:
        res = fh.FitModel(qpaint1mod, np.sqrt([500.,]), c, t)
        
        f = qpaint1mod(res[0], t_)
    
    plt.plot(t_, f)
    plt.grid()
    
    print(res[0]**2)
    
    #plt.semilogy()

    #plt.figure()
    plt.subplot(122)
    plt.plot(t, 1-c, 'x')
    plt.plot(t_, 1- f)
    
    plt.semilogy()
    
    plt.ylim((1-c).min(), 1)
    
    plt.grid()
    
    return res[0]**2
    
    
    #plt.plot()



        

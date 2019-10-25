"""

Additional kinetic fitting  stuff from Ben Rollins.

TODO - move out of core to plugin??? Esp. as technical accuracy of some fits debatable and duplicates more accurate
functions in photophysics / kinModels.

The following need to be resolved to ensure it's useful in the core:

- Fitting a simple exponential to a histogram of lifetimes behaves poorly for short lifetimes (< ~10 frames). The correct
model function needs to take bin width into account.
- Using unweighted least squares to fit an exponential will lead to undue influence of the very short time bins, and
generally poor performance.

"""

import numpy as np
from scipy.special import erf, gamma
from scipy.misc import comb

import wx #FIXME - not really appropriate for low level function module
from PYME.recipes.traits import Int # FIXME - ditto - is this needed?
import matplotlib as plt

USE_GUI = True

import scipy.optimize as optimize

def FitModel_N(res_fxn, startParameters, x_data, y_data): # use 'N' as a way to deteremine what fit you want to use, don't ned to specify start params here
    min_fxn = lambda p, x, y: np.mean(res_fxn(p, x, y)**2)
    p_guess = optimize.minimize(min_fxn, startParameters, args=(x_data, y_data), method='nelder-mead', options={'xtol':1e-8, 'disp':True})
    print('p_guess: ', p_guess.x)
    return optimize.least_squares(res_fxn, p_guess.x, args=(x_data, y_data))

def FitModel_NB(res_fxn, startParameters, x_data, y_data, prob): # use 'N' as a way to deteremine what fit you want to use, don't ned to specify start params here
    min_fxn = lambda p, x, y, prob: np.mean(res_fxn(p, x, y, prob)**2)
    p_guess = optimize.minimize(min_fxn, startParameters, args=(x_data, y_data, prob), method='nelder-mead', options={'xtol':1e-8, 'disp':True})
    print('p_guess: ', p_guess.x)
    return optimize.least_squares(res_fxn, p_guess.x, args=(x_data, y_data, prob))


def sefm(x, a, b): # single exponential fit model
    y = a * np.exp(-b * x)
    return y


def se_diff_l2_mean(p0, x, y):
    return np.mean(tfoef(p0, x, y) ** 2)


def defm(x, a, b, c, d): # double exponential fit model
    y = a * np.exp(-b * x) + c * np.exp(-d * x)
    return y


def gfm(x, a):
    return a * ((1 - a) ** x)


def tsoef(t, x, y): # == test second order exponential fit
    return t[0] * np.exp(-t[1] * x) + t[2] * np.exp(-t[3] * x) - y


def tfoef(t, x, y): # == test first order exponential fit
    return t[0] * np.exp(-t[1] * x) - y


def grff(t, x, y): #geometric residual fitting function
    return (t[0] * ((1 - t[0]) ** x)) - y


def nbrff(t, x, y, p): # negative binomial fitting function
    # return ((gamma(t[0] + x - 1) / (gamma(t[0] - 1) * gamma(x + 1))) * (t[1] ** t[0]) * ((1 - t[1]) ** x)) - y
    # return ((gamma(x+1)/(gamma(t[0]-1)*gamma(x-t[0]+1)))*(t[1]**t[0])*((1-t[1])**(x-t[0]+1))) - y
    return (comb(x, t[0] - 1, exact=False) * (p ** t[0]) * ((1 - p) ** (x - t[0] + 1))) - y


# def nbrff(t, x, y): # negative binomial fitting function
#     return ((math.factorial((t[0] + np.int(x)))/(math.factorial(t[0]) * math.factorial(np.int(x))))*(t[1]**t[0])*((1-t[1])**x)) - y

def nbmf(x, N, p): # negative binomial modelling function
    y = comb(x, N - 1, exact=False) * (p ** N) * ((1 - p) ** (x - N + 1))
    return y


def gen_nb_testdata(x, N, p, noise=0, n_outliers=1, random_state=0):
    y = (gamma(N + x - 1) / (gamma(N - 1) * gamma(x + 1))) * (p ** N) * ((1 - p) ** x)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(x.size)
    outliers = rnd.randint(0, x.size, n_outliers)
    error[outliers] *= 10
    
    return y + error


def neg_binom_histfitting(colourFilter, metadata, cluster_ids, blink_ids, N_bins,
                          P): #geometric and negative binomial fitting
    from PYME.IO import tabular
    # N_bins =
    # frame_duration = metadata.getEntry('Camera.IntegrationTime')
    # frame_duration = metadata.getEntry('Camera.CycleTime')
    
    """
    Important to note: blinks per cluster does nto exist as callable value in pipeline, have to create in this fxn call
    data from pipeline needed: dbscabcluster, blink id?
    """
    # I = np.argsort(colourFilter[cluster_ids])
    # print(pl['dbscanClustered'][I])
    # bpc = np.zeros_like(np.unique(colourFilter[cluster_ids]))# blinks per cluster
    # cid = np.arange(1, max(colourFilter[cluster_ids])+1, 1)
    # print('cid vec', cid)
    # print('just before loop', len(np.unique(colourFilter[cluster_ids])))
    # for i in range(1, len(np.unique(colourFilter[cluster_ids])) + 1):
    #     nblinks = len(colourFilter[blink_ids][I][colourFilter[cluster_ids] == i])
    #     bpc[i-1] = nblinks
    # print(i, nblinks)
    
    
    _, bpc_2 = np.unique(colourFilter[cluster_ids], return_counts=True)
    # print('any points where bps = 0?', np.where(bpc_2 == 0))
    
    
    
    # plt.hist(bpc_2, bins=20)
    # plt.figure()
    # plt.hist(bpc, bins=20)
    
    vals, bin_edges = np.histogram(bpc_2, bins=np.max(bpc_2) / 5, density=True)
    # print('hist stuff', vals, bin_edges)
    # x_data = on_times
    y_hist = vals
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]
    x_hist = (bin_starts + bin_ends) / 2
    # params_on = np.array([.1])
    # geofit = FitModel_N(grff, params_on, x_hist, y_hist)
    
    
    # print('p value from geo fit', geofit.x)
    # plt.figure()
    # plt.plot(np.linspace(min(x_hist), max(x_hist), 100), gfm(np.linspace(min(x_hist), max(x_hist), 100), geofit.x))
    
    
    nb_params = np.array([2.0])
    # print('nb_params: ', nb_params)
    nbfit = FitModel_NB(nbrff, nb_params, x_hist, y_hist, P)
    N = nbfit.x[0]
    # p = nbfit.x[1]
    x_fit = np.linspace(min(bin_edges), max(bin_edges), max(bin_edges) * 10)
    y_fit = nbmf(x_fit, N, P)
    
    hist_datasource = np.rec.fromarrays((y_hist, x_hist), dtype=[('y_hist', '<f4'), ('x_hist', '<f4')])
    
    hist_data = tabular.RecArraySource(hist_datasource.view(np.recarray))
    # filt_off = tabular.recArrayInput(off_hist_datasource.view(np.recarray))
    """
    have fit for both N and p simultaneously, also want to make one that fits based on p as found in geo fit
    maybe make a button to determine which one?
    """
    # geo_cov = np.linalg.inv(np.matmul(geofit.jac.T, geofit.jac)) * np.mean(
    #     (geofit.fun * geofit.fun).sum())
    nb_cov = np.linalg.inv(np.matmul(nbfit.jac.T, nbfit.jac)) * np.mean(
        (nbfit.fun * nbfit.fun).sum())
    # fitErrors_geo = np.sqrt(np.diag(geo_cov))
    fitErrors_nb = np.sqrt(np.diag(nb_cov))
    
    # print('bpc', len(bpc), max(bpc), min(bpc), len(np.unique(bpc)), bpc)
    if USE_GUI:
        # plt.bar(bin_starts, vals, width=bin_starts[1]-bin_starts[0], alpha=.4)
        # plt.plot(x_fit, y_fit)
        # xx3 = max(plt.xlim())
        # yy3 = max(plt.ylim())
        # cv = min(plt.xlim())
        # plt.text(((xx3-cv)*.1) + cv, yy3 * .9, 'y = p*(1-p)^x')
        # # plt.text(xx3 * .3, yy3 * .45, 'p= %5.2f' % (geofit.x))
        # plt.title('Geometric fit with p = %5.2f' % (geofit.x))
        
        # negative binomial fitting
        plt.figure()
        plt.bar(bin_starts, vals, width=bin_starts[1] - bin_starts[0], alpha=.4)
        plt.plot(x_fit, y_fit)
        plt.xlim([0, 200])
        xx4 = max(plt.xlim())
        yy4 = max(plt.ylim())
        cv2 = min(plt.xlim())
        plt.text(((xx4 - cv2) * .3) + cv2, yy4 * .7, 'Number of molecules N = %5.2f +/- %5.2f' % (N, fitErrors_nb))
        plt.title(r'$2xmEos3.2-CAAX$')
        plt.xlabel('Number of blinks per molecule')
        plt.ylabel('Probability')
        # plt.text(xx2 * .3, yy2 * .45, )
    """
    hist_data = hist_data
    params_geo = geofit.x
    params_nb = nbfit.x
    fitErrors_geo =
    fitErrors_nb =
    fit_eqn_geo =
    fit_eqn_nb =
    """
    return hist_data, nbfit.x, fitErrors_nb#, fit_eqn_geo, fit_eqn_nb


def g_histfitting(colourFilter, metadata, cluster_ids, blink_ids, N_bins): #geometric and negative binomial fitting
    from PYME.IO import tabular
    # N_bins =
    # frame_duration = metadata.getEntry('Camera.IntegrationTime')
    # frame_duration = metadata.getEntry('Camera.CycleTime')
    
    """
    Important to note: blinks per cluster does nto exist as callable value in pipeline, have to create in this fxn call
    data from pipeline needed: dbscabcluster, blink id?
    """
    # I = np.argsort(colourFilter[cluster_ids])
    # print(pl['dbscanClustered'][I])
    # bpc = np.zeros_like(np.unique(colourFilter[cluster_ids]))# blinks per cluster
    # cid = np.arange(1, max(colourFilter[cluster_ids])+1, 1)
    # print('cid vec', cid)
    # print('just before loop', len(np.unique(colourFilter[cluster_ids])))
    # for i in range(1, len(np.unique(colourFilter[cluster_ids])) + 1):
    #     nblinks = len(colourFilter[blink_ids][I][colourFilter[cluster_ids] == i])
    #     bpc[i-1] = nblinks
    # print(i, nblinks)
    
    _, bpc_2 = np.unique(colourFilter[cluster_ids], return_counts=True)
    print('any points where bps = 0?', np.where(bpc_2 == 0))
    
    plt.hist(bpc_2, bins=20)
    plt.figure()
    # plt.hist(bpc, bins=20)
    # binning = np.linspace(1, np.max(bpc_2), np.max(bpc_2))
    vals, bin_edges = np.histogram(bpc_2, bins=np.max(bpc_2) / 5, density=True)
    print('hist stuff', vals, bin_edges)
    # x_data = on_times
    y_hist = vals
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]
    x_hist = (bin_starts + bin_ends) / 2
    params_on = np.array([.1])
    geofit = FitModel_N(grff, params_on, x_hist, y_hist)
    
    # print('p value from geo fit', geofit.x)
    # plt.figure()
    # plt.plot(np.linspace(min(x_hist), max(x_hist), 100), gfm(np.linspace(min(x_hist), max(x_hist), 100), geofit.x))
    
    
    # nb_params = np.array([2.0, geofit.x[0]])
    # print('nb_params: ', nb_params)
    # nbfit = FitModel_N(nbrff, nb_params, x_hist, y_hist)
    # N = nbfit.x[0]
    # p = nbfit.x[1]
    x_fit = np.linspace(min(bin_edges), max(bin_edges), max(bin_edges) * 10)
    y_fit = gfm(x_fit, geofit.x)
    #
    hist_datasource = np.rec.fromarrays((y_hist, x_hist), dtype=[('y_hist', '<f4'), ('x_hist', '<f4')])
    #
    hist_data = tabular.RecArraySource(hist_datasource.view(np.recarray))
    # filt_off = tabular.recArrayInput(off_hist_datasource.view(np.recarray))
    """
    have fit for both N and p simultaneously, also want to make one that fits based on p as found in geo fit
    maybe make a button to determine which one?
    """
    geo_cov = np.linalg.inv(np.matmul(geofit.jac.T, geofit.jac)) * np.mean((geofit.fun * geofit.fun).sum())
    # nb_cov = np.linalg.inv(np.matmul(nbfit.jac.T, nbfit.jac)) * np.mean(
    #     (nbfit.fun * nbfit.fun).sum())
    fitErrors_geo = np.sqrt(np.diag(geo_cov))
    # fitErrors_nb = np.sqrt(np.diag(nb_cov))
    
    
    # print('bpc', len(bpc), max(bpc), min(bpc), len(np.unique(bpc)), bpc)
    if USE_GUI:
        plt.bar(bin_starts, vals, width=bin_starts[1] - bin_starts[0], alpha=.4)
        plt.plot(x_fit, y_fit)
        plt.xlim([0, 200])
        xx3 = max(plt.xlim())
        yy3 = max(plt.ylim())
        cv = min(plt.xlim())
        # r'$y = Ae ^{-Bx} + C$'
        r'$y = p(1-p)^{x}$'
        plt.text(((xx3 - cv) * .3) + cv, yy3 * .7, r'$y = p(1-p)^{x}$')
        plt.text(((xx3 - cv) * .3) + cv, yy3 * .6, 'p= %5.2f +/- %5.2f' % (geofit.x, fitErrors_geo))
        
        # plt.text(xx3 * .3, yy3 * .45, 'p= %5.2f' % (geofit.x))
        plt.title(r'$mE0s3.2-CAAX$')
        plt.xlabel('Number of blinks per molecule')
        plt.ylabel('Probability')
        
        # negative binomial fitting
        # plt.figure()
        # plt.bar(bin_starts, vals, width=bin_starts[1] - bin_starts[0], alpha=.4)
        # plt.plot(x_fit, nbmf(x_fit, N, p))
        # xx4 = max(plt.xlim())
        # yy4 = max(plt.ylim())
        # cv2 = min(plt.xlim())
        # plt.text(((xx4-cv2)*.1) + cv2, yy4 * .9, 'Number of molecules N = %5.2f, off probability = %5.2f' % (N, p))
        # plt.title('Negative Binomial fit for both N and p')
        # plt.text(xx2 * .3, yy2 * .45, )
    """
    hist_data = hist_data
    params_geo = geofit.x
    params_nb = nbfit.x
    fitErrors_geo =
    fitErrors_nb =
    fit_eqn_geo =
    fit_eqn_nb =
    """
    return hist_data, geofit.x, fitErrors_geo, #, fit_eqn_geo, fit_eqn_nb


def histfitting(colourFilter, metadata, cluster_idxs, fit_order, num_bins, blink_on_label, blink_off_label,
                to_json=False, log_bins=False, n_on_bins=1, n_off_bins=1, fixed_on_max=-1, fixed_off_max=-1):
    import matplotlib as plt
    from PYME.IO import tabular
    # for best number of bins, find number of different on duration times, set as number of bins
    # should probably include this as an option in traits ui, I.E. do you want to manually set num bins or set to
    #       max number of unique states (blink duration, time to next blink, etc)
    
    N = fit_order
    """
    should add traits thing where this line can be used or set to 1

    """
    frame_duration = metadata.getEntry('Camera.IntegrationTime')
    
    on_times = colourFilter[blink_on_label].astype('f') #* frame_duration
    on_times = [np.int(i) for i in on_times]
    # print('is it being loaded correctly 1?', on_times)
    
    """
    setting up binning for histogram(s)
    this might not be correct, check again
    """
    #    max_on = np.max(on_times) + frame_duration
    
    # if fixed_on_max == -1:
    #     fixed_on_max = on_times.max()
    
    # if log_bins:
    #     binning = np.logspace(0.5, fixed_on_max + 1, num=n_on_bins)
    # else:
    binning = np.arange(frame_duration, max(on_times), 1)
    print('max on time', max(on_times))
    # if len(binning) < 20
    # binning = np.linspace(0, 30, 30)
    
    vals, bin_edges = np.histogram(on_times, bins=binning) #here, take 2-ed vals = vals[2:]
    np.set_printoptions(threshold=100000, suppress=True)
    # print('on y hist', len(vals), vals)
    
    logonbins = np.logspace(0, 1.49136169383, num=max(on_times))
    
    logvals, logbinedges = np.histogram(on_times, bins=logonbins)
    np.set_printoptions(threshold=100000, suppress=True)
    print('log scale on vals', logvals)
    # logoff_vals, log_bin_edges_off = np.histogram()
    
    
    bin_edges *= frame_duration
    
    """
    clip vectors here
    """
    y_hist = vals
    bin_starts = bin_edges[:-1]
    x_hist = bin_starts
    max_xaxis = max(bin_edges)
    
    off_times = colourFilter[blink_off_label].astype('f')
    # print('is it being loaded right 2?', off_times)
    # min_off = min(off_times)
    #    max_off = max(off_times) + frame_duratio
    logoffbins = np.logspace(0, 4.92471852852, num=30)
    logoff_vals, log_bin_edges_off = np.histogram(off_times, bins=logoffbins)
    print('log scale off vals', logoff_vals)
    
    if fixed_off_max == -1:
        fixed_off_max = off_times.max()
    
    # if log_bins:
    # binning_off = np.logspace(frame_duration, fixed_off_max + 1, num=30)
    
    # binning_off = np.logspace(-.60205999132, 4.92471852852, num=30)
    #
    # print(binning_off)
    # else:
    binning_off = np.arange(0.5, fixed_off_max + 1, n_off_bins)
    
    vals_off, bin_edges_off = np.histogram(off_times, bins=binning_off)
    """
    brute force print statement for getting sims going before retreat
    """
    np.set_printoptions(threshold=100000, suppress=True)
    # print('off y hist vals', len(vals_off), vals_off)
    bin_edges_off *= frame_duration
    
    y_hist_off = vals_off
    bin_starts_off = bin_edges_off[:-1]
    x_hist_off = bin_starts_off
    max_xaxis_off = max(bin_edges_off)
    
    # getting start params from integrated fit
    vals = np.array(vals)
    on_t_in_t = [i * frame_duration for i in on_times]
    off_t_in_t = [i * frame_duration for i in off_times]
    if N == 1:
        res_fxn = tfoef
        fit_fxn = sefm
        # print('what is going wrong here?')
        # print(max(vals), vals)
        # print(on_times, frame_duration)
        # print(np.mean(on_t_in_t))
        
        start_params = [max(vals), 1.0 / np.mean(on_t_in_t)]
        # start_params = [40417, 5.608, 0]
        start_params_off = [max(vals_off), 1.0 / np.mean(off_t_in_t)]
        fit_eqn = 'A*e^(-B*x)'
    
    if N == 2:
        res_fxn = tsoef
        fit_fxn = defm
        start_params = [(np.max(vals), 1.0 / np.mean(on_t_in_t), np.max(vals), 5.0 / np.mean(on_t_in_t))]
        start_params_off = [(np.max(vals_off), 1.0 / np.mean(off_t_in_t), np.max(vals_off), 5.0 / np.mean(off_t_in_t))]
        fit_eqn = 'A*e^(-B*x) + C*e^(-D*x)'
    
    params = start_params
    fit_results = FitModel_N(res_fxn, params, x_hist[1:], y_hist[1:])
    #
    # plt.figure()
    # plt.plot(x_hist_off[1:], y_hist[1:])
    fit_results_off = FitModel_N(res_fxn, start_params_off, x_hist_off[1:], y_hist_off[1:])
    
    cov = np.linalg.inv(np.matmul(fit_results.jac.T, fit_results.jac)) * np.mean(
        (fit_results.fun * fit_results.fun).sum())
    # print('test 1', fit_results_off.jac.T)
    # print('test 2', fit_results_off.jac)
    # print(np.matmul(fit_results_off.jac.T, fit_results_off.jac))
    # print(np.linalg.inv(np.matmul(fit_results_off.jac.T, fit_results_off.jac)))
    cov_off = np.linalg.inv(np.matmul(fit_results_off.jac.T, fit_results_off.jac)) * np.mean(
        (fit_results_off.fun * fit_results_off.fun).sum())
    
    fitErrors_on = np.sqrt(np.diag(cov))
    fitErrors_off = np.sqrt(np.diag(cov_off))
    # fitErrors_off = 1.1
    
    x_on_fit = np.linspace(0, max_xaxis, 100)
    y_on_fit = fit_fxn(x_on_fit, *fit_results.x)
    
    x_off_fit = np.linspace(0, max_xaxis_off, 100)
    y_off_fit = fit_fxn(x_off_fit, *fit_results_off.x)
    
    on_hist_datasource = np.rec.fromarrays((y_hist, bin_edges[1:]), dtype=[('y_hist', '<f4'), ('x_hist', '<f4')])
    off_hist_datasource = np.rec.fromarrays((y_hist_off, bin_edges_off[1:]),
                                            dtype=[('y_hist', '<f4'), ('x_hist', '<f4')])
    
    filt_on = tabular.RecArraySource(on_hist_datasource.view(np.recarray))
    filt_off = tabular.RecArraySource(off_hist_datasource.view(np.recarray))
    
    """
    trying to get relevant files into metadata
    """
    
    if USE_GUI:
        plt.figure()
        plt.bar(bin_starts, vals, width=bin_starts[1] - bin_starts[0], alpha=.4)
        # plt.scatter(x_hist[1:], y_hist[1:])
        plt.plot(x_on_fit,
                   y_on_fit)
        # plt.xscale('')
        xx = max(plt.xlim())
        yy = max(plt.ylim())
        
        if N == 1:
            plt.text(xx * .3, yy * .5, r'$y = Ae ^{-Bx} + C$') #, transform=plt.gca())
            plt.text(xx * .3, yy * .45, 'A= %5.2f +/- %5.2f' % (fit_results.x[0], fitErrors_on[0]))
            plt.text(xx * .3, yy * .40, 'B= %5.2f +/- %5.2f' % (fit_results.x[1], fitErrors_on[1]))
            # plt.text(xx * .3, yy * .35, 'C= %5.2f +/- %5.2f' % (fit_results.x[2], fitErrors_on[2]))
        elif N == 2:
            plt.text(xx * .3, yy * .6, r'$y =  Ae^{-B*x} + C e^{-D*x}$')
            plt.text(xx * .3, yy * .55, 'A= %5.2f +/- %5.2f' % (fit_results.x[0], fitErrors_on[0]))
            plt.text(xx * .3, yy * .5, 'B= %5.2f +/- %5.2f' % (fit_results.x[1], fitErrors_on[1]))
            plt.text(xx * .3, yy * .45, 'C= %5.2f +/- %5.2f' % (fit_results.x[2], fitErrors_on[2]))
            plt.text(xx * .3, yy * .4, 'D= %5.2f +/- %5.2f' % (fit_results.x[3], fitErrors_on[3]))
        #            plt.text(xx * .3, yy * .35, 'E= %5.2f +/- %5.2f' % (fit_results.x[4], fitErrors_on[2]))
        
        
        plt.ylabel('Events')
        plt.xlabel('Blink duration')
        plt.title('Blink On Duration')
        
        fig, ax = plt.subplots(1, 1)
        ax.scatter(x_hist, y_hist - fit_fxn(x_hist, *fit_results.x))
        ax.set_ylabel('Residual')
        ax.set_xlabel('Blink duration')
        ax.set_title('Blink On Duration residuals')
        
        # now, this is getting hists & fits for time to next blink values
        plt.figure()
        plt.bar(bin_starts_off, vals_off, width=bin_starts_off[1] - bin_starts_off[0], alpha=.4)
        # plt.scatter(x_hist_off[1:], y_hist_off[1:])
        plt.plot(x_off_fit,
                   y_off_fit)
        xx2 = max(plt.xlim())
        yy2 = max(plt.ylim())
        if N == 1:
            plt.text(xx2 * .3, yy2 * .6, r'$y = Ae ^{-Bx} + C$')
            plt.text(xx2 * .3, yy2 * .55, 'A= %5.2f +/- %5.2f' % (fit_results_off.x[0], fitErrors_off[0]))
            plt.text(xx2 * .3, yy2 * .50, 'B= %5.2f +/- %5.2f' % (fit_results_off.x[1], fitErrors_off[1]))
            # plt.text(xx2 * .3, yy2 * .35, 'C= %5.2f +/- %5.2f' % (fit_results_off.x[2], fitErrors_off[2]))
        elif N == 2:
            plt.text(xx2 * .3, yy2 * .6, r'$y =  Ae^{-B*x} + C e^{-D*x}$')
            plt.text(xx2 * .3, yy2 * .55, 'A= %5.2f +/- %5.2f' % (fit_results_off.x[0], fitErrors_off[0]))
            plt.text(xx2 * .3, yy2 * .5, 'B= %5.2f +/- %5.2f' % (fit_results_off.x[1], fitErrors_off[1]))
            plt.text(xx2 * .3, yy2 * .45, 'C= %5.2f +/- %5.2f' % (fit_results_off.x[2], fitErrors_off[2]))
            plt.text(xx2 * .3, yy2 * .4, 'D= %5.2f +/- %5.2f' % (fit_results_off.x[3], fitErrors_off[3]))
        #            plt.text(xx2 * .3, yy2 * .35, 'E= %5.2f +/- %5.2f' % (fit_results_off.x[4], fitErrors_off[2]))
        
        plt.ylabel('Events')
        plt.xlabel('Time to next blink')
        plt.title('Time to next blink in the same cluster')
        
        fig, ax = plt.subplots(1, 1)
        ax.scatter(x_hist_off, y_hist_off - fit_fxn(x_hist_off, *fit_results_off.x))
        ax.set_ylabel('Residual')
        ax.set_xlabel('Time to next blink')
        ax.set_title('Time to next blink residuals')
    
    if to_json == True:
        import io
        import time
        import json
        try:
            to_unicode = unicode
        except NameError:
            to_unicode = str
        
        # Generate dye kinetics structure for JSON file
        dk = {}
        dk['plog'] = [0, 0]
        if log_bins:
            dk['tlog'] = [1, 1]
        else:
            dk['tlog'] = [0, 0]
        dk['tmin'] = [bin_edges[0], bin_edges_off[0]]
        dk['tmax'] = [bin_edges[-1], bin_edges_off[-1]]
        dk['off'] = y_hist_off.tolist()
        dk['on'] = y_hist.tolist()
        
        # Write JSON file
        timestr = time.strftime("%Y%m%d-%H%M%S")
        with io.open('empirical_histograms_' + timestr + '.json', 'w', encoding='utf8') as outfile:
            str_ = '{"dk": ' + json.dumps(dk) + '}'
            outfile.write(to_unicode(str_))
    
    params_on = fit_results.x
    params_off = fit_results_off.x
    return filt_on, filt_off, params_on, params_off, fitErrors_on, fitErrors_off, fit_eqn

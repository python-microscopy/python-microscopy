import numpy as np
from scipy.interpolate import UnivariateSpline #as we only use this function, interpolate it directly

from . import astiglookup


def lookup_astig_z(fres, astig_calibrations, plot=False):
    """
    Generates a look-up table of sorts for z based on sigma x/y fit results and calibration information. If a molecule
    appears on multiple planes, sigma values from both planes will be used in the look up.
    Args:
        fres: dictionary-like object containing relevant fit results
        astig_calibrations: list of astigmatism calibration dictionaries corresponding to each multiview channel, which are
            used to recreate shiftmap objects
        chanPlane: list of which plane each channel corresponds to, e.g. [0, 0, 1, 1]

    Returns:
        z: an array of z-positions for each molecule in nm (assuming proper units were used in astigmatism calibration)
        zerr: an array containing discrepancies between sigma values and the PSF calibration curves. Note that this
            array is in units of nm, but error may not be propagated from sigma fitResults properly as is.
    """
    # fres = pipeline.selectedDataSource.resultsSource.fitResults
    numMolecules = len(fres['x']) # there is no guarantee that fitResults_x0 will be present - change to x
    numChans = len(astig_calibrations)

    # find overall min and max z values
    z_min = 0
    z_max = 0
    for astig_cal in astig_calibrations: #more idiomatic way of looping through list - also avoids one list access / lookup
        r_min, r_max = astig_cal['zRange']
        z_min = min(z_min, r_min)
        z_max = max(z_max, r_max)

    # generate z vector for interpolation
    zVal = np.arange(z_min, z_max)


    #TODO - Is this a robust choice?
    smoothFac = 5 * len(astig_calibrations[0]['z'])

    # generate look up table of sorts
    sigCalX = []
    sigCalY = []
    for i, astig_cal in enumerate(astig_calibrations):
        zdat = np.array(astig_cal['z'])
        # find indices of range we trust
        #zrange = astig_cal['zRange']

        #z_valid_mask = (zdat > zrange[0])*(zdat < zrange[1])
        z_valid_mask = zdat > -1e6
        z_valid = zdat[z_valid_mask]

        sigCalX.append(UnivariateSpline(z_valid,np.array(astig_cal['sigmax'])[z_valid_mask],ext='const', s=smoothFac)(zVal))
        sigCalY.append(UnivariateSpline(z_valid,np.array(astig_cal['sigmay'])[z_valid_mask],ext='const', s=smoothFac)(zVal))

    sigCalX = np.array(sigCalX)
    sigCalY = np.array(sigCalY)

    #allocate arrays for the estimated z positions and their errors
    z = np.zeros(numMolecules)
    zerr = 1e4 * np.ones(numMolecules)

    failures = 0
    chans = np.arange(numChans)

    #extract our sigmas and their errors
    #doing this here means we only do the string operations and look-ups once, rather than once per molecule
    sxs = np.abs(np.array([fres['sigmax%i' % ci] for ci in chans]))
    sys = np.abs(np.array([fres['sigmay%i' % ci] for ci in chans]))
    esxs = [fres['error_sigmax%i' % ci] for ci in chans]
    esys = [fres['error_sigmay%i' % ci] for ci in chans]
    wXs = np.array([1. / (esx_i*esx_i) for esx_i in esxs])
    wYs = np.array([1. / (esy_i*esy_i) for esy_i in esys])

    if plot:
        from matplotlib import pyplot as plt

        plt.figure()
        plt.subplot(211)
        for astig_cal, interp_sigx, col in zip(astig_calibrations, sigCalX, ['r', 'g', 'b', 'c']):
            plt.plot(astig_cal['z'], astig_cal['sigmax'], ':', c=col)
            plt.plot(zVal, interp_sigx, c=col)

        plt.subplot(212)
        for astig_cal, interp_sigy, col in zip(astig_calibrations, sigCalY, ['r', 'g', 'b', 'c']):
            plt.plot(astig_cal['z'], astig_cal['sigmay'], ':', c=col)
            plt.plot(zVal, interp_sigy, c=col)

    # _lenz_chunked = np.floor(len(zVal)) - 1
    # sigCalX_chunked = np.ascontiguousarray(sigCalX[:,::100])
    # sigCalY_chunked = np.ascontiguousarray(sigCalY[:,::100])
    #
    # for i in range(numMolecules):
    #     #TODO - can we avoid this loop?
    #     wX = wXs[:, i]
    #     wY = wYs[:, i]
    #     sx = sxs[:, i]
    #     sy = sys[:, i]
    #
    #     wSum = (wX + wY).sum()
    #
    #     #estimate the position in two steps - coarse then fine
    #
    #     #coarse step:
    #     errX = (wX[:,None] * (sx[:, None] - sigCalX_chunked)**2).sum(0)
    #     errY = (wY[:, None] * (sy[:, None] - sigCalY_chunked)**2).sum(0)
    #
    #     err = (errX + errY) / wSum
    #     loc_coarse = min(max(np.argmin(err), 1), _lenz_chunked)
    #
    #     fine_s =  100*(loc_coarse - 1)
    #     fine_end = 100*(loc_coarse + 1)
    #
    #     #print loc_coarse, fine_s, fine_end, sigCalX.shape
    #
    #     #fine step
    #     errX = (wX[:, None] * (sx[:, None] - sigCalX[:,fine_s:fine_end]) ** 2).sum(0)
    #     errY = (wY[:, None] * (sy[:, None] - sigCalY[:,fine_s:fine_end]) ** 2).sum(0)
    #
    #     err = (errX + errY) / wSum
    #     minLoc = np.argmin(err)
    #
    #     z[i] = -zVal[fine_s + minLoc]
    #     zerr[i] = np.sqrt(err[minLoc])

    zi, ze = astiglookup.astig_lookup(sigCalX.T.astype('f'), sigCalY.T.astype('f'), sxs.T.astype('f'), sys.T.astype('f'), wXs.T.astype('f'), wYs.T.astype('f'))

    print 'used c lookup'

    z = -zVal[zi]
    zerr = np.sqrt(ze)


    #print('%i localizations did not have sigmas in acceptable range/planes (out of %i)' % (failures, numMolecules))

    return z, zerr
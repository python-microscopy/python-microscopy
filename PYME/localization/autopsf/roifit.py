#!/usr/bin/python

##################
# roifit.py
#
# Copyright David Baddeley, Kenny Chung, 2012-2017
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permission from David Baddeley
#
##################


from PYME.Analysis._fithelpers import FitModelWeighted_, FitModelWeighted
from PYME.Analysis.PSFGen import fourierHNA
from PYME.localization.FitFactories.Interpolators import LinearInterpolator, CubicSplineInterpolator, CSInterpolator
from PYME.localization.FitFactories.zEstimators import biplaneEstimator
import numpy as np
from PYME.IO import MetaDataHandler

from collections import namedtuple

#VS = namedtuple('VS', ['x', 'y', 'z'])
VS = MetaDataHandler.VoxelSize #Alias for backward compatibility

#interpolator = LinearInterpolator.interpolator
#interpolator = CubicSplineInterpolator.interpolator
interpolator = CSInterpolator.interpolator
estimator = biplaneEstimator


def f_Interp3d(p, interpolator, Xg, Yg, Zg, safeRegion, *args):
    """3D PSF model function with 0 background - parameter vector [A, x0, y0, z0, ]"""
    if len(p) == 5:
        A, x0, y0, z0, bg = p
    else:
        A, x0, y0, z0 = p
        bg = 0
    
    if np.isnan(A):
        return np.empty((len(Xg), len(Yg))) * np.nan
    
    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0]), safeRegion[2][1])
    
    return interpolator.interp(Xg - x0 + 1, Yg - y0 + 1, Zg - z0 + 1) * A + bg


def f_Interp3d2cr(p, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, axialShift, ratio, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    
    if len(p) == 7:
        Ag, Ar, x0, y0, z0, bg_g, bg_r = p
    elif len(p) == 5:
        Ag, Ar, x0, y0, z0 = p
        bg_g, bg_r = 0, 0
    else:
        A, x0, y0, z0 = p
        
        Ag = ratio * A
        Ar = (1 - ratio) * A
        
        bg_g, bg_r = 0, 0
    
    #FIXME - why is this here????
    if np.isnan(Ag):
        return np.empty((len(Xg), len(Yg), 2)) * np.nan
    
    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + axialShift), safeRegion[2][1] - axialShift)
    
    g = interpolator.interp(Xg - x0 + 1, Yg - y0 + 1, Zg - z0 + 1) * Ag + bg_g
    r = interpolator.interp(Xr - x0 + 1 + interpolator.PSF2Offset, Yr - y0 + 1, Zr - z0 + 1) * Ar + bg_r
    
    return np.concatenate((np.atleast_3d(g), np.atleast_3d(r)), 2)


def startEstROI(roi, md, interpolator, estimator, r, axialShift, fitBG=False):
    dataROI = roi['data']
    #sigma = roi['sigma']
    #sigma[sigma == 0] = 1e3
    
    Xg, Yg, Zg, safeRegion = interpolator.getCoords(md, slice(0, dataROI.shape[0]), slice(0, dataROI.shape[1]),
                                                    slice(0, 1))
    
    # Xr = roi['sp']['Xr0'] + Xg - roi['sp']['Xg0']
    # Yr = roi['sp']['Yr0'] + Yg - roi['sp']['Yg0']
    #
    # Zr = Zg + (axialShift if axialShift else 0)
    
    if len(Xg.shape) > 1: #X is a matrix
        X_ = Xg[:, 0, 0]
        Y_ = Yg[0, :, 0]
    else:
        X_ = Xg
        Y_ = Yg
        
        #estimate some start parameters...
    startParams = estimator.getStartParameters(dataROI, X_, Y_)
    
    if startParams[0].size > 1:
        if fitBG:
            return np.array(
                [startParams[0][0], startParams[0][1], startParams[1], startParams[2], startParams[3], 0, 0])
        else:
            return np.array([startParams[0][0], startParams[0][1], startParams[1], startParams[2], startParams[3]])
    else:
        if fitBG:
            return np.array([startParams[0], startParams[1], startParams[2], startParams[3], 0])
        else:
            return np.array([startParams[0], startParams[1], startParams[2], startParams[3]])
            
            #return [2*startParams[0], startParams[1], startParams[2], startParams[3]]


def fitROI(roi, md, interpolator, startParameters, r, axialShift):
    dataROI = roi['data']
    sigma = roi['sigma']
    #sigma[sigma == 0] = 1e3
    
    #print axialShift
    
    Xg, Yg, Zg, safeRegion = interpolator.getCoords(md, slice(0, dataROI.shape[0]), slice(0, dataROI.shape[1]),
                                                    slice(0, 1))
    
    if axialShift is None:
        (res, cov_x, infodict, mesg, resCode) = FitModelWeighted_(f_Interp3d, startParameters, dataROI, sigma,
                                                                  interpolator, Xg, Yg, Zg, safeRegion)
    else:
        Xr = roi['sp']['Xr0'] + Xg - roi['sp']['Xg0']
        Yr = roi['sp']['Yr0'] + Yg - roi['sp']['Yg0']
        Zr = Zg + axialShift
        
        (res, cov_x, infodict, mesg, resCode) = FitModelWeighted_(f_Interp3d2cr, startParameters, dataROI, sigma,
                                                                  interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion,
                                                                  axialShift, r)
    
    #normalised Chi-squared
    nchi2 = (infodict['fvec'] ** 2).sum() / (dataROI.size - res.size)
    
    return res, nchi2


def misfitROI(roi, md, interpolator, startParameters, r, axialShift):
    dataROI = roi['data']
    sigma = roi['sigma']
    #sigma[sigma == 0] = 1e3
    
    Xg, Yg, Zg, safeRegion = interpolator.getCoords(md, slice(0, dataROI.shape[0]), slice(0, dataROI.shape[1]),
                                                    slice(0, 1))
    
    Xr = roi['sp']['Xr0'] + Xg - roi['sp']['Xg0']
    Yr = roi['sp']['Yr0'] + Yg - roi['sp']['Yg0']
    
    Zr = Zg + axialShift
    
    misf = (dataROI - f_Interp3d2cr(startParameters, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, axialShift,
                                    r)) / sigma
    
    return misf


def misfallA(r2, mdh, zCoeffs, ns=1.51, axialShift=None, colourRatio=None, beadSize=0):
    mdh = MetaDataHandler.NestedClassMDHandler(mdh)
    if not axialShift == None:
        mdh['Analysis.AxialShift'] = axialShift
    if not colourRatio == None:
        mdh['Analysis.ColourRatio'] = colourRatio
    voxelsize = mdh.voxelsize_nm
    voxelsize.z = 50.
    zs = 50. * np.arange(-30., 31)
    p1 = fourierHNA.GenZernikeDPSF(zs, 70, zCoeffs, ns=ns)
    interpolator.setModel('foo', p1, voxelsize)
    
    estimator.splines.clear()
    estimator.calibrate(interpolator, mdh)
    
    sp_all = [startEstROI(r2[j], mdh, interpolator, estimator, mdh['Analysis.ColourRatio'], mdh['Analysis.AxialShift'])
              for j in range(len(r2))]
    
    mf = np.array(
        [(fitROI(r2[j], mdh, interpolator, sp_all[j], mdh['Analysis.ColourRatio'], mdh['Analysis.AxialShift'])[1]).sum()
         for j in range(len(r2))])
    
    #return mf[mf < (median(mf)+ 2*std(mf))].mean()
    return np.sqrt(mf).mean()


def GenMultiChanZernikeDPSF(Z, zerns2, wavelengths, dx=70., output_shape=[61, 61, 61], **kwargs):
    from PYME.Analysis.PSFGen import fourierHNA
    
    psfs = list()
    
    for i in range(len(zerns2)):
        psf = fourierHNA.GenZernikeDPSF(Z, dx=dx, zernikeCoeffs=zerns2[i], lamb=wavelengths[i],
                                        output_shape=output_shape,
                                        **kwargs)
        psf /= psf.sum(1).sum(0).max()#, 1)).max()
        
        psfs.append(psf)
    
    psfs = np.concatenate(psfs, 0)
    
    return psfs


def misfallB(r2, mdh, zCoeffs, interpolatorName, estimatorName, wavelengths=[700, 700], ns=1.51, axialShift=None,
             colourRatio=None, beadSize=0, fitBG=False,
             psf_model_settings={'NA': 1.45, 'apodization': 'sine', 'vectorial': True, 'n': 1.51}):
    from importlib import import_module
    interpolator = import_module(interpolatorName).interpolator
    estimator = import_module(estimatorName)
    
    mdh = MetaDataHandler.NestedClassMDHandler(mdh)
    if not axialShift is None:
        mdh['Analysis.AxialShift'] = axialShift
    if not colourRatio is None:
        mdh['Analysis.ColourRatio'] = colourRatio
    
    #axialShift = mdh.getOrDefault('Analysis.AxialShift', None)
    #colourRatio = mdh.getOrDefault('Analysis.ColourRatio', None)
    
    #fitBG = mdh.getOrDefault('AutoPSF.FitBackground', 0)
    
    #wavelengths = mdh.getOrDefault('AutoPSF.Wavelengths', [700,700])
    
    
    zs = 50. * np.arange(-30., 31)
    vs_nm = VS(*mdh.voxelsize_nm[:2], 50.)
    #p1 = fourierHNA.GenZernikeDPSF(zs,  zCoeffs, dx=1e3*voxelsize.x, ns=ns)
    
    p1 = GenMultiChanZernikeDPSF(zs, zerns2=zCoeffs, wavelengths=wavelengths, dx=vs_nm.x,
                                 ns=ns, output_shape=[61, 61, 61], **psf_model_settings)
    
    #print vs_nm, wavelengths,zCoeffs, p1.shape
    
    interpolator.setModel('foo', p1, vs_nm)
    
    roiSize = int(np.floor(r2[0]['data'].shape[0] / 2))
    
    estimator.splines.clear()
    estimator.calibrate(interpolator, mdh, roiSize=roiSize)
    
    sp_all = [startEstROI(r2[j], mdh, interpolator, estimator, colourRatio, axialShift, fitBG=fitBG) for j in
              range(len(r2))]
    
    #print sp_all
    #print 'cr:', colourRatio
    
    fr = [(fitROI(r2[j], mdh, interpolator, sp_all[j], colourRatio, axialShift)) for j in range(len(r2))]
    
    #mf = np.array([(fitROI(r2[j], mdh, interpolator, sp_all[j], colourRatio, axialShift)[1]).sum() for j in range(len(r2))])
    
    mf = np.array([f[1].sum() for f in fr])
    
    mfm = mf.argmax()
    
    #print 'sp:', sp_all[mfm], 'mfm:', mf[mfm], 'mf_j:', mfm, '\nfr:', fr[mfm][0]
    
    #return mf[mf < (median(mf)+ 2*std(mf))].mean()
    #return np.nanmean(np.sqrt(mf))
    return np.sqrt(mf).mean()


def misfallB_MP(task_queue, results_queue, data):
    while True:
        ci, args, kwargs = task_queue.get()
        results_queue.put((ci, misfallB(data, *args, **kwargs)))


def fitallA(r2, mdh, zCoeffs, ns=1.51, axialShift=200.):
    mdh = MetaDataHandler.NestedClassMDHandler(mdh)
    mdh['Analysis.AxialShift'] = axialShift
    voxelsize = mdh.voxelsize_nm
    voxelsize.z = 50.
    zs = 50. * np.arange(-30., 31)
    p1 = fourierHNA.GenZernikeDPSF(zs, 70, zCoeffs, ns=ns)
    interpolator.setModel('foo', p1, voxelsize)
    
    estimator.splines.clear()
    estimator.calibrate(interpolator, mdh)
    
    sp_all = [startEstROI(r2[j], mdh, interpolator, estimator, mdh['Analysis.ColourRatio'], mdh['Analysis.AxialShift'])
              for j in range(len(r2))]
    
    fr = np.array(
        [(fitROI(r2[j], mdh, interpolator, sp_all[j], mdh['Analysis.ColourRatio'], mdh['Analysis.AxialShift'])[0]) for j
         in range(len(r2))])
    
    #return mf[mf < (median(mf)+ 2*std(mf))].mean()
    return fr


def fmfROI(roi, mdh, p1, voxelsize, sa, astig):
    interpolator.setModel('foo', p1, voxelsize)
    estimator.splines.clear()
    estimator.calibrate(interpolator, mdh)
    
    sp = startEstROI(roi, mdh, interpolator, estimator, mdh['Analysis.ColourRatio'], mdh['Analysis.AxialShift'])
    
    mf = fitROI(roi, mdh, interpolator, sp, mdh['Analysis.ColourRatio'], mdh['Analysis.AxialShift'])[1]
    
    #return mf[mf < (median(mf)+ 2*std(mf))].mean()
    return np.sqrt(mf)
#!/usr/bin/python

##################
# _fithelpers.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import scipy
import scipy.optimize as optimize

FWHM_CONV_FACTOR = 2*scipy.sqrt(2*scipy.log(2))

#EPS_FCN = 1e-4

def missfit(p, fcn, data, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data)"""
    return data - fcn(p, *args).ravel()

def weightedMissfit(p, fcn, data, sigmas, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data), scaling with the errors in the measured data (sigmas)"""
    mod = fcn(p, *args).ravel()
    #print mod.shape
    #print data.shape
    #print sigmas.shape
    return (data - mod)/sigmas

def weightedMissfitF(p, fcn, data, weights, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data), scaling with precomputed weights corresponding to the errors in the measured data (weights)"""
    mod = fcn(p, *args).ravel()
    #print mod.shape
    #print data.shape
    #print sigmas.shape
    return (data - mod)*weights  

def weightedJacF(p, fcn, data, weights, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data), scaling with precomputed weights corresponding to the errors in the measured data (weights)"""
    r = weights[:,None]*fcn.D(p, *args)
    return -r
    

def FitModel(modelFcn, startParameters, data, *args):
    return optimize.leastsq(missfit, startParameters, (modelFcn, data.ravel()) + args, full_output=1)

def FitModel_(modelFcn, startParameters, data, *args):
    return optimize.leastsq(missfit, startParameters, (modelFcn, data.ravel()) + args, full_output=1, epsfcn=EPS_FCN)

def FitModelWeighted(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('f').ravel()) + args, full_output=1)

def FitModelWeighted_(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('f').ravel()) + args, full_output=1, epsfcn=1e-4)

def FitModelWeightedJac(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('d').ravel()) + args, Dfun = weightedJacF, full_output=1, col_deriv = 0)

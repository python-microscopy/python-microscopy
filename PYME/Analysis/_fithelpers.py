#!/usr/bin/python

##################
# _fithelpers.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

#import scipy
import scipy.optimize as optimize
import numpy as np

FWHM_CONV_FACTOR = 2*np.sqrt(2*np.log(2))

EPS_FCN = 1e-4

def missfit(p, fcn, data, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data)"""
    return data - fcn(p, *args).ravel()

def missfit_fixed(p, fit, sp, fcn, data, weights, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data)"""
    p_ = sp.copy()
    p_[fit] = p
    return (data.ravel() - fcn(p_, *args).ravel())*weights.ravel()

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
    mod = fcn(p, *args)
    mod = mod.ravel()
    #print mod.shape
    #print data.shape
    #print weights.shape
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
    
def FitModel_D(modelFcn, startParameters, data, diag, *args):
    return optimize.leastsq(missfit, startParameters, (modelFcn, data.ravel()) + args, full_output=1, epsfcn=EPS_FCN, diag=diag)
    
def FitModel_E(modelFcn, startParameters, data, eps, *args):
    return optimize.leastsq(missfit, startParameters, (modelFcn, data.ravel()) + args, full_output=1, epsfcn=eps)

def FitModelWeighted(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('f').ravel()) + args, full_output=1)

def FitModelWeighted_(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('f').ravel()) + args, full_output=1, epsfcn=EPS_FCN)
    
def FitModelWeighted_D(modelFcn, startParameters, data, sigmas, diag, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('f').ravel()) + args, full_output=1, epsfcn=EPS_FCN, diag=diag)


def FitModelWeightedJac(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('d').ravel()) + args, Dfun = weightedJacF, full_output=1, col_deriv = 0)
    
def FitModelWeightedJac_(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).astype('d').ravel()) + args, Dfun = weightedJacF, full_output=1, col_deriv = 0, epsfcn=EPS_FCN)


def FitWeightedMisfitFcn(misfitFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(misfitFcn, np.array(startParameters), (np.array(data, order='F'), np.array(1.0/sigmas, order='F')) + args, full_output=1)


def poisson_lhood(p, fcn, data, bg, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data)"""
    mu = (fcn(p, *args) + bg)
    return -(data*np.log(mu) - mu).sum()
    
def poisson_lhoodJ(p, fcn, data, bg, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data)"""
    f0 = poisson_lhood(p, fcn, data, bg, *args)
    
    df = 0*p
    for i in range(len(p)):
        dpi = 0.1*p[i] + 1
        pt = 1.0*p + 0
        pt[i] += dpi
        ft = poisson_lhood(pt, fcn, data, bg, *args)
        df[i] = (ft - f0)/dpi
        
    return df
    
    
def poisson_lhood2(p, fcn, data, bg, *args):
    """Helper function which evaluates a model function (fcn) with parameters (p) and additional arguments
    (*args) and compares this with measured data (data)"""
    mu = (fcn(p, *args) + bg)
    return -(data*np.log(mu) - mu)
    
def FitModelPoisson(modelFcn, startParmeters, data, *args, **kwargs):
    try:
        bg = kwargs['bg']
    except KeyError:
        bg = 0
    return [optimize.fmin_powell(poisson_lhood, startParmeters, ((modelFcn, data,bg) +args))]
    
def FitModelPoissonBFGS(modelFcn, startParmeters, data, *args, **kwargs):
    try:
        bg = kwargs['bg']
    except KeyError:
        bg = 0
    return [optimize.fmin_bfgs(poisson_lhood, startParmeters, args=((modelFcn, data,bg) +args), epsilon=0.1)]


def FitModelFixed(modelFcn, startParameters, fitWhich, data, *args, **kwargs):
    eps = kwargs.get('eps', EPS_FCN)
    weights = kwargs.get('weights', np.array(1.0))

    startParameters = np.array(startParameters)
    fitWhich = np.where(fitWhich)
    p = startParameters[fitWhich]
    #print p, fitWhich
    res =  optimize.leastsq(missfit_fixed, p, (fitWhich, startParameters, modelFcn, data, weights) + args, full_output=1, epsfcn=eps)[0]

    out = startParameters.copy()
    out[fitWhich] = res
    return [out]

    
#def FitModelPoissonS(modelFcn, startParmeters, data, *args):
import scipy
import scipy.optimize as optimize

FWHM_CONV_FACTOR = 2*scipy.sqrt(2*scipy.log(2))

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

def FitModel(modelFcn, startParameters, data, *args):
    return optimize.leastsq(missfit, startParameters, (modelFcn, data.ravel()) + args, full_output=1)

def FitModelWeighted(modelFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).ravel()) + args, full_output=1)

def FitModelWeightedJac(modelFcn, jacFcn, startParameters, data, sigmas, *args):
    return optimize.leastsq(weightedMissfitF, startParameters, (modelFcn, data.ravel(), (1.0/sigmas).ravel()) + args, Dfun = jacFcn, full_output=1)

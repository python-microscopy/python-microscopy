# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:58:51 2014

@author: david
"""
import numpy as np

try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

def replNoneWith1(n):
        if n is None:
            return 1
        else:
            return n

def fmtSlicesUsed(slicesUsed):
    if slicesUsed is None:
        return ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
    else:
        return tuple([(sl.start, sl.stop, replNoneWith1(sl.step)) for sl in slicesUsed] )

def _tuplify(var):
    try:
        return tuple(var)
    except TypeError:
        return var


def pack_results(dtype, tIndex, fitResults, fitError=None, startParams=None, slicesUsed=None, resultCode=-1, **kwargs):
    """ Pack fit results into a structured array of the given dtype
    
    Collects logic from fit factories to a central place which hopefully makes it easier to
    maintain.
    
    Parameters
    ----------
    dtype  : np.dtype
        the numpy dtype of the structured array we want to pack into
    tIndex : int 
        the current frame number
    fitResults : np.ndarray
        the fit parameters in the order they are defined in the dtype
    fitError   : np.ndarray
        the fit errors in the order they are defined in the dtype
    startParams : np.ndarray, optional 
        the start parameters in the order they are defined in the dtype
    slicesUsed : tuple, optional 
        a 3-tuple of slice objects (xslice, yslice, zslice) that define the ROI used for this molecule
    resultCode : int, optional 
        the result code as returned by the fitting routine
    **kwargs :  dict, optional
        any additional information which gets stored in the structured array, either a scalar or a numpy array
    
    Returns
    -------
    np.recarray
        The packed results array
    
    TODOS:
    - Support length mismatch on data
    
    FIXME: This currently uses tuples which is really gross for a number of reasons (e.g. moves what should be a numpy
    low level c loop into python, relies on implicitly coercing types rather than doing it explicitly). For some
    reason it is currently faster than assigning to views into an array even though it really should be quite a
    lot slower. If numpy history is anything to go by, it's also quite likely to break at some point in the future.
    """
    
    dtype = np.dtype(dtype)
    if fitError is None:
        fitError = -5e3 + 0 * fitResults
    
    if startParams is None:
        startParams = -5e3 + 0 * fitResults
    
    slicesUsed = fmtSlicesUsed(slicesUsed)
    
    ns = locals()
    ns.update(kwargs)
    
    return np.array(tuple([_tuplify(ns[n]) for n in dtype.names]), dtype=dtype)
    

###############################################
# Below are various experimental alternatives to pack_results. They are still a work in progress, but should
# hopefully let us replace some of the tuple madness in the above one. Of the alternatives, _pack_results4, which
# pushes stuff into a pre-allocated array is ~2 times faster than the tuple based code above, but would need quite
# a lot of additional refactoring in the calling code to make it actually work (the exceptions here are the Multifit
# and GPU fitting classes. Punting that to some point in the future for now.

def _pack_results1(dtype, flatdtype, tIndex, fitResults, fitError=None, startParams=None, slicesUsed=None, resultCode=-1, **kwargs):
    dtype = np.dtype(dtype)
    if fitError is None:
        fitError = -5e3 + 0 * fitResults
    
    if startParams is None:
        startParams = -5e3 + 0 * fitResults
    
    slicesUsed = np.ravel(fmtSlicesUsed(slicesUsed))
    
    ns = locals()
    ns.update(kwargs)
    
    res = np.zeros(1, dtype=flatdtype)
    
    for n in dtype.names:
        res[n] = ns[n]
        
    return res.view(dtype)


def _pack_results4(out, flat_out, tIndex, fitResults, fitError=None, startParams=None, slicesUsed=None, resultCode=-1,
                  **kwargs):
    if fitError is None:
        fitError = -5e3 + 0 * fitResults
    
    if startParams is None:
        startParams = -5e3 + 0 * fitResults
    
    slicesUsed = np.ravel(fmtSlicesUsed(slicesUsed))
    
    ns = locals()
    ns.update(kwargs)
    
    
    for n in out.dtype.names:
        flat_out[n] = ns[n]
    
    return out


def _pack_results3(dtype, flatdtype, tIndex, fitResults, fitError=None, startParams=None, slicesUsed=None, resultCode=-1,
                  **kwargs):
    dtype = np.dtype(dtype)
    if fitError is None:
        fitError = -5e3 + 0 * fitResults
    
    if startParams is None:
        startParams = -5e3 + 0 * fitResults
    
    slicesUsed = np.ravel(fmtSlicesUsed(slicesUsed))
    
    ns = locals()
    ns.update(kwargs)
    
    #res = np.zeros(1, dtype=flatdtype)
    
    #for n in dtype.names:
    #    d = ns[n]
    
    return np.array(tuple([ns[n] for n in dtype.names]), flatdtype).view(dtype)


def _pack_results2(dtype, tIndex, fitResults, fitError=None, startParams=None, slicesUsed=None, resultCode=-1, **kwargs):
    dtype = np.dtype(dtype)
    if fitError is None:
        fitError = -5e3 + 0 * fitResults
    
    if startParams is None:
        startParams = -5e3 + 0 * fitResults
    
    slicesUsed = fmtSlicesUsed(slicesUsed)
    
    ns = locals()
    ns.update(kwargs)
    
    res = np.zeros(1, dtype=dtype)
    
    for n in dtype.names:
        res[n] = _tuplify(ns[n])
    
    return res
    

#generate a flat dtype from a standard nested one (incomplete)
def _gen_flat_dtype(dtype):
    dtype = np.dtype(dtype)
    
    out_dt = []
    
    for n in dtype.names:
        field_dt = dtype.fields[n][0]
        
        
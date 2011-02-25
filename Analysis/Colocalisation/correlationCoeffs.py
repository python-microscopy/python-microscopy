#!/usr/bin/python

##################
# correlationCoeffs.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
# Computes the standard correlation/colocalisation coeficients
#
##################
import numpy as np

def pearson(X, Y):
    X = X - X.mean()
    Y = Y-Y.mean()
    return (X*Y).sum()/np.sqrt((X*X).sum()*(Y*Y).sum())

def overlap(X, Y):
    return (X*Y).sum()/np.sqrt((X*X).sum()*(Y*Y).sum())


def thresholdedManders(A, B, tA, tB):
    '''Manders, as practically used with threshold determined masks'''

    MA = ((B > tB)*A).sum()/A.sum()
    MB = ((A > tA)*B).sum()/B.sum()

    return MA, MB
#!/usr/bin/python

##################
# piecewise.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy


def piecewiseLinear(t, tvals, grads):
    tvals = numpy.hstack([0., tvals, 1e9])

    res = numpy.zeros(t.shape)

    a = numpy.hstack((0, numpy.cumsum(numpy.diff(tvals)*grads)))

    for i in range(len(grads)):
        res += (t >= tvals[i])*(t <  tvals[i+1])*(a[i] + grads[i]*(t-tvals[i]))

    return res
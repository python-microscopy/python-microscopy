#!/usr/bin/python
##################
# moments.py
#
# Copyright David Baddeley, 2011
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
import numpy as np

def calcCenteredMoments(x, y, order=4):
    """Calculate centered moments of a point cloud"""
    xm = x.mean()
    ym = y.mean()
    
    xc = x - xm
    yc = y - ym
    
    N = order + 1
    
    m = np.zeros((N,N))
    #i_s = np.zeros_like(m)
    #j_s = np.zeros_like(m)
    
    for i in range(N):
        for j in range(N):
            m[i,j] = ((xc**i)*(yc**j)).sum()/x.size
            #i_s[i,j] = i
            #j_s[i,j] = j


    return m #, i_s, j_s

def calcMCCenteredMoments(x, y, order=4, nSamples=10):
    """Calculate centered moments with monte-carlo resampling to allow error estimation"""

    ms = []

    for n in range(nSamples):
        ind = np.random.uniform(size=x.shape) > .5
        #print ind
        ms.append(calcCenteredMoments(x[ind], y[ind], order))

    ms = np.array(ms)
    #print ms.shape

    return ms.mean(0), np.std(ms, 0)

def genIndexAndLabels(order=4):
    N = order + 1
    i,j = np.mgrid[:N,:N]

    i = i.ravel()
    j = j.ravel()

    ind = np.argsort(i+j+0.1*np.maximum(i,j))

    labs = ['%d,%d' % a for a in zip(i,j)]

    return ind, labs


def ttest_ind(m1, m2, s1, s2, n1, n2):
    from scipy.stats import distributions
    """Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

    copied from scipy/stats/stats.py & modified to work with pre-calulated means and
    std. deviations.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values.

    Parameters
    ----------
    m1, m2 : means of the two samples.
    s1, s2 : std deviations of the two samples
    n1, n2 : sample sizes

    Returns
    -------
    t : float or array
        t-statistic
    prob : float or array
        two-tailed p-value


    Notes
    -----

    We can use this test, if we observe two independent samples from
    the same or different population, e.g. exam scores of boys and
    girls or of two ethnic groups. The test measures whether the
    average (expected) value differs significantly across samples. If
    we observe a large p-value, for example larger than 0.05 or 0.1,
    then we cannot reject the null hypothesis of identical average scores.
    If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%,
    then we reject the null hypothesis of equal averages.

    References
    ----------

       http://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test

    """
    
    v1 = s1**2
    v2 = s2**2
    
    df = n1+n2-2

    d = m1 - m2
    svar = ((n1-1)*v1+(n2-1)*v2) / float(df)

    t = d/np.sqrt(svar*(1.0/n1 + 1.0/n2))
    t = np.where((d==0)*(svar==0), 1.0, t) #define t=0/0 = 0, identical means
    prob = distributions.t.sf(np.abs(t),df)*2#use np.abs to get upper tail

    #distributions.t.sf currently does not propagate nans
    #this can be dropped, if distributions.t.sf propagates nans
    #if this is removed, then prob = prob[()] needs to be removed
    prob = np.where(np.isnan(t), np.nan, prob)

    if t.ndim == 0:
        t = t[()]
        prob = prob[()]

    return t, prob




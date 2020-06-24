#!/usr/bin/python
##################
# objectMeasurements.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# c.soeller@exeter.ac.uk
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

import wx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def Warn(parent, message, caption = 'Warning!'):
    dlg = wx.MessageDialog(parent, message, caption, wx.OK | wx.ICON_WARNING)
    dlg.ShowModal()
    dlg.Destroy()

def subsampidx(arraylen, percentage=10):
    newlen = percentage*1e-2*arraylen
    idx = np.random.choice(arraylen,newlen)
    return idx

def scatterdens(x,y,subsample=1.0, s=40, **kwargs):
    xf = x.flatten()
    yf = y.flatten()
    if subsample < 1.0:
        idx = subsampidx(xf.size,percentage = 100*subsample)
        xs = xf[idx]
        ys = yf[idx]
    else:
        xs = xf
        ys = yf
        
    estimator = gaussian_kde([xs,ys]) 
    density = estimator.evaluate([xf,yf])
    print("density min, max: %f, %f" % (density.min(), density.max()))
    plt.scatter(xf,yf,c=density,marker='o',linewidth='0',zorder=3,s=s,**kwargs)
    return estimator

def multicolcheck(pipeline,subsample=0.03,dA=20,xrange=[-1000,3000],yrange=[-1000,6000]):
    p = pipeline
    plt.figure()
    plt.subplot(1, 2, 1)
    estimator = scatterdens(p['fitResults_Ag'],p['fitResults_Ar'],subsample=subsample,s=10)
    plt.xlim(xrange)
    plt.ylim(yrange)
    
    x1d = np.arange(xrange[0],xrange[1],dA)
    y1d = np.arange(yrange[0],yrange[1],dA)
    x2d = x1d[:,None] * np.ones_like(y1d)[None,:]
    y2d = np.ones_like(x1d)[:,None] * y1d[None,:]
    
    imd = estimator.evaluate([x2d.flatten(),y2d.flatten()])
    imd2d = imd.reshape(x2d.shape)
    imd2d /= imd2d.max()

    #plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(imd2d[:,::-1].transpose(),cmap=plt.get_cmap('jet'),extent=[xrange[0],xrange[1],yrange[0],yrange[1]])
    plt.grid(True)
    
    return imd2d

class MultiColAnalyzer:
    def __init__(self, visFr):
        self.visFr = visFr
        self.pipeline = self.visFr.pipeline

        visFr.AddMenuItem('Extras',  "Analyse Multicolour Events", self.OnMCA)

    def OnMCA(self, event):
        from PYME.DSView import dsviewer

        visFr = self.visFr
        pipeline = visFr.pipeline

        # we will have to see if checking for RuntimeError is good enough 
        try:
            Ag = pipeline['fitResults_Ag']
        except RuntimeError:
            Warn(None,'This does not appear to be a multi colour data set (no fitResults_Ag) - aborting')
        else:
            img = multicolcheck(pipeline)

def Plug(visFr):
    '''Plugs this module into the gui'''
    MultiColAnalyzer(visFr)

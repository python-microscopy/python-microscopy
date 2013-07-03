#!/usr/bin/python
##################
# photophysics.py
#
# Copyright David Baddeley, 2010
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

#import wx
#import numpy as np

class VibrationAnalyser:
    def __init__(self, visFr):
        self.visFr = visFr
        
        visFr.AddExtrasMenuItem('Plot vibration spectra', self.VibrationSpecgram)

    def VibrationSpecgram(self, event):
        import pylab as pl
        
        pipeline = self.visFr.pipeline
        
        x = pipeline['x']
        y = pipeline['y']
        
        x -= x.mean()
        y -= y.mean()
        
        pl.figure()
        
        pl.subplot(121)
        pl.specgram(x, clim=[0, 50])
       
        pl.subplot(122)
        pl.specgram(y, clim=[0, 50])
        
        pl.figure()
        pl.plot(x)
        pl.plot(y)


def Plug(visFr):
    '''Plugs this module into the gui'''
    VibrationAnalyser(visFr)




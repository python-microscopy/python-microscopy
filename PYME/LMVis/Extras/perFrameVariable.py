#!/usr/bin/python
##################
# perFrameVariable.py
#
# Copyright David Baddeley, 2014
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

import wx

class PerFrameVar:
    '''
    Allows a per frame measurement (e.g. Intensity / Channel assignment / etc ... 
    which is extracted from each frame seperately from the fitting process to be 
    included in the event data.
    
    The variable should be stored as a pickled numpy object (.npy) and should have
    the same number of entries as there are frames, as it will be indexed using the
    frame number to generate per event values.
    '''
    def __init__(self, visFr):
        self.visFr = visFr

        ID_FRAMEVAR = wx.NewId()
        visFr.extras_menu.Append(ID_FRAMEVAR, "Load per-frame variable")
        visFr.Bind(wx.EVT_MENU, self.OnPerFrame, id=ID_FRAMEVAR)

    def OnPerFrame(self, event):
        import numpy as np
        import os
        from PYME.FileUtils import nameUtils
        
        filename = wx.FileSelector("Choose a per frame variable to open", 
                                   nameUtils.genResultDirectoryPath(), 
                                   default_extension='npy', 
                                   wildcard='Saved numpy array (*.npy)|*.npy')

        #print filename
        if not filename == '':
            #open file
            var = np.load(filename).astype('f')
            
            dlg = wx.TextEntryDialog(self.visFr, 'What do you want the new variable to be called?', 'Enter variable name', os.path.split(filename)[-1])
            
            if dlg.ShowModal() == wx.ID_OK:
                varname = dlg.GetValue().encode()
                pipeline = self.visFr.pipeline
                
                #perform lookup            
                values = var[pipeline.inputMapping['t'].astype('i')]
                
                #print values
                
                
                #add looked up values to input mapping
                pipeline.inputMapping.__dict__[varname + '_'] = values
                pipeline.inputMapping.setMapping(varname, '%s_' % varname)
                
                print type('%s_' % varname)
                
                print pipeline.inputMapping.mappings
                
                #regenerate the pipeline
                self.visFr.RegenFilter()
                self.visFr.CreateFoldPanel()



def Plug(visFr):
    '''Plugs this module into the gui'''
    PerFrameVar(visFr)



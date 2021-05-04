#!/usr/bin/python

###############
# preflight.py
#
# Copyright David Baddeley, 2012
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
################

import logging
logger = logging.getLogger(__name__)

def ShowPreflightResults(failedChecks, preflight_mode='interactive', parent=None):
    if len(failedChecks) == 0:
        return True

    else:
        if preflight_mode == 'interactive':
            import wx
            #print failedChecks
            errormsgs = '\n'.join(['- ' + c.message for c in failedChecks])
            msg = 'Preflight check found the following potential problems:\n\n' + errormsgs + '\n\nDo you wish to continue?'
            dlg = wx.MessageDialog(parent, msg, 'Preflight Check:', wx.YES_NO|wx.NO_DEFAULT|wx.ICON_ERROR)
    
            ret = dlg.ShowModal()
            dlg.Destroy()
    
            return ret == wx.ID_YES
        elif preflight_mode == 'warn':
            import warnings
            for c in failedChecks:
                warnings.warn('PREFLIGHT FAILURE: ' + c.message)
                logger.warning('PREFLIGHT FAILURE: ' + c.message)
                
            return True
        else: # 'abort'
            for c in failedChecks:
                logger.error('PREFLIGHT FAILURE: ' + c.message)
                
            logger.error('Aborting series due to preflight failures')
        




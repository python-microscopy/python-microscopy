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

def ShowPreflightResults(parent, failedChecks):
    import wx
    if len(failedChecks) == 0:
        return True

    else:
        #print failedChecks
        errormsgs = '\n'.join(['- ' + c.message for c in failedChecks])
        msg = 'Preflight check found the following potential problems:\n\n' + errormsgs + '\n\nDo you wish to continue?'
        dlg = wx.MessageDialog(parent, msg, 'Preflight Check:', wx.YES_NO|wx.NO_DEFAULT|wx.ICON_ERROR)

        ret = dlg.ShowModal()
        dlg.Destroy()

        return ret == wx.ID_YES




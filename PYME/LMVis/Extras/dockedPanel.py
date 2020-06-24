#!/usr/bin/python

# dockedPanel.py
#
# Copyright Michael Graff
#   graff@hm.edu
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

import wx
import wx.lib.agw.aui as aui

# This is legacy-imported as afp since much of the code was written using
# PYME.ui.autoFoldPanel and when we switched to manual fold panel it was 
# easiest to change import PYME.ui.autoFoldPanel as afp to 
# import PYME.ui.manualFoldPanel as afp. Maintaining this convention here
# to make it easier to swap out the fold panel code again in the future 
# for something which is better behaved.
import PYME.ui.manualFoldPanel as afp


class DockedPanel(afp.foldingPane):


    def __init__(self, parent_panel, **kwargs):
        kwargs['style'] = wx.TAB_TRAVERSAL
        kwargs['pinned'] = True
        afp.foldingPane.__init__(self, parent_panel, **kwargs)

        self.parent_panel = parent_panel

    @staticmethod
    def show(vis_fr, panel, p_info_name, caption):
        frame_manager = vis_fr._mgr
        panel.SetSize(panel.GetBestSize())
        p_info = aui.AuiPaneInfo().Name(p_info_name).Right().Caption(caption).CloseButton(True).MinimizeButton(
            True).DestroyOnClose(True).Dock().MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART | aui.AUI_MINIMIZE_POS_RIGHT)
        frame_manager.AddPane(panel, p_info)
        # frame_manager.ShowPane(panel, True)
        frame_manager.Update()

    @staticmethod
    def add_menu_item(vis_fr, menu_name, panel_class, p_info_name, caption=None):
        if caption is None:
            caption = menu_name
        vis_fr.AddMenuItem('Extras', menu_name, lambda e: DockedPanel.show(vis_fr, panel_class(vis_fr), p_info_name, caption))

    def get_canvas(self):
        return self.parent_panel.glCanvas

def Plug(vis_fr):
    """
    Within sub-classes add a line like this:
    DockedPanel.add_menu_item(vis_fr, 'Video Panel',  VideoPanel, 'video_panel')

    """
    pass

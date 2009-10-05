#!/usr/bin/python

##################
# logview.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:FramePanel:wxPanel1

from wxPython.wx import *
from wxPython.gizmos import *

[wxID_WXPANEL1, wxID_WXPANEL1TREELISTCTRL1, 
] = map(lambda _init_ctrls: wxNewId(), range(2))

class wxPanel1(wxPanel):
    def _init_coll_treeListCtrl1_Columns(self, parent):
        # generated method, don't edit

        parent.AddColumn(text='Columns0')
        parent.AddColumn(text='Columns1')

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wxPanel.__init__(self, id=wxID_WXPANEL1, name='', parent=prnt,
              pos=wxPoint(329, 322), size=wxSize(617, 476),
              style=wxTAB_TRAVERSAL)
        self.SetClientSize(wxSize(609, 449))

        self.treeListCtrl1 = wxTreeListCtrl(id=wxID_WXPANEL1TREELISTCTRL1,
              name='treeListCtrl1', parent=self, pos=wxPoint(80, 40),
              size=wxSize(424, 312), style=wxTR_HAS_BUTTONS)
        self._init_coll_treeListCtrl1_Columns(self.treeListCtrl1)

    def __init__(self, parent, id, pos, size, style, name):
        self._init_ctrls(parent)

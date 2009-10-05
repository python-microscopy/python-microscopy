#!/usr/bin/python

##################
# draw2fr.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Frame:wxFrame1

from wxPython.wx import *
#from scipy import *


def create(parent):
    return wxFrame1(parent)

[wxID_WXFRAME1, wxID_WXFRAME1PANEL1, wxID_WXFRAME1PANEL2, 
 wxID_WXFRAME1STATICBITMAP1, wxID_WXFRAME1STATICTEXT1, 
] = map(lambda _init_ctrls: wxNewId(), range(5))

class wxFrame1(wxFrame):
    def _init_utils(self):
        # generated method, don't edit
        pass

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wxFrame.__init__(self, id=wxID_WXFRAME1, name='', parent=prnt,
              pos=wxPoint(275, 175), size=wxSize(960, 560),
              style=wxDEFAULT_FRAME_STYLE, title='wxFrame1')
        self._init_utils()
        self.SetClientSize(wxSize(952, 533))

        self.panel1 = wxPanel(id=wxID_WXFRAME1PANEL1, name='panel1',
              parent=self, pos=wxPoint(24, 16), size=wxSize(240, 224),
              style=wxTAB_TRAVERSAL)

        self.panel2 = wxPanel(id=wxID_WXFRAME1PANEL2, name='panel2',
              parent=self, pos=wxPoint(288, 16), size=wxSize(240, 224),
              style=wxTAB_TRAVERSAL)

        self.staticText1 = wxStaticText(id=wxID_WXFRAME1STATICTEXT1,
              label='staticText1', name='staticText1', parent=self,
              pos=wxPoint(24, 264), size=wxSize(240, 168), style=0)

        self.staticBitmap1 = wxStaticBitmap(bitmap=wxNullBitmap,
              id=wxID_WXFRAME1STATICBITMAP1, name='staticBitmap1', parent=self,
              pos=wxPoint(592, 88), size=wxSize(240, 208), style=0)

    def __init__(self, parent):
        self._init_ctrls(parent)
        
    def change_im(self,im):
        self.staticBitmap1.SetBitmap(im)
        

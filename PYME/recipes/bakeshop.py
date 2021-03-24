#!/bin/bash /Users/david/anaconda/bin/pythonw
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:06:17 2015

@author: david
"""
from PYME.misc import big_sur_fix
import os
os.environ['ETS_TOOLKIT'] = 'wx'

import matplotlib
matplotlib.use('WxAgg')

import wx
from PYME.recipes import recipeGui

class BakeshopApp(wx.App):        
    def OnInit(self):
        self.main = recipeGui.BatchFrame()
        self.main.Show()
        self.SetTopWindow(self.main)
        return True


def main():
    application = BakeshopApp(0)
    application.MainLoop()


if __name__ == '__main__':
    main()

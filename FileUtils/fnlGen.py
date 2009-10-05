#!/usr/bin/python

##################
# fnlGen.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import os
import wx

def genFnl(inDir):
    '''Generates a list of files in inDir with full paths'''
    if not (inDir[-1] == os.sep):
        inDir += os.sep #append a / to directroy name if necessary

    fnl = os.listdir(inDir)
    return [inDir + f for f in fnl]

def GuiGetDirList(promptString='Select directory containing image files', defaultPath = ''):
    inDir = wx.DirSelector(promptString, defaultPath)
    if inDir == '':
        return []
    else:
        return genFnl(inDir)

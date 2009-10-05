#!/usr/bin/python

##################
# PhasePreviewAquisator.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import PreviewAquisator

class PhasePreviewAquisator(PreviewAquisator):
    def __init__(self, _chans, _cam, _ds = None):
        PreviewAquisator.__init__(self, _chans, _cam, _ds)
#!/usr/bin/python
##################
# metadata.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from PYME.Analysis import MetadataTree

def Plug(dsviewer):
    mdv = MetadataTree.MetadataPanel(dsviewer, dsviewer.mdh)
    dsviewer.AddPage(page=mdv, select=False, caption='Metadata')

    dsviewer.mdv = mdv
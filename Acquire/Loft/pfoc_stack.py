#!/usr/bin/python

##################
# pfoc_stack.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

def aq_refr(source):
    dfr.update()

def aq_end(source):
    pa.start()

pa.stop()

sa = simplesequenceaquisator.SimpleSequenceAquisitor(chaninfo, cam, pfoc)
sa.SetStartMode(sa.START_AND_END)

#alter these lines as appropriate - no need to restart python
sa.SetStepSize(0.2)
sa.SetStartPos(40)
sa.SetEndPos(50)

sa.Prepare()

dfr = dsviewer.DSViewFrame(None, "New Aquisition", sa.ds)

sa.WantFrameNotification.append(aq_refr)

sa.Want

dfr.Show()

sa.start()


# to save the stack:
#
# sa.ds.SaveToFile('filename.kdf')
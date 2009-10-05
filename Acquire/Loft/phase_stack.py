#!/usr/bin/python

##################
# phase_stack.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

pa.stop()

sa = simplesequenceaquisator.SimpleSequenceAquisitor(chaninfo, cam, pphase)
sa.SetStartMode(sa.START_AND_END)

#alter these lines as appropriate - no need to restart python
sa.SetStepSize(1)
sa.SetStartPos(2)
sa.SetEndPos(6)

sa.Prepare()
sa.start()

# to view  the stack:
#
# shell.runfile('showstack.py')

# to save the stack:
#
# sa.ds.SaveToFile('filename.kdf')
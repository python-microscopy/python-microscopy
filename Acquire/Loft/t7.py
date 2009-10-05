#!/usr/bin/python

##################
# t7.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

pa.stop()
import piezo_e662
pfoc = piezo_e662.piezo_e662()
pfoc.initialise()

import simplesequenceaquisator
sa = simplesequenceaquisator.SimpleSequenceAquisitor(chaninfo, cam, pfoc)
sa.SetStartMode(sa.START_AND_END)
sa.SetStepSize(2)
sa.SetStartPos(50)
sa.SetEndPos(70)
sa.Prepare()
sa.start()
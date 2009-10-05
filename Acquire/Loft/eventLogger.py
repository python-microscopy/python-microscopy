#!/usr/bin/python

##################
# eventLogger.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

WantEventNotification = []

def logEvent(eventName, eventDescr = ''):
    for evl in WantEventNotification:
            evl.logEvent(eventName, eventDescr)
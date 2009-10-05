#!/usr/bin/python

##################
# statusLog.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import random

statusTexts = {}
statusKeys = []

statusDispFcn = None


class StatusLogger:
    def __init__(self, initText=''):
        global statusKeys
        self.key = random.random()
        statusKeys.append(self.key)
        self.setStatus(initText)

    def __del__(self):
        global statusTexts, statusKeys
        statusTexts.pop(self.key)
        statusKeys.remove(self.key)

    def setStatus(self, statusText):
        global statusTexts
        statusTexts[self.key] = statusText
        if not statusDispFcn == None:
            statusDispFcn(GenStatusText())

def GenStatusText():    
    st = '\t'.join([statusTexts[key] for key in statusKeys])

    return st

def SetStatusDispFcn(dispFcn):
    global statusDispFcn
    statusDispFcn = dispFcn

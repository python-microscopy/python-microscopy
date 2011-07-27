#!/usr/bin/python
##################
# computerName.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import os
import sys

def GetComputerName():
    if 'PYME_COMPUTERNAME' in os.environ.keys():
        return os.environ['PYME_COMPUTERNAME']
    elif sys.platform == 'win32':
        return os.environ['COMPUTERNAME']
    else:
        return os.uname()[1]


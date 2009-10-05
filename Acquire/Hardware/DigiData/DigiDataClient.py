#!/usr/bin/python

##################
# DigiDataClient.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Pyro.core

def getDDClient(): #for client machine
    return Pyro.core.getProxyForURI('PYRONAME://DigiData')

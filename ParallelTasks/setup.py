#!/usr/bin/python

##################
# setup.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from distutils.core import setup
try:
	import py2exe

	import matplotlib

	setup(console=['taskWorkerME.py'], data_files=matplotlib.get_py2exe_datafiles())
except:
	pass

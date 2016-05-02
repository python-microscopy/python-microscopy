'''
PYME.resources
==============

A storage area for non-code based resources.

'''
import os
dirname = os.path.dirname(__file__)

def getIconPath(name):
	'''Returns the full path to the icon with the given name'''	

	return os.path.join(dirname, 'icons', name)
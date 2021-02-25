"""
PYME.resources
==============

A storage area for non-code based resources.

"""
import os
dirname = os.path.dirname(__file__)

def getIconPath(name):
	"""Returns the full path to the icon with the given name"""

	return os.path.join(dirname, 'icons', name)

def get_web_static_dir():
	return os.path.join(dirname, 'web', 'static')

def get_web_dir():
	return os.path.join(dirname, 'web')

def get_test_data_dir():
	return os.path.join(dirname, 'test_datasets')
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


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('Analysis',parent_package,top_path)
    config.add_subpackage('cModels')
    config.add_subpackage('FitFactories')
    config.add_subpackage('QuadTree')
    config.add_subpackage('LMVis')
    config.add_subpackage('DataSources')
    config.add_subpackage('SoftRend')
    config.add_subpackage('DistHist')
    
    #config.make_svn_version_py()  # installs __svn_version__.py
    #config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(data_files = ['LMVis/dh5view.cmd'], **configuration(top_path='').todict())

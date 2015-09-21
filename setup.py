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
    config = Configuration('PYME',parent_package,top_path)
    config.add_subpackage('PYME')
    
    #config.make_svn_version_py()  # installs __svn_version__.py
    #config.make_config_py()
    config.get_version('PYME/version.py')
    return config

entry_points={
    'console_scripts': [
        'PYMEBatch = PYME.Analysis.Modules.batchProcess:main',
    ],
    'gui_scripts': [
        'dh5view = PYME.DSView.dsviewer_npy_nb:main',
        'PYMEAcquire = PYME.Acquire.PYMEAcquire:main',
        'VisGUI = PYME.Analysis.LMVis.VisGUI:main',
        'launchWorkers = PYME.ParallelTasks.launchWorkers:main',
        'bakeshop = PYME.Analysis.Modules.bakeshop:main',
    ]
}

if __name__ == '__main__':
    import setuptools
    from numpy.distutils.core import setup
    conf = configuration(top_path='').todict()
    conf['entry_points'] = entry_points
    setup(**conf)

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
import setuptools #to monkey-patch distutils for ms visualc for python
from PYME.misc import cython_numpy_monkey

#from PYME.setup import configuration

# def configuration(parent_package='',top_path=None):
#     from numpy.distutils.misc_util import Configuration
#     config = Configuration('PYME',parent_package,top_path)
#     config.add_subpackage('PYME')
#
#     #config.make_svn_version_py()  # installs __svn_version__.py
#     #config.make_config_py()
#     config.get_version('PYME/version.py')
#     return config

import yaml
import os
with open(os.path.join(os.path.dirname(__file__), 'conda-recipes', 'python-microscopy', 'entry_points.yaml')) as f:
    ep = yaml.safe_load(f)['entry_points']

entry_points={'console_scripts':ep}

# if __name__ == '__main__':
#     from numpy.distutils.core import setup
#     conf = configuration(top_path='').todict()
#     conf['entry_points'] = entry_points
#     setup(**conf)

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')
    
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        #quiet=True,
    )

    #config['entry_points'] = entry_points
    
    config.add_subpackage('PYME')
    config.get_version('PYME/version.py')
    return config

if __name__ == '__main__':
    import setuptools
    import os
    from distutils.command.sdist import sdist
    from numpy.distutils.core import setup


    with open('README.md', 'r') as f:
        long_description = f.read()

    setup(name='python-microscopy',
        description='Tools for (super-resolution) microscopy data analysis and microsope control',
        author='David Baddeley',
        author_email='david.baddeley@auckland.ac.nz',
        url='https://github.com/python-microscopy/python-microscopy',
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="GNU General Public License v3 (GPLv3)",
        install_requires=['numpy'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', # Again, pick a license
            'Programming Language :: Python :: 2.7', #Specify which python versions that you want to support
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        cmdclass={'sdist': sdist},
        entry_points=entry_points,
        configuration=configuration)

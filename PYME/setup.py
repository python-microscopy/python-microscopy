#!/usr/bin/python

##################
# setup.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################
import glob
import sys
import os
import setuptools #to monkey-patch distutils for ms visualc for python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('PYME',parent_package,top_path)
    config.add_subpackage('Analysis')
    config.add_subpackage('Acquire')
    config.add_subpackage('DSView')
    #config.add_subpackage('PSFGen')
    #config.add_subpackage('cSMI')
    config.add_subpackage('ParallelTasks')
    config.add_subpackage('IO')
    config.add_subpackage('Deconv')
    #config.add_subpackage('PSFEst')
    config.add_subpackage('simulation')
    config.add_subpackage('misc')
    config.add_subpackage('LMVis')
    config.add_subpackage('ui')
    config.add_subpackage('util')
    config.add_subpackage('util.shmarray')
    config.add_subpackage('util.mProfile')
    config.add_subpackage('util.fProfile')
    config.add_data_dir('util/fProfile/html')
    config.add_subpackage('localization')
    
    config.add_subpackage('recipes')
    config.add_data_dir('recipes/Recipes')
    
    config.add_subpackage('cluster')
    config.add_data_dir('cluster/clusterUI')
    
    #config.add_subpackage('pad')
    #config.add_subpackage('bcl')
    config.add_subpackage('resources')
    config.add_subpackage('contrib')
    config.add_subpackage('experimental')
    config.add_subpackage('tileviewer')
    config.add_data_dir('tileviewer/templates')
    #config.add_subpackage('dataBrowser')
    #config.add_subpackage('shmarray')
    #config.add_subpackage('SampleDB2')
    #config.add_subpackage('clusterUI')
    
    #config.add_scripts(glob.glob('scripts/*'))
    if False:#not 'CONDA_BUILD' in os.environ.keys():
        #entry points are defined in the condas meta.yaml - not needed here
        #if running under conda
        if sys.platform == 'win32':
            config.add_scripts('scripts/*')
        else:
            #don't add .cmd files
            config.add_scripts('scripts/*.py')
        
    config.get_version()
    
    #config.set_options()
    
    #config.make_svn_version_py()  # installs __svn_version__.py
    #config.make_config_py()
    return config

if __name__ == '__main__':
    #import setuptools
    from numpy.distutils.core import setup
    setup(description='Tools for (super-resolution) microscopy data analysis and microsope control',
        author='David Baddeley',
        author_email='david.baddeley@auckland.ac.nz',
        url='https://github.com/python-microscopy/python-microscopy',
        long_description='Tools for (super-resolution) microscopy data analysis and microsope control',
        long_description_content_type="text/markdown",
        license="GPLv3",
        install_requires=['numpy'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', # Again, pick a license
            'Programming Language :: Python :: 2.7', #Specify which python versions that you want to support
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
          **configuration(top_path='').todict())
    #setup(options = {'bdist_wininst':{'install-script=' : 'pyme_win_postinstall.py',
    #                    'user-access-control=' : 'force'}}, **configuration(top_path='').todict())

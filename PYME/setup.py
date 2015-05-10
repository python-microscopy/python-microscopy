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

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('PYME',parent_package,top_path)
    config.add_subpackage('Analysis')
    config.add_subpackage('Acquire')
    config.add_subpackage('DSView')
    config.add_subpackage('PSFGen')
    config.add_subpackage('cSMI')
    config.add_subpackage('ParallelTasks')
    config.add_subpackage('FileUtils')
    config.add_subpackage('Deconv')
    config.add_subpackage('PSFEst')
    config.add_subpackage('mProfile')
    config.add_subpackage('misc')
    config.add_subpackage('pad')
    config.add_subpackage('gohlke')
    config.add_subpackage('dataBrowser')
    config.add_subpackage('shmarray')
    config.add_subpackage('SampleDB2')
    
    #config.add_scripts(glob.glob('scripts/*'))
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
    setup(author='David Baddeley',
          description = 'PYthon (localisation) Microscopy Environment',
          **configuration(top_path='').todict())
    #setup(options = {'bdist_wininst':{'install-script=' : 'pyme_win_postinstall.py',
    #                    'user-access-control=' : 'force'}}, **configuration(top_path='').todict())

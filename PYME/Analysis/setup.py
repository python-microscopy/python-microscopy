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


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('Analysis',parent_package,top_path)
    #config.add_subpackage('cModels')
    #config.add_subpackage('cInterp')
    #config.add_subpackage('FitFactories')
    #config.add_subpackage('FitFactories.Interpolators')
    #config.add_subpackage('FitFactories.zEstimators')
    #config.add_subpackage('QuadTree')
    #config.add_subpackage('LMVis')
    #config.add_subpackage('DataSources')
    #config.add_subpackage('SoftRend')
    #config.add_subpackage('DistHist')
    #config.add_subpackage('DeClump')
    #config.add_subpackage('EdgeDB')
    #config.add_subpackage('qHull')
    config.add_subpackage('points')
    config.add_subpackage('BleachProfile')
    config.add_subpackage('Colocalisation')
    #config.add_subpackage('Modules')
    config.add_subpackage('Tracking')
    
    config.add_subpackage('PSFEst')
    config.add_subpackage('PSFGen')
    
    #config.add_data_dir('Modules/Recipes')

    #config.add_scripts(['LMVis/VisGUI.py', 'LMVis/VisGUI.cmd'])
    
    #config.make_svn_version_py()  # installs __svn_version__.py
    #config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(data_files = [], **configuration(top_path='').todict())
